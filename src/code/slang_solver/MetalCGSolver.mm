// MetalCGSolver implementation. Loads three .metallib files
// (spmv, saxpby, dot_reduce — the parallel df32 reduce), creates
// MTLComputePipelineState for each, and runs a CG loop on the GPU.
//
// CG layout (Shewchuk § B.2), all on-device buffers:
//   r₀ = b − A·x₀          spmv into a temp, then saxpby
//   p  = r₀
//   δ_new = dot(r, r)       dot_reduce reads to a tiny shared buffer
//   loop:
//     q     = A·p           spmv
//     pq    = dot(p, q)     dot_reduce
//     α     = δ_new / pq    on the host (sync after dot_reduce)
//     x    += α·p           saxpby
//     r    -= α·q           saxpby
//     δ_old = δ_new
//     δ_new = dot(r, r)     dot_reduce
//     if δ_new < tol²·δ₀ break
//     β     = δ_new / δ_old
//     p     = r + β·p       saxpby

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "MetalCGSolver.h"

#include <cassert>
#include <cstdio>
#include <string>

namespace cloth {

struct MetalCGSolver::Impl {
    id<MTLDevice>               device   = nil;
    id<MTLCommandQueue>         queue    = nil;
    id<MTLComputePipelineState> psoSpmv  = nil;
    id<MTLComputePipelineState> psoSax   = nil;
    id<MTLComputePipelineState> psoDot   = nil;

    // CSR matrix (uploaded once).
    id<MTLBuffer> bufRowPtr = nil;
    id<MTLBuffer> bufColIdx = nil;
    id<MTLBuffer> bufVals   = nil;
    uint32_t      rows      = 0;

    // CG workspace (allocated once, length = rows).
    id<MTLBuffer> bufX      = nil;     // solution
    id<MTLBuffer> bufR      = nil;     // residual
    id<MTLBuffer> bufP      = nil;     // search direction
    id<MTLBuffer> bufQ      = nil;     // A·p workspace
    id<MTLBuffer> bufB      = nil;     // RHS (uploaded per solve)
    id<MTLBuffer> bufDotDst = nil;     // dot_reduce (hi, lo) sink, length 2

    // Per-kernel parameter buffers (rebound per dispatch).
    id<MTLBuffer> bufSpmvParams = nil;   // { uint rows; }
    id<MTLBuffer> bufSaxParams  = nil;   // { uint n; float alpha; float beta; }
    id<MTLBuffer> bufDotParams  = nil;   // { uint n; }

    bool ok = false;

    static id<MTLLibrary> loadLib(id<MTLDevice> dev, const std::string& path,
                                   const char* name) {
        NSError* err = nil;
        NSURL* url = [NSURL fileURLWithPath:[NSString stringWithUTF8String:path.c_str()]];
        id<MTLLibrary> lib = [dev newLibraryWithURL:url error:&err];
        if (!lib) {
            std::fprintf(stderr, "MetalCGSolver: failed to load %s from %s: %s\n",
                         name, path.c_str(),
                         err ? err.localizedDescription.UTF8String : "(null)");
        }
        return lib;
    }

    static id<MTLComputePipelineState> makePSO(id<MTLDevice> dev,
                                                 id<MTLLibrary> lib,
                                                 const char* fn,
                                                 const char* tag) {
        NSError* err = nil;
        id<MTLFunction> f = [lib newFunctionWithName:[NSString stringWithUTF8String:fn]];
        if (!f) {
            std::fprintf(stderr, "MetalCGSolver: no fn %s in %s\n", fn, tag);
            return nil;
        }
        id<MTLComputePipelineState> pso =
            [dev newComputePipelineStateWithFunction:f error:&err];
        if (!pso) {
            std::fprintf(stderr, "MetalCGSolver: PSO %s failed: %s\n",
                         tag, err ? err.localizedDescription.UTF8String : "(null)");
        }
        return pso;
    }
};

namespace {
struct SpmvParams { uint32_t rows; };
struct SaxParams  { uint32_t n; float alpha; float beta; };
struct DotParams  { uint32_t n; };
}

MetalCGSolver::MetalCGSolver(const CSRSpMatF& A, const char* metallibPath)
    : impl_(new Impl()) {
    impl_->device = MTLCreateSystemDefaultDevice();
    if (!impl_->device) {
        std::fprintf(stderr, "MetalCGSolver: no Metal device\n");
        return;
    }
    impl_->queue = [impl_->device newCommandQueue];

    std::string root = metallibPath;
    if (!root.empty() && root.back() != '/') root.push_back('/');
    id<MTLLibrary> libSpmv = Impl::loadLib(impl_->device, root + "spmv.metallib",       "spmv");
    id<MTLLibrary> libSax  = Impl::loadLib(impl_->device, root + "saxpby.metallib",     "saxpby");
    id<MTLLibrary> libDot  = Impl::loadLib(impl_->device, root + "dot_reduce.metallib", "dot_reduce");
    if (!libSpmv || !libSax || !libDot) return;

    impl_->psoSpmv = Impl::makePSO(impl_->device, libSpmv, "main_0", "spmv");
    impl_->psoSax  = Impl::makePSO(impl_->device, libSax,  "main_0", "saxpby");
    impl_->psoDot  = Impl::makePSO(impl_->device, libDot,  "main_0", "dot_reduce");
    if (!impl_->psoSpmv || !impl_->psoSax || !impl_->psoDot) return;

    impl_->rows = A.rows;
    const uint32_t N    = A.rows;
    const uint32_t nnz  = uint32_t(A.colIdx.size());
    const NSUInteger bytesRow = (N + 1) * sizeof(int32_t);
    const NSUInteger bytesNnz = nnz * sizeof(int32_t);
    const NSUInteger bytesVal = nnz * sizeof(float);
    const NSUInteger bytesVec = N * sizeof(float);

    impl_->bufRowPtr = [impl_->device newBufferWithBytes:A.rowPtr.data()
                                                  length:bytesRow
                                                 options:MTLResourceStorageModeShared];
    impl_->bufColIdx = [impl_->device newBufferWithBytes:A.colIdx.data()
                                                  length:bytesNnz
                                                 options:MTLResourceStorageModeShared];
    impl_->bufVals   = [impl_->device newBufferWithBytes:A.values.data()
                                                  length:bytesVal
                                                 options:MTLResourceStorageModeShared];

    impl_->bufX = [impl_->device newBufferWithLength:bytesVec options:MTLResourceStorageModeShared];
    impl_->bufR = [impl_->device newBufferWithLength:bytesVec options:MTLResourceStorageModeShared];
    impl_->bufP = [impl_->device newBufferWithLength:bytesVec options:MTLResourceStorageModeShared];
    impl_->bufQ = [impl_->device newBufferWithLength:bytesVec options:MTLResourceStorageModeShared];
    impl_->bufB = [impl_->device newBufferWithLength:bytesVec options:MTLResourceStorageModeShared];
    impl_->bufDotDst = [impl_->device newBufferWithLength:2 * sizeof(float)
                                                  options:MTLResourceStorageModeShared];

    impl_->bufSpmvParams = [impl_->device newBufferWithLength:sizeof(SpmvParams)
                                                      options:MTLResourceStorageModeShared];
    impl_->bufSaxParams  = [impl_->device newBufferWithLength:sizeof(SaxParams)
                                                      options:MTLResourceStorageModeShared];
    impl_->bufDotParams  = [impl_->device newBufferWithLength:sizeof(DotParams)
                                                      options:MTLResourceStorageModeShared];
    *static_cast<SpmvParams*>(impl_->bufSpmvParams.contents) = {N};
    *static_cast<DotParams*>(impl_->bufDotParams.contents)   = {N};

    impl_->ok = true;
}

MetalCGSolver::~MetalCGSolver() { delete impl_; }

bool MetalCGSolver::ok() const { return impl_ && impl_->ok; }

int MetalCGSolver::solve(const std::vector<double>& b,
                         std::vector<double>& x,
                         double tol, int max_iter) {
    if (!ok()) return -1;
    const uint32_t N = impl_->rows;
    if (b.size() != N) {
        std::fprintf(stderr, "MetalCGSolver: b.size()=%zu != rows=%u\n",
                     b.size(), N);
        return -1;
    }

    // ---- Upload b (fp64 → fp32) -------------------------------------------
    {
        float* bptr = static_cast<float*>(impl_->bufB.contents);
        for (uint32_t i = 0; i < N; ++i) bptr[i] = float(b[i]);
    }
    // Initial guess x₀ = 0; r₀ = b − A·0 = b; p₀ = r₀.
    {
        float* xptr = static_cast<float*>(impl_->bufX.contents);
        float* rptr = static_cast<float*>(impl_->bufR.contents);
        float* pptr = static_cast<float*>(impl_->bufP.contents);
        float* bptr = static_cast<float*>(impl_->bufB.contents);
        for (uint32_t i = 0; i < N; ++i) {
            xptr[i] = 0.0f;
            rptr[i] = bptr[i];
            pptr[i] = bptr[i];
        }
    }

    auto encDispatch = [&](id<MTLComputeCommandEncoder> enc,
                           id<MTLComputePipelineState> pso,
                           NSArray* buffers,
                           MTLSize grid, MTLSize tg) {
        [enc setComputePipelineState:pso];
        for (NSUInteger i = 0; i < buffers.count; ++i) {
            [enc setBuffer:buffers[i] offset:0 atIndex:i];
        }
        [enc dispatchThreads:grid threadsPerThreadgroup:tg];
    };

    auto dispatchDot = [&](id<MTLBuffer> a, id<MTLBuffer> bb) -> double {
        id<MTLCommandBuffer> cb = [impl_->queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        encDispatch(enc, impl_->psoDot,
                    @[impl_->bufDotParams, a, bb, impl_->bufDotDst],
                    MTLSizeMake(256, 1, 1), MTLSizeMake(256, 1, 1));
        [enc endEncoding];
        [cb commit];
        [cb waitUntilCompleted];
        const float* dst = static_cast<const float*>(impl_->bufDotDst.contents);
        return double(dst[0]) + double(dst[1]);
    };

    double delta_new = dispatchDot(impl_->bufR, impl_->bufR);
    const double delta_0 = delta_new;
    if (delta_0 == 0.0) {
        // b ≡ 0 → x ≡ 0 already satisfies A·x = b.
        x.assign(N, 0.0);
        return 0;
    }

    int iter = 0;
    SaxParams* spy = static_cast<SaxParams*>(impl_->bufSaxParams.contents);
    spy->n = N;

    for (; iter < max_iter; ++iter) {
        // q = A · p
        {
            id<MTLCommandBuffer> cb = [impl_->queue commandBuffer];
            id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
            encDispatch(enc, impl_->psoSpmv,
                        @[impl_->bufSpmvParams, impl_->bufRowPtr,
                          impl_->bufColIdx, impl_->bufVals,
                          impl_->bufP, impl_->bufQ],
                        MTLSizeMake(N, 1, 1), MTLSizeMake(256, 1, 1));
            [enc endEncoding];
            [cb commit];
            [cb waitUntilCompleted];
        }

        const double pq    = dispatchDot(impl_->bufP, impl_->bufQ);
        const double alpha = delta_new / pq;

        // x := x + α·p    (saxpby with α'=1, β'=α, src x, src p, dst x)
        spy->alpha = 1.0f; spy->beta = float(alpha);
        {
            id<MTLCommandBuffer> cb = [impl_->queue commandBuffer];
            id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
            encDispatch(enc, impl_->psoSax,
                        @[impl_->bufSaxParams, impl_->bufX, impl_->bufP, impl_->bufX],
                        MTLSizeMake(N, 1, 1), MTLSizeMake(256, 1, 1));
            [enc endEncoding];
            [cb commit];
            [cb waitUntilCompleted];
        }
        // r := r − α·q
        spy->alpha = 1.0f; spy->beta = float(-alpha);
        {
            id<MTLCommandBuffer> cb = [impl_->queue commandBuffer];
            id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
            encDispatch(enc, impl_->psoSax,
                        @[impl_->bufSaxParams, impl_->bufR, impl_->bufQ, impl_->bufR],
                        MTLSizeMake(N, 1, 1), MTLSizeMake(256, 1, 1));
            [enc endEncoding];
            [cb commit];
            [cb waitUntilCompleted];
        }

        const double delta_old = delta_new;
        delta_new = dispatchDot(impl_->bufR, impl_->bufR);
        if (delta_new < tol * tol * delta_0) { ++iter; break; }
        const double beta = delta_new / delta_old;

        // p := r + β·p   (saxpby with α'=1, β'=β, src r, src p, dst p)
        spy->alpha = 1.0f; spy->beta = float(beta);
        {
            id<MTLCommandBuffer> cb = [impl_->queue commandBuffer];
            id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
            encDispatch(enc, impl_->psoSax,
                        @[impl_->bufSaxParams, impl_->bufR, impl_->bufP, impl_->bufP],
                        MTLSizeMake(N, 1, 1), MTLSizeMake(256, 1, 1));
            [enc endEncoding];
            [cb commit];
            [cb waitUntilCompleted];
        }
    }

    // Read back x (fp32 → fp64).
    x.resize(N);
    const float* xptr = static_cast<const float*>(impl_->bufX.contents);
    for (uint32_t i = 0; i < N; ++i) x[i] = double(xptr[i]);
    return iter;
}

}  // namespace cloth
