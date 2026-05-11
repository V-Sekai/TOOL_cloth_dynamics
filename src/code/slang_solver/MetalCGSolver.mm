// MetalCGSolver implementation. Loads six .metallib files
// (spmv, saxpby_indirect, dot_reduce, cg_alpha, cg_beta, plus an
// optional saxpby fallback), creates one MTLComputePipelineState
// each, and runs a GPU-fused CG loop.
//
// Batching strategy: K CG iterations are encoded into ONE Metal
// command buffer with NO CPU intervention between dispatches. α and
// β are computed by the cg_alpha/cg_beta kernels writing to scalar
// buffers that saxpby_indirect reads from on the next dispatch.
// commit + waitUntilCompleted is paid once per K iters; convergence
// is checked by reading dotOldBuf (which holds the most recent
// delta_new after cg_beta's in-place copy) at the end of each batch.
//
// Per-iter encoded sequence (8 dispatches, all in one CB):
//
//   spmv               q = A·p
//   dot_reduce         dotPQ = dot(p, q)
//   cg_alpha           alphaArr = [+α, −α]
//   saxpby_indirect    x = 1·x + α·p       (alpha=ones, beta=alphaArr[0])
//   saxpby_indirect    r = 1·r + (−α)·q    (alpha=ones, beta=alphaArr[1])
//   dot_reduce         dotNew = dot(r, r)
//   cg_beta            betaBuf = β; dotOld := dotNew  (in place)
//   saxpby_indirect    p = 1·r + β·p       (alpha=ones, beta=betaBuf)
//
// dotOld is initialised to dot_reduce(r₀,r₀) before the first batch
// and is updated in-place by cg_beta every iter — no CPU ping-pong.

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "MetalCGSolver.h"

#include <algorithm>
#include <cassert>
#include <cstdio>
#include <cstring>
#include <string>

namespace cloth {

struct MetalCGSolver::Impl {
    id<MTLDevice>               device   = nil;
    id<MTLCommandQueue>         queue    = nil;
    id<MTLComputePipelineState> psoSpmv      = nil;
    id<MTLComputePipelineState> psoSaxIndir  = nil;   // saxpby_indirect
    id<MTLComputePipelineState> psoDot       = nil;   // parallel dot_reduce (256 threads)
    id<MTLComputePipelineState> psoCGAlpha   = nil;
    id<MTLComputePipelineState> psoCGBeta    = nil;

    // CSR matrix (uploaded once).
    id<MTLBuffer> bufRowPtr = nil;
    id<MTLBuffer> bufColIdx = nil;
    id<MTLBuffer> bufVals   = nil;
    uint32_t      rows      = 0;
    uint32_t      nnz       = 0;

    // CG workspace (allocated once, length = rows).
    id<MTLBuffer> bufX = nil;
    id<MTLBuffer> bufR = nil;
    id<MTLBuffer> bufP = nil;
    id<MTLBuffer> bufQ = nil;
    id<MTLBuffer> bufB = nil;

    // Scalar workspace buffers.
    id<MTLBuffer> bufDotPQ   = nil;   // 2 floats: dot(p, q) df32 hi/lo
    id<MTLBuffer> bufDotNew  = nil;   // 2 floats: dot(r, r) df32 hi/lo
    id<MTLBuffer> bufDotOld  = nil;   // 2 floats: previous iter's dot(r, r)
    id<MTLBuffer> bufAlpha2  = nil;   // 2 floats: [+α, −α] from cg_alpha
    id<MTLBuffer> bufBeta    = nil;   // 1 float : β from cg_beta
    id<MTLBuffer> bufOnes    = nil;   // 1 float : 1.0 (saxpby_indirect's α'=1)

    // Per-kernel parameter buffers.
    id<MTLBuffer> bufSpmvParams      = nil;   // { uint rows; }
    id<MTLBuffer> bufSaxIndirParams  = nil;   // { uint n; }
    id<MTLBuffer> bufDotParams       = nil;   // { uint n; }

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

    // Encode one CG iteration into the given encoder. No commit; the
    // caller batches K iters before commit + wait.
    void encodeOneIter(id<MTLComputeCommandEncoder> enc) {
        // 1. q = A · p
        [enc setComputePipelineState:psoSpmv];
        [enc setBuffer:bufSpmvParams offset:0 atIndex:0];
        [enc setBuffer:bufRowPtr     offset:0 atIndex:1];
        [enc setBuffer:bufColIdx     offset:0 atIndex:2];
        [enc setBuffer:bufVals       offset:0 atIndex:3];
        [enc setBuffer:bufP          offset:0 atIndex:4];
        [enc setBuffer:bufQ          offset:0 atIndex:5];
        [enc dispatchThreads:MTLSizeMake(rows, 1, 1)
       threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];

        // 2. dotPQ = dot(p, q)   (parallel df32 reduce)
        [enc setComputePipelineState:psoDot];
        [enc setBuffer:bufDotParams offset:0 atIndex:0];
        [enc setBuffer:bufP         offset:0 atIndex:1];
        [enc setBuffer:bufQ         offset:0 atIndex:2];
        [enc setBuffer:bufDotPQ     offset:0 atIndex:3];
        [enc dispatchThreads:MTLSizeMake(256, 1, 1)
       threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];

        // 3. cg_alpha: alphaArr = [+α, −α]
        [enc setComputePipelineState:psoCGAlpha];
        [enc setBuffer:bufDotPQ  offset:0 atIndex:0];
        [enc setBuffer:bufDotNew offset:0 atIndex:1];
        [enc setBuffer:bufAlpha2 offset:0 atIndex:2];
        [enc dispatchThreads:MTLSizeMake(1, 1, 1)
       threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];

        // 4. x = 1·x + α·p       (saxpby_indirect, beta = alphaArr[0] = +α)
        [enc setComputePipelineState:psoSaxIndir];
        [enc setBuffer:bufSaxIndirParams offset:0       atIndex:0];
        [enc setBuffer:bufOnes           offset:0       atIndex:1];   // alpha
        [enc setBuffer:bufAlpha2         offset:0       atIndex:2];   // beta = +α at byte 0
        [enc setBuffer:bufX              offset:0       atIndex:3];
        [enc setBuffer:bufP              offset:0       atIndex:4];
        [enc setBuffer:bufX              offset:0       atIndex:5];   // dst = x (in-place)
        [enc dispatchThreads:MTLSizeMake(rows, 1, 1)
       threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];

        // 5. r = 1·r + (−α)·q    (saxpby_indirect, beta = alphaArr[1] = −α at offset 4)
        [enc setComputePipelineState:psoSaxIndir];
        [enc setBuffer:bufSaxIndirParams offset:0                    atIndex:0];
        [enc setBuffer:bufOnes           offset:0                    atIndex:1];
        [enc setBuffer:bufAlpha2         offset:sizeof(float)        atIndex:2];   // beta = −α
        [enc setBuffer:bufR              offset:0                    atIndex:3];
        [enc setBuffer:bufQ              offset:0                    atIndex:4];
        [enc setBuffer:bufR              offset:0                    atIndex:5];   // dst = r (in-place)
        [enc dispatchThreads:MTLSizeMake(rows, 1, 1)
       threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];

        // 6. dotNew = dot(r, r)
        [enc setComputePipelineState:psoDot];
        [enc setBuffer:bufDotParams offset:0 atIndex:0];
        [enc setBuffer:bufR         offset:0 atIndex:1];
        [enc setBuffer:bufR         offset:0 atIndex:2];
        [enc setBuffer:bufDotNew    offset:0 atIndex:3];
        [enc dispatchThreads:MTLSizeMake(256, 1, 1)
       threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];

        // 7. cg_beta: betaBuf = β; dotOld := dotNew  (in place)
        [enc setComputePipelineState:psoCGBeta];
        [enc setBuffer:bufDotNew offset:0 atIndex:0];
        [enc setBuffer:bufDotOld offset:0 atIndex:1];
        [enc setBuffer:bufBeta   offset:0 atIndex:2];
        [enc dispatchThreads:MTLSizeMake(1, 1, 1)
       threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];

        // 8. p = 1·r + β·p       (saxpby_indirect, beta = betaBuf)
        [enc setComputePipelineState:psoSaxIndir];
        [enc setBuffer:bufSaxIndirParams offset:0 atIndex:0];
        [enc setBuffer:bufOnes           offset:0 atIndex:1];
        [enc setBuffer:bufBeta           offset:0 atIndex:2];
        [enc setBuffer:bufR              offset:0 atIndex:3];
        [enc setBuffer:bufP              offset:0 atIndex:4];
        [enc setBuffer:bufP              offset:0 atIndex:5];
        [enc dispatchThreads:MTLSizeMake(rows, 1, 1)
       threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    }
};

namespace {
struct SpmvParams         { uint32_t rows; };
struct SaxIndirParams     { uint32_t n; };
struct DotParams          { uint32_t n; };
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
    id<MTLLibrary> libSpmv  = Impl::loadLib(impl_->device, root + "spmv.metallib",            "spmv");
    id<MTLLibrary> libSaxI  = Impl::loadLib(impl_->device, root + "saxpby_indirect.metallib", "saxpby_indirect");
    id<MTLLibrary> libDot   = Impl::loadLib(impl_->device, root + "dot_reduce.metallib",      "dot_reduce");
    id<MTLLibrary> libAlpha = Impl::loadLib(impl_->device, root + "cg_alpha.metallib",        "cg_alpha");
    id<MTLLibrary> libBeta  = Impl::loadLib(impl_->device, root + "cg_beta.metallib",         "cg_beta");
    if (!libSpmv || !libSaxI || !libDot || !libAlpha || !libBeta) return;

    impl_->psoSpmv      = Impl::makePSO(impl_->device, libSpmv,  "main_0", "spmv");
    impl_->psoSaxIndir  = Impl::makePSO(impl_->device, libSaxI,  "main_0", "saxpby_indirect");
    impl_->psoDot       = Impl::makePSO(impl_->device, libDot,   "main_0", "dot_reduce");
    impl_->psoCGAlpha   = Impl::makePSO(impl_->device, libAlpha, "main_0", "cg_alpha");
    impl_->psoCGBeta    = Impl::makePSO(impl_->device, libBeta,  "main_0", "cg_beta");
    if (!impl_->psoSpmv || !impl_->psoSaxIndir || !impl_->psoDot ||
        !impl_->psoCGAlpha || !impl_->psoCGBeta) return;

    impl_->rows = A.rows;
    impl_->nnz  = uint32_t(A.colIdx.size());
    const uint32_t N    = A.rows;
    const NSUInteger bytesRow = (N + 1) * sizeof(int32_t);
    const NSUInteger bytesNnz = impl_->nnz * sizeof(int32_t);
    const NSUInteger bytesVal = impl_->nnz * sizeof(float);
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

    impl_->bufDotPQ  = [impl_->device newBufferWithLength:2 * sizeof(float) options:MTLResourceStorageModeShared];
    impl_->bufDotNew = [impl_->device newBufferWithLength:2 * sizeof(float) options:MTLResourceStorageModeShared];
    impl_->bufDotOld = [impl_->device newBufferWithLength:2 * sizeof(float) options:MTLResourceStorageModeShared];
    impl_->bufAlpha2 = [impl_->device newBufferWithLength:2 * sizeof(float) options:MTLResourceStorageModeShared];
    impl_->bufBeta   = [impl_->device newBufferWithLength:1 * sizeof(float) options:MTLResourceStorageModeShared];
    impl_->bufOnes   = [impl_->device newBufferWithLength:1 * sizeof(float) options:MTLResourceStorageModeShared];
    *static_cast<float*>(impl_->bufOnes.contents) = 1.0f;

    impl_->bufSpmvParams = [impl_->device newBufferWithLength:sizeof(SpmvParams)
                                                      options:MTLResourceStorageModeShared];
    impl_->bufSaxIndirParams = [impl_->device newBufferWithLength:sizeof(SaxIndirParams)
                                                          options:MTLResourceStorageModeShared];
    impl_->bufDotParams = [impl_->device newBufferWithLength:sizeof(DotParams)
                                                     options:MTLResourceStorageModeShared];
    *static_cast<SpmvParams*>(impl_->bufSpmvParams.contents)         = {N};
    *static_cast<SaxIndirParams*>(impl_->bufSaxIndirParams.contents) = {N};
    *static_cast<DotParams*>(impl_->bufDotParams.contents)           = {N};

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

    // [experiment] MOCK_SOLVE=1 short-circuits the entire Metal solve
    // path. Returns x = 0 without touching any Metal buffer or
    // dispatching any kernel. Used to test whether the 7-23x
    // non-solve phase slowdown is caused by Metal's memory touches
    // (unified-memory contention hypothesis) or something else.
    // The simulation will be physically wrong but step 20's
    // BENCH_PHASE_AT output gives the clean isolation.
    static const bool s_mockSolve = (std::getenv("MOCK_SOLVE") != nullptr);
    if (s_mockSolve) {
        x.assign(N, 0.0);
        return 0;
    }

    // ---- Upload b, init x = 0, r = b, p = b -------------------------------
    {
        float* bptr = static_cast<float*>(impl_->bufB.contents);
        float* xptr = static_cast<float*>(impl_->bufX.contents);
        float* rptr = static_cast<float*>(impl_->bufR.contents);
        float* pptr = static_cast<float*>(impl_->bufP.contents);
        for (uint32_t i = 0; i < N; ++i) {
            const float v = float(b[i]);
            bptr[i] = v;
            xptr[i] = 0.0f;
            rptr[i] = v;
            pptr[i] = v;
        }
    }

    // ---- Initial dot_new = dot(r, r); copy to dotOld for first iter -------
    double delta_0 = 0.0;
    {
        id<MTLCommandBuffer> cb = [impl_->queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:impl_->psoDot];
        [enc setBuffer:impl_->bufDotParams offset:0 atIndex:0];
        [enc setBuffer:impl_->bufR         offset:0 atIndex:1];
        [enc setBuffer:impl_->bufR         offset:0 atIndex:2];
        [enc setBuffer:impl_->bufDotNew    offset:0 atIndex:3];
        [enc dispatchThreads:MTLSizeMake(256, 1, 1)
       threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        [enc endEncoding];
        [cb commit];
        [cb waitUntilCompleted];

        const float* dn = static_cast<const float*>(impl_->bufDotNew.contents);
        float* dold = static_cast<float*>(impl_->bufDotOld.contents);
        dold[0] = dn[0]; dold[1] = dn[1];

        delta_0 = double(dn[0]) + double(dn[1]);
        if (delta_0 == 0.0) {
            x.assign(N, 0.0);
            return 0;
        }
    }

    // ---- Batched CG loop: K iters per command buffer ----------------------
    const int K = 30;
    const double convTol2 = tol * tol * delta_0;
    int iter = 0;

    while (iter < max_iter) {
        const int thisBatch = std::min(K, max_iter - iter);

        id<MTLCommandBuffer> cb = [impl_->queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        for (int j = 0; j < thisBatch; ++j) impl_->encodeOneIter(enc);
        [enc endEncoding];
        [cb commit];
        [cb waitUntilCompleted];
        iter += thisBatch;

        // After the batch, dotOld holds the most recent delta_new
        // (cg_beta's in-place copy on the last iter of the batch).
        const float* dold = static_cast<const float*>(impl_->bufDotOld.contents);
        const double delta_new = double(dold[0]) + double(dold[1]);
        if (delta_new < convTol2) break;
    }

    x.resize(N);
    const float* xptr = static_cast<const float*>(impl_->bufX.contents);
    for (uint32_t i = 0; i < N; ++i) x[i] = double(xptr[i]);
    return iter;
}

}  // namespace cloth
