// AvbdSolver implementation (PR-E stub). Loads the six AVBD kernel
// metallibs and builds compute pipeline state objects. Per-step
// dispatch is a stub — actual GPU loop wiring lands in follow-up PRs.

#include "AvbdSolver.h"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <cstdio>
#include <cstring>
#include <string>

namespace cloth {

// Matches the VbdInitParams struct emitted by Cloth.SlangCodegen.VbdInit
// (single float field `invHSquared`).
struct VbdInitParams {
    float invHSquared;
};

struct AvbdSolver::Impl {
    id<MTLDevice>               device   = nil;
    id<MTLCommandQueue>         queue    = nil;

    // Six AVBD pipeline kernels. Each loaded from a separate
    // metallib at construction time.
    id<MTLComputePipelineState> psoInit             = nil;
    id<MTLComputePipelineState> psoGatherSpring     = nil;
    id<MTLComputePipelineState> psoGatherAttachment = nil;
    id<MTLComputePipelineState> psoGatherTriangle   = nil;
    id<MTLComputePipelineState> psoGatherBending    = nil;
    id<MTLComputePipelineState> psoSolveApply       = nil;

    // Per-constraint-type force kernels (compute force/Hess from
    // current positions before the gather kernel scatters them
    // per-vertex). PR-E loads spring_force first; others follow.
    id<MTLComputePipelineState> psoSpringForce      = nil;

    // Per-vertex state buffers (allocated by setupMesh).
    id<MTLBuffer> bufPositions = nil;   // length = 3 * nVerts (float)
    id<MTLBuffer> bufPredicted = nil;   // length = 3 * nVerts (float)
    id<MTLBuffer> bufMass      = nil;   // length = nVerts (float)
    id<MTLBuffer> bufGScratch  = nil;   // length = 3 * nVerts (float)
    id<MTLBuffer> bufHScratch  = nil;   // length = 6 * nVerts (float)
    id<MTLBuffer> bufInitParams = nil;  // sizeof(VbdInitParams)

    // Spring constraint buffers (allocated by uploadSprings).
    id<MTLBuffer> bufSpringP1Idx     = nil;  // uint per spring
    id<MTLBuffer> bufSpringP2Idx     = nil;  // uint per spring
    id<MTLBuffer> bufSpringRestLen   = nil;  // float per spring
    id<MTLBuffer> bufSpringStiffness = nil;  // float per spring
    id<MTLBuffer> bufSpringGradA     = nil;  // 3 floats per spring
    id<MTLBuffer> bufSpringHess      = nil;  // 6 floats per spring

    uint32_t nVerts   = 0;
    uint32_t nSprings = 0;

    bool ok = false;
    bool meshReady = false;

    static id<MTLLibrary> loadLib(id<MTLDevice> dev, const std::string& path,
                                   const char* name) {
        NSError* err = nil;
        NSURL* url = [NSURL fileURLWithPath:[NSString stringWithUTF8String:path.c_str()]];
        id<MTLLibrary> lib = [dev newLibraryWithURL:url error:&err];
        if (!lib) {
            std::fprintf(stderr, "AvbdSolver: failed to load %s from %s: %s\n",
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
            std::fprintf(stderr, "AvbdSolver: no fn %s in %s\n", fn, tag);
            return nil;
        }
        id<MTLComputePipelineState> pso =
            [dev newComputePipelineStateWithFunction:f error:&err];
        if (!pso) {
            std::fprintf(stderr, "AvbdSolver: PSO %s failed: %s\n",
                         tag, err ? err.localizedDescription.UTF8String : "(null)");
        }
        return pso;
    }
};

AvbdSolver::AvbdSolver(const char* metallibPath)
    : impl_(new Impl()) {
    impl_->device = MTLCreateSystemDefaultDevice();
    if (!impl_->device) {
        std::fprintf(stderr, "AvbdSolver: no Metal device\n");
        return;
    }
    impl_->queue = [impl_->device newCommandQueue];

    std::string root = metallibPath;
    if (!root.empty() && root.back() != '/') root.push_back('/');

    id<MTLLibrary> libInit       = Impl::loadLib(impl_->device, root + "vbd_init.metallib",             "vbd_init");
    id<MTLLibrary> libSpring     = Impl::loadLib(impl_->device, root + "vbd_gather_spring.metallib",    "vbd_gather_spring");
    id<MTLLibrary> libAttachment = Impl::loadLib(impl_->device, root + "vbd_gather_attachment.metallib","vbd_gather_attachment");
    id<MTLLibrary> libTriangle   = Impl::loadLib(impl_->device, root + "vbd_gather_triangle.metallib",  "vbd_gather_triangle");
    id<MTLLibrary> libBending    = Impl::loadLib(impl_->device, root + "vbd_gather_bending.metallib",   "vbd_gather_bending");
    id<MTLLibrary> libSolve      = Impl::loadLib(impl_->device, root + "vbd_solve_apply.metallib",      "vbd_solve_apply");
    id<MTLLibrary> libSpringForce = Impl::loadLib(impl_->device, root + "spring_force.metallib",        "spring_force");
    if (!libInit || !libSpring || !libAttachment || !libTriangle || !libBending || !libSolve ||
        !libSpringForce) return;

    impl_->psoInit             = Impl::makePSO(impl_->device, libInit,       "main_0", "vbd_init");
    impl_->psoGatherSpring     = Impl::makePSO(impl_->device, libSpring,     "main_0", "vbd_gather_spring");
    impl_->psoGatherAttachment = Impl::makePSO(impl_->device, libAttachment, "main_0", "vbd_gather_attachment");
    impl_->psoGatherTriangle   = Impl::makePSO(impl_->device, libTriangle,   "main_0", "vbd_gather_triangle");
    impl_->psoGatherBending    = Impl::makePSO(impl_->device, libBending,    "main_0", "vbd_gather_bending");
    impl_->psoSolveApply       = Impl::makePSO(impl_->device, libSolve,      "main_0", "vbd_solve_apply");
    impl_->psoSpringForce      = Impl::makePSO(impl_->device, libSpringForce, "main_0", "spring_force");
    if (!impl_->psoInit || !impl_->psoGatherSpring || !impl_->psoGatherAttachment ||
        !impl_->psoGatherTriangle || !impl_->psoGatherBending || !impl_->psoSolveApply ||
        !impl_->psoSpringForce) return;

    impl_->ok = true;
    std::fprintf(stderr,
        "AvbdSolver: 7 kernels loaded (init, gather_{spring,attachment,triangle,bending}, solve_apply, spring_force)\n");
}

AvbdSolver::~AvbdSolver() { delete impl_; }

bool AvbdSolver::ok() const { return impl_ && impl_->ok; }

void AvbdSolver::setupMesh(uint32_t nVerts,
                           const float* positions,
                           const float* predicted,
                           const float* mass,
                           float invHSquared) {
    if (!ok()) return;
    impl_->nVerts = nVerts;
    const NSUInteger bytesVec3 = 3 * nVerts * sizeof(float);
    const NSUInteger bytesScalar = nVerts * sizeof(float);
    const NSUInteger bytesHess = 6 * nVerts * sizeof(float);
    const MTLResourceOptions opts = MTLResourceStorageModeShared;

    impl_->bufPositions = [impl_->device newBufferWithBytes:positions length:bytesVec3 options:opts];
    impl_->bufPredicted = [impl_->device newBufferWithBytes:predicted length:bytesVec3 options:opts];
    impl_->bufMass      = [impl_->device newBufferWithBytes:mass      length:bytesScalar options:opts];
    impl_->bufGScratch  = [impl_->device newBufferWithLength:bytesVec3 options:opts];
    impl_->bufHScratch  = [impl_->device newBufferWithLength:bytesHess options:opts];

    impl_->bufInitParams = [impl_->device newBufferWithLength:sizeof(VbdInitParams) options:opts];
    *static_cast<VbdInitParams*>(impl_->bufInitParams.contents) = {invHSquared};

    impl_->meshReady = true;
}

void AvbdSolver::uploadSprings(uint32_t nSprings,
                                const uint32_t* p1Idx,
                                const uint32_t* p2Idx,
                                const float* restLen,
                                const float* stiffness) {
    if (!ok()) return;
    impl_->nSprings = nSprings;
    const NSUInteger bytesU = nSprings * sizeof(uint32_t);
    const NSUInteger bytesF = nSprings * sizeof(float);
    const NSUInteger bytesGrad = 3 * nSprings * sizeof(float);
    const NSUInteger bytesHess = 6 * nSprings * sizeof(float);
    const MTLResourceOptions opts = MTLResourceStorageModeShared;

    impl_->bufSpringP1Idx     = [impl_->device newBufferWithBytes:p1Idx     length:bytesU options:opts];
    impl_->bufSpringP2Idx     = [impl_->device newBufferWithBytes:p2Idx     length:bytesU options:opts];
    impl_->bufSpringRestLen   = [impl_->device newBufferWithBytes:restLen   length:bytesF options:opts];
    impl_->bufSpringStiffness = [impl_->device newBufferWithBytes:stiffness length:bytesF options:opts];
    impl_->bufSpringGradA     = [impl_->device newBufferWithLength:bytesGrad options:opts];
    impl_->bufSpringHess      = [impl_->device newBufferWithLength:bytesHess options:opts];
}

int AvbdSolver::step() {
    if (!ok() || !impl_->meshReady) return -1;

    // PR-E slice 3: vbd_init + spring_force in one command buffer.
    // Gather + solve_apply land in follow-up PRs once CSR adjacency
    // is wired into the dispatch.
    id<MTLCommandBuffer> cb = [impl_->queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];

    // 1. vbd_init: inertial term per vertex.
    [enc setComputePipelineState:impl_->psoInit];
    [enc setBuffer:impl_->bufInitParams offset:0 atIndex:0];
    [enc setBuffer:impl_->bufPositions  offset:0 atIndex:1];
    [enc setBuffer:impl_->bufPredicted  offset:0 atIndex:2];
    [enc setBuffer:impl_->bufMass       offset:0 atIndex:3];
    [enc setBuffer:impl_->bufGScratch   offset:0 atIndex:4];
    [enc setBuffer:impl_->bufHScratch   offset:0 atIndex:5];
    [enc dispatchThreads:MTLSizeMake(impl_->nVerts, 1, 1)
   threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];

    // 2. spring_force: per-spring force + GN Hessian into gradA/hess.
    if (impl_->nSprings > 0) {
        [enc setComputePipelineState:impl_->psoSpringForce];
        [enc setBuffer:impl_->bufPositions       offset:0 atIndex:0];
        [enc setBuffer:impl_->bufSpringP1Idx     offset:0 atIndex:1];
        [enc setBuffer:impl_->bufSpringP2Idx     offset:0 atIndex:2];
        [enc setBuffer:impl_->bufSpringRestLen   offset:0 atIndex:3];
        [enc setBuffer:impl_->bufSpringStiffness offset:0 atIndex:4];
        [enc setBuffer:impl_->bufSpringGradA     offset:0 atIndex:5];
        [enc setBuffer:impl_->bufSpringHess      offset:0 atIndex:6];
        [enc dispatchThreads:MTLSizeMake(impl_->nSprings, 1, 1)
       threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
    }

    [enc endEncoding];
    [cb commit];
    [cb waitUntilCompleted];

    return 0;
}

void AvbdSolver::readScratch(std::vector<float>& gScratch_out,
                             std::vector<float>& hScratch_out) const {
    if (!ok() || !impl_->meshReady) {
        gScratch_out.clear();
        hScratch_out.clear();
        return;
    }
    const uint32_t n = impl_->nVerts;
    gScratch_out.assign(3 * n, 0.0f);
    hScratch_out.assign(6 * n, 0.0f);
    std::memcpy(gScratch_out.data(), impl_->bufGScratch.contents, 3 * n * sizeof(float));
    std::memcpy(hScratch_out.data(), impl_->bufHScratch.contents, 6 * n * sizeof(float));
}

void AvbdSolver::readSpringForce(std::vector<float>& gradA_out,
                                  std::vector<float>& hess_out) const {
    if (!ok() || impl_->nSprings == 0) {
        gradA_out.clear();
        hess_out.clear();
        return;
    }
    const uint32_t n = impl_->nSprings;
    gradA_out.assign(3 * n, 0.0f);
    hess_out.assign(6 * n, 0.0f);
    std::memcpy(gradA_out.data(), impl_->bufSpringGradA.contents, 3 * n * sizeof(float));
    std::memcpy(hess_out.data(),  impl_->bufSpringHess.contents,  6 * n * sizeof(float));
}

}  // namespace cloth
