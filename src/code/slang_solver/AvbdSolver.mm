// AvbdSolver implementation (PR-E stub). Loads the six AVBD kernel
// metallibs and builds compute pipeline state objects. Per-step
// dispatch is a stub — actual GPU loop wiring lands in follow-up PRs.

#include "AvbdSolver.h"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

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
    // per-vertex).
    id<MTLComputePipelineState> psoSpringForce      = nil;
    id<MTLComputePipelineState> psoAttachmentForce  = nil;
    id<MTLComputePipelineState> psoTriangleMembraneForce = nil;
    id<MTLComputePipelineState> psoTriangleBendingForce  = nil;

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

    // Vertex → spring CSR adjacency (built host-side in uploadSprings,
    // mirroring Cloth.Avbd.AdjacencySpring's `native_decide`-pinned
    // algorithm).
    id<MTLBuffer> bufVertSpringOffset = nil; // uint per (nVerts+1)
    id<MTLBuffer> bufVertSpringIdx    = nil; // uint per (2*nSprings)
    id<MTLBuffer> bufVertSpringRole   = nil; // uint per (2*nSprings)

    // Attachment constraint buffers (allocated by uploadAttachments).
    // Each attachment pins one vertex to a fixed world anchor.
    id<MTLBuffer> bufAttachVertIdx     = nil; // uint per attachment
    id<MTLBuffer> bufAttachFixedPos    = nil; // padded float3 per attachment
    id<MTLBuffer> bufAttachStiffness   = nil; // float per attachment
    id<MTLBuffer> bufAttachGradV       = nil; // padded float3 per attachment
    id<MTLBuffer> bufAttachHessScalar  = nil; // float per attachment

    // Vertex → attachment CSR. Role is implicit (always 0 for K=1).
    id<MTLBuffer> bufVertAttachOffset  = nil; // uint per (nVerts+1)
    id<MTLBuffer> bufVertAttachIdx     = nil; // uint per nAttach

    // Triangle membrane constraint buffers (allocated by uploadTriangles).
    id<MTLBuffer> bufTriIdx            = nil; // uint per (3 * nTri) — corners
    id<MTLBuffer> bufTriStiffness      = nil; // float per nTri
    id<MTLBuffer> bufTriGrad           = nil; // padded float3 per (3 * nTri)
    id<MTLBuffer> bufTriHessScalar     = nil; // float per (3 * nTri)

    // Vertex → triangle CSR. K=3, so role ∈ {0, 1, 2}.
    id<MTLBuffer> bufVertTriOffset     = nil; // uint per (nVerts+1)
    id<MTLBuffer> bufVertTriIdx        = nil; // uint per (3 * nTri)
    id<MTLBuffer> bufVertTriRole       = nil; // uint per (3 * nTri)

    // Bending constraint buffers (allocated by uploadBendings).
    id<MTLBuffer> bufBendIdx           = nil; // uint per (4 * nBend) — 4-vert stencil
    id<MTLBuffer> bufBendWeight        = nil; // float per (4 * nBend) — Laplacian weights
    id<MTLBuffer> bufBendNTarget       = nil; // float per nBend — rest magnitude
    id<MTLBuffer> bufBendStiffness     = nil; // float per nBend
    id<MTLBuffer> bufBendGrad          = nil; // padded float3 per (4 * nBend)
    id<MTLBuffer> bufBendHessScalar    = nil; // float per (4 * nBend)

    // Vertex → bending CSR. K=4, role ∈ {0, 1, 2, 3}.
    id<MTLBuffer> bufVertBendOffset    = nil; // uint per (nVerts+1)
    id<MTLBuffer> bufVertBendIdx       = nil; // uint per (4 * nBend)
    id<MTLBuffer> bufVertBendRole      = nil; // uint per (4 * nBend)

    uint32_t nVerts   = 0;
    uint32_t nSprings = 0;
    uint32_t nAttach  = 0;
    uint32_t nTri     = 0;
    uint32_t nBend    = 0;

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
    id<MTLLibrary> libSpringForce         = Impl::loadLib(impl_->device, root + "spring_force.metallib",            "spring_force");
    id<MTLLibrary> libAttachmentForce     = Impl::loadLib(impl_->device, root + "attachment_force.metallib",        "attachment_force");
    id<MTLLibrary> libTriMembraneForce    = Impl::loadLib(impl_->device, root + "triangle_membrane_force.metallib", "triangle_membrane_force");
    id<MTLLibrary> libTriBendingForce     = Impl::loadLib(impl_->device, root + "triangle_bending_force.metallib",  "triangle_bending_force");
    if (!libInit || !libSpring || !libAttachment || !libTriangle || !libBending || !libSolve ||
        !libSpringForce || !libAttachmentForce || !libTriMembraneForce || !libTriBendingForce) return;

    impl_->psoInit             = Impl::makePSO(impl_->device, libInit,       "main_0", "vbd_init");
    impl_->psoGatherSpring     = Impl::makePSO(impl_->device, libSpring,     "main_0", "vbd_gather_spring");
    impl_->psoGatherAttachment = Impl::makePSO(impl_->device, libAttachment, "main_0", "vbd_gather_attachment");
    impl_->psoGatherTriangle   = Impl::makePSO(impl_->device, libTriangle,   "main_0", "vbd_gather_triangle");
    impl_->psoGatherBending    = Impl::makePSO(impl_->device, libBending,    "main_0", "vbd_gather_bending");
    impl_->psoSolveApply       = Impl::makePSO(impl_->device, libSolve,      "main_0", "vbd_solve_apply");
    impl_->psoSpringForce            = Impl::makePSO(impl_->device, libSpringForce,         "main_0", "spring_force");
    impl_->psoAttachmentForce        = Impl::makePSO(impl_->device, libAttachmentForce,     "main_0", "attachment_force");
    impl_->psoTriangleMembraneForce  = Impl::makePSO(impl_->device, libTriMembraneForce,    "main_0", "triangle_membrane_force");
    impl_->psoTriangleBendingForce   = Impl::makePSO(impl_->device, libTriBendingForce,     "main_0", "triangle_bending_force");
    if (!impl_->psoInit || !impl_->psoGatherSpring || !impl_->psoGatherAttachment ||
        !impl_->psoGatherTriangle || !impl_->psoGatherBending || !impl_->psoSolveApply ||
        !impl_->psoSpringForce || !impl_->psoAttachmentForce ||
        !impl_->psoTriangleMembraneForce || !impl_->psoTriangleBendingForce) return;

    impl_->ok = true;
    std::fprintf(stderr,
        "AvbdSolver: 10 kernels loaded — full AVBD pipeline (init, gather_*, solve_apply, spring/attach/tri/bend_force)\n");
}

AvbdSolver::~AvbdSolver() { delete impl_; }

bool AvbdSolver::ok() const { return impl_ && impl_->ok; }

// Metal's `float3` has 16-byte alignment, so `StructuredBuffer<float3>`
// strides by 16 bytes (4 floats per vec3, last is padding). Host arrays
// using tight 3-float-per-vec3 layout are scattered/gathered with the
// padding inserted on upload and stripped on readback.
static void uploadVec3Padded(id<MTLBuffer> dst, const float* src, uint32_t n) {
    float* p = static_cast<float*>(dst.contents);
    for (uint32_t i = 0; i < n; ++i) {
        p[4*i + 0] = src[3*i + 0];
        p[4*i + 1] = src[3*i + 1];
        p[4*i + 2] = src[3*i + 2];
        p[4*i + 3] = 0.0f;
    }
}

static void readVec3Padded(std::vector<float>& dst, const id<MTLBuffer>& src, uint32_t n) {
    dst.assign(3 * n, 0.0f);
    const float* p = static_cast<const float*>(src.contents);
    for (uint32_t i = 0; i < n; ++i) {
        dst[3*i + 0] = p[4*i + 0];
        dst[3*i + 1] = p[4*i + 1];
        dst[3*i + 2] = p[4*i + 2];
    }
}

void AvbdSolver::setupMesh(uint32_t nVerts,
                           const float* positions,
                           const float* predicted,
                           const float* mass,
                           float invHSquared) {
    if (!ok()) return;
    impl_->nVerts = nVerts;
    const NSUInteger bytesVec3Padded = 4 * nVerts * sizeof(float);
    const NSUInteger bytesScalar = nVerts * sizeof(float);
    const NSUInteger bytesHess = 6 * nVerts * sizeof(float);
    const MTLResourceOptions opts = MTLResourceStorageModeShared;

    impl_->bufPositions = [impl_->device newBufferWithLength:bytesVec3Padded options:opts];
    impl_->bufPredicted = [impl_->device newBufferWithLength:bytesVec3Padded options:opts];
    impl_->bufGScratch  = [impl_->device newBufferWithLength:bytesVec3Padded options:opts];
    impl_->bufMass      = [impl_->device newBufferWithBytes:mass length:bytesScalar options:opts];
    impl_->bufHScratch  = [impl_->device newBufferWithLength:bytesHess options:opts];

    uploadVec3Padded(impl_->bufPositions, positions, nVerts);
    uploadVec3Padded(impl_->bufPredicted, predicted, nVerts);

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
    const NSUInteger bytesGradPadded = 4 * nSprings * sizeof(float);
    const NSUInteger bytesHess = 6 * nSprings * sizeof(float);
    const MTLResourceOptions opts = MTLResourceStorageModeShared;

    impl_->bufSpringP1Idx     = [impl_->device newBufferWithBytes:p1Idx     length:bytesU options:opts];
    impl_->bufSpringP2Idx     = [impl_->device newBufferWithBytes:p2Idx     length:bytesU options:opts];
    impl_->bufSpringRestLen   = [impl_->device newBufferWithBytes:restLen   length:bytesF options:opts];
    impl_->bufSpringStiffness = [impl_->device newBufferWithBytes:stiffness length:bytesF options:opts];
    // gradA is `StructuredBuffer<float3>` → padded 16-byte stride per spring.
    impl_->bufSpringGradA     = [impl_->device newBufferWithLength:bytesGradPadded options:opts];
    impl_->bufSpringHess      = [impl_->device newBufferWithLength:bytesHess options:opts];

    // Build vertex→spring CSR on host. Faithful port of
    // Cloth.Avbd.AdjacencySpring.SpringCsr.build: count incidences per
    // vertex (each spring touches both endpoints), prefix-sum into
    // offsets, second pass scatters spring id + role (0 for a, 1 for b).
    const uint32_t nV = impl_->nVerts;
    std::vector<uint32_t> counts(nV, 0u);
    for (uint32_t i = 0; i < nSprings; ++i) {
        counts[p1Idx[i]]++;
        counts[p2Idx[i]]++;
    }
    std::vector<uint32_t> offsets(nV + 1, 0u);
    for (uint32_t v = 0; v < nV; ++v) offsets[v + 1] = offsets[v] + counts[v];
    const uint32_t totalInc = offsets[nV];
    std::vector<uint32_t> springIdx(totalInc, 0u);
    std::vector<uint32_t> role(totalInc, 0u);
    std::vector<uint32_t> cursor(nV, 0u);
    for (uint32_t i = 0; i < nSprings; ++i) {
        const uint32_t a = p1Idx[i];
        const uint32_t pa = offsets[a] + cursor[a];
        springIdx[pa] = i;  role[pa] = 0u;  cursor[a]++;
        const uint32_t b = p2Idx[i];
        const uint32_t pb = offsets[b] + cursor[b];
        springIdx[pb] = i;  role[pb] = 1u;  cursor[b]++;
    }

    impl_->bufVertSpringOffset =
        [impl_->device newBufferWithBytes:offsets.data()
                                   length:offsets.size() * sizeof(uint32_t)
                                  options:opts];
    impl_->bufVertSpringIdx =
        [impl_->device newBufferWithBytes:springIdx.data()
                                   length:springIdx.size() * sizeof(uint32_t)
                                  options:opts];
    impl_->bufVertSpringRole =
        [impl_->device newBufferWithBytes:role.data()
                                   length:role.size() * sizeof(uint32_t)
                                  options:opts];
}

void AvbdSolver::uploadAttachments(uint32_t nAttach,
                                    const uint32_t* vertIdx,
                                    const float* fixedPos,
                                    const float* stiffness) {
    if (!ok()) return;
    impl_->nAttach = nAttach;
    const NSUInteger bytesU = nAttach * sizeof(uint32_t);
    const NSUInteger bytesF = nAttach * sizeof(float);
    const NSUInteger bytesVec3Padded = 4 * nAttach * sizeof(float);
    const MTLResourceOptions opts = MTLResourceStorageModeShared;

    impl_->bufAttachVertIdx    = [impl_->device newBufferWithBytes:vertIdx   length:bytesU options:opts];
    impl_->bufAttachFixedPos   = [impl_->device newBufferWithLength:bytesVec3Padded options:opts];
    impl_->bufAttachStiffness  = [impl_->device newBufferWithBytes:stiffness length:bytesF options:opts];
    impl_->bufAttachGradV      = [impl_->device newBufferWithLength:bytesVec3Padded options:opts];
    impl_->bufAttachHessScalar = [impl_->device newBufferWithLength:bytesF options:opts];
    uploadVec3Padded(impl_->bufAttachFixedPos, fixedPos, nAttach);

    // Build vertex→attachment CSR. K=1, so each attachment contributes
    // exactly one entry to vertIdx[c]. No role buffer — implicit 0.
    const uint32_t nV = impl_->nVerts;
    std::vector<uint32_t> counts(nV, 0u);
    for (uint32_t i = 0; i < nAttach; ++i) counts[vertIdx[i]]++;
    std::vector<uint32_t> offsets(nV + 1, 0u);
    for (uint32_t v = 0; v < nV; ++v) offsets[v + 1] = offsets[v] + counts[v];
    const uint32_t totalInc = offsets[nV];
    std::vector<uint32_t> attachIdx(totalInc, 0u);
    std::vector<uint32_t> cursor(nV, 0u);
    for (uint32_t i = 0; i < nAttach; ++i) {
        const uint32_t v = vertIdx[i];
        const uint32_t p = offsets[v] + cursor[v];
        attachIdx[p] = i;
        cursor[v]++;
    }

    impl_->bufVertAttachOffset =
        [impl_->device newBufferWithBytes:offsets.data()
                                   length:offsets.size() * sizeof(uint32_t)
                                  options:opts];
    impl_->bufVertAttachIdx =
        [impl_->device newBufferWithBytes:attachIdx.data()
                                   length:attachIdx.size() * sizeof(uint32_t)
                                  options:opts];
}

void AvbdSolver::uploadTriangles(uint32_t nTri,
                                  const uint32_t* triIdx,
                                  const float* stiffness) {
    if (!ok()) return;
    impl_->nTri = nTri;
    const NSUInteger bytesTriIdx = 3 * nTri * sizeof(uint32_t);
    const NSUInteger bytesF      = nTri * sizeof(float);
    const NSUInteger bytesGradPadded   = 4 * 3 * nTri * sizeof(float);
    const NSUInteger bytesHessScalar   = 3 * nTri * sizeof(float);
    const MTLResourceOptions opts = MTLResourceStorageModeShared;

    impl_->bufTriIdx        = [impl_->device newBufferWithBytes:triIdx    length:bytesTriIdx options:opts];
    impl_->bufTriStiffness  = [impl_->device newBufferWithBytes:stiffness length:bytesF      options:opts];
    impl_->bufTriGrad       = [impl_->device newBufferWithLength:bytesGradPadded options:opts];
    impl_->bufTriHessScalar = [impl_->device newBufferWithLength:bytesHessScalar options:opts];

    // Build vertex→triangle CSR. K=3, role ∈ {0,1,2} per corner.
    const uint32_t nV = impl_->nVerts;
    std::vector<uint32_t> counts(nV, 0u);
    for (uint32_t c = 0; c < nTri; ++c)
        for (uint32_t r = 0; r < 3; ++r) counts[triIdx[3*c + r]]++;
    std::vector<uint32_t> offsets(nV + 1, 0u);
    for (uint32_t v = 0; v < nV; ++v) offsets[v + 1] = offsets[v] + counts[v];
    const uint32_t totalInc = offsets[nV];
    std::vector<uint32_t> idxArr(totalInc, 0u);
    std::vector<uint32_t> roleArr(totalInc, 0u);
    std::vector<uint32_t> cursor(nV, 0u);
    for (uint32_t c = 0; c < nTri; ++c) {
        for (uint32_t r = 0; r < 3; ++r) {
            const uint32_t v = triIdx[3*c + r];
            const uint32_t p = offsets[v] + cursor[v];
            idxArr[p]  = c;
            roleArr[p] = r;
            cursor[v]++;
        }
    }

    impl_->bufVertTriOffset =
        [impl_->device newBufferWithBytes:offsets.data()
                                   length:offsets.size() * sizeof(uint32_t)
                                  options:opts];
    impl_->bufVertTriIdx =
        [impl_->device newBufferWithBytes:idxArr.data()
                                   length:idxArr.size() * sizeof(uint32_t)
                                  options:opts];
    impl_->bufVertTriRole =
        [impl_->device newBufferWithBytes:roleArr.data()
                                   length:roleArr.size() * sizeof(uint32_t)
                                  options:opts];
}

void AvbdSolver::uploadBendings(uint32_t nBend,
                                 const uint32_t* bendIdx,
                                 const float* weight,
                                 const float* nTarget,
                                 const float* stiffness) {
    if (!ok()) return;
    impl_->nBend = nBend;
    const NSUInteger bytesBendIdx    = 4 * nBend * sizeof(uint32_t);
    const NSUInteger bytesBendWeight = 4 * nBend * sizeof(float);
    const NSUInteger bytesF          = nBend * sizeof(float);
    const NSUInteger bytesGradPadded = 4 * 4 * nBend * sizeof(float);
    const NSUInteger bytesHessScalar = 4 * nBend * sizeof(float);
    const MTLResourceOptions opts = MTLResourceStorageModeShared;

    impl_->bufBendIdx        = [impl_->device newBufferWithBytes:bendIdx   length:bytesBendIdx    options:opts];
    impl_->bufBendWeight     = [impl_->device newBufferWithBytes:weight    length:bytesBendWeight options:opts];
    impl_->bufBendNTarget    = [impl_->device newBufferWithBytes:nTarget   length:bytesF          options:opts];
    impl_->bufBendStiffness  = [impl_->device newBufferWithBytes:stiffness length:bytesF          options:opts];
    impl_->bufBendGrad       = [impl_->device newBufferWithLength:bytesGradPadded options:opts];
    impl_->bufBendHessScalar = [impl_->device newBufferWithLength:bytesHessScalar options:opts];

    // Build vertex→bending CSR. K=4, role ∈ {0,1,2,3} per stencil corner.
    const uint32_t nV = impl_->nVerts;
    std::vector<uint32_t> counts(nV, 0u);
    for (uint32_t c = 0; c < nBend; ++c)
        for (uint32_t r = 0; r < 4; ++r) counts[bendIdx[4*c + r]]++;
    std::vector<uint32_t> offsets(nV + 1, 0u);
    for (uint32_t v = 0; v < nV; ++v) offsets[v + 1] = offsets[v] + counts[v];
    const uint32_t totalInc = offsets[nV];
    std::vector<uint32_t> idxArr(totalInc, 0u);
    std::vector<uint32_t> roleArr(totalInc, 0u);
    std::vector<uint32_t> cursor(nV, 0u);
    for (uint32_t c = 0; c < nBend; ++c) {
        for (uint32_t r = 0; r < 4; ++r) {
            const uint32_t v = bendIdx[4*c + r];
            const uint32_t p = offsets[v] + cursor[v];
            idxArr[p]  = c;
            roleArr[p] = r;
            cursor[v]++;
        }
    }

    impl_->bufVertBendOffset =
        [impl_->device newBufferWithBytes:offsets.data()
                                   length:offsets.size() * sizeof(uint32_t)
                                  options:opts];
    impl_->bufVertBendIdx =
        [impl_->device newBufferWithBytes:idxArr.data()
                                   length:idxArr.size() * sizeof(uint32_t)
                                  options:opts];
    impl_->bufVertBendRole =
        [impl_->device newBufferWithBytes:roleArr.data()
                                   length:roleArr.size() * sizeof(uint32_t)
                                  options:opts];
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

        // 3. vbd_gather_spring: per-vertex gather of spring contributions
        //    via CSR adjacency. Reads gradA/hess produced by step 2 and
        //    accumulates into the gScratch/hScratch written by step 1.
        [enc setComputePipelineState:impl_->psoGatherSpring];
        [enc setBuffer:impl_->bufSpringGradA      offset:0 atIndex:0];
        [enc setBuffer:impl_->bufSpringHess       offset:0 atIndex:1];
        [enc setBuffer:impl_->bufVertSpringOffset offset:0 atIndex:2];
        [enc setBuffer:impl_->bufVertSpringIdx    offset:0 atIndex:3];
        [enc setBuffer:impl_->bufVertSpringRole   offset:0 atIndex:4];
        [enc setBuffer:impl_->bufGScratch         offset:0 atIndex:5];
        [enc setBuffer:impl_->bufHScratch         offset:0 atIndex:6];
        [enc dispatchThreads:MTLSizeMake(impl_->nVerts, 1, 1)
       threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
    }

    // 4. attachment_force + vbd_gather_attachment (if nAttach > 0).
    if (impl_->nAttach > 0) {
        [enc setComputePipelineState:impl_->psoAttachmentForce];
        [enc setBuffer:impl_->bufPositions          offset:0 atIndex:0];
        [enc setBuffer:impl_->bufAttachVertIdx      offset:0 atIndex:1];
        [enc setBuffer:impl_->bufAttachFixedPos     offset:0 atIndex:2];
        [enc setBuffer:impl_->bufAttachStiffness    offset:0 atIndex:3];
        [enc setBuffer:impl_->bufAttachGradV        offset:0 atIndex:4];
        [enc setBuffer:impl_->bufAttachHessScalar   offset:0 atIndex:5];
        [enc dispatchThreads:MTLSizeMake(impl_->nAttach, 1, 1)
       threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];

        [enc setComputePipelineState:impl_->psoGatherAttachment];
        [enc setBuffer:impl_->bufAttachGradV        offset:0 atIndex:0];
        [enc setBuffer:impl_->bufAttachHessScalar   offset:0 atIndex:1];
        [enc setBuffer:impl_->bufVertAttachOffset   offset:0 atIndex:2];
        [enc setBuffer:impl_->bufVertAttachIdx      offset:0 atIndex:3];
        [enc setBuffer:impl_->bufGScratch           offset:0 atIndex:4];
        [enc setBuffer:impl_->bufHScratch           offset:0 atIndex:5];
        [enc dispatchThreads:MTLSizeMake(impl_->nVerts, 1, 1)
       threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
    }

    // 5. triangle_membrane_force + vbd_gather_triangle (if nTri > 0).
    if (impl_->nTri > 0) {
        [enc setComputePipelineState:impl_->psoTriangleMembraneForce];
        [enc setBuffer:impl_->bufPositions      offset:0 atIndex:0];
        [enc setBuffer:impl_->bufTriIdx         offset:0 atIndex:1];
        [enc setBuffer:impl_->bufTriStiffness   offset:0 atIndex:2];
        [enc setBuffer:impl_->bufTriGrad        offset:0 atIndex:3];
        [enc setBuffer:impl_->bufTriHessScalar  offset:0 atIndex:4];
        [enc dispatchThreads:MTLSizeMake(impl_->nTri, 1, 1)
       threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];

        [enc setComputePipelineState:impl_->psoGatherTriangle];
        [enc setBuffer:impl_->bufTriGrad        offset:0 atIndex:0];
        [enc setBuffer:impl_->bufTriHessScalar  offset:0 atIndex:1];
        [enc setBuffer:impl_->bufVertTriOffset  offset:0 atIndex:2];
        [enc setBuffer:impl_->bufVertTriIdx     offset:0 atIndex:3];
        [enc setBuffer:impl_->bufVertTriRole    offset:0 atIndex:4];
        [enc setBuffer:impl_->bufGScratch       offset:0 atIndex:5];
        [enc setBuffer:impl_->bufHScratch       offset:0 atIndex:6];
        [enc dispatchThreads:MTLSizeMake(impl_->nVerts, 1, 1)
       threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
    }

    // 6. triangle_bending_force + vbd_gather_bending (if nBend > 0).
    if (impl_->nBend > 0) {
        [enc setComputePipelineState:impl_->psoTriangleBendingForce];
        [enc setBuffer:impl_->bufPositions       offset:0 atIndex:0];
        [enc setBuffer:impl_->bufBendIdx         offset:0 atIndex:1];
        [enc setBuffer:impl_->bufBendWeight      offset:0 atIndex:2];
        [enc setBuffer:impl_->bufBendNTarget     offset:0 atIndex:3];
        [enc setBuffer:impl_->bufBendStiffness   offset:0 atIndex:4];
        [enc setBuffer:impl_->bufBendGrad        offset:0 atIndex:5];
        [enc setBuffer:impl_->bufBendHessScalar  offset:0 atIndex:6];
        [enc dispatchThreads:MTLSizeMake(impl_->nBend, 1, 1)
       threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];

        [enc setComputePipelineState:impl_->psoGatherBending];
        [enc setBuffer:impl_->bufBendGrad        offset:0 atIndex:0];
        [enc setBuffer:impl_->bufBendHessScalar  offset:0 atIndex:1];
        [enc setBuffer:impl_->bufVertBendOffset  offset:0 atIndex:2];
        [enc setBuffer:impl_->bufVertBendIdx     offset:0 atIndex:3];
        [enc setBuffer:impl_->bufVertBendRole    offset:0 atIndex:4];
        [enc setBuffer:impl_->bufGScratch        offset:0 atIndex:5];
        [enc setBuffer:impl_->bufHScratch        offset:0 atIndex:6];
        [enc dispatchThreads:MTLSizeMake(impl_->nVerts, 1, 1)
       threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
    }

    // 7. vbd_solve_apply: invert 3x3 H, write positions += -H⁻¹ g.
    [enc setComputePipelineState:impl_->psoSolveApply];
    [enc setBuffer:impl_->bufGScratch  offset:0 atIndex:0];
    [enc setBuffer:impl_->bufHScratch  offset:0 atIndex:1];
    [enc setBuffer:impl_->bufPositions offset:0 atIndex:2];
    [enc dispatchThreads:MTLSizeMake(impl_->nVerts, 1, 1)
   threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];

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
    readVec3Padded(gScratch_out, impl_->bufGScratch, n);
    hScratch_out.assign(6 * n, 0.0f);
    std::memcpy(hScratch_out.data(), impl_->bufHScratch.contents, 6 * n * sizeof(float));
}

void AvbdSolver::updateState(const float* positions, const float* predicted) {
    if (!ok() || !impl_->meshReady) return;
    uploadVec3Padded(impl_->bufPositions, positions, impl_->nVerts);
    uploadVec3Padded(impl_->bufPredicted, predicted, impl_->nVerts);
}

void AvbdSolver::readPositions(std::vector<float>& positions_out) const {
    if (!ok() || !impl_->meshReady) {
        positions_out.clear();
        return;
    }
    readVec3Padded(positions_out, impl_->bufPositions, impl_->nVerts);
}

void AvbdSolver::readSpringForce(std::vector<float>& gradA_out,
                                  std::vector<float>& hess_out) const {
    if (!ok() || impl_->nSprings == 0) {
        gradA_out.clear();
        hess_out.clear();
        return;
    }
    const uint32_t n = impl_->nSprings;
    readVec3Padded(gradA_out, impl_->bufSpringGradA, n);
    hess_out.assign(6 * n, 0.0f);
    std::memcpy(hess_out.data(),  impl_->bufSpringHess.contents,  6 * n * sizeof(float));
}

}  // namespace cloth
