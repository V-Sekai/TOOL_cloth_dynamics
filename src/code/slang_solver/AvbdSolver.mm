// AvbdSolver implementation (PR-E stub). Loads the six AVBD kernel
// metallibs and builds compute pipeline state objects. Per-step
// dispatch is a stub — actual GPU loop wiring lands in follow-up PRs.

#include "AvbdSolver.h"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace cloth {

// Matches the VbdInitParams struct emitted by Cloth.SlangCodegen.VbdInit
// (now includes `colorOffset` for coloring-aware dispatch).
struct VbdInitParams {
    float invHSquared;
    uint32_t colorOffset;
};

// Params for each of the 4 vbd_gather_* kernels — same single-uint
// payload. setBytes per-dispatch captures these by value.
struct VbdGatherParams {
    uint32_t colorOffset;
};

// Matches the SelfCollisionScanParams struct emitted by
// Cloth.SlangCodegen.SelfCollisionScan.
struct SelfCollisionScanParams {
    uint32_t nVerts;
    uint32_t K;
};

// Matches the VbdSolveApplyParams struct emitted by
// Cloth.SlangCodegen.VbdSolveApply (single uint colorOffset that
// shifts the lane-to-vertex index lookup, supporting coloring-aware
// dispatch). With colorOffset=0 and identity vertPerm the kernel is
// bit-equivalent to the pre-coloring version.
struct VbdSolveApplyParams {
    uint32_t colorOffset;
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
    id<MTLComputePipelineState> psoAttachmentForce  = nil;        // legacy non-AL path (kept for fallback)
    id<MTLComputePipelineState> psoAttachmentForceAl = nil;       // AL-augmented (active)
    id<MTLComputePipelineState> psoAttachmentDualUpdate = nil;    // per-iter λ ramp
    id<MTLComputePipelineState> psoTriangleMembraneForce = nil;        // legacy non-AL
    id<MTLComputePipelineState> psoTriangleMembraneForceAl = nil;      // AL-augmented (active)
    id<MTLComputePipelineState> psoTriangleMembraneDualUpdate = nil;
    id<MTLComputePipelineState> psoTriangleBendingForce  = nil;        // legacy non-AL
    id<MTLComputePipelineState> psoTriangleBendingForceAl = nil;       // AL-augmented (active)
    id<MTLComputePipelineState> psoTriangleBendingDualUpdate = nil;
    id<MTLComputePipelineState> psoSelfCollisionScan = nil;

    // Self-collision GPU scan state (allocated by
    // uploadSelfCollisionRadii). bufNeighbors holds nVerts·K uints,
    // each row listing up to K overlapping neighbor indices for the
    // vertex (UINT_MAX sentinel for empty slots).
    id<MTLBuffer> bufRadii       = nil;
    id<MTLBuffer> bufNeighbors   = nil;
    uint32_t      selfCollK      = 0;     // 0 = not uploaded
    bool          selfCollReady  = false;

    // Per-vertex state buffers (allocated by setupMesh).
    id<MTLBuffer> bufPositions = nil;   // length = 3 * nVerts (float)
    id<MTLBuffer> bufPredicted = nil;   // length = 3 * nVerts (float)
    id<MTLBuffer> bufMass      = nil;   // length = nVerts (float)
    id<MTLBuffer> bufGScratch  = nil;   // length = 3 * nVerts (float)
    id<MTLBuffer> bufHScratch  = nil;   // length = 6 * nVerts (float)
    id<MTLBuffer> bufInitParams = nil;  // sizeof(VbdInitParams)

    // Coloring-aware dispatch state. `bufVertPerm` is a permutation of
    // [0..nVerts), grouped by color. `colorOffsets` (host-side) has
    // length numColors+1: color k holds vertices
    // vertPerm[colorOffsets[k] .. colorOffsets[k+1]). Default = single
    // identity color (numColors=1, identity perm) — equivalent to
    // pre-PR block Jacobi. uploadTriangles builds the real coloring
    // from triangle adjacency.
    id<MTLBuffer> bufVertPerm         = nil;
    id<MTLBuffer> bufSolveApplyParams = nil;  // sizeof(VbdSolveApplyParams), kept for compat
    uint32_t      numColors           = 1;
    std::vector<uint32_t> colorOffsets;  // length numColors+1
    // Cached for assembling the VbdInitParams payload per dispatch
    // (the kernel struct now packs invHSquared + colorOffset). Set in
    // setupMesh; never changes after.
    float invHSqCached = 0.0f;
    // True once buildVertexColoring has been called (lazily on first
    // step() or stepColored()).
    bool coloringBuilt = false;

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

    // Augmented-Lagrangian state per attachment (PR-F).
    // λ_c — dual multiplier, accumulates across outer iters (padded float3).
    // γ_c — AL penalty (per-attachment, defaults to stiffness).
    id<MTLBuffer> bufAttachLambda      = nil;
    id<MTLBuffer> bufAttachGamma       = nil;

    // Vertex → attachment CSR. Role is implicit (always 0 for K=1).
    id<MTLBuffer> bufVertAttachOffset  = nil; // uint per (nVerts+1)
    id<MTLBuffer> bufVertAttachIdx     = nil; // uint per nAttach

    // Triangle membrane constraint buffers (allocated by uploadTriangles).
    id<MTLBuffer> bufTriIdx            = nil; // uint per (3 * nTri) — corners
    id<MTLBuffer> bufTriStiffness      = nil; // float per nTri
    id<MTLBuffer> bufTriInvUV          = nil; // float per (4 * nTri) — rest material inv (2x2 per tri)
    id<MTLBuffer> bufTriGrad           = nil; // padded float3 per (3 * nTri)
    id<MTLBuffer> bufTriHessScalar     = nil; // float per (3 * nTri)

    // AL state per membrane triangle (PR-F).
    id<MTLBuffer> bufTriLambda0        = nil; // padded float3 per tri (col 0)
    id<MTLBuffer> bufTriLambda1        = nil; // padded float3 per tri (col 1)
    id<MTLBuffer> bufTriGamma          = nil; // float per tri

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

    // AL state per bending constraint (PR-F).
    id<MTLBuffer> bufBendLambda        = nil; // padded float3 per bend
    id<MTLBuffer> bufBendGamma         = nil; // float per bend

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

    // Greedy first-fit vertex coloring on the conflict graph induced
    // by all uploaded constraints (spring 2-clique, triangle 3-clique,
    // bending 4-clique — attachment is 1-vert, contributes nothing).
    // Two vertices share a color only if no constraint references both.
    // Idempotent — early-returns if already built.
    void buildVertexColoringIfNeeded() {
        if (coloringBuilt) return;
        const uint32_t nV = nVerts;

        std::vector<std::vector<uint32_t>> adj(nV);
        auto edge = [&](uint32_t a, uint32_t b) {
            if (a == b) return;
            adj[a].push_back(b);
            adj[b].push_back(a);
        };

        if (nSprings > 0) {
            auto* p1 = static_cast<const uint32_t*>(bufSpringP1Idx.contents);
            auto* p2 = static_cast<const uint32_t*>(bufSpringP2Idx.contents);
            for (uint32_t c = 0; c < nSprings; ++c) edge(p1[c], p2[c]);
        }
        if (nTri > 0) {
            auto* idx = static_cast<const uint32_t*>(bufTriIdx.contents);
            for (uint32_t c = 0; c < nTri; ++c) {
                const uint32_t i0 = idx[3*c+0];
                const uint32_t i1 = idx[3*c+1];
                const uint32_t i2 = idx[3*c+2];
                edge(i0, i1); edge(i0, i2); edge(i1, i2);
            }
        }
        if (nBend > 0) {
            auto* idx = static_cast<const uint32_t*>(bufBendIdx.contents);
            for (uint32_t c = 0; c < nBend; ++c) {
                const uint32_t i0 = idx[4*c+0];
                const uint32_t i1 = idx[4*c+1];
                const uint32_t i2 = idx[4*c+2];
                const uint32_t i3 = idx[4*c+3];
                edge(i0,i1); edge(i0,i2); edge(i0,i3);
                edge(i1,i2); edge(i1,i3); edge(i2,i3);
            }
        }

        std::vector<int32_t> color(nV, -1);
        std::vector<bool> used;
        for (uint32_t v = 0; v < nV; ++v) {
            used.assign(adj[v].size() + 1, false);
            for (uint32_t n : adj[v]) {
                const int32_t cn = color[n];
                if (cn >= 0 && uint32_t(cn) < used.size()) used[cn] = true;
            }
            uint32_t c = 0;
            while (c < used.size() && used[c]) ++c;
            color[v] = int32_t(c);
        }

        uint32_t nc = 0;
        for (int32_t c : color) if (uint32_t(c + 1) > nc) nc = uint32_t(c + 1);
        if (nc == 0) nc = 1;

        std::vector<uint32_t> counts(nc, 0);
        for (int32_t c : color) counts[uint32_t(c)]++;
        std::vector<uint32_t> offsets(nc + 1, 0);
        for (uint32_t c = 0; c < nc; ++c) offsets[c + 1] = offsets[c] + counts[c];

        auto* perm = static_cast<uint32_t*>(bufVertPerm.contents);
        std::vector<uint32_t> cursor(nc, 0);
        for (uint32_t v = 0; v < nV; ++v) {
            const uint32_t c = uint32_t(color[v]);
            perm[offsets[c] + cursor[c]] = v;
            cursor[c]++;
        }

        numColors = nc;
        colorOffsets = std::move(offsets);
        coloringBuilt = true;

        std::printf("[avbd] vertex coloring: %u verts, %u colors\n", nV, nc);
        for (uint32_t c = 0; c < nc; ++c) {
            std::printf("[avbd]   color %2u: %u verts\n", c,
                        colorOffsets[c + 1] - colorOffsets[c]);
        }
    }

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
    id<MTLLibrary> libAttachmentForceAl   = Impl::loadLib(impl_->device, root + "attachment_force_al.metallib",     "attachment_force_al");
    id<MTLLibrary> libAttachmentDualUpdate= Impl::loadLib(impl_->device, root + "attachment_dual_update.metallib",  "attachment_dual_update");
    id<MTLLibrary> libTriMembraneForceAl     = Impl::loadLib(impl_->device, root + "triangle_membrane_force_al.metallib",     "triangle_membrane_force_al");
    id<MTLLibrary> libTriMembraneDualUpdate  = Impl::loadLib(impl_->device, root + "triangle_membrane_dual_update.metallib",  "triangle_membrane_dual_update");
    id<MTLLibrary> libTriBendingForceAl      = Impl::loadLib(impl_->device, root + "triangle_bending_force_al.metallib",      "triangle_bending_force_al");
    id<MTLLibrary> libTriBendingDualUpdate   = Impl::loadLib(impl_->device, root + "triangle_bending_dual_update.metallib",   "triangle_bending_dual_update");
    id<MTLLibrary> libSelfCollisionScan      = Impl::loadLib(impl_->device, root + "self_collision_scan.metallib",            "self_collision_scan");
    if (!libInit || !libSpring || !libAttachment || !libTriangle || !libBending || !libSolve ||
        !libSpringForce || !libAttachmentForceAl ||
        !libAttachmentDualUpdate || !libTriMembraneForceAl ||
        !libTriMembraneDualUpdate || !libTriBendingForceAl ||
        !libTriBendingDualUpdate || !libSelfCollisionScan) return;

    impl_->psoInit             = Impl::makePSO(impl_->device, libInit,       "main_0", "vbd_init");
    impl_->psoGatherSpring     = Impl::makePSO(impl_->device, libSpring,     "main_0", "vbd_gather_spring");
    impl_->psoGatherAttachment = Impl::makePSO(impl_->device, libAttachment, "main_0", "vbd_gather_attachment");
    impl_->psoGatherTriangle   = Impl::makePSO(impl_->device, libTriangle,   "main_0", "vbd_gather_triangle");
    impl_->psoGatherBending    = Impl::makePSO(impl_->device, libBending,    "main_0", "vbd_gather_bending");
    impl_->psoSolveApply       = Impl::makePSO(impl_->device, libSolve,      "main_0", "vbd_solve_apply");
    impl_->psoSpringForce            = Impl::makePSO(impl_->device, libSpringForce,         "main_0", "spring_force");
    impl_->psoAttachmentForceAl      = Impl::makePSO(impl_->device, libAttachmentForceAl,   "main_0", "attachment_force_al");
    impl_->psoAttachmentDualUpdate   = Impl::makePSO(impl_->device, libAttachmentDualUpdate,"main_0", "attachment_dual_update");
    impl_->psoTriangleMembraneForceAl     = Impl::makePSO(impl_->device, libTriMembraneForceAl,     "main_0", "triangle_membrane_force_al");
    impl_->psoTriangleMembraneDualUpdate  = Impl::makePSO(impl_->device, libTriMembraneDualUpdate,  "main_0", "triangle_membrane_dual_update");
    impl_->psoTriangleBendingForceAl      = Impl::makePSO(impl_->device, libTriBendingForceAl,      "main_0", "triangle_bending_force_al");
    impl_->psoTriangleBendingDualUpdate   = Impl::makePSO(impl_->device, libTriBendingDualUpdate,   "main_0", "triangle_bending_dual_update");
    impl_->psoSelfCollisionScan           = Impl::makePSO(impl_->device, libSelfCollisionScan,      "main_0", "self_collision_scan");
    if (!impl_->psoInit || !impl_->psoGatherSpring || !impl_->psoGatherAttachment ||
        !impl_->psoGatherTriangle || !impl_->psoGatherBending || !impl_->psoSolveApply ||
        !impl_->psoSpringForce ||
        !impl_->psoAttachmentForceAl || !impl_->psoAttachmentDualUpdate ||
        !impl_->psoTriangleMembraneForceAl ||
        !impl_->psoTriangleMembraneDualUpdate ||
        !impl_->psoTriangleBendingForceAl ||
        !impl_->psoTriangleBendingDualUpdate ||
        !impl_->psoSelfCollisionScan) return;

    impl_->ok = true;
    std::fprintf(stderr,
        "AvbdSolver: 17 kernels loaded — full AVBD + AL + GPU self-collision\n");
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
    *static_cast<VbdInitParams*>(impl_->bufInitParams.contents) = {invHSquared, 0u};
    impl_->invHSqCached = invHSquared;

    // Default vertex permutation = identity (lane i → vertex i) and
    // solve_apply colorOffset = 0. Equivalent to a single color spanning
    // all vertices — same behavior as pre-coloring. buildVertexColoring()
    // overwrites these with greedy first-fit when stepColored() runs.
    const NSUInteger bytesPerm = nVerts * sizeof(uint32_t);
    impl_->bufVertPerm = [impl_->device newBufferWithLength:bytesPerm options:opts];
    uint32_t* perm = static_cast<uint32_t*>(impl_->bufVertPerm.contents);
    for (uint32_t i = 0; i < nVerts; ++i) perm[i] = i;

    impl_->bufSolveApplyParams = [impl_->device
        newBufferWithLength:sizeof(VbdSolveApplyParams) options:opts];
    static_cast<VbdSolveApplyParams*>(impl_->bufSolveApplyParams.contents)
        ->colorOffset = 0u;

    impl_->numColors = 1;
    impl_->colorOffsets.assign({0u, nVerts});
    impl_->coloringBuilt = false;

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
    // AL state: λ_c starts at zero (any prior accumulation discarded
    // on re-upload); γ_c defaults to k_c so dual ramp tracks the
    // physical stiffness.
    impl_->bufAttachLambda     = [impl_->device newBufferWithLength:bytesVec3Padded options:opts];
    impl_->bufAttachGamma      = [impl_->device newBufferWithBytes:stiffness length:bytesF options:opts];
    std::memset(impl_->bufAttachLambda.contents, 0, bytesVec3Padded);
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
                                  const float* invUV,
                                  const float* stiffness) {
    if (!ok()) return;
    impl_->nTri = nTri;
    const NSUInteger bytesTriIdx = 3 * nTri * sizeof(uint32_t);
    const NSUInteger bytesInvUV  = 4 * nTri * sizeof(float);
    const NSUInteger bytesF      = nTri * sizeof(float);
    const NSUInteger bytesGradPadded   = 4 * 3 * nTri * sizeof(float);
    const NSUInteger bytesHessScalar   = 3 * nTri * sizeof(float);
    const MTLResourceOptions opts = MTLResourceStorageModeShared;

    impl_->bufTriIdx        = [impl_->device newBufferWithBytes:triIdx    length:bytesTriIdx options:opts];
    impl_->bufTriInvUV      = [impl_->device newBufferWithBytes:invUV     length:bytesInvUV  options:opts];
    impl_->bufTriStiffness  = [impl_->device newBufferWithBytes:stiffness length:bytesF      options:opts];
    impl_->bufTriGrad       = [impl_->device newBufferWithLength:bytesGradPadded options:opts];
    impl_->bufTriHessScalar = [impl_->device newBufferWithLength:bytesHessScalar options:opts];

    // AL state: λ_col0 / λ_col1 zero-init, γ defaults to stiffness.
    const NSUInteger bytesTriLam = 4 * nTri * sizeof(float);  // padded float3 per tri
    impl_->bufTriLambda0 = [impl_->device newBufferWithLength:bytesTriLam options:opts];
    impl_->bufTriLambda1 = [impl_->device newBufferWithLength:bytesTriLam options:opts];
    impl_->bufTriGamma   = [impl_->device newBufferWithBytes:stiffness length:bytesF options:opts];
    std::memset(impl_->bufTriLambda0.contents, 0, bytesTriLam);
    std::memset(impl_->bufTriLambda1.contents, 0, bytesTriLam);

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

    // AL state: λ zero-init, γ defaults to stiffness.
    const NSUInteger bytesBendLam = 4 * nBend * sizeof(float);
    impl_->bufBendLambda = [impl_->device newBufferWithLength:bytesBendLam options:opts];
    impl_->bufBendGamma  = [impl_->device newBufferWithBytes:stiffness length:bytesF options:opts];
    std::memset(impl_->bufBendLambda.contents, 0, bytesBendLam);

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

    // Coloring-aware AVBD outer iter. For each color k:
    //   1. vbd_init for color-k verts (g/H = inertial)
    //   2. force kernels (constraint-indexed, full pass — they see the
    //      latest positions, including updates from earlier colors in
    //      this same iter)
    //   3. gather kernels for color-k verts (write color-k g/H)
    //   4. vbd_solve_apply for color-k verts (positions[v in C_k] += Δx)
    // With numColors=1 + identity vertPerm (default state set by
    // setupMesh), this is one full sweep equivalent to the pre-PR
    // block-Jacobi behavior. When buildVertexColoringIfNeeded() has
    // been called, the per-color loop gives true Gauss-Seidel.
    id<MTLCommandBuffer> cb = [impl_->queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];

    const uint32_t numColors = impl_->numColors;
    for (uint32_t k = 0; k < numColors; ++k) {
        const uint32_t offset = impl_->colorOffsets[k];
        const uint32_t end    = impl_->colorOffsets[k + 1];
        const uint32_t count  = end - offset;
        if (count == 0) continue;

        // 1. vbd_init for color k.
        [enc setComputePipelineState:impl_->psoInit];
        {
            VbdInitParams ip{impl_->invHSqCached, offset};
            [enc setBytes:&ip length:sizeof(ip) atIndex:0];
        }
        [enc setBuffer:impl_->bufPositions  offset:0 atIndex:1];
        [enc setBuffer:impl_->bufPredicted  offset:0 atIndex:2];
        [enc setBuffer:impl_->bufMass       offset:0 atIndex:3];
        [enc setBuffer:impl_->bufGScratch   offset:0 atIndex:4];
        [enc setBuffer:impl_->bufHScratch   offset:0 atIndex:5];
        [enc setBuffer:impl_->bufVertPerm   offset:0 atIndex:6];
        [enc dispatchThreads:MTLSizeMake(count, 1, 1)
       threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];

        // 2a. spring_force (constraint-indexed).
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

            // 3a. vbd_gather_spring for color k.
            [enc setComputePipelineState:impl_->psoGatherSpring];
            [enc setBuffer:impl_->bufSpringGradA      offset:0 atIndex:0];
            [enc setBuffer:impl_->bufSpringHess       offset:0 atIndex:1];
            [enc setBuffer:impl_->bufVertSpringOffset offset:0 atIndex:2];
            [enc setBuffer:impl_->bufVertSpringIdx    offset:0 atIndex:3];
            [enc setBuffer:impl_->bufVertSpringRole   offset:0 atIndex:4];
            [enc setBuffer:impl_->bufGScratch         offset:0 atIndex:5];
            [enc setBuffer:impl_->bufHScratch         offset:0 atIndex:6];
            [enc setBuffer:impl_->bufVertPerm         offset:0 atIndex:7];
            {
                VbdGatherParams gp{offset};
                [enc setBytes:&gp length:sizeof(gp) atIndex:8];
            }
            [enc dispatchThreads:MTLSizeMake(count, 1, 1)
           threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
        }

        // 2b/3b. attachment_force_al + vbd_gather_attachment.
        if (impl_->nAttach > 0) {
            [enc setComputePipelineState:impl_->psoAttachmentForceAl];
            [enc setBuffer:impl_->bufPositions          offset:0 atIndex:0];
            [enc setBuffer:impl_->bufAttachVertIdx      offset:0 atIndex:1];
            [enc setBuffer:impl_->bufAttachFixedPos     offset:0 atIndex:2];
            [enc setBuffer:impl_->bufAttachStiffness    offset:0 atIndex:3];
            [enc setBuffer:impl_->bufAttachLambda       offset:0 atIndex:4];
            [enc setBuffer:impl_->bufAttachGradV        offset:0 atIndex:5];
            [enc setBuffer:impl_->bufAttachHessScalar   offset:0 atIndex:6];
            [enc dispatchThreads:MTLSizeMake(impl_->nAttach, 1, 1)
           threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];

            [enc setComputePipelineState:impl_->psoGatherAttachment];
            [enc setBuffer:impl_->bufAttachGradV        offset:0 atIndex:0];
            [enc setBuffer:impl_->bufAttachHessScalar   offset:0 atIndex:1];
            [enc setBuffer:impl_->bufVertAttachOffset   offset:0 atIndex:2];
            [enc setBuffer:impl_->bufVertAttachIdx      offset:0 atIndex:3];
            [enc setBuffer:impl_->bufGScratch           offset:0 atIndex:4];
            [enc setBuffer:impl_->bufHScratch           offset:0 atIndex:5];
            [enc setBuffer:impl_->bufVertPerm           offset:0 atIndex:6];
            {
                VbdGatherParams gp{offset};
                [enc setBytes:&gp length:sizeof(gp) atIndex:7];
            }
            [enc dispatchThreads:MTLSizeMake(count, 1, 1)
           threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
        }

        // 2c/3c. triangle_membrane_force_al + vbd_gather_triangle.
        if (impl_->nTri > 0) {
            [enc setComputePipelineState:impl_->psoTriangleMembraneForceAl];
            [enc setBuffer:impl_->bufPositions      offset:0 atIndex:0];
            [enc setBuffer:impl_->bufTriIdx         offset:0 atIndex:1];
            [enc setBuffer:impl_->bufTriStiffness   offset:0 atIndex:2];
            [enc setBuffer:impl_->bufTriLambda0     offset:0 atIndex:3];
            [enc setBuffer:impl_->bufTriLambda1     offset:0 atIndex:4];
            [enc setBuffer:impl_->bufTriGrad        offset:0 atIndex:5];
            [enc setBuffer:impl_->bufTriHessScalar  offset:0 atIndex:6];
            [enc setBuffer:impl_->bufTriInvUV       offset:0 atIndex:7];
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
            [enc setBuffer:impl_->bufVertPerm       offset:0 atIndex:7];
            {
                VbdGatherParams gp{offset};
                [enc setBytes:&gp length:sizeof(gp) atIndex:8];
            }
            [enc dispatchThreads:MTLSizeMake(count, 1, 1)
           threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
        }

        // 2d/3d. triangle_bending_force_al + vbd_gather_bending.
        if (impl_->nBend > 0) {
            [enc setComputePipelineState:impl_->psoTriangleBendingForceAl];
            [enc setBuffer:impl_->bufPositions       offset:0 atIndex:0];
            [enc setBuffer:impl_->bufBendIdx         offset:0 atIndex:1];
            [enc setBuffer:impl_->bufBendWeight      offset:0 atIndex:2];
            [enc setBuffer:impl_->bufBendNTarget     offset:0 atIndex:3];
            [enc setBuffer:impl_->bufBendStiffness   offset:0 atIndex:4];
            [enc setBuffer:impl_->bufBendLambda      offset:0 atIndex:5];
            [enc setBuffer:impl_->bufBendGrad        offset:0 atIndex:6];
            [enc setBuffer:impl_->bufBendHessScalar  offset:0 atIndex:7];
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
            [enc setBuffer:impl_->bufVertPerm        offset:0 atIndex:7];
            {
                VbdGatherParams gp{offset};
                [enc setBytes:&gp length:sizeof(gp) atIndex:8];
            }
            [enc dispatchThreads:MTLSizeMake(count, 1, 1)
           threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
        }

        // 4. vbd_solve_apply for color k.
        [enc setComputePipelineState:impl_->psoSolveApply];
        [enc setBuffer:impl_->bufGScratch  offset:0 atIndex:0];
        [enc setBuffer:impl_->bufHScratch  offset:0 atIndex:1];
        [enc setBuffer:impl_->bufPositions offset:0 atIndex:2];
        [enc setBuffer:impl_->bufVertPerm  offset:0 atIndex:3];
        {
            VbdSolveApplyParams sp{offset};
            [enc setBytes:&sp length:sizeof(sp) atIndex:4];
        }
        [enc dispatchThreads:MTLSizeMake(count, 1, 1)
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
    readVec3Padded(gScratch_out, impl_->bufGScratch, n);
    hScratch_out.assign(6 * n, 0.0f);
    std::memcpy(hScratch_out.data(), impl_->bufHScratch.contents, 6 * n * sizeof(float));
}

void AvbdSolver::updateState(const float* positions, const float* predicted) {
    if (!ok() || !impl_->meshReady) return;
    uploadVec3Padded(impl_->bufPositions, positions, impl_->nVerts);
    uploadVec3Padded(impl_->bufPredicted, predicted, impl_->nVerts);
}

int AvbdSolver::stepDualAttachments() {
    if (!ok() || !impl_->meshReady) return -1;
    if (impl_->nAttach == 0) return 0;
    id<MTLCommandBuffer> cb = [impl_->queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    [enc setComputePipelineState:impl_->psoAttachmentDualUpdate];
    [enc setBuffer:impl_->bufPositions       offset:0 atIndex:0];
    [enc setBuffer:impl_->bufAttachVertIdx   offset:0 atIndex:1];
    [enc setBuffer:impl_->bufAttachFixedPos  offset:0 atIndex:2];
    [enc setBuffer:impl_->bufAttachGamma     offset:0 atIndex:3];
    [enc setBuffer:impl_->bufAttachLambda    offset:0 atIndex:4];
    [enc dispatchThreads:MTLSizeMake(impl_->nAttach, 1, 1)
   threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
    [enc endEncoding];
    [cb commit];
    [cb waitUntilCompleted];
    return 0;
}

int AvbdSolver::stepDualMembrane() {
    if (!ok() || !impl_->meshReady) return -1;
    if (impl_->nTri == 0) return 0;
    id<MTLCommandBuffer> cb = [impl_->queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    [enc setComputePipelineState:impl_->psoTriangleMembraneDualUpdate];
    [enc setBuffer:impl_->bufPositions  offset:0 atIndex:0];
    [enc setBuffer:impl_->bufTriIdx     offset:0 atIndex:1];
    [enc setBuffer:impl_->bufTriGamma   offset:0 atIndex:2];
    [enc setBuffer:impl_->bufTriLambda0 offset:0 atIndex:3];
    [enc setBuffer:impl_->bufTriLambda1 offset:0 atIndex:4];
    [enc setBuffer:impl_->bufTriInvUV   offset:0 atIndex:5];
    [enc dispatchThreads:MTLSizeMake(impl_->nTri, 1, 1)
   threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
    [enc endEncoding];
    [cb commit];
    [cb waitUntilCompleted];
    return 0;
}

void AvbdSolver::setGammaScale(float scale) {
    if (!ok()) return;
    auto scaleBuf = [scale](id<MTLBuffer> buf, uint32_t n) {
        if (!buf || n == 0) return;
        float* p = static_cast<float*>(buf.contents);
        for (uint32_t i = 0; i < n; ++i) p[i] *= scale;
    };
    scaleBuf(impl_->bufAttachGamma, impl_->nAttach);
    scaleBuf(impl_->bufTriGamma,    impl_->nTri);
    scaleBuf(impl_->bufBendGamma,   impl_->nBend);
}

int AvbdSolver::stepDualBending() {
    if (!ok() || !impl_->meshReady) return -1;
    if (impl_->nBend == 0) return 0;
    id<MTLCommandBuffer> cb = [impl_->queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    [enc setComputePipelineState:impl_->psoTriangleBendingDualUpdate];
    [enc setBuffer:impl_->bufPositions    offset:0 atIndex:0];
    [enc setBuffer:impl_->bufBendIdx      offset:0 atIndex:1];
    [enc setBuffer:impl_->bufBendWeight   offset:0 atIndex:2];
    [enc setBuffer:impl_->bufBendNTarget  offset:0 atIndex:3];
    [enc setBuffer:impl_->bufBendGamma    offset:0 atIndex:4];
    [enc setBuffer:impl_->bufBendLambda   offset:0 atIndex:5];
    [enc dispatchThreads:MTLSizeMake(impl_->nBend, 1, 1)
   threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
    [enc endEncoding];
    [cb commit];
    [cb waitUntilCompleted];
    return 0;
}

void AvbdSolver::buildColoring() {
    if (!ok() || !impl_->meshReady) return;
    impl_->buildVertexColoringIfNeeded();
}

void AvbdSolver::uploadSelfCollisionRadii(const float* radii,
                                          uint32_t maxNeighborsPerVert) {
    if (!ok() || !impl_->meshReady) return;
    const uint32_t nV = impl_->nVerts;
    const MTLResourceOptions opts = MTLResourceStorageModeShared;
    impl_->bufRadii =
        [impl_->device newBufferWithBytes:radii
                                   length:nV * sizeof(float)
                                  options:opts];
    impl_->selfCollK = maxNeighborsPerVert;
    const NSUInteger bytesNeighbors =
        NSUInteger(nV) * NSUInteger(maxNeighborsPerVert) * sizeof(uint32_t);
    impl_->bufNeighbors =
        [impl_->device newBufferWithLength:bytesNeighbors options:opts];
    impl_->selfCollReady = true;
}

int AvbdSolver::detectSelfCollisions(
        std::vector<std::pair<uint32_t, uint32_t>>& out_pairs) {
    out_pairs.clear();
    if (!ok() || !impl_->meshReady || !impl_->selfCollReady) return -1;

    const uint32_t nV = impl_->nVerts;
    const uint32_t K  = impl_->selfCollK;

    id<MTLCommandBuffer> cb = [impl_->queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    [enc setComputePipelineState:impl_->psoSelfCollisionScan];
    [enc setBuffer:impl_->bufPositions offset:0 atIndex:0];
    [enc setBuffer:impl_->bufRadii     offset:0 atIndex:1];
    [enc setBuffer:impl_->bufNeighbors offset:0 atIndex:2];
    SelfCollisionScanParams sp{nV, K};
    [enc setBytes:&sp length:sizeof(sp) atIndex:3];
    [enc dispatchThreads:MTLSizeMake(nV, 1, 1)
   threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
    [enc endEncoding];
    [cb commit];
    [cb waitUntilCompleted];

    // Read pairs from neighbors buffer. Dedupe by (lo, hi) ordering.
    const uint32_t* neighbors =
        static_cast<const uint32_t*>(impl_->bufNeighbors.contents);
    constexpr uint32_t SENTINEL = 0xFFFFFFFFu;
    out_pairs.reserve(nV);
    for (uint32_t i = 0; i < nV; ++i) {
        const uint32_t* row = neighbors + (size_t(i) * K);
        for (uint32_t k = 0; k < K; ++k) {
            const uint32_t j = row[k];
            if (j == SENTINEL) break;
            if (j > i) out_pairs.emplace_back(i, j);  // skip mirror (j, i)
        }
    }
    return 0;
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
