// GENERATED from AvbdSolver.mm by port_avbd.py, then maintained by hand.
// slang-rhi backend (Vulkan / D3D12 / CUDA / Metal). Same cloth::AvbdSolver
// interface as the Metal host; build exactly one of the two.
// AvbdSolver implementation (PR-E stub). Loads the six AVBD kernel
// metallibs and builds compute pipeline state objects. Per-step
// dispatch is a stub — actual GPU loop wiring lands in follow-up PRs.

#include "AvbdSolver.h"

#include "RhiContext.h"

#include <algorithm>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace cloth {

// Backend selection for slang-rhi. Override with AVBD_RHI_BACKEND=vulkan|d3d12|cuda|metal|cpu.
static rhi::DeviceType pickDeviceType() {
    if (const char* e = std::getenv("AVBD_RHI_BACKEND")) {
        std::string v = e;
        if (v == "d3d12") return rhi::DeviceType::D3D12;
        if (v == "cuda") return rhi::DeviceType::CUDA;
        if (v == "metal") return rhi::DeviceType::Metal;
        if (v == "cpu") return rhi::DeviceType::CPU;
        return rhi::DeviceType::Vulkan;
    }
#ifdef __APPLE__
    return rhi::DeviceType::Metal;
#else
    return rhi::DeviceType::Vulkan;
#endif
}

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

// Matches the VbdInitBackwardParams struct emitted by
// Cloth.SlangCodegen.VbdInitBackward (the adjoint of vbd_init's
// inertial seed; needs invHSquared to scale w = m_v · invHSq).
struct VbdInitBackwardParams {
    float invHSquared;
};

struct AvbdSolver::Impl {
    RhiContext ctx{pickDeviceType()};
    Buf mk(const void* init, size_t bytes) { return ctx.makeBuffer(init, bytes); }

    // Six AVBD pipeline kernels. Each loaded from a separate
    // metallib at construction time.

    // Per-constraint-type force kernels (compute force/Hess from
    // current positions before the gather kernel scatters them
    // per-vertex).

    // ---------------- Reverse-mode adjoint kernels (PR-G / CHI-13) ----------------
    // 10 backward PSOs. Loaded at construction; nil until then. Mirrors
    // the forward pipeline in reverse order (apply → gathers → force).

    // Self-collision GPU scan state (allocated by
    // uploadSelfCollisionRadii). bufNeighbors holds nVerts·K uints,
    // each row listing up to K overlapping neighbor indices for the
    // vertex (UINT_MAX sentinel for empty slots).
    Buf bufRadii;
    Buf bufNeighbors;
    uint32_t      selfCollK      = 0;     // 0 = not uploaded
    bool          selfCollReady  = false;

    // Per-vertex state buffers (allocated by setupMesh).
    Buf bufPositions;   // length = 3 * nVerts (float)
    Buf bufPredicted;   // length = 3 * nVerts (float)
    Buf bufMass;   // length = nVerts (float)
    Buf bufGScratch;   // length = 3 * nVerts (float)
    Buf bufHScratch;   // length = 6 * nVerts (float)
    Buf bufInitParams;  // sizeof(VbdInitParams)

    // Coloring-aware dispatch state. `bufVertPerm` is a permutation of
    // [0..nVerts), grouped by color. `colorOffsets` (host-side) has
    // length numColors+1: color k holds vertices
    // vertPerm[colorOffsets[k] .. colorOffsets[k+1]). Default = single
    // identity color (numColors=1, identity perm) — equivalent to
    // pre-PR block Jacobi. uploadTriangles builds the real coloring
    // from triangle adjacency.
    Buf bufVertPerm;
    Buf bufSolveApplyParams;  // sizeof(VbdSolveApplyParams), kept for compat
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
    Buf bufSpringP1Idx;  // uint per spring
    Buf bufSpringP2Idx;  // uint per spring
    Buf bufSpringRestLen;  // float per spring
    Buf bufSpringStiffness;  // float per spring
    Buf bufSpringGradA;  // 3 floats per spring
    Buf bufSpringHess;  // 6 floats per spring

    // Vertex → spring CSR adjacency (built host-side in uploadSprings,
    // mirroring Cloth.Avbd.AdjacencySpring's `native_decide`-pinned
    // algorithm).
    Buf bufVertSpringOffset; // uint per (nVerts+1)
    Buf bufVertSpringIdx; // uint per (2*nSprings)
    Buf bufVertSpringRole; // uint per (2*nSprings)

    // Attachment constraint buffers (allocated by uploadAttachments).
    // Each attachment pins one vertex to a fixed world anchor.
    Buf bufAttachVertIdx; // uint per attachment
    Buf bufAttachFixedPos; // padded float3 per attachment
    Buf bufAttachStiffness; // float per attachment
    Buf bufAttachGradV; // padded float3 per attachment
    Buf bufAttachHessScalar; // float per attachment

    // Augmented-Lagrangian state per attachment (PR-F).
    // λ_c — dual multiplier, accumulates across outer iters (padded float3).
    // γ_c — AL penalty (per-attachment, defaults to stiffness).
    Buf bufAttachLambda;
    Buf bufAttachGamma;

    // Vertex → attachment CSR. Role is implicit (always 0 for K=1).
    Buf bufVertAttachOffset; // uint per (nVerts+1)
    Buf bufVertAttachIdx; // uint per nAttach

    // Triangle membrane constraint buffers (allocated by uploadTriangles).
    Buf bufTriIdx; // uint per (3 * nTri) — corners
    Buf bufTriStiffness; // float per nTri
    Buf bufTriInvUV; // float per (4 * nTri) — rest material inv (2x2 per tri)
    Buf bufTriGrad; // padded float3 per (3 * nTri)
    Buf bufTriHessScalar; // float per (3 * nTri)

    // AL state per membrane triangle (PR-F).
    Buf bufTriLambda0; // padded float3 per tri (col 0)
    Buf bufTriLambda1; // padded float3 per tri (col 1)
    Buf bufTriGamma; // float per tri

    // Vertex → triangle CSR. K=3, so role ∈ {0, 1, 2}.
    Buf bufVertTriOffset; // uint per (nVerts+1)
    Buf bufVertTriIdx; // uint per (3 * nTri)
    Buf bufVertTriRole; // uint per (3 * nTri)

    // Bending constraint buffers (allocated by uploadBendings).
    Buf bufBendIdx; // uint per (4 * nBend) — 4-vert stencil
    Buf bufBendWeight; // float per (4 * nBend) — Laplacian weights
    Buf bufBendNTarget; // float per nBend — rest magnitude
    Buf bufBendStiffness; // float per nBend
    Buf bufBendGrad; // padded float3 per (4 * nBend)
    Buf bufBendHessScalar; // float per (4 * nBend)

    // AL state per bending constraint (PR-F).
    Buf bufBendLambda; // padded float3 per bend
    Buf bufBendGamma; // float per bend

    // Vertex → bending CSR. K=4, role ∈ {0, 1, 2, 3}.
    Buf bufVertBendOffset; // uint per (nVerts+1)
    Buf bufVertBendIdx; // uint per (4 * nBend)
    Buf bufVertBendRole; // uint per (4 * nBend)

    // ---------------- Backward cotangent buffers (PR-G / CHI-13) ----------------
    // Per-vertex cotangents (allocated lazily on first stepBackward()).
    Buf bufPositionsPreStep; // padded float3 per vert — snapshot at start of step()
    Buf bufVPositionsLoss; // padded float3 per vert — input v_x cotangent
    Buf bufVG; // padded float3 per vert — v_g (∂L/∂gScratch)
    Buf bufVH; // 6 floats per vert      — v_H (∂L/∂hScratch, sym 3x3)
    Buf bufDeltaX; // padded float3 per vert — cached forward Δx
    Buf bufVPositionsGrad; // padded float3 per vert — output ∂L/∂x_in
    // Scratch H buffer used as a junk write target when the forward
    // vbd_gather_* kernels are re-purposed to additively scatter
    // per-constraint position cotangents into bufVPositionsGrad.
    // (The gather kernels are RW on hScratch but we don't read it
    // after — having a dedicated buffer avoids racing with
    // solve_apply_backward which reads bufHScratch.)
    Buf bufHScratchJunk; // 6 floats per vert

    // PR-G Brick B (CHI-14): per-vertex cotangents from vbd_init_backward.
    // The init kernel writes the inertial seed g_v = w·(x_v − y_v),
    // H_v = w·I where w = m_v · invHSquared, so the inertial path is the
    // only place where density / mass flows. Outputs:
    //   v_x_init   — cotangent on x via the init g term (3-vec per vert).
    //                Added to bufVPositionsGrad on the CPU after dispatch.
    //   v_y_init   — cotangent on predicted (3-vec per vert). Unused for
    //                single-step inverse design; allocated as a junk
    //                write target.
    //   v_mass     — ∂L/∂m_v (scalar per vert). Read via readMassGrad.
    Buf bufVPositionsInit;
    Buf bufVPredictedJunk;
    Buf bufVMass;
    // VbdInitBackwardParams { invHSquared }; one float, packed via setBytes.
    // No buffer needed since setBytes can carry small constant buffers.

    // Per-constraint backward cotangents.
    Buf bufVSpringGradA; // padded float3 per spring   (input from gather bwd)
    Buf bufVSpringHess; // float per spring           (input from gather bwd)
    Buf bufVSpringPd; // padded float3 per spring   (output: ∂L/∂d)
    Buf bufVSpringRestLen; // float per spring           (output: ∂L/∂L)
    Buf bufVSpringStiff; // float per spring           (output: ∂L/∂k)

    Buf bufVAttachGradV;
    Buf bufVAttachHessScalar;
    Buf bufVAttachFixedPos;
    Buf bufVAttachLambda;
    Buf bufVAttachStiff;

    Buf bufVTriGrad;
    Buf bufVTriHessScalar;
    Buf bufVTriP; // padded float3 per (3 * nTri)
    Buf bufVTriStiff;
    Buf bufVTriLambda0;
    Buf bufVTriLambda1;

    Buf bufVBendGrad;
    Buf bufVBendHessScalar;
    Buf bufVBendP; // padded float3 per (4 * nBend)
    Buf bufVBendNTarget;
    Buf bufVBendStiff;
    Buf bufVBendLambda;

    // True once stepBackward() has been called at least once (lazy alloc).
    bool backwardReady = false;

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
            auto* p1 = static_cast<const uint32_t*>(bufSpringP1Idx.cdata());
            auto* p2 = static_cast<const uint32_t*>(bufSpringP2Idx.cdata());
            for (uint32_t c = 0; c < nSprings; ++c) edge(p1[c], p2[c]);
        }
        if (nTri > 0) {
            auto* idx = static_cast<const uint32_t*>(bufTriIdx.cdata());
            for (uint32_t c = 0; c < nTri; ++c) {
                const uint32_t i0 = idx[3*c+0];
                const uint32_t i1 = idx[3*c+1];
                const uint32_t i2 = idx[3*c+2];
                edge(i0, i1); edge(i0, i2); edge(i1, i2);
            }
        }
        if (nBend > 0) {
            auto* idx = static_cast<const uint32_t*>(bufBendIdx.cdata());
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

        auto* perm = static_cast<uint32_t*>(bufVertPerm.contents());
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
};

AvbdSolver::AvbdSolver(const char* shaderDir)
    : impl_(new Impl()) {
    if (!impl_->ctx.ok()) {
        std::fprintf(stderr, "AvbdSolver: no slang-rhi device (%s)\n",
                     impl_->ctx.error().c_str());
        return;
    }
    static const char* const kForward[] = { "vbd_init", "vbd_gather_spring", "vbd_gather_attachment", "vbd_gather_triangle", "vbd_gather_bending", "vbd_solve_apply", "spring_force", "attachment_force_al", "attachment_dual_update", "triangle_membrane_force_al", "triangle_membrane_dual_update", "triangle_bending_force_al", "triangle_bending_dual_update", "self_collision_scan" };
    static const char* const kBackward[] = { "vbd_solve_apply_backward", "vbd_init_backward", "vbd_gather_spring_backward", "vbd_gather_attachment_backward", "vbd_gather_triangle_backward", "vbd_gather_bending_backward", "spring_force_backward", "attachment_force_al_backward", "triangle_membrane_force_al_backward", "triangle_bending_force_al_backward" };
    for (const char* k : kForward) {
        if (!impl_->ctx.loadKernel(shaderDir, k)) {
            std::fprintf(stderr, "AvbdSolver: %s\n", impl_->ctx.error().c_str());
            return;
        }
    }
    // Backward kernels are optional: forward stays usable without them.
    int backwardLoaded = 0;
    for (const char* k : kBackward) {
        if (impl_->ctx.loadKernel(shaderDir, k)) ++backwardLoaded;
        else std::fprintf(stderr, "AvbdSolver: %s\n", impl_->ctx.error().c_str());
    }
    impl_->ok = true;
    std::fprintf(stderr,
        "AvbdSolver[slang-rhi %s]: 14 forward kernels + %d/10 backward kernels loaded\n",
        impl_->ctx.adapterName(), backwardLoaded);
}

AvbdSolver::~AvbdSolver() { delete impl_; }

bool AvbdSolver::ok() const { return impl_ && impl_->ok; }

// On slang-rhi backends StructuredBuffer<float3> is tightly packed (12 B
// stride), so host arrays upload and read back with a plain copy.
static void uploadVec3Padded(Buf& dst, const float* src, uint32_t n) {
    std::memcpy(dst.contents(), src, size_t(n) * 3 * sizeof(float));
}

static void readVec3Padded(std::vector<float>& dst, const Buf& src, uint32_t n) {
    dst.assign(3 * size_t(n), 0.0f);
    std::memcpy(dst.data(), src.cdata(), size_t(n) * 3 * sizeof(float));
}

void AvbdSolver::setupMesh(uint32_t nVerts,
                           const float* positions,
                           const float* predicted,
                           const float* mass,
                           float invHSquared) {
    if (!ok()) return;
    impl_->nVerts = nVerts;
    const size_t bytesVec3Padded = 3 * nVerts * sizeof(float);
    const size_t bytesScalar = nVerts * sizeof(float);
    const size_t bytesHess = 6 * nVerts * sizeof(float);

    impl_->bufPositions = impl_->mk(nullptr, bytesVec3Padded);
    impl_->bufPredicted = impl_->mk(nullptr, bytesVec3Padded);
    impl_->bufGScratch  = impl_->mk(nullptr, bytesVec3Padded);
    impl_->bufMass      = impl_->mk(mass, bytesScalar);
    impl_->bufHScratch  = impl_->mk(nullptr, bytesHess);
    // Snapshot buffer for the backward pass — step() copies bufPositions
    // here before any solve_apply mutates it, so stepBackward can
    // recompute Δx = positions_after − positions_before.
    impl_->bufPositionsPreStep = impl_->mk(nullptr, bytesVec3Padded);

    uploadVec3Padded(impl_->bufPositions, positions, nVerts);
    uploadVec3Padded(impl_->bufPredicted, predicted, nVerts);

    impl_->bufInitParams = impl_->mk(nullptr, sizeof(VbdInitParams));
    *static_cast<VbdInitParams*>(impl_->bufInitParams.contents()) = {invHSquared, 0u};
    impl_->invHSqCached = invHSquared;

    // Default vertex permutation = identity (lane i → vertex i) and
    // solve_apply colorOffset = 0. Equivalent to a single color spanning
    // all vertices — same behavior as pre-coloring. buildVertexColoring()
    // overwrites these with greedy first-fit when stepColored() runs.
    const size_t bytesPerm = nVerts * sizeof(uint32_t);
    impl_->bufVertPerm = impl_->mk(nullptr, bytesPerm);
    uint32_t* perm = static_cast<uint32_t*>(impl_->bufVertPerm.contents());
    for (uint32_t i = 0; i < nVerts; ++i) perm[i] = i;

    impl_->bufSolveApplyParams = impl_->mk(nullptr, sizeof(VbdSolveApplyParams));
    static_cast<VbdSolveApplyParams*>(impl_->bufSolveApplyParams.contents())
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
    const size_t bytesU = nSprings * sizeof(uint32_t);
    const size_t bytesF = nSprings * sizeof(float);
    const size_t bytesGradPadded = 3 * nSprings * sizeof(float);
    const size_t bytesHess = 6 * nSprings * sizeof(float);

    impl_->bufSpringP1Idx     = impl_->mk(p1Idx, bytesU);
    impl_->bufSpringP2Idx     = impl_->mk(p2Idx, bytesU);
    impl_->bufSpringRestLen   = impl_->mk(restLen, bytesF);
    impl_->bufSpringStiffness = impl_->mk(stiffness, bytesF);
    // gradA is `StructuredBuffer<float3>` → padded 16-byte stride per spring.
    impl_->bufSpringGradA     = impl_->mk(nullptr, bytesGradPadded);
    impl_->bufSpringHess      = impl_->mk(nullptr, bytesHess);

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
        impl_->mk(offsets.data(), offsets.size() * sizeof(uint32_t));
    impl_->bufVertSpringIdx =
        impl_->mk(springIdx.data(), springIdx.size() * sizeof(uint32_t));
    impl_->bufVertSpringRole =
        impl_->mk(role.data(), role.size() * sizeof(uint32_t));
}

void AvbdSolver::uploadAttachments(uint32_t nAttach,
                                    const uint32_t* vertIdx,
                                    const float* fixedPos,
                                    const float* stiffness) {
    if (!ok()) return;
    impl_->nAttach = nAttach;
    const size_t bytesU = nAttach * sizeof(uint32_t);
    const size_t bytesF = nAttach * sizeof(float);
    const size_t bytesVec3Padded = 3 * nAttach * sizeof(float);

    impl_->bufAttachVertIdx    = impl_->mk(vertIdx, bytesU);
    impl_->bufAttachFixedPos   = impl_->mk(nullptr, bytesVec3Padded);
    impl_->bufAttachStiffness  = impl_->mk(stiffness, bytesF);
    impl_->bufAttachGradV      = impl_->mk(nullptr, bytesVec3Padded);
    impl_->bufAttachHessScalar = impl_->mk(nullptr, bytesF);
    // AL state: λ_c starts at zero (any prior accumulation discarded
    // on re-upload); γ_c defaults to k_c so dual ramp tracks the
    // physical stiffness.
    impl_->bufAttachLambda     = impl_->mk(nullptr, bytesVec3Padded);
    impl_->bufAttachGamma      = impl_->mk(stiffness, bytesF);
    std::memset(impl_->bufAttachLambda.contents(), 0, bytesVec3Padded);
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
        impl_->mk(offsets.data(), offsets.size() * sizeof(uint32_t));
    impl_->bufVertAttachIdx =
        impl_->mk(attachIdx.data(), attachIdx.size() * sizeof(uint32_t));
}

void AvbdSolver::uploadTriangles(uint32_t nTri,
                                  const uint32_t* triIdx,
                                  const float* invUV,
                                  const float* stiffness) {
    if (!ok()) return;
    impl_->nTri = nTri;
    const size_t bytesTriIdx = 3 * nTri * sizeof(uint32_t);
    const size_t bytesInvUV  = 4 * nTri * sizeof(float);
    const size_t bytesF      = nTri * sizeof(float);
    const size_t bytesGradPadded   = 3 * 3 * nTri * sizeof(float);
    const size_t bytesHessScalar   = 3 * nTri * sizeof(float);

    impl_->bufTriIdx        = impl_->mk(triIdx, bytesTriIdx);
    impl_->bufTriInvUV      = impl_->mk(invUV, bytesInvUV);
    impl_->bufTriStiffness  = impl_->mk(stiffness, bytesF);
    impl_->bufTriGrad       = impl_->mk(nullptr, bytesGradPadded);
    impl_->bufTriHessScalar = impl_->mk(nullptr, bytesHessScalar);

    // AL state: λ_col0 / λ_col1 zero-init, γ defaults to stiffness.
    const size_t bytesTriLam = 3 * nTri * sizeof(float);  // padded float3 per tri
    impl_->bufTriLambda0 = impl_->mk(nullptr, bytesTriLam);
    impl_->bufTriLambda1 = impl_->mk(nullptr, bytesTriLam);
    impl_->bufTriGamma   = impl_->mk(stiffness, bytesF);
    std::memset(impl_->bufTriLambda0.contents(), 0, bytesTriLam);
    std::memset(impl_->bufTriLambda1.contents(), 0, bytesTriLam);

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
        impl_->mk(offsets.data(), offsets.size() * sizeof(uint32_t));
    impl_->bufVertTriIdx =
        impl_->mk(idxArr.data(), idxArr.size() * sizeof(uint32_t));
    impl_->bufVertTriRole =
        impl_->mk(roleArr.data(), roleArr.size() * sizeof(uint32_t));
}

void AvbdSolver::uploadBendings(uint32_t nBend,
                                 const uint32_t* bendIdx,
                                 const float* weight,
                                 const float* nTarget,
                                 const float* stiffness) {
    if (!ok()) return;
    impl_->nBend = nBend;
    const size_t bytesBendIdx    = 4 * nBend * sizeof(uint32_t);
    const size_t bytesBendWeight = 4 * nBend * sizeof(float);
    const size_t bytesF          = nBend * sizeof(float);
    const size_t bytesGradPadded = 3 * 4 * nBend * sizeof(float);
    const size_t bytesHessScalar = 4 * nBend * sizeof(float);

    impl_->bufBendIdx        = impl_->mk(bendIdx, bytesBendIdx);
    impl_->bufBendWeight     = impl_->mk(weight, bytesBendWeight);
    impl_->bufBendNTarget    = impl_->mk(nTarget, bytesF);
    impl_->bufBendStiffness  = impl_->mk(stiffness, bytesF);
    impl_->bufBendGrad       = impl_->mk(nullptr, bytesGradPadded);
    impl_->bufBendHessScalar = impl_->mk(nullptr, bytesHessScalar);

    // AL state: λ zero-init, γ defaults to stiffness.
    const size_t bytesBendLam = 3 * nBend * sizeof(float);
    impl_->bufBendLambda = impl_->mk(nullptr, bytesBendLam);
    impl_->bufBendGamma  = impl_->mk(stiffness, bytesF);
    std::memset(impl_->bufBendLambda.contents(), 0, bytesBendLam);

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
        impl_->mk(offsets.data(), offsets.size() * sizeof(uint32_t));
    impl_->bufVertBendIdx =
        impl_->mk(idxArr.data(), idxArr.size() * sizeof(uint32_t));
    impl_->bufVertBendRole =
        impl_->mk(roleArr.data(), roleArr.size() * sizeof(uint32_t));
}

int AvbdSolver::step() {
    if (!ok() || !impl_->meshReady) return -1;

    // PR-G / CHI-13: snapshot positions before any iteration mutates
    // them. stepBackward() reads this to recompute Δx = positions_after
    // − positions_before for the vbd_solve_apply backward kernel.
    // Cheap blit; no synchronisation cost outside the existing command
    // buffer's wait.
    if (impl_->bufPositionsPreStep) {
        impl_->ctx.copyBuffer(impl_->bufPositionsPreStep, impl_->bufPositions);
    }

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

    const uint32_t numColors = impl_->numColors;
    for (uint32_t k = 0; k < numColors; ++k) {
        const uint32_t offset = impl_->colorOffsets[k];
        const uint32_t end    = impl_->colorOffsets[k + 1];
        const uint32_t count  = end - offset;
        if (count == 0) continue;

        // 1. vbd_init for color k.
        impl_->ctx.setPipeline("vbd_init");
        {
            VbdInitParams ip{impl_->invHSqCached, offset};
            impl_->ctx.setBytes(0, &ip, sizeof(ip));
        }
        impl_->ctx.setBuffer(1, impl_->bufPositions);
        impl_->ctx.setBuffer(2, impl_->bufPredicted);
        impl_->ctx.setBuffer(3, impl_->bufMass);
        impl_->ctx.setBuffer(4, impl_->bufGScratch);
        impl_->ctx.setBuffer(5, impl_->bufHScratch);
        impl_->ctx.setBuffer(6, impl_->bufVertPerm);
        impl_->ctx.dispatchThreads(count);

        // 2a. spring_force (constraint-indexed).
        if (impl_->nSprings > 0) {
            impl_->ctx.setPipeline("spring_force");
            impl_->ctx.setBuffer(0, impl_->bufPositions);
            impl_->ctx.setBuffer(1, impl_->bufSpringP1Idx);
            impl_->ctx.setBuffer(2, impl_->bufSpringP2Idx);
            impl_->ctx.setBuffer(3, impl_->bufSpringRestLen);
            impl_->ctx.setBuffer(4, impl_->bufSpringStiffness);
            impl_->ctx.setBuffer(5, impl_->bufSpringGradA);
            impl_->ctx.setBuffer(6, impl_->bufSpringHess);
            impl_->ctx.dispatchThreads(impl_->nSprings);

            // 3a. vbd_gather_spring for color k.
            impl_->ctx.setPipeline("vbd_gather_spring");
            impl_->ctx.setBuffer(0, impl_->bufSpringGradA);
            impl_->ctx.setBuffer(1, impl_->bufSpringHess);
            impl_->ctx.setBuffer(2, impl_->bufVertSpringOffset);
            impl_->ctx.setBuffer(3, impl_->bufVertSpringIdx);
            impl_->ctx.setBuffer(4, impl_->bufVertSpringRole);
            impl_->ctx.setBuffer(5, impl_->bufGScratch);
            impl_->ctx.setBuffer(6, impl_->bufHScratch);
            impl_->ctx.setBuffer(7, impl_->bufVertPerm);
            {
                VbdGatherParams gp{offset};
                impl_->ctx.setBytes(8, &gp, sizeof(gp));
            }
            impl_->ctx.dispatchThreads(count);
        }

        // 2b/3b. attachment_force_al + vbd_gather_attachment.
        if (impl_->nAttach > 0) {
            impl_->ctx.setPipeline("attachment_force_al");
            impl_->ctx.setBuffer(0, impl_->bufPositions);
            impl_->ctx.setBuffer(1, impl_->bufAttachVertIdx);
            impl_->ctx.setBuffer(2, impl_->bufAttachFixedPos);
            impl_->ctx.setBuffer(3, impl_->bufAttachStiffness);
            impl_->ctx.setBuffer(4, impl_->bufAttachLambda);
            impl_->ctx.setBuffer(5, impl_->bufAttachGradV);
            impl_->ctx.setBuffer(6, impl_->bufAttachHessScalar);
            impl_->ctx.dispatchThreads(impl_->nAttach);

            impl_->ctx.setPipeline("vbd_gather_attachment");
            impl_->ctx.setBuffer(0, impl_->bufAttachGradV);
            impl_->ctx.setBuffer(1, impl_->bufAttachHessScalar);
            impl_->ctx.setBuffer(2, impl_->bufVertAttachOffset);
            impl_->ctx.setBuffer(3, impl_->bufVertAttachIdx);
            impl_->ctx.setBuffer(4, impl_->bufGScratch);
            impl_->ctx.setBuffer(5, impl_->bufHScratch);
            impl_->ctx.setBuffer(6, impl_->bufVertPerm);
            {
                VbdGatherParams gp{offset};
                impl_->ctx.setBytes(7, &gp, sizeof(gp));
            }
            impl_->ctx.dispatchThreads(count);
        }

        // 2c/3c. triangle_membrane_force_al + vbd_gather_triangle.
        if (impl_->nTri > 0) {
            impl_->ctx.setPipeline("triangle_membrane_force_al");
            impl_->ctx.setBuffer(0, impl_->bufPositions);
            impl_->ctx.setBuffer(1, impl_->bufTriIdx);
            impl_->ctx.setBuffer(2, impl_->bufTriStiffness);
            impl_->ctx.setBuffer(3, impl_->bufTriLambda0);
            impl_->ctx.setBuffer(4, impl_->bufTriLambda1);
            impl_->ctx.setBuffer(5, impl_->bufTriGrad);
            impl_->ctx.setBuffer(6, impl_->bufTriHessScalar);
            impl_->ctx.setBuffer(7, impl_->bufTriInvUV);
            impl_->ctx.dispatchThreads(impl_->nTri);

            impl_->ctx.setPipeline("vbd_gather_triangle");
            impl_->ctx.setBuffer(0, impl_->bufTriGrad);
            impl_->ctx.setBuffer(1, impl_->bufTriHessScalar);
            impl_->ctx.setBuffer(2, impl_->bufVertTriOffset);
            impl_->ctx.setBuffer(3, impl_->bufVertTriIdx);
            impl_->ctx.setBuffer(4, impl_->bufVertTriRole);
            impl_->ctx.setBuffer(5, impl_->bufGScratch);
            impl_->ctx.setBuffer(6, impl_->bufHScratch);
            impl_->ctx.setBuffer(7, impl_->bufVertPerm);
            {
                VbdGatherParams gp{offset};
                impl_->ctx.setBytes(8, &gp, sizeof(gp));
            }
            impl_->ctx.dispatchThreads(count);
        }

        // 2d/3d. triangle_bending_force_al + vbd_gather_bending.
        if (impl_->nBend > 0) {
            impl_->ctx.setPipeline("triangle_bending_force_al");
            impl_->ctx.setBuffer(0, impl_->bufPositions);
            impl_->ctx.setBuffer(1, impl_->bufBendIdx);
            impl_->ctx.setBuffer(2, impl_->bufBendWeight);
            impl_->ctx.setBuffer(3, impl_->bufBendNTarget);
            impl_->ctx.setBuffer(4, impl_->bufBendStiffness);
            impl_->ctx.setBuffer(5, impl_->bufBendLambda);
            impl_->ctx.setBuffer(6, impl_->bufBendGrad);
            impl_->ctx.setBuffer(7, impl_->bufBendHessScalar);
            impl_->ctx.dispatchThreads(impl_->nBend);

            impl_->ctx.setPipeline("vbd_gather_bending");
            impl_->ctx.setBuffer(0, impl_->bufBendGrad);
            impl_->ctx.setBuffer(1, impl_->bufBendHessScalar);
            impl_->ctx.setBuffer(2, impl_->bufVertBendOffset);
            impl_->ctx.setBuffer(3, impl_->bufVertBendIdx);
            impl_->ctx.setBuffer(4, impl_->bufVertBendRole);
            impl_->ctx.setBuffer(5, impl_->bufGScratch);
            impl_->ctx.setBuffer(6, impl_->bufHScratch);
            impl_->ctx.setBuffer(7, impl_->bufVertPerm);
            {
                VbdGatherParams gp{offset};
                impl_->ctx.setBytes(8, &gp, sizeof(gp));
            }
            impl_->ctx.dispatchThreads(count);
        }

        // 4. vbd_solve_apply for color k.
        impl_->ctx.setPipeline("vbd_solve_apply");
        impl_->ctx.setBuffer(0, impl_->bufGScratch);
        impl_->ctx.setBuffer(1, impl_->bufHScratch);
        impl_->ctx.setBuffer(2, impl_->bufPositions);
        impl_->ctx.setBuffer(3, impl_->bufVertPerm);
        {
            VbdSolveApplyParams sp{offset};
            impl_->ctx.setBytes(4, &sp, sizeof(sp));
        }
        impl_->ctx.dispatchThreads(count);
    }

    impl_->ctx.flush();

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
    std::memcpy(hScratch_out.data(), impl_->bufHScratch.cdata(), 6 * n * sizeof(float));
}

void AvbdSolver::updateState(const float* positions, const float* predicted) {
    if (!ok() || !impl_->meshReady) return;
    uploadVec3Padded(impl_->bufPositions, positions, impl_->nVerts);
    uploadVec3Padded(impl_->bufPredicted, predicted, impl_->nVerts);
}

void AvbdSolver::updateAttachmentFixedPos(const float* fixedPos) {
    if (!ok() || impl_->nAttach == 0 || !impl_->bufAttachFixedPos) return;
    uploadVec3Padded(impl_->bufAttachFixedPos, fixedPos, impl_->nAttach);
}

int AvbdSolver::stepDualAttachments() {
    if (!ok() || !impl_->meshReady) return -1;
    if (impl_->nAttach == 0) return 0;
    impl_->ctx.setPipeline("attachment_dual_update");
    impl_->ctx.setBuffer(0, impl_->bufPositions);
    impl_->ctx.setBuffer(1, impl_->bufAttachVertIdx);
    impl_->ctx.setBuffer(2, impl_->bufAttachFixedPos);
    impl_->ctx.setBuffer(3, impl_->bufAttachGamma);
    impl_->ctx.setBuffer(4, impl_->bufAttachLambda);
    impl_->ctx.dispatchThreads(impl_->nAttach);
    impl_->ctx.flush();
    return 0;
}

int AvbdSolver::stepDualMembrane() {
    if (!ok() || !impl_->meshReady) return -1;
    if (impl_->nTri == 0) return 0;
    impl_->ctx.setPipeline("triangle_membrane_dual_update");
    impl_->ctx.setBuffer(0, impl_->bufPositions);
    impl_->ctx.setBuffer(1, impl_->bufTriIdx);
    impl_->ctx.setBuffer(2, impl_->bufTriGamma);
    impl_->ctx.setBuffer(3, impl_->bufTriLambda0);
    impl_->ctx.setBuffer(4, impl_->bufTriLambda1);
    impl_->ctx.setBuffer(5, impl_->bufTriInvUV);
    impl_->ctx.dispatchThreads(impl_->nTri);
    impl_->ctx.flush();
    return 0;
}

void AvbdSolver::setGammaScale(float scale) {
    if (!ok()) return;
    auto scaleBuf = [scale](Buf& buf, uint32_t n) {
        if (!buf || n == 0) return;
        float* p = static_cast<float*>(buf.contents());
        for (uint32_t i = 0; i < n; ++i) p[i] *= scale;
    };
    scaleBuf(impl_->bufAttachGamma, impl_->nAttach);
    scaleBuf(impl_->bufTriGamma,    impl_->nTri);
    scaleBuf(impl_->bufBendGamma,   impl_->nBend);
}

int AvbdSolver::stepDualBending() {
    if (!ok() || !impl_->meshReady) return -1;
    if (impl_->nBend == 0) return 0;
    impl_->ctx.setPipeline("triangle_bending_dual_update");
    impl_->ctx.setBuffer(0, impl_->bufPositions);
    impl_->ctx.setBuffer(1, impl_->bufBendIdx);
    impl_->ctx.setBuffer(2, impl_->bufBendWeight);
    impl_->ctx.setBuffer(3, impl_->bufBendNTarget);
    impl_->ctx.setBuffer(4, impl_->bufBendGamma);
    impl_->ctx.setBuffer(5, impl_->bufBendLambda);
    impl_->ctx.dispatchThreads(impl_->nBend);
    impl_->ctx.flush();
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
    impl_->bufRadii =
        impl_->mk(radii, nV * sizeof(float));
    impl_->selfCollK = maxNeighborsPerVert;
    const size_t bytesNeighbors =
        size_t(nV) * size_t(maxNeighborsPerVert) * sizeof(uint32_t);
    impl_->bufNeighbors =
        impl_->mk(nullptr, bytesNeighbors);
    impl_->selfCollReady = true;
}

int AvbdSolver::detectSelfCollisions(
        std::vector<std::pair<uint32_t, uint32_t>>& out_pairs) {
    out_pairs.clear();
    if (!ok() || !impl_->meshReady || !impl_->selfCollReady) return -1;

    const uint32_t nV = impl_->nVerts;
    const uint32_t K  = impl_->selfCollK;

    impl_->ctx.setPipeline("self_collision_scan");
    impl_->ctx.setBuffer(0, impl_->bufPositions);
    impl_->ctx.setBuffer(1, impl_->bufRadii);
    impl_->ctx.setBuffer(2, impl_->bufNeighbors);
    SelfCollisionScanParams sp{nV, K};
    impl_->ctx.setBytes(3, &sp, sizeof(sp));
    impl_->ctx.dispatchThreads(nV);
    impl_->ctx.flush();

    // Read pairs from neighbors buffer. Dedupe by (lo, hi) ordering.
    const uint32_t* neighbors =
        static_cast<const uint32_t*>(impl_->bufNeighbors.cdata());
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
    std::memcpy(hess_out.data(),  impl_->bufSpringHess.cdata(),  6 * n * sizeof(float));
}

// ============================================================================
// PR-G / CHI-13 — reverse-mode adjoint
// ============================================================================

// Allocate cotangent buffers on first stepBackward(). Sized to match the
// uploaded mesh + constraints. Idempotent: no-op once backwardReady.
namespace {
template <class Impl>
void ensureBackwardBuffers_impl(Impl* impl) {
    if (impl->backwardReady) return;
    const uint32_t nV = impl->nVerts;
    const size_t bytesVec3Padded_v = 3 * nV * sizeof(float);
    const size_t bytesHess_v       = 6 * nV * sizeof(float);

    impl->bufVPositionsLoss = impl->mk(nullptr, bytesVec3Padded_v);
    impl->bufVG             = impl->mk(nullptr, bytesVec3Padded_v);
    impl->bufVH             = impl->mk(nullptr, bytesHess_v);
    impl->bufDeltaX         = impl->mk(nullptr, bytesVec3Padded_v);
    impl->bufVPositionsGrad = impl->mk(nullptr, bytesVec3Padded_v);
    impl->bufHScratchJunk   = impl->mk(nullptr, bytesHess_v);
    impl->bufVPositionsInit = impl->mk(nullptr, bytesVec3Padded_v);
    impl->bufVPredictedJunk = impl->mk(nullptr, bytesVec3Padded_v);
    impl->bufVMass          = impl->mk(nullptr, nV * sizeof(float));

    if (impl->nSprings > 0) {
        const uint32_t n = impl->nSprings;
        const size_t v3 = 3 * n * sizeof(float);
        const size_t f  = n * sizeof(float);
        impl->bufVSpringGradA   = impl->mk(nullptr, v3);
        impl->bufVSpringHess    = impl->mk(nullptr, f);
        impl->bufVSpringPd      = impl->mk(nullptr, v3);
        impl->bufVSpringRestLen = impl->mk(nullptr, f);
        impl->bufVSpringStiff   = impl->mk(nullptr, f);
    }
    if (impl->nAttach > 0) {
        const uint32_t n = impl->nAttach;
        const size_t v3 = 3 * n * sizeof(float);
        const size_t f  = n * sizeof(float);
        impl->bufVAttachGradV      = impl->mk(nullptr, v3);
        impl->bufVAttachHessScalar = impl->mk(nullptr, f);
        impl->bufVAttachFixedPos   = impl->mk(nullptr, v3);
        impl->bufVAttachLambda     = impl->mk(nullptr, v3);
        impl->bufVAttachStiff      = impl->mk(nullptr, f);
    }
    if (impl->nTri > 0) {
        const uint32_t n = impl->nTri;
        const size_t v3K = 3 * 3 * n * sizeof(float);  // per-corner
        const size_t fK  = 3 * n * sizeof(float);
        const size_t v3  = 3 * n * sizeof(float);
        const size_t f   = n * sizeof(float);
        impl->bufVTriGrad       = impl->mk(nullptr, v3K);
        impl->bufVTriHessScalar = impl->mk(nullptr, fK);
        impl->bufVTriP          = impl->mk(nullptr, v3K);
        impl->bufVTriStiff      = impl->mk(nullptr, f);
        impl->bufVTriLambda0    = impl->mk(nullptr, v3);
        impl->bufVTriLambda1    = impl->mk(nullptr, v3);
    }
    if (impl->nBend > 0) {
        const uint32_t n = impl->nBend;
        const size_t v3K = 3 * 4 * n * sizeof(float);  // per-corner, K=4
        const size_t fK  = 4 * n * sizeof(float);
        const size_t v3  = 3 * n * sizeof(float);
        const size_t f   = n * sizeof(float);
        impl->bufVBendGrad       = impl->mk(nullptr, v3K);
        impl->bufVBendHessScalar = impl->mk(nullptr, fK);
        impl->bufVBendP          = impl->mk(nullptr, v3K);
        impl->bufVBendNTarget    = impl->mk(nullptr, f);
        impl->bufVBendStiff      = impl->mk(nullptr, f);
        impl->bufVBendLambda     = impl->mk(nullptr, v3);
    }
    impl->backwardReady = true;
}
}  // namespace

int AvbdSolver::stepBackward(const float* v_positions_loss) {
    if (!ok() || !impl_->meshReady) return -1;
    if (!impl_->ctx.hasKernel("vbd_solve_apply_backward") || !impl_->ctx.hasKernel("spring_force_backward")) return -1;

    ensureBackwardBuffers_impl(impl_);
    const uint32_t nV = impl_->nVerts;

    // Upload v_positions_loss (cotangent on x_out) and compute Δx on CPU.
    uploadVec3Padded(impl_->bufVPositionsLoss, v_positions_loss, nV);
    {
        const float* x_after  = static_cast<const float*>(impl_->bufPositions.cdata());
        const float* x_before = static_cast<const float*>(impl_->bufPositionsPreStep.cdata());
        float*       deltaX   = static_cast<float*>(impl_->bufDeltaX.contents());
        for (uint32_t v = 0; v < nV; ++v) {
            deltaX[3*v + 0] = x_after[3*v + 0] - x_before[3*v + 0];
            deltaX[3*v + 1] = x_after[3*v + 1] - x_before[3*v + 1];
            deltaX[3*v + 2] = x_after[3*v + 2] - x_before[3*v + 2];
        }
    }
    // Zero v_positions_grad — spring/tri/bend gathers will additively
    // scatter into it; attachment writes attached vertices directly.
    std::memset(impl_->bufVPositionsGrad.contents(), 0,
                impl_->bufVPositionsGrad.length);


    // ---- 1. vbd_solve_apply_backward: from v_x_out → (v_g, v_H) ---------
    impl_->ctx.setPipeline("vbd_solve_apply_backward");
    impl_->ctx.setBuffer(0, impl_->bufVPositionsLoss);
    impl_->ctx.setBuffer(1, impl_->bufHScratch);
    impl_->ctx.setBuffer(2, impl_->bufDeltaX);
    impl_->ctx.setBuffer(3, impl_->bufVG);
    impl_->ctx.setBuffer(4, impl_->bufVH);
    impl_->ctx.dispatchThreads(nV);


    // ---- 2. Per-constraint gather backward: (v_g, v_H) → per-constraint cotangents
    if (impl_->nSprings > 0 && impl_->ctx.hasKernel("vbd_gather_spring_backward")) {
        impl_->ctx.setPipeline("vbd_gather_spring_backward");
        impl_->ctx.setBuffer(0, impl_->bufSpringP1Idx);
        impl_->ctx.setBuffer(1, impl_->bufSpringP2Idx);
        impl_->ctx.setBuffer(2, impl_->bufVG);
        impl_->ctx.setBuffer(3, impl_->bufVH);
        impl_->ctx.setBuffer(4, impl_->bufVSpringGradA);
        impl_->ctx.setBuffer(5, impl_->bufVSpringHess);
        impl_->ctx.dispatchThreads(impl_->nSprings);
    }
    if (impl_->nAttach > 0 && impl_->ctx.hasKernel("vbd_gather_attachment_backward")) {
        impl_->ctx.setPipeline("vbd_gather_attachment_backward");
        impl_->ctx.setBuffer(0, impl_->bufAttachVertIdx);
        impl_->ctx.setBuffer(1, impl_->bufVG);
        impl_->ctx.setBuffer(2, impl_->bufVH);
        impl_->ctx.setBuffer(3, impl_->bufVAttachGradV);
        impl_->ctx.setBuffer(4, impl_->bufVAttachHessScalar);
        impl_->ctx.dispatchThreads(impl_->nAttach);
    }
    if (impl_->nTri > 0 && impl_->ctx.hasKernel("vbd_gather_triangle_backward")) {
        impl_->ctx.setPipeline("vbd_gather_triangle_backward");
        impl_->ctx.setBuffer(0, impl_->bufTriIdx);
        impl_->ctx.setBuffer(1, impl_->bufVG);
        impl_->ctx.setBuffer(2, impl_->bufVH);
        impl_->ctx.setBuffer(3, impl_->bufVTriGrad);
        impl_->ctx.setBuffer(4, impl_->bufVTriHessScalar);
        impl_->ctx.dispatchThreads(3 * impl_->nTri);
    }
    if (impl_->nBend > 0 && impl_->ctx.hasKernel("vbd_gather_bending_backward")) {
        impl_->ctx.setPipeline("vbd_gather_bending_backward");
        impl_->ctx.setBuffer(0, impl_->bufBendIdx);
        impl_->ctx.setBuffer(1, impl_->bufVG);
        impl_->ctx.setBuffer(2, impl_->bufVH);
        impl_->ctx.setBuffer(3, impl_->bufVBendGrad);
        impl_->ctx.setBuffer(4, impl_->bufVBendHessScalar);
        impl_->ctx.dispatchThreads(4 * impl_->nBend);
    }

    // ---- 3. Attachment force backward: writes v_positions[v=attached]
    // directly (1:1 attachment, no contention) + per-attachment param
    // cotangents. Run BEFORE spring/tri/bend gathers so their additive
    // scatter on top is correct.
    //
    // IMPORTANT: per-constraint force backward kernels read positions
    // as the forward did — i.e., BEFORE solve_apply mutated them. Use
    // bufPositionsPreStep (snapshotted at start of step()), NOT bufPositions.
    if (impl_->nAttach > 0 && impl_->ctx.hasKernel("attachment_force_al_backward")) {
        impl_->ctx.setPipeline("attachment_force_al_backward");
        impl_->ctx.setBuffer(0, impl_->bufPositionsPreStep);
        impl_->ctx.setBuffer(1, impl_->bufAttachVertIdx);
        impl_->ctx.setBuffer(2, impl_->bufAttachFixedPos);
        impl_->ctx.setBuffer(3, impl_->bufAttachStiffness);
        impl_->ctx.setBuffer(4, impl_->bufVAttachGradV);
        impl_->ctx.setBuffer(5, impl_->bufVAttachHessScalar);
        impl_->ctx.setBuffer(6, impl_->bufVPositionsGrad);
        impl_->ctx.setBuffer(7, impl_->bufVAttachFixedPos);
        impl_->ctx.setBuffer(8, impl_->bufVAttachLambda);
        impl_->ctx.setBuffer(9, impl_->bufVAttachStiff);
        impl_->ctx.dispatchThreads(impl_->nAttach);
    }

    // ---- 4. Spring force backward: per-spring outputs incl. v_p_d.
    if (impl_->nSprings > 0 && impl_->ctx.hasKernel("spring_force_backward")) {
        impl_->ctx.setPipeline("spring_force_backward");
        impl_->ctx.setBuffer(0, impl_->bufPositionsPreStep);
        impl_->ctx.setBuffer(1, impl_->bufSpringP1Idx);
        impl_->ctx.setBuffer(2, impl_->bufSpringP2Idx);
        impl_->ctx.setBuffer(3, impl_->bufSpringRestLen);
        impl_->ctx.setBuffer(4, impl_->bufSpringStiffness);
        impl_->ctx.setBuffer(5, impl_->bufVSpringGradA);
        impl_->ctx.setBuffer(6, impl_->bufVSpringHess);
        impl_->ctx.setBuffer(7, impl_->bufVSpringPd);
        impl_->ctx.setBuffer(8, impl_->bufVSpringRestLen);
        impl_->ctx.setBuffer(9, impl_->bufVSpringStiff);
        impl_->ctx.dispatchThreads(impl_->nSprings);
    }
    // ---- 5. Triangle membrane force backward.
    if (impl_->nTri > 0 && impl_->ctx.hasKernel("triangle_membrane_force_al_backward")) {
        impl_->ctx.setPipeline("triangle_membrane_force_al_backward");
        impl_->ctx.setBuffer(0, impl_->bufPositionsPreStep);
        impl_->ctx.setBuffer(1, impl_->bufTriIdx);
        impl_->ctx.setBuffer(2, impl_->bufTriStiffness);
        impl_->ctx.setBuffer(3, impl_->bufTriLambda0);
        impl_->ctx.setBuffer(4, impl_->bufTriLambda1);
        impl_->ctx.setBuffer(5, impl_->bufTriInvUV);
        impl_->ctx.setBuffer(6, impl_->bufVTriGrad);
        impl_->ctx.setBuffer(7, impl_->bufVTriHessScalar);
        impl_->ctx.setBuffer(8, impl_->bufVTriP);
        impl_->ctx.setBuffer(9, impl_->bufVTriStiff);
        impl_->ctx.setBuffer(10, impl_->bufVTriLambda0);
        impl_->ctx.setBuffer(11, impl_->bufVTriLambda1);
        impl_->ctx.dispatchThreads(impl_->nTri);
    }
    // ---- 6. Triangle bending force backward.
    if (impl_->nBend > 0 && impl_->ctx.hasKernel("triangle_bending_force_al_backward")) {
        impl_->ctx.setPipeline("triangle_bending_force_al_backward");
        impl_->ctx.setBuffer(0, impl_->bufPositionsPreStep);
        impl_->ctx.setBuffer(1, impl_->bufBendIdx);
        impl_->ctx.setBuffer(2, impl_->bufBendWeight);
        impl_->ctx.setBuffer(3, impl_->bufBendNTarget);
        impl_->ctx.setBuffer(4, impl_->bufBendStiffness);
        impl_->ctx.setBuffer(5, impl_->bufBendLambda);
        impl_->ctx.setBuffer(6, impl_->bufVBendGrad);
        impl_->ctx.setBuffer(7, impl_->bufVBendHessScalar);
        impl_->ctx.setBuffer(8, impl_->bufVBendP);
        impl_->ctx.setBuffer(9, impl_->bufVBendNTarget);
        impl_->ctx.setBuffer(10, impl_->bufVBendStiff);
        impl_->ctx.setBuffer(11, impl_->bufVBendLambda);
        impl_->ctx.dispatchThreads(impl_->nBend);
    }

    // ---- 6b. vbd_init_backward — inertial path. Writes per-vertex
    //          cotangents on positions (v_x_init), predicted (v_y, junk),
    //          and mass (v_mass). Reads bufVG / bufVH from solve_apply_bwd.
    //          PR-G Brick B (CHI-14) — density gradient flows only here.
    if (impl_->ctx.hasKernel("vbd_init_backward")) {
        impl_->ctx.setPipeline("vbd_init_backward");
        impl_->ctx.setBuffer(0, impl_->bufPositionsPreStep);
        impl_->ctx.setBuffer(1, impl_->bufPredicted);
        impl_->ctx.setBuffer(2, impl_->bufMass);
        impl_->ctx.setBuffer(3, impl_->bufVG);
        impl_->ctx.setBuffer(4, impl_->bufVH);
        impl_->ctx.setBuffer(5, impl_->bufVPositionsInit);
        impl_->ctx.setBuffer(6, impl_->bufVPredictedJunk);
        impl_->ctx.setBuffer(7, impl_->bufVMass);
        VbdInitBackwardParams ibp{impl_->invHSqCached};
        impl_->ctx.setBytes(8, &ibp, sizeof(ibp));
        impl_->ctx.dispatchThreads(nV);
    }

    // ---- 7. Accumulate per-constraint position cotangents into v_positions
    //         by RE-USING the forward gather kernels. They additively
    //         scatter springGradA / triGrad / bendGrad into the per-vertex
    //         gScratch input — passing v_p_d / v_p as the gradA buffer and
    //         bufVPositionsGrad as the gScratch buffer does exactly what
    //         we need. We pass `bufHScratchJunk` as the hScratch slot —
    //         the gather kernels RW that buffer but we don't read it
    //         after, and using a SEPARATE buffer avoids racing with
    //         solve_apply_backward's read of bufHScratch earlier in this
    //         same encoder.
    std::memset(impl_->bufHScratchJunk.contents(), 0, impl_->bufHScratchJunk.length);
    if (impl_->nSprings > 0) {
        impl_->ctx.setPipeline("vbd_gather_spring");
        impl_->ctx.setBuffer(0, impl_->bufVSpringPd);
        impl_->ctx.setBuffer(1, impl_->bufVSpringHess);  // gather adds (scalar reused as 6-vec; values discarded)
        impl_->ctx.setBuffer(2, impl_->bufVertSpringOffset);
        impl_->ctx.setBuffer(3, impl_->bufVertSpringIdx);
        impl_->ctx.setBuffer(4, impl_->bufVertSpringRole);
        impl_->ctx.setBuffer(5, impl_->bufVPositionsGrad);
        impl_->ctx.setBuffer(6, impl_->bufHScratchJunk);
        impl_->ctx.setBuffer(7, impl_->bufVertPerm);
        {
            VbdGatherParams gp{0u};
            impl_->ctx.setBytes(8, &gp, sizeof(gp));
        }
        impl_->ctx.dispatchThreads(nV);
    }
    if (impl_->nTri > 0) {
        impl_->ctx.setPipeline("vbd_gather_triangle");
        impl_->ctx.setBuffer(0, impl_->bufVTriP);
        impl_->ctx.setBuffer(1, impl_->bufVTriHessScalar);
        impl_->ctx.setBuffer(2, impl_->bufVertTriOffset);
        impl_->ctx.setBuffer(3, impl_->bufVertTriIdx);
        impl_->ctx.setBuffer(4, impl_->bufVertTriRole);
        impl_->ctx.setBuffer(5, impl_->bufVPositionsGrad);
        impl_->ctx.setBuffer(6, impl_->bufHScratchJunk);
        impl_->ctx.setBuffer(7, impl_->bufVertPerm);
        {
            VbdGatherParams gp{0u};
            impl_->ctx.setBytes(8, &gp, sizeof(gp));
        }
        impl_->ctx.dispatchThreads(nV);
    }
    if (impl_->nBend > 0) {
        impl_->ctx.setPipeline("vbd_gather_bending");
        impl_->ctx.setBuffer(0, impl_->bufVBendP);
        impl_->ctx.setBuffer(1, impl_->bufVBendHessScalar);
        impl_->ctx.setBuffer(2, impl_->bufVertBendOffset);
        impl_->ctx.setBuffer(3, impl_->bufVertBendIdx);
        impl_->ctx.setBuffer(4, impl_->bufVertBendRole);
        impl_->ctx.setBuffer(5, impl_->bufVPositionsGrad);
        impl_->ctx.setBuffer(6, impl_->bufHScratchJunk);
        impl_->ctx.setBuffer(7, impl_->bufVertPerm);
        {
            VbdGatherParams gp{0u};
            impl_->ctx.setBytes(8, &gp, sizeof(gp));
        }
        impl_->ctx.dispatchThreads(nV);
    }

    impl_->ctx.flush();

    // PR-G Brick B (CHI-14): add the inertial path's position cotangent
    // to bufVPositionsGrad. CPU saxpby is fine — nV·3 float adds, fully
    // bandwidth-bound, dwarfed by the GPU dispatches we just waited on.
    if (impl_->ctx.hasKernel("vbd_init_backward")) {
        float* pg       = static_cast<float*>(impl_->bufVPositionsGrad.contents());
        const float* pi = static_cast<const float*>(impl_->bufVPositionsInit.cdata());
        for (uint32_t v = 0; v < nV; ++v) {
            pg[3*v + 0] += pi[3*v + 0];
            pg[3*v + 1] += pi[3*v + 1];
            pg[3*v + 2] += pi[3*v + 2];
        }
    }

    return 0;
}

void AvbdSolver::readPositionsGrad(std::vector<float>& out) const {
    if (!ok() || !impl_->meshReady || !impl_->backwardReady) {
        out.clear();
        return;
    }
    readVec3Padded(out, impl_->bufVPositionsGrad, impl_->nVerts);
}

void AvbdSolver::readMassGrad(std::vector<float>& mass_grad) const {
    if (!ok() || !impl_->meshReady || !impl_->backwardReady ||
        !impl_->bufVMass) {
        mass_grad.clear();
        return;
    }
    const uint32_t n = impl_->nVerts;
    mass_grad.assign(n, 0.0f);
    std::memcpy(mass_grad.data(), impl_->bufVMass.cdata(), n * sizeof(float));
}

void AvbdSolver::readPredictedGrad(std::vector<float>& predicted_grad) const {
    if (!ok() || !impl_->meshReady || !impl_->backwardReady ||
        !impl_->bufVPredictedJunk) {
        predicted_grad.clear();
        return;
    }
    readVec3Padded(predicted_grad, impl_->bufVPredictedJunk, impl_->nVerts);
}

void AvbdSolver::readSpringGrad(std::vector<float>& restLen_grad,
                                std::vector<float>& stiff_grad) const {
    if (!ok() || impl_->nSprings == 0 || !impl_->backwardReady) {
        restLen_grad.clear();
        stiff_grad.clear();
        return;
    }
    const uint32_t n = impl_->nSprings;
    restLen_grad.assign(n, 0.0f);
    stiff_grad.assign(n, 0.0f);
    std::memcpy(restLen_grad.data(), impl_->bufVSpringRestLen.cdata(), n * sizeof(float));
    std::memcpy(stiff_grad.data(),   impl_->bufVSpringStiff.cdata(),   n * sizeof(float));
}

void AvbdSolver::readAttachGrad(std::vector<float>& fixedPos_grad,
                                std::vector<float>& stiff_grad,
                                std::vector<float>& lambda_grad) const {
    if (!ok() || impl_->nAttach == 0 || !impl_->backwardReady) {
        fixedPos_grad.clear();
        stiff_grad.clear();
        lambda_grad.clear();
        return;
    }
    const uint32_t n = impl_->nAttach;
    readVec3Padded(fixedPos_grad, impl_->bufVAttachFixedPos, n);
    stiff_grad.assign(n, 0.0f);
    std::memcpy(stiff_grad.data(), impl_->bufVAttachStiff.cdata(), n * sizeof(float));
    readVec3Padded(lambda_grad, impl_->bufVAttachLambda, n);
}

void AvbdSolver::readTriGrad(std::vector<float>& stiff_grad,
                             std::vector<float>& lambda0_grad,
                             std::vector<float>& lambda1_grad) const {
    if (!ok() || impl_->nTri == 0 || !impl_->backwardReady) {
        stiff_grad.clear();
        lambda0_grad.clear();
        lambda1_grad.clear();
        return;
    }
    const uint32_t n = impl_->nTri;
    stiff_grad.assign(n, 0.0f);
    std::memcpy(stiff_grad.data(), impl_->bufVTriStiff.cdata(), n * sizeof(float));
    readVec3Padded(lambda0_grad, impl_->bufVTriLambda0, n);
    readVec3Padded(lambda1_grad, impl_->bufVTriLambda1, n);
}

void AvbdSolver::readBendGrad(std::vector<float>& nTarget_grad,
                              std::vector<float>& stiff_grad,
                              std::vector<float>& lambda_grad) const {
    if (!ok() || impl_->nBend == 0 || !impl_->backwardReady) {
        nTarget_grad.clear();
        stiff_grad.clear();
        lambda_grad.clear();
        return;
    }
    const uint32_t n = impl_->nBend;
    nTarget_grad.assign(n, 0.0f);
    stiff_grad.assign(n, 0.0f);
    std::memcpy(nTarget_grad.data(), impl_->bufVBendNTarget.cdata(), n * sizeof(float));
    std::memcpy(stiff_grad.data(),   impl_->bufVBendStiff.cdata(),   n * sizeof(float));
    readVec3Padded(lambda_grad, impl_->bufVBendLambda, n);
}

}  // namespace cloth
