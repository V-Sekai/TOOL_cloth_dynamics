// AvbdSolver — pure-C++ interface to the Slang/Metal AVBD
// vertex-block-update pipeline (vbd_init + 4 gather kernels +
// vbd_solve_apply). Wraps the Metal dispatcher so DiffCloth's
// simulation code can call into the AVBD path without pulling
// Foundation/Metal headers into Eigen-using TUs.
//
// This PR (PR-E stub) ships only the constructor — it loads the
// six kernel metallibs and reports `ok()` if all PSOs built. The
// per-step `step()` dispatch is intentionally a no-op stub that
// returns immediately; the actual GPU loop wiring lands in follow-up
// PRs (PR-E continued, then PR-F augmented Lagrangian).
//
// AVBD is one algorithm covering cloth, attachment, triangle membrane,
// and dihedral bending in a single per-vertex block update. See
// todo.md for the full roadmap.
//
// Lifecycle:
//   - Construct once per simulation scene (loads metallibs, builds PSOs).
//   - `step()` per timestep — currently a stub.
//   - Destruct to free Metal resources.

#ifndef CLOTH_AVBD_SOLVER_H
#define CLOTH_AVBD_SOLVER_H

#include <cstddef>
#include <cstdint>
#include <vector>

namespace cloth {

class AvbdSolver {
public:
    // Construct from a metallib directory path. The directory must
    // contain all six AVBD kernels:
    //   vbd_init.metallib
    //   vbd_gather_spring.metallib
    //   vbd_gather_attachment.metallib
    //   vbd_gather_triangle.metallib
    //   vbd_gather_bending.metallib
    //   vbd_solve_apply.metallib
    //
    // Builds compute pipeline state objects for each and reports
    // `ok() == true` only if every PSO built successfully.
    explicit AvbdSolver(const char* metallibPath);
    ~AvbdSolver();

    // True if construction succeeded and all six kernels are loaded.
    bool ok() const;

    // Upload mesh state. Allocates per-vertex GPU buffers (positions,
    // predicted, mass, gScratch, hScratch) sized to `nVerts`, uploads
    // initial state. Subsequent step() calls operate on this buffer
    // set. Calling setupMesh again resizes/replaces the buffers.
    //
    // `positions`, `predicted` are flat float arrays of length 3*nVerts
    // (xyz triplets); `mass` is length nVerts. `invHSquared = 1/h²`.
    void setupMesh(uint32_t nVerts,
                   const float* positions,
                   const float* predicted,
                   const float* mass,
                   float invHSquared);

    // Upload spring constraints. `nSprings` constraints with endpoints
    // (p1Idx[i], p2Idx[i]) into the per-vertex buffer, rest length
    // `restLen[i]`, stiffness `stiffness[i]`. Allocates output buffers
    // for spring_force (gradA, hess).
    void uploadSprings(uint32_t nSprings,
                       const uint32_t* p1Idx,
                       const uint32_t* p2Idx,
                       const float* restLen,
                       const float* stiffness);

    // Upload attachment (point-pin) constraints. `nAttach` constraints
    // pinning vertex `vertIdx[c]` to world anchor `fixedPos[c]` with
    // stiffness `stiffness[c]`. `fixedPos` is flat float array of length
    // 3 * nAttach (xyz). Allocates output buffers for attachment_force
    // (gradV, hessScalar).
    void uploadAttachments(uint32_t nAttach,
                           const uint32_t* vertIdx,
                           const float* fixedPos,
                           const float* stiffness);

    // Upload triangle membrane (in-plane stretch) constraints.
    // `nTri` triangles with corner vertex indices `triIdx` (length
    // 3*nTri, interleaved (i0,i1,i2) per triangle), per-triangle
    // rest material inverse `invUV` (length 4*nTri, row-major 2x2
    // per triangle: [m00, m01, m10, m11] from DiffCloth's
    // `Triangle::inv_deltaUV`), and stiffness `stiffness[c]`.
    // Allocates output buffers for triangle_membrane_force
    // (grad, hessScalar at slot 3*c+r).
    void uploadTriangles(uint32_t nTri,
                         const uint32_t* triIdx,
                         const float* invUV,
                         const float* stiffness);

    // Upload dihedral bending constraints. `nBend` bending stencils
    // with 4-vertex indices `bendIdx` (length 4*nBend), bind-time
    // Laplacian `weight` (length 4*nBend), rest magnitude
    // `nTarget[c]` (degenerate stencils have nTarget=0 → no
    // contribution), and stiffness `stiffness[c]`. Allocates output
    // buffers for triangle_bending_force (grad, hessScalar at slot
    // 4*c+r).
    void uploadBendings(uint32_t nBend,
                        const uint32_t* bendIdx,
                        const float* weight,
                        const float* nTarget,
                        const float* stiffness);

    // Dispatch one full AVBD outer iteration covering every constraint
    // type DiffCloth uses:
    //   1. vbd_init                       inertial term per vertex
    //   2. spring_force + gather          (if nSprings > 0)
    //   3. attachment_force_al + gather   (if nAttach > 0)
    //                                      AL-augmented: gradient
    //                                      includes accumulated λ_c
    //                                      (initially zero; ramped by
    //                                      stepDualAttachments())
    //   4. triangle_membrane_force + gather  (if nTri > 0)
    //   5. triangle_bending_force + gather   (if nBend > 0)
    //   6. vbd_solve_apply                3x3 inverse + position update
    //
    // After step() returns, `readPositions(out)` returns the updated
    // vertex positions. Returns 0 on success, -1 if not set up.
    int step();

    // Augmented-Lagrangian dual ramp for attachments. Per attachment
    // c, updates λ_c ← λ_c + γ_c · (p_v − fixedPos[c]). Call after
    // step() each AVBD outer iter to suppress oscillation on hard
    // pins (see PR #73 convergence probe for the failure mode this
    // fixes). No-op if nAttach == 0. Returns 0 on success.
    int stepDualAttachments();

    // AL dual ramp for triangle membrane (ARAP) constraints.
    // λ_c.col0 ← λ_c.col0 + γ_c · (F.col0 − R.col0)
    // λ_c.col1 ← λ_c.col1 + γ_c · (F.col1 − R.col1)
    // Targets the dress-mesh oscillation (#77 finding).
    int stepDualMembrane();

    // AL dual ramp for dihedral bending constraints.
    // λ_c ← λ_c + γ_c · (s − target)
    // where s = Σ w·p_r is the 4-vertex weighted sum.
    int stepDualBending();

    // Multiply all γ values across attachment / membrane / bending by
    // `scale`. Default γ on upload = constraint stiffness, which is
    // typically ~1e3-1e4 for the dress (membrane k_stiff). PR #83
    // showed this is too aggressive — λ over-ramps and diverges.
    // Call this AFTER uploadAttachments / uploadTriangles /
    // uploadBendings to scale γ down to a stable range. Calling
    // again multiplies on top of the previous scale.
    void setGammaScale(float scale);

    // Per-step state refresh: re-upload positions + predicted into the
    // existing GPU buffers (no realloc). Cheap when called every step
    // from the simulation. Caller is responsible for keeping nVerts
    // stable since setupMesh; passing arrays of a different length is
    // undefined.
    void updateState(const float* positions, const float* predicted);

    // Build greedy first-fit vertex coloring from all uploaded
    // constraints' incidence graphs (spring 2-clique, triangle
    // 3-clique, bending 4-clique). After this call, step() dispatches
    // each kernel per color, giving real Gauss-Seidel semantics
    // across vertices within an iter. Idempotent — safe to call after
    // each upload* method or just once before the first step().
    // Without it, step() runs as a single Jacobi sweep (numColors=1 +
    // identity vertPerm = pre-coloring behavior).
    void buildColoring();

    // Read back current vertex positions to a host array. Used by tests.
    // `positions_out` length = 3 * nVerts (xyz).
    void readPositions(std::vector<float>& positions_out) const;

    // Read back current scratch state to host arrays. Used by tests.
    // `gScratch_out` length 3*nVerts (xyz); `hScratch_out` length 6*nVerts
    // (packed sym 3x3: [Hxx, Hxy, Hxz, Hyy, Hyz, Hzz] per vertex).
    void readScratch(std::vector<float>& gScratch_out,
                     std::vector<float>& hScratch_out) const;

    // Read back spring_force outputs. `gradA_out` length 3*nSprings
    // (xyz); `hess_out` length 6*nSprings (packed sym 3x3 per spring).
    // Used by tests.
    void readSpringForce(std::vector<float>& gradA_out,
                         std::vector<float>& hess_out) const;

private:
    struct Impl;
    Impl* impl_;
};

}  // namespace cloth

#endif  // CLOTH_AVBD_SOLVER_H
