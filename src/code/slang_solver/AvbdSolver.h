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

    // Dispatch one AVBD outer iteration. Currently runs:
    //   1. vbd_init           writes inertial term to (gScratch, hScratch)
    //   2. spring_force       per-spring force + GN Hessian into output buffers
    //   3. vbd_gather_spring  per-vertex accumulates spring contributions
    //                         into (gScratch, hScratch) via CSR adjacency
    // solve_apply lands in a follow-up PR. Returns 0 on success,
    // -1 if not set up.
    int step();

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
