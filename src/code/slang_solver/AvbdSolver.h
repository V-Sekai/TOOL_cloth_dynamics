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

    // PR-E stub: one outer AVBD iteration (one pass over all color
    // batches dispatching vbd_init → gathers → vbd_solve_apply).
    // Currently a no-op that returns 0 without touching any buffer.
    // Real implementation lands in PR-E continued.
    int step();

private:
    struct Impl;
    Impl* impl_;
};

}  // namespace cloth

#endif  // CLOTH_AVBD_SOLVER_H
