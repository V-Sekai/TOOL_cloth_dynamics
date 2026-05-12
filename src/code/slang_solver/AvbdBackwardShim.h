// AvbdBackwardShim — aggregates AvbdSolver::stepBackward outputs into a
// DiffCloth-style per-type gradient struct (CHI-14 Brick C).
//
// AvbdSolver emits cotangents per CONSTRAINT (one stiffness per spring,
// per triangle, etc) plus per-vertex mass / position. DiffCloth's
// `BackwardInformation` stores per-CONSTRAINT-TYPE scalars
// (`dL_dk_pertype[CONSTRAINT_SPRING]`, …) and a uniform-density scalar
// (`dL_ddensity`). This shim performs the sum:
//
//   dL_dk_pertype[SPRING]    = Σ_i  v_stiffness_spring[i]
//   dL_dk_pertype[ATTACH]    = Σ_c  v_stiffness_attach[c]
//   dL_dk_pertype[TRIANGLE]  = Σ_c  v_stiffness_tri[c]
//   dL_dk_pertype[BENDING]   = Σ_c  v_stiffness_bend[c]
//   dL_ddensity              = Σ_v  v_mass[v] · area_per_vertex[v]
//   dL_dxfixed[3c..3c+2]     = v_fixedPos[c]
//   dL_dx                    = v_positions[v]
//
// Decoupled from DiffCloth's `Simulation` so it can be unit-tested
// without a full DiffCloth build. The downstream DiffCloth-side hook
// (Brick D: `Simulation::stepBackwardAvbd`) is a thin 1-to-1 copy of
// these fields into `BackwardInformation`.

#ifndef CLOTH_AVBD_BACKWARD_SHIM_H
#define CLOTH_AVBD_BACKWARD_SHIM_H

#include "AvbdSolver.h"

#include <cstdint>
#include <vector>

namespace cloth {

struct AvbdParamGradients {
    // Aggregated scalar gradients per constraint type. The indices
    // match DiffCloth's `Constraint::ConstraintType` enum:
    //   [0] CONSTRAINT_SPRING_STRETCH
    //   [1] CONSTRAINT_ATTACHMENT
    //   [2] CONSTRAINT_TRIANGLE
    //   [3] CONSTRAINT_TRIANGLE_BENDING
    double dL_dk_pertype[4] = {0.0, 0.0, 0.0, 0.0};

    // Density gradient — Σ_v v_mass[v]·area_per_vertex[v]. Zero if the
    // caller passes nullptr for `vertex_area`.
    double dL_ddensity = 0.0;

    // Per-attachment anchor cotangent (xyz tightly packed; length
    // 3·nAttach).
    std::vector<double> dL_dxfixed;

    // Per-vertex position cotangent on x_after-step (xyz tightly
    // packed; length 3·nVerts).
    std::vector<double> dL_dx;

    // Per-attachment lambda cotangent (xyz tight; length 3·nAttach).
    // AL multiplier state; downstream optimizers can ignore unless
    // fitting the dual ramp directly.
    std::vector<double> dL_dlambda_attach;

    // Per-triangle membrane lambda cotangents (col0, col1; each xyz
    // tight; length 3·nTri each).
    std::vector<double> dL_dlambda0_tri;
    std::vector<double> dL_dlambda1_tri;

    // Per-bending lambda cotangent (xyz tight; length 3·nBend).
    std::vector<double> dL_dlambda_bend;
};

// Runs `solver.stepBackward(v_loss_in_float)` and aggregates outputs
// into `out`. `dL_dx_in` is the per-vertex cotangent on x_after-step,
// xyz-tight, length 3·nVerts (double for compatibility with LBFGSpp /
// DiffCloth's Eigen path; internally downcast to float for the GPU
// dispatch).
//
// `vertex_area` (optional, length nVerts) maps per-vertex mass
// cotangent to density gradient via the DiffCloth identity
//   mass[v] = density · area[v]
// so ∂L/∂density = Σ v_mass[v]·area[v]. Pass nullptr to skip — useful
// when the caller doesn't optimise density.
//
// Returns 0 on success, -1 if the solver isn't set up.
int avbdBackwardShim(AvbdSolver& solver,
                     uint32_t nVerts,
                     uint32_t nAttach,
                     uint32_t nTri,
                     uint32_t nBend,
                     const double* dL_dx_in,
                     const double* vertex_area,
                     AvbdParamGradients& out);

}  // namespace cloth

#endif  // CLOTH_AVBD_BACKWARD_SHIM_H
