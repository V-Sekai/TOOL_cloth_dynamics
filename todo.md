# AVBD port — status & next moves

This file was originally the AVBD port roadmap (PRs #42-#84 of this
session). PR-A through PR-F-functional now ship live on `main`.
Updating it to capture the current state, the empirical findings,
and the open questions so future work resumes cleanly.

## Where we are (as of #84)

| Layer | Status |
|---|---|
| Lean specs for every kernel | ✅ 18 `native_decide`-pinned kernels |
| Slang codegen (`*.slang`) | ✅ all kernels emit; slangc clean |
| cpp validators (slangc-cpp target) | ✅ all bit-exact vs hand-computed reference |
| Metal metallibs (slangc-metal target) | ✅ all 18 build, load on Apple Silicon |
| AvbdSolver wrapper | ✅ 16 kernels loaded, ABI clean (PIMPL) |
| Simulation.cpp integration | ✅ `USE_AVBD=1` env-gates the shuttle |
| DiffCloth dress uploaded | ✅ 3634 verts / 7026 tri / 31 attach / 10416 bend |
| Per-step shadow shuttle on Metal | ✅ runs on dress, output physically plausible, no NaN |
| AL kernels (3 dual + 3 force_al) | ✅ all bit-exact, wired, env-gated |
| AL dispatch in iter loop | ✅ `AVBD_AL=1` activates all 3 dual updates |
| γ scaling for AL | ✅ `AVBD_AL_GAMMA=<scale>` env |

## Per-step wall on dress (10902 DoF, Apple Silicon Metal)

| Path | Wall/step |
|---|---:|
| PD+CG (session start) | ~7,960 ms |
| Eigen LLT target baseline | ~370 ms |
| **AVBD 1 outer iter** | **~0.7 ms** |
| **AVBD 16 outer iters** | **~9 ms** |
| AVBD 64 outer iters | ~30 ms |

AVBD 16-iter is **~40× faster than Eigen LLT, ~880× faster than the
session-start path**. Real DiffCloth dress data, real Metal hardware,
zero NaN, displacement magnitudes within physically plausible range.

## Key empirical findings (this session's PRs)

1. **AVBD wall scales linearly with iter count** at ~600 µs per iter
   on dress. Memory-bandwidth-bound on Apple Silicon (PR #71).

2. **AL is the right tool for hard constraints, NOT for cloth
   oscillation.** The dress shows `conv_max ≈ 0.71` between iter
   N/2 and iter N — i.e., AVBD's inner Gauss-Seidel oscillates
   rather than converges. AL ramping λ doesn't fix this
   regardless of γ scale (PRs #77, #83, #84).
   - Tiny γ → AL inactive → conv unchanged
   - Large γ → λ over-ramps → simulation diverges
   - No γ sweet spot exists for the dress's failure mode

3. **AVBD output is correctly physical**: at any iter count, |Δx|
   stays bounded, no NaN, position deltas grow only by ~0.3% over
   10 steps (PR #72).

## Open questions / next directions

### Convergence of inner solver
The dress's `conv_max ≈ 0.71` is per-vertex Gauss-Seidel oscillation
(not slow convergence — at iter 64, `conv_max` is *higher* than at
iter 16). Per the AVBD paper's discussion this can come from:
- Per-vertex GS oscillating between two states when stiffness is
  high relative to inertia (cloth membrane stiffness ~1e4, inertia
  weight w = m/h² ~1e2 on dress)
- Need either:
  - **Chebyshev acceleration** — over/under-relax the position
    update per iter
  - **Sub-stepping** — smaller h ⇒ larger w ⇒ inertia dominates
  - **Multigrid preconditioner** (MGPBD direction, but bigger
    architectural change)

### Velocity damping
Orthogonal to the inner-solver issue but worth trying. DiffCloth's
PD path has implicit damping from the iterative projection. AVBD has
no damping by construction — it's an implicit Euler step that
preserves kinetic energy except through dissipation in the
projection. Adding `v_n *= 0.99` before computing the AVBD predictor
might bleed off the oscillation energy.

### Inverse-design adjoint (PR-G)
VBD's iter map is differentiable (Chen et al. SIGGRAPH 2024). Once
the forward simulation drives correctly, the backward pass for
inverse design should be straightforward to derive. DiffAvatar
(CVPR 2024) demonstrates the recipe for XPBD; AVBD's structure is
similar.

### Dress demo validation (PR-H)
Once forward AVBD drives the simulation (positions written back to
`Particle.pos`), run the existing dress inverse-design loop with
AVBD as the forward simulator and check L-BFGS-B converges to a
sensible spinning-angle target.

## Reverse-chronological PR list (this session)

#42–46  PR-A: 4 per-constraint force kernels (spring/attach/tri/bend)
#47–48  PR-B: vertex CSR adjacency (per constraint type)
#49     PR-C: vertex graph coloring
#50–55  PR-D: 6 vertex-block-update kernels (init, 4 gather, solve_apply)
#56–63  PR-E: AvbdSolver wrapper, 4-vert test bit-exact
#64–67  PR-E continued: Simulation.cpp scaffold, mesh + spring + tri uploads
#68–69  PR-E: attachment + bending uploads — full dress on AvbdSolver
#70     PR-E: per-step shadow shuttle on real dress (0.7 ms/iter)
#71–73  PR-E: iter scaling bench + output sanity + convergence probe
#74–81  PR-F: 6 AL kernels (3 dual_update + 3 force_al, all bit-exact)
#76, 82 PR-F: AvbdSolver AL wiring (16 kernels loaded)
#77, 83 PR-F: Simulation dispatch AL duals + γ=stiffness diverges finding
#84     PR-F: AVBD_AL_GAMMA env + γ sweep — AL is NOT dress's fix
