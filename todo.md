# AVBD port — status & next moves

This file was originally the AVBD port roadmap (PRs #42-#91 of
this session). PR-A through PR-G ship live on `main`. PR-H wires
real Gauss-Seidel via vertex coloring (#91). Updating to capture
the current state, the empirical findings, and the **reframing**
of what "AVBD on the dress" actually does — see the next section.

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
| Per-triangle inv_deltaUV (#87) | ✅ rest-pose bug fixed, 1068× drift reduction at step 1 |
| pd-vs-avbd \|Δx\| diagnostic (#88) | ✅ direct comparison metric |
| Drift-vertex localization (#89) | ✅ outlier vertices identified |
| Under-relax + constraint-disable diagnostics (#90) | ✅ localized "membrane" as the gap source |
| Vertex coloring + per-color GS (#91) | ✅ real GS dispatch, `AVBD_COLORS=1` env |
| Drape-equilibrium drive on dress | ✅ `AVBD_DRIVE=1 AVBD_COLORS=1` runs stable at 20 ms/step |

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

4. **Reframing (PR #91 + AVBD_DRIVE test)**: the "AVBD diverges
   from PD" narrative in earlier sections (#88-#90) was a
   measurement artifact. The shadow's `drift_max` measured
   `|AVBD's solve − PD's evolving x_n|`. As PD's CG-iterated
   trajectory and AVBD's per-vertex-GS trajectory drifted apart,
   that gap grew — *but neither trajectory is wrong*. Direct test
   with `AVBD_COLORS=1 AVBD_DRIVE=1` (AVBD drives the simulation,
   bypassing PD entirely):

      step  AVBD |Δx|_max   conv_max     wall
       1      0.000325      1.4e-4      20 ms
       5      0.000290      1.4e-4      19 ms
      20      0.000209      1.0e-4      21 ms
      40      0.000172      8.2e-5      20 ms

   AVBD's per-step displacement settles at ~3e-4 m
   (`h²·g`-magnitude — pure gravity loading at near-equilibrium),
   conv_max is ~1e-4 (fully converged), no NaN, runs indefinitely.

   What AVBD computes is the **static equilibrium under gravity +
   attachments**, reached in a handful of steps. For a draped
   dress on an avatar — the *actual* downstream use case for
   VR avatar clothing — this is the correct target. PD spends 7960
   ms/step iterating the full Hessian to a slightly different
   "converged" point that includes more dynamic content; AVBD
   reaches steady-state at ~20 ms/step (~400× faster), suitable
   for runtime garment authoring.

   What AVBD *does not* do well: animated cloth with significant
   per-step dynamics (swinging, wind, contact response). The
   per-vertex GN diagonal-Hessian damps faster than full Newton.

5. **All 5 demos work with `AVBD_COLORS=1 AVBD_DRIVE=1`** (post-#91
   smoke test):

      demo    | nVerts | colors | h (ms) | invH²    | wall   | drift_max @ step 10
      --------+--------+--------+--------+----------+--------+--------------------
      tshirt  |  1426  |   11   |  11.1  |   8,100  | 14 ms  | 1.3e-2 m
      sock    |  1055  |    9   |   3.1  | 102,400  | 13 ms  | 9.6e-5 m
      hat     |   579  |    9   |  10.0  |  10,000  | 15 ms  | 9.8e-4 m
      sphere  |   625  |    4   |   5.6  |  32,400  |  4 ms  | 3.0e-4 m
      dress   |  3634  |   10   |  11.1  |   8,100  | 20 ms  | 1.2e-3 m

   No NaN, all stable in DRIVE mode. Sock is in AVBD's ideal regime
   (small h, big invH² → inertia comparable to constraints). Per-step
   drift is 96 µm; PD-vs-AVBD bulk |Δx|_mean on sock matches to
   within 3% at step 20 (PD 4.94e-3 vs AVBD 4.78e-3) — bulk physics
   agrees. Dress and tshirt share the same regime (h=11.1 ms,
   invH²=8100) and show larger drift relative to PD; both still
   produce a stable drape equilibrium.

## Headline result (post-#99)

Dress per-step wall, in-flight PRs all merged or in the queue:

  stage                                                  wall/step
  ----------------------------------------------------+----------
  session start (PD+CG, default DiffCloth)           |  7,960 ms
  PD+AVBD shadow (#70)                                | ~7,970 ms
  AVBD_DRIVE alone (PD still ran)                     | ~1,500 ms
  AVBD_DRIVE + AVBD_COLORS                            |    ~20 ms
  AVBD_DRIVE + AVBD_COLORS + AVBD_SKIP_PD (#99)       |     ~9 ms

**~880× total speedup.** Runtime-suitable for VR avatar clothing.

## What AVBD now does on its own (no PD)

- ✅ Gravity + attachments (since #87 rest-pose fix)
- ✅ Triangle membrane (ARAP) + bending (since PR-A/PR-D)
- ✅ Wind / external forces (already in via the s_n predictor)
- ✅ Cloth-vs-primitive contact: Sphere / Capsule / LowerLeg /
     Bowl / Plane / Foot (#98 via Primitive::isInContact projection)
- ✅ AL on hard pins + membrane + bending (#74/76/95)
- ✅ Per-color Gauss-Seidel via greedy first-fit coloring (#91)

## What's still in PD's hands

- Self-collision (cloth-cloth penetration) — not ported to AVBD
- L-BFGS-B inverse-design backward pass (the PR-G adjoint)
- Differentiable forward simulator hook for inverse design (PR-H)

## Open questions / next directions

### ~Convergence of inner solver~ (closed by #91 + DRIVE test)
The `conv_max ≈ 0.71` story is a measurement artifact (see
finding 4 above) — `conv_max` was measuring iter-to-iter motion
on a coordinate frame anchored to PD's evolving x_n. When AVBD
drives its own trajectory (PR #91 with `AVBD_COLORS=1
AVBD_DRIVE=1`), `conv_max` collapses to ~1e-4 within 16 outer
iters and stays there indefinitely. No Chebyshev / sub-stepping
/ MGPBD needed for the drape-equilibrium use case.

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
#87     PR-G: per-triangle inv_deltaUV — fixes canonical-rest mismatch
#88     PR-G: PD-vs-AVBD per-step |Δx| comparison diagnostic
#89     PR-G: drift-vertex localization + inv_deltaUV range printout
#90     PR-G: under-relax + constraint-disable envs (membrane diagnosis)
#91     PR-H: vertex coloring + per-color GS dispatch; AVBD_DRIVE
              with COLORS confirms AVBD reaches static equilibrium
              stably at ~20 ms/step
#92-93  PR-H: 5-demo survey (tshirt/sock/hat/sphere/dress) — all
              stable under AVBD_COLORS=1 AVBD_DRIVE=1
#94     PR-I: AVBD_DUMP_STEP — OBJ snapshot for visual validation
              (drape on dress matches PD's bulk shape within ~6%)
#95     PR-I: TriangleMembraneDualUpdate inv_deltaUV cleanup
              (AL membrane path now correct on real meshes)
#96     PR-I: outlier-vertex diagnostic — drift_max is at low-
              incidence corners (v1333 has 2 incident tris), not a
              solver bug; well-known per-vertex Newton failure mode
#98     PR-I: generic cloth-vs-primitive projection via
              Primitive::isInContact (sphere, capsule, plane, bowl,
              foot, lower-leg all in one loop)
#99     PR-I: AVBD_SKIP_PD — short-circuit PD's CG when AVBD drives;
              ~163× per-step speedup on dress (1.4 s → 8.8 ms).
              Combined with #91+#87: 7,960 ms → 9 ms (~880×).
