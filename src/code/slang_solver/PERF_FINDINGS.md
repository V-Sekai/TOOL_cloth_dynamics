# MetalCGSolver vs Eigen LLT — perf findings

Result of running both solver paths through DiffCloth's `Simulation::step()`
on the same M2 Pro, with the wiring + optimizations from PRs #26–#33.

## TL;DR

The Slang/Metal CG solver works correctly (no NaN, equivalent simulation
quality), but **is not competitive with Eigen LLT for DiffCloth's
workload**. The gap widens with N, not narrows.

| Demo | DoF | Eigen LLT | Slang CG | Gap |
|---|---|---|---|---|
| Hat | 1737 | ~120 ms / step | ~300 ms / step (post-warmup) | 2.5× slower |
| Dress | 10902 | ~220 ms / step | ~3.8 s / step | 17× slower |

Same machine, same demo seed, same build except for the
`USE_SLANG_CG=1` env var. Both paths run for 60–90 s.

## Why per-kernel "100× faster" doesn't translate to per-step

The Tier-3 perf table from #24 (batched Metal dispatch) showed real GPU
kernels running 4-12× faster per dispatch than Eigen's CPU primitives.
That number is **per kernel call**, with all reps batched into one
command buffer. It's the *steady-state GPU compute throughput* once
synchronisation costs are amortised.

A real `Simulation::step()` doesn't get to amortise like that. Each PD
outer iteration calls `MetalCGSolver::solve()` once, and each solve is
its own `commit + waitUntilCompleted`. The number of solves per step
turns out to be much higher than I'd estimated:

```
DiffCloth dress demo, 20 steps:
  total Slang-CG solves observed     ~29 500
  → solves per step                  ~1 475
```

DiffCloth's PD outer loop runs hundreds to thousands of iters per step
on a complex scene like dress. Each one is a separate `MetalCGSolver`
dispatch through Metal's command queue. The per-solve cost in our
implementation is ~1.5 ms (steady-state), vs Eigen LLT's ~100 µs back-
substitution on a precomputed Cholesky factor.

```
per-step solver cost (estimated):
  Eigen LLT:  1475 solves × 100 µs ≈ 150 ms
  Slang CG:   1475 solves × 1.5 ms ≈ 2.2 s
```

Plus the rest of `step()` (collision detection, friction, constraint
reassembly) which is the same for both — ~200 ms on dress.

## Bottleneck breakdown for one MetalCGSolver::solve()

| Phase | Per solve | Why it's there |
|---|---|---|
| Upload `b` (fp64→fp32) | ~50 µs | N=10902 doubles → floats |
| Init x=0, r=b, p=b | ~100 µs | 4·N fp stores in shared memory |
| First `dot(r, r)` | ~150 µs | 1 commit + 1 wait + GPU compute |
| K=30 batched CG iters | ~600 µs | 1 commit + 1 wait + GPU compute |
| Read `x` back | ~50 µs | fp32→fp64 |
| **Sum (one solve)** | **~1 ms** | not bad in isolation |
| × 1475 solves/step | **~1.5 s** | dispatch overhead × every PD outer iter |

The kernel arithmetic is fast (the Tier-3 numbers from #24 hold). It's
the **per-solve dispatch cost** that doesn't amortise: every PD outer
iter does one `commit + wait` (or two — one for the init dot, one for
the K-iter batch).

## Why this can't be trivially closed

The PD outer iters are **algorithmically serial** within a step:
each iter's `b` is derived from the previous iter's `x` (via the
constraint projections that DiffCloth does between solver calls).

That means we can't naively batch multiple solves into one Metal
command buffer — solve N+1 needs solve N's `x` on the CPU side to
re-project constraints.

Two paths could close the gap, both substantial:

1. **Move the entire PD outer loop onto the GPU.** The constraint
   projections themselves are GPU-friendly (we already have
   `spring_project`, `triangle_bending`, `attachment_project`,
   `triangle_project`, `assemble_b` from earlier PRs). If the projection
   step is a GPU kernel that feeds into the next solve's `b`, we never
   leave the GPU between PD outer iters. The whole step becomes one
   long command buffer.

2. **Replace CG with on-GPU Cholesky back-substitution.** Eigen's
   advantage is precomputed factorisation + cheap back-sub. We could
   factorise once at bind time on the CPU (using Eigen), upload L to
   GPU, and dispatch a GPU back-substitution kernel per solve. That
   eliminates the CG-iteration loop entirely. Triangular solve kernels
   are a real area of research; not trivial but tractable.

Both are multi-PR architectural changes, not parameter tunes.

## Honest verdict on "rip out Eigen"

**Not yet.** The wiring is correct and the kernels are validated, but
the per-step performance gap is structural, not incremental. Eigen LLT
stays as the default; `USE_SLANG_CG=1` lives as opt-in infrastructure
for future work.

What was actually achieved across PRs #8 → #33:

- 10 Slang shader kernels (4 constraint projections, 5 CG primitives,
  assemble_b) pinned by native_decide and bit-validated through both
  slang_validate (cpp target) and SPIR-V round-trip.
- Full Lean → LeanSlang → slangc pipeline for the Cloth namespace,
  scaffolded from zero.
- Tier-2 (slangc-cpp) and Tier-3 (real Metal GPU) perf baselines,
  the latter showing kernel-level speedups of 4-12× over the cpp
  fallback in #24.
- MetalCGSolver C++/Obj-C++ wrapper that exposes the GPU CG path to
  DiffCloth's Eigen-typed `Simulation::step()` (PRs #26, #27).
- GPU-side α/β scalar kernels (`cg_alpha`, `cg_beta`,
  `saxpby_indirect`) so a full CG iter can run with no CPU sync
  (PRs #28, #29, #30).
- Fused CG dispatcher batching K iters per command buffer (#31).
- NaN-safe `cg_beta` for over-convergence robustness (#32).
- Per-step wall comparison instrumentation (#33).
- This document (#34).

The infrastructure is real and reusable for the architectural changes
above. The "competitive perf" target wasn't met; the work to get there
is now visible and tractable.
