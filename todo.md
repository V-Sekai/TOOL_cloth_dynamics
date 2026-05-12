# AVBD port roadmap

DiffCloth's PD-with-CG path tops out at ~3.9× behind Eigen LLT after
PR #41 (df32 + env-gated MetalCGSolver). The literature has moved on
twice since DiffCloth (2022): XPBD (DiffAvatar 2024) and then
Vertex Block Descent / Augmented VBD (SIGGRAPH 2024–2025). AVBD is
*one* algorithm that covers cloth, muscle, soft body, rigid contact,
and hard constraints in a single per-vertex block update.

Plan: port AVBD as the unified simulator backend, env-gated
`USE_AVBD=1` so it lives alongside the PD path. Each PR is small,
testable in isolation, and `native_decide`-pinned.

## Why AVBD, not MGPBD, not subspace-PD

- **MGPBD** is multigrid + XPBD — two algorithms stacked, faster at
  >100K DoF, irrelevant at DiffCloth's 1.7K-10.9K scale.
- **Subspace-PD** (Li-Wang 2023) is great for cloth but only cloth.
- **AVBD** is one algorithm: per-vertex block update + augmented
  Lagrangian for hard constraints. Covers every constraint type
  (stretch / bend / volume / joint / contact / stitch) with a single
  vertex update kernel.
- Aligned with DiffAvatar's production direction (XPBD-family for
  "drape clothing on avatars").

## Existing infrastructure that carries over

The PD-local-step projection kernels we shipped are the *same*
per-constraint primitives AVBD needs — just gathered per-vertex
instead of dispatched per-constraint:

- `spring_project`        → spring force + 3x3 Hessian contribution
- `triangle_bending`      → bending force + Hessian
- `triangle_project`      → membrane force + Hessian
- `attachment_project`    → soft attachment force + Hessian
- `assemble_b`            → folds into the vertex-block RHS

What gets dropped (after AVBD lands):
- `spmv`, `saxpby`, `cg_alpha`, `cg_beta`, `dot_reduce`, the whole
  `MetalCGSolver` linear-solver stack. AVBD has no global solve.

## PRs

### PR-A — constraint-force kernels (replaces project kernels)

For each constraint type already shipped as a `*_project` kernel, add
a sibling `*_force` kernel that writes per-endpoint force + GN Hessian
block instead of the PD-style projection. One thread per constraint,
output indexed by constraint id.

- `spring_force`          (single 3D force per endpoint + 3x3 GN block)
- `triangle_membrane_force`
- `triangle_bending_force`
- `attachment_force`

Each as its own PR with `native_decide` pinning and a cpp validator
against a tiny fixture (1 constraint, 2-4 vertices).

### PR-B — vertex adjacency CSR precompute

Lean module that, given the constraint topology (already known at
build time), produces:

- `vert_to_spring_offset[N_verts+1]`     CSR-style offset array
- `vert_to_spring_idx[total_incidences]` flattened spring ids
- `vert_to_spring_role[total_incidences]` 0 if vert == a-endpoint,
  1 if b-endpoint (so the gather knows which sign to apply)

Same for each constraint type. Validated by a `native_decide` example
that counts edges around a known fixture and matches a hand-checked
table.

### PR-C — vertex graph coloring

For parallel-safe vertex-block updates we need a vertex coloring such
that no two vertices in the same color share a constraint. Lean module
that emits the coloring as a `vert_color[N_verts]` array plus
`color_offsets[N_colors+1]` (so we can dispatch one color at a time).

Greedy first-fit coloring is adequate; usually yields 6-12 colors on
triangle meshes. Pinned reference for a small fixture.

### PR-D — vertex-block-update kernel

Per vertex (one thread per vertex within a color batch):
  1. Read x̃ (predictor), m (mass), x (current), h.
  2. g  := (m / h²) (x - x̃)
     H  := (m / h²) I₃
  3. For each constraint type, for each incident constraint:
       fetch (force_a or force_b based on role), (3x3 GN Hess block).
       g += fetched force
       H += fetched Hessian block
  4. Solve 3x3:  Δx = -H⁻¹ g    (closed-form determinant / adjugate)
  5. x ← x + Δx ;   write back position.

Single Slang kernel. Color-batched dispatch. `native_decide`-pinned
+ cpp validator on a hanging-mass fixture (3 verts, 2 springs, 1
gravity → reach equilibrium in ~5 outer iters).

### PR-E — outer iteration loop wiring in Simulation.cpp

`USE_AVBD=1` env switch routes `Simulation::step()` through:

```
for color c in colors:
    dispatch vertex-block-update kernel on vertices of color c
repeat N_outer (default 4) times
```

No global solve. No PD outer loop. No `MetalCGSolver`.

### PR-F — augmented Lagrangian dual update (the "A" in AVBD)

For each hard constraint (rigid joint, contact, attachment), maintain
a Lagrange multiplier λ_c and stiffness γ_c. After each outer iter
sweep:

```
λ_c ← λ_c + γ_c · C(x)        # dual ascent
γ_c ← clamp(γ_c · β, γ_max)    # stiffness adaption
```

Single Slang kernel. The vertex-block-update kernel reads the λ_c and
γ_c values and treats them as additional force terms.

### PR-G — differentiable adjoint pass

VBD's vertex-block update is a closed-form map x_{n+1} = f(x_n, params).
The adjoint is autograd-style reverse-mode through f, replacing
DiffCloth's PD complementarity gradient. Follow DiffAvatar's XPBD
adjoint recipe (Sec 4 of their paper).

### PR-H — dress-demo validation

Run dress + sock + hat + sphere + T-shirt under both `USE_AVBD=1`
and the existing PD path. Compare:

- Forward sim trajectories (visual sanity)
- L-BFGS-B convergence per demo
- Total wall per optimization run

Goal: match-or-beat the PD path on every demo with USE_AVBD=1 at
default settings. Stretch goal: 100-1000× faster wall.

## Scope

~6-10 PRs, ~3000 LOC across Lean + Slang + cpp. Estimated 4-8 weeks of
focused work. Each PR ships independently with `native_decide` and
cpp-validator coverage.

## Open questions

- AVBD adjoint hasn't been published. Need to derive ours from VBD's
  iter-map structure. Risk: tractable but unproven for cloth-scale
  scenes.
- AVBD's hard-constraint penalty parameter `γ` tuning may differ
  between cloth (mostly soft) and rigid (mostly hard) demos.
- Vertex coloring quality affects parallel efficiency. May need
  hierarchical coloring on dense meshes.
