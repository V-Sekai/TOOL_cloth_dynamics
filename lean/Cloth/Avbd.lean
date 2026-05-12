import Cloth.Avbd.AdjacencySpring

/-!
# `Cloth.Avbd` — AVBD port data structures and precomputes

Lean-side primitives for the AVBD solver port (`todo.md`). Companion
to `Cloth.SlangCodegen`'s per-constraint *_force kernels (PR-A).

This umbrella covers the host-side data the GPU vertex-block-update
kernel (PR-D) consumes:

- `AdjacencySpring`  vertex → incident-spring CSR (this PR)
- ...
-/
