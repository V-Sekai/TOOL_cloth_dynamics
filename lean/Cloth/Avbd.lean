import Cloth.Avbd.AdjacencySpring
import Cloth.Avbd.AdjacencyKwise
import Cloth.Avbd.Coloring

/-!
# `Cloth.Avbd` — AVBD port data structures and precomputes

Lean-side primitives for the AVBD solver port (`todo.md`). Companion
to `Cloth.SlangCodegen`'s per-constraint *_force kernels (PR-A).

This umbrella covers the host-side data the GPU vertex-block-update
kernel (PR-D) consumes:

- `AdjacencySpring`  vertex → incident-spring CSR (springs only, K=2)
- `AdjacencyKwise`   vertex → incident-constraint CSR (K=1, 3, 4 —
                     attachment / triangle / bending)
- `Coloring`         greedy first-fit vertex coloring for parallel-safe
                     dispatch of the vertex-block-update kernel
-/
