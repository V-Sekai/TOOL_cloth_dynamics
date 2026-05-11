import Cloth.SlangCodegen

/-!
# `Cloth` — DiffCloth → Slang compute-shader codegen umbrella

Lean 4 spec + LeanSlang DSL for the V-Sekai DiffCloth fork's GPU
kernels. Mirrors the structure used by V-Sekai-fire's `Curvenet`
project (Pixar Profile Curves articulation): per-kernel `native_decide`
pinning of the emitted Slang source, paired with a `tests/slang_validate/`
host-diff harness that bit-checks the slangc-cpp-target output against
a hand-written CPU reference.

Submodules live under `Cloth.SlangCodegen.*`. The first kernel is
`Cloth.SlangCodegen.SpringProject`, the Projective-Dynamics local step
for stretch springs (`src/code/simulation/Spring.{h,cpp}`).
-/
