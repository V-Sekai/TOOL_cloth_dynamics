import Cloth.SlangCodegen.SpringProject
import Cloth.SlangCodegen.TriangleBending
import Cloth.SlangCodegen.AttachmentProject
import Cloth.SlangCodegen.TriangleProject
import Cloth.SlangCodegen.Saxpby

/-!
# `Cloth.SlangCodegen` — Slang shader codegen umbrella

Each submodule produces a `LeanSlang.SlangShaderModule` for one of
the GPU kernels the DiffCloth runtime needs. Pinned `native_decide`
fixtures assert the emission text against a hand-checked reference
per kernel.

Layout convention: every kernel module exports

- `shader   : SlangShaderModule`
- `expected : String`
- two `example` lemmas: `emit shader = expected` and
  `shader.entryPointName = "main"`.
-/
