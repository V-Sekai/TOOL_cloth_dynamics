import Cloth.SlangCodegen.SpringProject
import Cloth.SlangCodegen.TriangleBending
import Cloth.SlangCodegen.AttachmentProject
import Cloth.SlangCodegen.TriangleProject
import Cloth.SlangCodegen.Saxpby
import Cloth.SlangCodegen.Spmv
import Cloth.SlangCodegen.DotReduce
import Cloth.SlangCodegen.DotReduceSerial
import Cloth.SlangCodegen.AssembleB
import Cloth.SlangCodegen.CGAlpha
import Cloth.SlangCodegen.CGBeta
import Cloth.SlangCodegen.SaxpbyIndirect
import Cloth.SlangCodegen.SpmvDf32
import Cloth.SlangCodegen.SaxpbyIndirectDf32
import Cloth.SlangCodegen.SpringForce
import Cloth.SlangCodegen.AttachmentForce
import Cloth.SlangCodegen.TriangleMembraneForce
import Cloth.SlangCodegen.TriangleBendingForce
import Cloth.SlangCodegen.VbdInit
import Cloth.SlangCodegen.VbdSolveApply
import Cloth.SlangCodegen.VbdGatherSpring
import Cloth.SlangCodegen.VbdGatherAttachment
import Cloth.SlangCodegen.VbdGatherTriangle
import Cloth.SlangCodegen.VbdGatherBending
import Cloth.SlangCodegen.AttachmentDualUpdate
import Cloth.SlangCodegen.AttachmentForceAl
import Cloth.SlangCodegen.TriangleMembraneDualUpdate
import Cloth.SlangCodegen.TriangleMembraneForceAl

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
