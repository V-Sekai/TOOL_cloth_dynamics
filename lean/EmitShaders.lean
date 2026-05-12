import LeanSlang
import Cloth.SlangCodegen

/-!
# `emit_shaders` — Lake exe dumping every Slang shader to disk

For each `Cloth.SlangCodegen.*` kernel, writes the emitted Slang
source to `<outDir>/<name>.slang`. Used as input to slangc for
syntactic / semantic validation beyond the `native_decide` text
fixtures (see `tests/slang_validate/Makefile`).

Usage:

    lake exe emit_shaders /path/to/output/dir
-/

open Cloth.SlangCodegen
open LeanSlang

private def kernels : List (String × SlangShaderModule) :=
  [ ("spring_project",     SpringProject.shader)
  , ("triangle_bending",   TriangleBending.shader)
  , ("attachment_project", AttachmentProject.shader)
  , ("triangle_project",   TriangleProject.shader)
  , ("saxpby",             Saxpby.shader)
  , ("spmv",               Spmv.shader)
  , ("dot_reduce",         DotReduce.shader)
  , ("dot_reduce_serial",  DotReduceSerial.shader)
  , ("assemble_b",         AssembleB.shader)
  , ("cg_alpha",           CGAlpha.shader)
  , ("cg_beta",            CGBeta.shader)
  , ("saxpby_indirect",    SaxpbyIndirect.shader)
  , ("spmv_df32",          SpmvDf32.shader)
  , ("saxpby_indirect_df32", SaxpbyIndirectDf32.shader)
  , ("spring_force",       SpringForce.shader)
  , ("attachment_force",   AttachmentForce.shader)
  , ("triangle_membrane_force", TriangleMembraneForce.shader)
  , ("triangle_bending_force", TriangleBendingForce.shader)
  , ("vbd_init",           VbdInit.shader)
  , ("vbd_solve_apply",    VbdSolveApply.shader)
  , ("vbd_gather_spring",  VbdGatherSpring.shader)
  , ("vbd_gather_attachment", VbdGatherAttachment.shader)
  , ("vbd_gather_triangle", VbdGatherTriangle.shader)
  , ("vbd_gather_bending", VbdGatherBending.shader)
  , ("attachment_dual_update", AttachmentDualUpdate.shader)
  , ("attachment_force_al", AttachmentForceAl.shader)
  , ("triangle_membrane_dual_update", TriangleMembraneDualUpdate.shader)
  , ("triangle_membrane_force_al", TriangleMembraneForceAl.shader)
  , ("triangle_bending_dual_update", TriangleBendingDualUpdate.shader)
  , ("triangle_bending_force_al", TriangleBendingForceAl.shader)
  ]

def main (args : List String) : IO UInt32 := do
  let outDir := args.headD "."
  IO.FS.createDirAll outDir
  for (name, m) in kernels do
    let path := outDir ++ "/" ++ name ++ ".slang"
    IO.FS.writeFile path (LeanSlang.emit m ++ "\n")
    IO.println s!"wrote {path}"
  return 0
