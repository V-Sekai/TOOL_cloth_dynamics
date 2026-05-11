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
  ]

def main (args : List String) : IO UInt32 := do
  let outDir := args.headD "."
  IO.FS.createDirAll outDir
  for (name, m) in kernels do
    let path := outDir ++ "/" ++ name ++ ".slang"
    IO.FS.writeFile path (LeanSlang.emit m ++ "\n")
    IO.println s!"wrote {path}"
  return 0
