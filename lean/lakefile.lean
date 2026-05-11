import Lake
open Lake DSL

package Cloth where

require LeanSlang from git
  "https://github.com/V-Sekai-fire/lean-slang.git" @ "v0.0.5"

@[default_target] lean_lib Cloth where

lean_exe emit_shaders where
  root := `EmitShaders
