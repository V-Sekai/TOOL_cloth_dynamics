import LeanSlang

/-!
# `Cloth.SlangCodegen.AttachmentForceAl` — AVBD augmented-Lagrangian attachment force

Companion to `AttachmentDualUpdate` (#74). After the dual-ramp kernel
updates `λ_c` each outer-iter sweep, the next force-compute pass reads
the accumulated multiplier and folds it into the per-attachment
gradient.

Augmented-Lagrangian energy for attachment c pinning vertex
`v = vertIdx[c]` to anchor `fixedPos[c]` with effective stiffness
`k_eff[c] = k + γ_c`:

```
E_al(x_v)   = (1/2) · k_eff · |C(x_v)|²  +  λ_cᵀ C(x_v)
              where C(x_v) = x_v − fixedPos[c]

∇_v E_al    = k_eff · C(x_v)  +  λ_c
∇²_v E_al   = k_eff · I₃                 -- same scalar·I shape as
                                            the original
                                            attachment_force
```

Drop-in replacement for `attachment_force` (PR #44) with one extra
input buffer (`lambda`) and one extra add-per-axis in the gradient.
Hessian shape unchanged — the AL augmentation only shifts the
gradient by `λ_c`, not the curvature.

Bindings (set 0):

  0  StructuredBuffer<float3>   positions    length = N_verts
  1  StructuredBuffer<uint>     vertIdx      length = N_attach
  2  StructuredBuffer<float3>   fixedPos     length = N_attach
  3  StructuredBuffer<float>    stiffness    length = N_attach   (= k_eff)
  4  StructuredBuffer<float3>   lambda       length = N_attach   (accumulated
                                                                   by attachment_dual_update)
  5  RWStructuredBuffer<float3> gradV        length = N_attach
  6  RWStructuredBuffer<float>  hessScalar   length = N_attach
-/

namespace Cloth.SlangCodegen.AttachmentForceAl

open LeanSlang

private def f3 : SlangType := .vec .float 3
private def u  : SlangType := .scalar .uint
private def f  : SlangType := .scalar .float

private def bnd (n : Nat) (name : String) (t : SlangType) : SlangBinding :=
  { name := name, type := t, semantic := Semantic.none
  , binding := some n, space := some 0 }

private def body : List SlangStmt :=
  [ .declInit u  "c"   (.member (.var "tid") "x")
  , .declInit u  "v"   (.index (.var "vertIdx") (.var "c"))
  , .declInit f3 "p"   (.index (.var "positions") (.var "v"))
  , .declInit f3 "fp"  (.index (.var "fixedPos") (.var "c"))
  , .declInit f  "k"   (.index (.var "stiffness") (.var "c"))
  , .declInit f3 "lam" (.index (.var "lambda") (.var "c"))
  , .assign (.index (.var "gradV") (.var "c"))
      (.call "float3"
        [ .bin "+"
            (.bin "*" (.var "k")
              (.bin "-" (.member (.var "p") "x") (.member (.var "fp") "x")))
            (.member (.var "lam") "x")
        , .bin "+"
            (.bin "*" (.var "k")
              (.bin "-" (.member (.var "p") "y") (.member (.var "fp") "y")))
            (.member (.var "lam") "y")
        , .bin "+"
            (.bin "*" (.var "k")
              (.bin "-" (.member (.var "p") "z") (.member (.var "fp") "z")))
            (.member (.var "lam") "z") ])
  , .assign (.index (.var "hessScalar") (.var "c")) (.var "k")
  ]

def shader : SlangShaderModule :=
  { globals :=
      [ bnd 0 "positions"  (.roBuf f3)
      , bnd 1 "vertIdx"    (.roBuf u)
      , bnd 2 "fixedPos"   (.roBuf f3)
      , bnd 3 "stiffness"  (.roBuf f)
      , bnd 4 "lambda"     (.roBuf f3)
      , bnd 5 "gradV"      (.rwBuf f3)
      , bnd 6 "hessScalar" (.rwBuf f)
      ]
  , functions := [{
      attrs  := [.shaderCompute, .numthreads 64 1 1]
      name   := "main"
      params := [{ name := "tid", type := .vec .uint 3
                 , semantic := Semantic.svDispatchThreadId
                 , binding := none, space := none }]
      body   := body
    }] }

def expected : String :=
"[[vk::binding(0, 0)]]
StructuredBuffer<float3> positions;
[[vk::binding(1, 0)]]
StructuredBuffer<uint> vertIdx;
[[vk::binding(2, 0)]]
StructuredBuffer<float3> fixedPos;
[[vk::binding(3, 0)]]
StructuredBuffer<float> stiffness;
[[vk::binding(4, 0)]]
StructuredBuffer<float3> lambda;
[[vk::binding(5, 0)]]
RWStructuredBuffer<float3> gradV;
[[vk::binding(6, 0)]]
RWStructuredBuffer<float> hessScalar;

[shader(\"compute\")] [numthreads(64, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
  uint c = tid.x;
  uint v = vertIdx[c];
  float3 p = positions[v];
  float3 fp = fixedPos[c];
  float k = stiffness[c];
  float3 lam = lambda[c];
  gradV[c] = float3(((k * (p.x - fp.x)) + lam.x), ((k * (p.y - fp.y)) + lam.y), ((k * (p.z - fp.z)) + lam.z));
  hessScalar[c] = k;
}"

example : LeanSlang.emit shader = expected := by native_decide
example : shader.entryPointName = "main" := by native_decide

end Cloth.SlangCodegen.AttachmentForceAl
