import LeanSlang

/-!
# `Cloth.SlangCodegen.AttachmentForce` — AVBD attachment force + Hessian

Second constraint-force kernel for the AVBD port (todo.md PR-A). For
each attachment `c` pinning vertex `v = vertIdx[c]` to world-space
anchor `fixedPos[c]` with stiffness `k`:

```
gradV[c]      = k · (p_v − fixedPos[c])
hessScalar[c] = k                          -- Hessian is k · I₃, fully
                                              described by the scalar k
```

Attachment is the simplest constraint geometry: one vertex, no role
question (it's always "this vertex"), and the Hessian is a scalar
multiple of the identity, so the per-constraint write is just one
float instead of `spring_force`'s packed 6-float block.

Energy reference:

```
E_c = (1/2) · k · |p_v − fixedPos|²
```

Gradient on `p_v`:

```
∂E/∂p_v = k · (p_v − fixedPos)
```

Hessian on `p_v`:

```
∇²_v E = k · I₃
```

The per-vertex AVBD gather (PR-D) for an incident attachment `c` on
vertex `v` adds `gradV[c]` to `g` and `hessScalar[c]` to each diagonal
entry of the local 3x3 `H`.

Bindings (set 0):

  0  StructuredBuffer<float3>   positions    length = N_verts
  1  StructuredBuffer<uint>     vertIdx      length = N_attach
  2  StructuredBuffer<float3>   fixedPos     length = N_attach
  3  StructuredBuffer<float>    stiffness    length = N_attach
  4  RWStructuredBuffer<float3> gradV        length = N_attach
  5  RWStructuredBuffer<float>  hessScalar   length = N_attach
-/

namespace Cloth.SlangCodegen.AttachmentForce

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
  , .assign (.index (.var "gradV") (.var "c"))
      (.call "float3"
        [ .bin "*" (.var "k")
            (.bin "-" (.member (.var "p") "x") (.member (.var "fp") "x"))
        , .bin "*" (.var "k")
            (.bin "-" (.member (.var "p") "y") (.member (.var "fp") "y"))
        , .bin "*" (.var "k")
            (.bin "-" (.member (.var "p") "z") (.member (.var "fp") "z")) ])
  , .assign (.index (.var "hessScalar") (.var "c")) (.var "k")
  ]

def shader : SlangShaderModule :=
  { globals :=
      [ bnd 0 "positions"  (.roBuf f3)
      , bnd 1 "vertIdx"    (.roBuf u)
      , bnd 2 "fixedPos"   (.roBuf f3)
      , bnd 3 "stiffness"  (.roBuf f)
      , bnd 4 "gradV"      (.rwBuf f3)
      , bnd 5 "hessScalar" (.rwBuf f)
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
RWStructuredBuffer<float3> gradV;
[[vk::binding(5, 0)]]
RWStructuredBuffer<float> hessScalar;

[shader(\"compute\")] [numthreads(64, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
  uint c = tid.x;
  uint v = vertIdx[c];
  float3 p = positions[v];
  float3 fp = fixedPos[c];
  float k = stiffness[c];
  gradV[c] = float3((k * (p.x - fp.x)), (k * (p.y - fp.y)), (k * (p.z - fp.z)));
  hessScalar[c] = k;
}"

example : LeanSlang.emit shader = expected := by native_decide
example : shader.entryPointName = "main" := by native_decide

end Cloth.SlangCodegen.AttachmentForce
