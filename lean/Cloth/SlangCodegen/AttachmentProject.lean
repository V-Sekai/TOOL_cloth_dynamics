import LeanSlang

/-!
# `Cloth.SlangCodegen.AttachmentProject` — DiffCloth PD attachment local-step

Third kernel ported. Implements the Projective-Dynamics local step for
a single-vertex attachment (pin) constraint, as defined in
`src/code/simulation/AttachmentSpring.cpp:project`:

```cpp
Eigen::VectorXd AttachmentSpring::project(const VecXd &x_vec) const {
  Vec3d ret = fixedPointPos();
  return sqrtConstraintWeight * ret;
}
```

Per attachment `c`:

```
projected[c] = sqrtWeight[c] · fixedPos[c]
```

The current particle position is intentionally unused — the PD update
drives the pinned vertex toward the world-space anchor, and the local
step's job is just to emit that anchor (scaled).

One thread per attachment. Host pads with `sqrtWeight = 0` so unused
slots produce `(0, 0, 0)`.

Bindings (set 0):

  0  RWStructuredBuffer<float3> projected   length = N_attach (rounded up)
  1  StructuredBuffer<float3>   fixedPos    length = N_attach
  2  StructuredBuffer<float>    sqrtWeight  length = N_attach
-/

namespace Cloth.SlangCodegen.AttachmentProject

open LeanSlang

/-- PD per-attachment projection kernel. -/
def shader : SlangShaderModule :=
  let f3 : SlangType := .vec .float 3
  let u  : SlangType := .scalar .uint
  let f  : SlangType := .scalar .float
  let bnd (n : Nat) (name : String) (t : SlangType) : SlangBinding :=
    { name := name, type := t, semantic := Semantic.none
    , binding := some n, space := some 0 }
  let body : List SlangStmt :=
    [ .declInit u  "c"   (.member (.var "tid") "x")
    , .declInit f3 "fp"  (.index (.var "fixedPos") (.var "c"))
    , .declInit f  "sw"  (.index (.var "sqrtWeight") (.var "c"))
    , .assign (.index (.var "projected") (.var "c"))
        (.call "float3"
          [ .bin "*" (.var "sw") (.member (.var "fp") "x")
          , .bin "*" (.var "sw") (.member (.var "fp") "y")
          , .bin "*" (.var "sw") (.member (.var "fp") "z") ])
    ]
  { globals :=
      [ bnd 0 "projected"  (.rwBuf f3)
      , bnd 1 "fixedPos"   (.roBuf f3)
      , bnd 2 "sqrtWeight" (.roBuf f)
      ]
  , functions := [{
      attrs  := [.shaderCompute, .numthreads 64 1 1]
      name   := "main"
      params := [{ name := "tid", type := .vec .uint 3
                 , semantic := Semantic.svDispatchThreadId
                 , binding := none, space := none }]
      body   := body
    }] }

/-- Pinned reference emission. -/
def expected : String :=
"[[vk::binding(0, 0)]]
RWStructuredBuffer<float3> projected;
[[vk::binding(1, 0)]]
StructuredBuffer<float3> fixedPos;
[[vk::binding(2, 0)]]
StructuredBuffer<float> sqrtWeight;

[shader(\"compute\")] [numthreads(64, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
  uint c = tid.x;
  float3 fp = fixedPos[c];
  float sw = sqrtWeight[c];
  projected[c] = float3((sw * fp.x), (sw * fp.y), (sw * fp.z));
}"

example : LeanSlang.emit shader = expected := by native_decide

example : shader.entryPointName = "main" := by native_decide

end Cloth.SlangCodegen.AttachmentProject
