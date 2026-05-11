import LeanSlang

/-!
# `Cloth.SlangCodegen.SpringProject` — DiffCloth PD spring local-step

First kernel ported into the Lean → LeanSlang DSL pipeline. Implements
the Projective-Dynamics local step for a stretch spring, as defined in
`src/code/simulation/Spring.cpp:108–113`:

```cpp
Eigen::VectorXd Spring::project(const VecXd &x_vec) const {
  Vec3d pos_diff = (p1_vec3(x_vec) - p2_vec3(x_vec));
  Vec3d dir      = pos_diff.normalized();
  Vec3d p        = dir * l0;
  return sqrtConstraintWeight * p;
}
```

Per spring `i`, with rest length `l0[i]`, sqrt weight `w[i]`, and
endpoint indices `(a, b)` into the position buffer:

```
projected[i] = (w[i] * l0[i] / |p_a − p_b|) · (p_a − p_b)
```

One thread per spring. The host pads the dispatch to a multiple of
the group size (64) and seeds unused slots with safe values
(`p1Idx = 0`, `p2Idx` = any index with a distinct position,
`restLen = 0`, `sqrtWeight = 0` → output `(0,0,0)`).

Bindings (set 0):

  0  StructuredBuffer<float3>   positions   length = N_verts
  1  RWStructuredBuffer<float3> projected   length = N_springs (rounded up)
  2  StructuredBuffer<uint>     p1Idx       length = N_springs
  3  StructuredBuffer<uint>     p2Idx       length = N_springs
  4  StructuredBuffer<float>    restLen     length = N_springs
  5  StructuredBuffer<float>    sqrtWeight  length = N_springs

Per-spring sqrtWeight (rather than a uniform scalar) keeps the kernel
extensible to non-uniform stiffness without an ABI change.

The pinned reference text below is asserted by `native_decide`. Any
drift in `LeanSlang.Emit` that affects this output trips here. The
matching slangc-cpp host-diff harness lives in
`tests/slang_validate/spring_project_test.cpp`.
-/

namespace Cloth.SlangCodegen.SpringProject

open LeanSlang

/-- PD per-spring projection kernel as a Slang shader module. -/
def shader : SlangShaderModule :=
  let f3 : SlangType := .vec .float 3
  let u  : SlangType := .scalar .uint
  let f  : SlangType := .scalar .float
  let bnd (n : Nat) (name : String) (t : SlangType) : SlangBinding :=
    { name := name, type := t, semantic := Semantic.none
    , binding := some n, space := some 0 }
  let body : List SlangStmt :=
    [ .declInit u  "i"   (.member (.var "tid") "x")
    , .declInit u  "a"   (.index (.var "p1Idx") (.var "i"))
    , .declInit u  "b"   (.index (.var "p2Idx") (.var "i"))
    , .declInit f3 "p1"  (.index (.var "positions") (.var "a"))
    , .declInit f3 "p2"  (.index (.var "positions") (.var "b"))
    , .declInit f  "dx"  (.bin "-" (.member (.var "p1") "x")
                                   (.member (.var "p2") "x"))
    , .declInit f  "dy"  (.bin "-" (.member (.var "p1") "y")
                                   (.member (.var "p2") "y"))
    , .declInit f  "dz"  (.bin "-" (.member (.var "p1") "z")
                                   (.member (.var "p2") "z"))
    , .declInit f  "len2"
        (.bin "+"
          (.bin "+" (.bin "*" (.var "dx") (.var "dx"))
                    (.bin "*" (.var "dy") (.var "dy")))
          (.bin "*" (.var "dz") (.var "dz")))
    , .declInit f  "len" (.call "sqrt" [.var "len2"])
    , .declInit f  "scale"
        (.bin "/"
          (.bin "*" (.index (.var "sqrtWeight") (.var "i"))
                    (.index (.var "restLen") (.var "i")))
          (.var "len"))
    , .assign (.index (.var "projected") (.var "i"))
        (.call "float3"
          [ .bin "*" (.var "scale") (.var "dx")
          , .bin "*" (.var "scale") (.var "dy")
          , .bin "*" (.var "scale") (.var "dz") ])
    ]
  { globals :=
      [ bnd 0 "positions"  (.roBuf f3)
      , bnd 1 "projected"  (.rwBuf f3)
      , bnd 2 "p1Idx"      (.roBuf u)
      , bnd 3 "p2Idx"      (.roBuf u)
      , bnd 4 "restLen"    (.roBuf f)
      , bnd 5 "sqrtWeight" (.roBuf f)
      ]
  , functions := [{
      attrs  := [.shaderCompute, .numthreads 64 1 1]
      name   := "main"
      params := [{ name := "tid", type := .vec .uint 3
                 , semantic := Semantic.svDispatchThreadId
                 , binding := none, space := none }]
      body   := body
    }] }

/-- Pinned reference emission. Drift in `LeanSlang.Emit` trips
    `native_decide` below. Update both this string and the kernel
    in lockstep. -/
def expected : String :=
"[[vk::binding(0, 0)]]
StructuredBuffer<float3> positions;
[[vk::binding(1, 0)]]
RWStructuredBuffer<float3> projected;
[[vk::binding(2, 0)]]
StructuredBuffer<uint> p1Idx;
[[vk::binding(3, 0)]]
StructuredBuffer<uint> p2Idx;
[[vk::binding(4, 0)]]
StructuredBuffer<float> restLen;
[[vk::binding(5, 0)]]
StructuredBuffer<float> sqrtWeight;

[shader(\"compute\")] [numthreads(64, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
  uint i = tid.x;
  uint a = p1Idx[i];
  uint b = p2Idx[i];
  float3 p1 = positions[a];
  float3 p2 = positions[b];
  float dx = (p1.x - p2.x);
  float dy = (p1.y - p2.y);
  float dz = (p1.z - p2.z);
  float len2 = (((dx * dx) + (dy * dy)) + (dz * dz));
  float len = sqrt(len2);
  float scale = ((sqrtWeight[i] * restLen[i]) / len);
  projected[i] = float3((scale * dx), (scale * dy), (scale * dz));
}"

/-- The pretty-printer matches the pinned reference. -/
example : LeanSlang.emit shader = expected := by native_decide

/-- The kernel's entry-point name is `main`. -/
example : shader.entryPointName = "main" := by native_decide

end Cloth.SlangCodegen.SpringProject
