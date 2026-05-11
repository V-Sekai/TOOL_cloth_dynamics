import LeanSlang

/-!
# `Cloth.SlangCodegen.TriangleBending` — DiffCloth PD bending local-step

Second kernel ported. Implements the Projective-Dynamics local step for
a dihedral-bending constraint, as defined in
`src/code/simulation/TriangleBending.cpp:project`:

```cpp
VecXd TriangleBending::project(const VecXd &x_vec) const {
  Vec3d e; e.setZero();
  if (n > 1e-6) {
    for (int i = 0; i < 4; i++)
      e += weightVert[i] * x_vec.segment(3 * idx_arr[i], 3);
    e = e.normalized() * n;
  }
  return constrainWeightSqrt * e;
}
```

Per constraint `c` (4-vertex stencil over the two triangles sharing an
edge, with bind-time cotangent-Laplacian weights):

```
e_c = Σ_{j=0..3} weight[4c + j] · positions[idx[4c + j]]
projected[c] = sqrtWeight[c] · ( e_c / |e_c| · nTarget[c] )     if nTarget[c] > 1e-6
             = (0, 0, 0)                                         otherwise
```

`nTarget` is captured at bind time and represents the rest-pose bending
magnitude; the local step pulls the current weighted sum toward that
magnitude while preserving its direction. The `nTarget == 0` branch
covers degenerate / collinear stencils.

One thread per constraint. The host pads the dispatch to a multiple of
the group size (64) and seeds unused slots with `nTarget = 0` so they
short-circuit to `(0,0,0)` without touching the (idx, weight) buffers.

Bindings (set 0):

  0  StructuredBuffer<float3>   positions   length = N_verts
  1  RWStructuredBuffer<float3> projected   length = N_constraints (rounded up)
  2  StructuredBuffer<uint>     idx         length = 4 · N_constraints
  3  StructuredBuffer<float>    weight      length = 4 · N_constraints
  4  StructuredBuffer<float>    nTarget     length = N_constraints
  5  StructuredBuffer<float>    sqrtWeight  length = N_constraints

The pinned reference text below is asserted by `native_decide`. The
matching slangc-cpp host-diff harness lives in
`tests/slang_validate/triangle_bending_test.cpp`.
-/

namespace Cloth.SlangCodegen.TriangleBending

open LeanSlang

/-- PD per-constraint bending projection kernel. -/
def shader : SlangShaderModule :=
  let f3 : SlangType := .vec .float 3
  let u  : SlangType := .scalar .uint
  let f  : SlangType := .scalar .float
  let bnd (n : Nat) (name : String) (t : SlangType) : SlangBinding :=
    { name := name, type := t, semantic := Semantic.none
    , binding := some n, space := some 0 }
  let body : List SlangStmt :=
    [ .declInit u  "c"    (.member (.var "tid") "x")
    , .declInit f  "n_c"  (.index (.var "nTarget") (.var "c"))
    , .declInit f  "ex"   (.litFloat 0.0)
    , .declInit f  "ey"   (.litFloat 0.0)
    , .declInit f  "ez"   (.litFloat 0.0)
    , .ifNoElse (.bin ">" (.var "n_c") (.litFloat 0.000001))
        [ .declInit u  "base" (.bin "*" (.var "c") (.litUint 4))
        , .declInit u  "i0"   (.index (.var "idx") (.var "base"))
        , .declInit u  "i1"   (.index (.var "idx")
                                (.bin "+" (.var "base") (.litUint 1)))
        , .declInit u  "i2"   (.index (.var "idx")
                                (.bin "+" (.var "base") (.litUint 2)))
        , .declInit u  "i3"   (.index (.var "idx")
                                (.bin "+" (.var "base") (.litUint 3)))
        , .declInit f  "w0"   (.index (.var "weight") (.var "base"))
        , .declInit f  "w1"   (.index (.var "weight")
                                (.bin "+" (.var "base") (.litUint 1)))
        , .declInit f  "w2"   (.index (.var "weight")
                                (.bin "+" (.var "base") (.litUint 2)))
        , .declInit f  "w3"   (.index (.var "weight")
                                (.bin "+" (.var "base") (.litUint 3)))
        , .declInit f3 "p0"   (.index (.var "positions") (.var "i0"))
        , .declInit f3 "p1"   (.index (.var "positions") (.var "i1"))
        , .declInit f3 "p2"   (.index (.var "positions") (.var "i2"))
        , .declInit f3 "p3"   (.index (.var "positions") (.var "i3"))
        , .declInit f  "sx"
            (.bin "+"
              (.bin "+" (.bin "*" (.var "w0") (.member (.var "p0") "x"))
                        (.bin "*" (.var "w1") (.member (.var "p1") "x")))
              (.bin "+" (.bin "*" (.var "w2") (.member (.var "p2") "x"))
                        (.bin "*" (.var "w3") (.member (.var "p3") "x"))))
        , .declInit f  "sy"
            (.bin "+"
              (.bin "+" (.bin "*" (.var "w0") (.member (.var "p0") "y"))
                        (.bin "*" (.var "w1") (.member (.var "p1") "y")))
              (.bin "+" (.bin "*" (.var "w2") (.member (.var "p2") "y"))
                        (.bin "*" (.var "w3") (.member (.var "p3") "y"))))
        , .declInit f  "sz"
            (.bin "+"
              (.bin "+" (.bin "*" (.var "w0") (.member (.var "p0") "z"))
                        (.bin "*" (.var "w1") (.member (.var "p1") "z")))
              (.bin "+" (.bin "*" (.var "w2") (.member (.var "p2") "z"))
                        (.bin "*" (.var "w3") (.member (.var "p3") "z"))))
        , .declInit f  "len"
            (.call "sqrt"
              [ .bin "+"
                  (.bin "+" (.bin "*" (.var "sx") (.var "sx"))
                            (.bin "*" (.var "sy") (.var "sy")))
                  (.bin "*" (.var "sz") (.var "sz")) ])
        , .declInit f  "scale" (.bin "/" (.var "n_c") (.var "len"))
        , .assign (.var "ex") (.bin "*" (.var "scale") (.var "sx"))
        , .assign (.var "ey") (.bin "*" (.var "scale") (.var "sy"))
        , .assign (.var "ez") (.bin "*" (.var "scale") (.var "sz"))
        ]
    , .declInit f  "sw"  (.index (.var "sqrtWeight") (.var "c"))
    , .assign (.index (.var "projected") (.var "c"))
        (.call "float3"
          [ .bin "*" (.var "sw") (.var "ex")
          , .bin "*" (.var "sw") (.var "ey")
          , .bin "*" (.var "sw") (.var "ez") ])
    ]
  { globals :=
      [ bnd 0 "positions"  (.roBuf f3)
      , bnd 1 "projected"  (.rwBuf f3)
      , bnd 2 "idx"        (.roBuf u)
      , bnd 3 "weight"     (.roBuf f)
      , bnd 4 "nTarget"    (.roBuf f)
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

/-- Pinned reference emission. -/
def expected : String :=
"[[vk::binding(0, 0)]]
StructuredBuffer<float3> positions;
[[vk::binding(1, 0)]]
RWStructuredBuffer<float3> projected;
[[vk::binding(2, 0)]]
StructuredBuffer<uint> idx;
[[vk::binding(3, 0)]]
StructuredBuffer<float> weight;
[[vk::binding(4, 0)]]
StructuredBuffer<float> nTarget;
[[vk::binding(5, 0)]]
StructuredBuffer<float> sqrtWeight;

[shader(\"compute\")] [numthreads(64, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
  uint c = tid.x;
  float n_c = nTarget[c];
  float ex = 0.000000;
  float ey = 0.000000;
  float ez = 0.000000;
  if ((n_c > 0.000001)) {
    uint base = (c * 4u);
    uint i0 = idx[base];
    uint i1 = idx[(base + 1u)];
    uint i2 = idx[(base + 2u)];
    uint i3 = idx[(base + 3u)];
    float w0 = weight[base];
    float w1 = weight[(base + 1u)];
    float w2 = weight[(base + 2u)];
    float w3 = weight[(base + 3u)];
    float3 p0 = positions[i0];
    float3 p1 = positions[i1];
    float3 p2 = positions[i2];
    float3 p3 = positions[i3];
    float sx = (((w0 * p0.x) + (w1 * p1.x)) + ((w2 * p2.x) + (w3 * p3.x)));
    float sy = (((w0 * p0.y) + (w1 * p1.y)) + ((w2 * p2.y) + (w3 * p3.y)));
    float sz = (((w0 * p0.z) + (w1 * p1.z)) + ((w2 * p2.z) + (w3 * p3.z)));
    float len = sqrt((((sx * sx) + (sy * sy)) + (sz * sz)));
    float scale = (n_c / len);
    ex = (scale * sx);
    ey = (scale * sy);
    ez = (scale * sz);
  }
  float sw = sqrtWeight[c];
  projected[c] = float3((sw * ex), (sw * ey), (sw * ez));
}"

example : LeanSlang.emit shader = expected := by native_decide

example : shader.entryPointName = "main" := by native_decide

end Cloth.SlangCodegen.TriangleBending
