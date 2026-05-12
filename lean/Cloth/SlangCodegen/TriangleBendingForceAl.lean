import LeanSlang

/-!
# `Cloth.SlangCodegen.TriangleBendingForceAl` — AL-augmented dihedral bending force

Last kernel in PR-F's AL set. Companion to
`TriangleBendingDualUpdate` (#80). Same per-constraint dihedral bending
force + GN Hessian as `TriangleBendingForce` (#46) but adds the
accumulated dual multiplier `λ_c` to each per-vertex gradient,
weighted by the corresponding Laplacian weight.

The constraint Jacobian for bending is `∂C/∂p_r = w_r · I` (each
vertex contributes to the weighted sum `s` linearly in its
weight). So the AL gradient contribution at vertex `r` is:

```
∂(λᵀ C)/∂p_r  =  ∂C/∂p_r · λ  =  w_r · λ
```

Added to the original physics gradient:

```
grad[4c+r]       = k · w_r · resid  +  w_r · λ      (= w_r · (k·resid + λ))
hessScalar[4c+r] = k · w_r²                          -- unchanged
```

Degenerate stencils (`n_target ≤ eps`) zero both `k_eff` AND `λ_eff`
so the entire contribution disappears — matches the early-out in the
original bending force kernel.

Bindings (set 0):

  0  StructuredBuffer<float3>   positions    length = N_verts
  1  StructuredBuffer<uint>     idx          length = 4 · N_bend
  2  StructuredBuffer<float>    weight       length = 4 · N_bend
  3  StructuredBuffer<float>    nTarget      length = N_bend
  4  StructuredBuffer<float>    stiffness    length = N_bend
  5  StructuredBuffer<float3>   lambda       length = N_bend
  6  RWStructuredBuffer<float3> grad         length = 4 · N_bend
  7  RWStructuredBuffer<float>  hessScalar   length = 4 · N_bend
-/

namespace Cloth.SlangCodegen.TriangleBendingForceAl

open LeanSlang

private def f3 : SlangType := .vec .float 3
private def u  : SlangType := .scalar .uint
private def f  : SlangType := .scalar .float

private def bnd (n : Nat) (name : String) (t : SlangType) : SlangBinding :=
  { name := name, type := t, semantic := Semantic.none
  , binding := some n, space := some 0 }

private def body : List SlangStmt :=
  [ .declInit u  "c"     (.member (.var "tid") "x")
  , .declInit f  "n_c"   (.index (.var "nTarget") (.var "c"))
  , .declInit u  "base"  (.bin "*" (.var "c") (.litUint 4))
  , .declInit f  "w0"    (.index (.var "weight") (.var "base"))
  , .declInit f  "w1"    (.index (.var "weight") (.bin "+" (.var "base") (.litUint 1)))
  , .declInit f  "w2"    (.index (.var "weight") (.bin "+" (.var "base") (.litUint 2)))
  , .declInit f  "w3"    (.index (.var "weight") (.bin "+" (.var "base") (.litUint 3)))
  , .declInit f  "ex"    (.litFloat 0.0)
  , .declInit f  "ey"    (.litFloat 0.0)
  , .declInit f  "ez"    (.litFloat 0.0)
  , .declInit f  "k_eff" (.litFloat 0.0)
  , .declInit f  "lx"    (.litFloat 0.0)
  , .declInit f  "ly"    (.litFloat 0.0)
  , .declInit f  "lz"    (.litFloat 0.0)
  , .ifNoElse (.bin ">" (.var "n_c") (.litFloat 0.000001))
      [ .declInit u  "i0"   (.index (.var "idx") (.var "base"))
      , .declInit u  "i1"   (.index (.var "idx") (.bin "+" (.var "base") (.litUint 1)))
      , .declInit u  "i2"   (.index (.var "idx") (.bin "+" (.var "base") (.litUint 2)))
      , .declInit u  "i3"   (.index (.var "idx") (.bin "+" (.var "base") (.litUint 3)))
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
      , .declInit f  "om"    (.bin "-" (.litFloat 1.0) (.var "scale"))
      , .assign (.var "ex")  (.bin "*" (.var "sx") (.var "om"))
      , .assign (.var "ey")  (.bin "*" (.var "sy") (.var "om"))
      , .assign (.var "ez")  (.bin "*" (.var "sz") (.var "om"))
      , .assign (.var "k_eff") (.index (.var "stiffness") (.var "c"))
      , .declInit f3 "lam" (.index (.var "lambda") (.var "c"))
      , .assign (.var "lx") (.member (.var "lam") "x")
      , .assign (.var "ly") (.member (.var "lam") "y")
      , .assign (.var "lz") (.member (.var "lam") "z")
      ]
  , .assign (.index (.var "grad") (.var "base"))
      (.call "float3"
        [ .bin "+" (.bin "*" (.bin "*" (.var "k_eff") (.var "w0")) (.var "ex")) (.bin "*" (.var "w0") (.var "lx"))
        , .bin "+" (.bin "*" (.bin "*" (.var "k_eff") (.var "w0")) (.var "ey")) (.bin "*" (.var "w0") (.var "ly"))
        , .bin "+" (.bin "*" (.bin "*" (.var "k_eff") (.var "w0")) (.var "ez")) (.bin "*" (.var "w0") (.var "lz")) ])
  , .assign (.index (.var "grad") (.bin "+" (.var "base") (.litUint 1)))
      (.call "float3"
        [ .bin "+" (.bin "*" (.bin "*" (.var "k_eff") (.var "w1")) (.var "ex")) (.bin "*" (.var "w1") (.var "lx"))
        , .bin "+" (.bin "*" (.bin "*" (.var "k_eff") (.var "w1")) (.var "ey")) (.bin "*" (.var "w1") (.var "ly"))
        , .bin "+" (.bin "*" (.bin "*" (.var "k_eff") (.var "w1")) (.var "ez")) (.bin "*" (.var "w1") (.var "lz")) ])
  , .assign (.index (.var "grad") (.bin "+" (.var "base") (.litUint 2)))
      (.call "float3"
        [ .bin "+" (.bin "*" (.bin "*" (.var "k_eff") (.var "w2")) (.var "ex")) (.bin "*" (.var "w2") (.var "lx"))
        , .bin "+" (.bin "*" (.bin "*" (.var "k_eff") (.var "w2")) (.var "ey")) (.bin "*" (.var "w2") (.var "ly"))
        , .bin "+" (.bin "*" (.bin "*" (.var "k_eff") (.var "w2")) (.var "ez")) (.bin "*" (.var "w2") (.var "lz")) ])
  , .assign (.index (.var "grad") (.bin "+" (.var "base") (.litUint 3)))
      (.call "float3"
        [ .bin "+" (.bin "*" (.bin "*" (.var "k_eff") (.var "w3")) (.var "ex")) (.bin "*" (.var "w3") (.var "lx"))
        , .bin "+" (.bin "*" (.bin "*" (.var "k_eff") (.var "w3")) (.var "ey")) (.bin "*" (.var "w3") (.var "ly"))
        , .bin "+" (.bin "*" (.bin "*" (.var "k_eff") (.var "w3")) (.var "ez")) (.bin "*" (.var "w3") (.var "lz")) ])
  , .assign (.index (.var "hessScalar") (.var "base"))
      (.bin "*" (.var "k_eff") (.bin "*" (.var "w0") (.var "w0")))
  , .assign (.index (.var "hessScalar") (.bin "+" (.var "base") (.litUint 1)))
      (.bin "*" (.var "k_eff") (.bin "*" (.var "w1") (.var "w1")))
  , .assign (.index (.var "hessScalar") (.bin "+" (.var "base") (.litUint 2)))
      (.bin "*" (.var "k_eff") (.bin "*" (.var "w2") (.var "w2")))
  , .assign (.index (.var "hessScalar") (.bin "+" (.var "base") (.litUint 3)))
      (.bin "*" (.var "k_eff") (.bin "*" (.var "w3") (.var "w3")))
  ]

def shader : SlangShaderModule :=
  { globals :=
      [ bnd 0 "positions"  (.roBuf f3)
      , bnd 1 "idx"        (.roBuf u)
      , bnd 2 "weight"     (.roBuf f)
      , bnd 3 "nTarget"    (.roBuf f)
      , bnd 4 "stiffness"  (.roBuf f)
      , bnd 5 "lambda"     (.roBuf f3)
      , bnd 6 "grad"       (.rwBuf f3)
      , bnd 7 "hessScalar" (.rwBuf f)
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
StructuredBuffer<uint> idx;
[[vk::binding(2, 0)]]
StructuredBuffer<float> weight;
[[vk::binding(3, 0)]]
StructuredBuffer<float> nTarget;
[[vk::binding(4, 0)]]
StructuredBuffer<float> stiffness;
[[vk::binding(5, 0)]]
StructuredBuffer<float3> lambda;
[[vk::binding(6, 0)]]
RWStructuredBuffer<float3> grad;
[[vk::binding(7, 0)]]
RWStructuredBuffer<float> hessScalar;

[shader(\"compute\")] [numthreads(64, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
  uint c = tid.x;
  float n_c = nTarget[c];
  uint base = (c * 4u);
  float w0 = weight[base];
  float w1 = weight[(base + 1u)];
  float w2 = weight[(base + 2u)];
  float w3 = weight[(base + 3u)];
  float ex = 0.000000;
  float ey = 0.000000;
  float ez = 0.000000;
  float k_eff = 0.000000;
  float lx = 0.000000;
  float ly = 0.000000;
  float lz = 0.000000;
  if ((n_c > 0.000001)) {
    uint i0 = idx[base];
    uint i1 = idx[(base + 1u)];
    uint i2 = idx[(base + 2u)];
    uint i3 = idx[(base + 3u)];
    float3 p0 = positions[i0];
    float3 p1 = positions[i1];
    float3 p2 = positions[i2];
    float3 p3 = positions[i3];
    float sx = (((w0 * p0.x) + (w1 * p1.x)) + ((w2 * p2.x) + (w3 * p3.x)));
    float sy = (((w0 * p0.y) + (w1 * p1.y)) + ((w2 * p2.y) + (w3 * p3.y)));
    float sz = (((w0 * p0.z) + (w1 * p1.z)) + ((w2 * p2.z) + (w3 * p3.z)));
    float len = sqrt((((sx * sx) + (sy * sy)) + (sz * sz)));
    float scale = (n_c / len);
    float om = (1.000000 - scale);
    ex = (sx * om);
    ey = (sy * om);
    ez = (sz * om);
    k_eff = stiffness[c];
    float3 lam = lambda[c];
    lx = lam.x;
    ly = lam.y;
    lz = lam.z;
  }
  grad[base] = float3((((k_eff * w0) * ex) + (w0 * lx)), (((k_eff * w0) * ey) + (w0 * ly)), (((k_eff * w0) * ez) + (w0 * lz)));
  grad[(base + 1u)] = float3((((k_eff * w1) * ex) + (w1 * lx)), (((k_eff * w1) * ey) + (w1 * ly)), (((k_eff * w1) * ez) + (w1 * lz)));
  grad[(base + 2u)] = float3((((k_eff * w2) * ex) + (w2 * lx)), (((k_eff * w2) * ey) + (w2 * ly)), (((k_eff * w2) * ez) + (w2 * lz)));
  grad[(base + 3u)] = float3((((k_eff * w3) * ex) + (w3 * lx)), (((k_eff * w3) * ey) + (w3 * ly)), (((k_eff * w3) * ez) + (w3 * lz)));
  hessScalar[base] = (k_eff * (w0 * w0));
  hessScalar[(base + 1u)] = (k_eff * (w1 * w1));
  hessScalar[(base + 2u)] = (k_eff * (w2 * w2));
  hessScalar[(base + 3u)] = (k_eff * (w3 * w3));
}"

example : LeanSlang.emit shader = expected := by native_decide
example : shader.entryPointName = "main" := by native_decide

end Cloth.SlangCodegen.TriangleBendingForceAl
