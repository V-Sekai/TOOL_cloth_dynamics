import LeanSlang

/-!
# `Cloth.SlangCodegen.TriangleBendingDualUpdate` — AVBD AL ramp for dihedral bending

Completes the AL dual-update set: attachment (#74), triangle membrane
(#78), and now dihedral bending. The bending energy targets a fixed
4-vertex weighted sum magnitude `n_target`; AL adds a single Vec3
multiplier per stencil that ramps with the constraint violation.

Per bending constraint `c` with 4-vertex stencil indices, cotangent
Laplacian weights `w[0..3]`, rest magnitude `n_target`, and AL
penalty `γ_c`:

```
s         = Σ_r  w[4c+r] · positions[idx[4c+r]]
C(x)      = s − target  =  s · (1 − n_target/|s|)
                                                 -- residual vector
                                                    (zero if |s| == n_target)
λ_c ← λ_c + γ_c · C(x)                           -- dual ascent
```

Degenerate stencils (`n_target = 0`) are no-op — λ stays unchanged.
This matches the `n_target ≤ eps` early-out semantics in
`TriangleBendingForce` (#46).

Bindings (set 0):

  0  StructuredBuffer<float3>   positions   length = N_verts
  1  StructuredBuffer<uint>     idx         length = 4 · N_bend
  2  StructuredBuffer<float>    weight      length = 4 · N_bend
  3  StructuredBuffer<float>    nTarget     length = N_bend
  4  StructuredBuffer<float>    gamma       length = N_bend
  5  RWStructuredBuffer<float3> lambda      length = N_bend

Mirrors the weighted-sum + projection math from TriangleBendingForce.
No output forces — caller dispatches `triangle_bending_force_al`
separately to consume λ in the gradient.
-/

namespace Cloth.SlangCodegen.TriangleBendingDualUpdate

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
  , .declInit f  "g_eff" (.litFloat 0.0)
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
      , .assign (.var "g_eff") (.index (.var "gamma") (.var "c"))
      ]
  , .declInit f3 "lam" (.index (.var "lambda") (.var "c"))
  , .assign (.index (.var "lambda") (.var "c"))
      (.call "float3"
        [ .bin "+" (.member (.var "lam") "x") (.bin "*" (.var "g_eff") (.var "ex"))
        , .bin "+" (.member (.var "lam") "y") (.bin "*" (.var "g_eff") (.var "ey"))
        , .bin "+" (.member (.var "lam") "z") (.bin "*" (.var "g_eff") (.var "ez")) ])
  ]

def shader : SlangShaderModule :=
  { globals :=
      [ bnd 0 "positions" (.roBuf f3)
      , bnd 1 "idx"       (.roBuf u)
      , bnd 2 "weight"    (.roBuf f)
      , bnd 3 "nTarget"   (.roBuf f)
      , bnd 4 "gamma"     (.roBuf f)
      , bnd 5 "lambda"    (.rwBuf f3)
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
StructuredBuffer<float> gamma;
[[vk::binding(5, 0)]]
RWStructuredBuffer<float3> lambda;

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
  float g_eff = 0.000000;
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
    g_eff = gamma[c];
  }
  float3 lam = lambda[c];
  lambda[c] = float3((lam.x + (g_eff * ex)), (lam.y + (g_eff * ey)), (lam.z + (g_eff * ez)));
}"

example : LeanSlang.emit shader = expected := by native_decide
example : shader.entryPointName = "main" := by native_decide

end Cloth.SlangCodegen.TriangleBendingDualUpdate
