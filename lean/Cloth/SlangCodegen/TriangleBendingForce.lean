import LeanSlang

/-!
# `Cloth.SlangCodegen.TriangleBendingForce` — AVBD dihedral bending force

Fourth and final per-constraint force kernel for the AVBD port
(todo.md PR-A). For each bending constraint `c` (4-vertex stencil
over the two triangles sharing an edge, with bind-time cotangent
Laplacian weights `w[4c+r]`), with rest bending magnitude `n_target`
and stiffness `k`:

```
s        = Σ_{r=0..3}  w[4c+r] · positions[idx[4c+r]]
target   = (n_target / |s|) · s             -- closest point on
                                               |·| = n_target sphere
resid    = s − target = s · (1 − n_target/|s|)

E_c      = (k/2) · |s − target|²
∂E/∂p_v  = k · w[role(v,c)] · resid          -- ∂s/∂p_v = w · I₃

grad[4c+r]       = k · w[4c+r] · resid       -- 4 per-vertex gradients
hessScalar[4c+r] = k · (w[4c+r])²            -- diagonal block scalar
```

The Gauss-Newton Hessian holds `target` fixed (the same R-fixed trick
PD's local step uses for rotation-invariant elastic energies). Each
diagonal block ∇²_v E reduces to a scalar multiple of `I₃`.

Degenerate stencils — bind-time collinear neighbors — are marked by
`n_target = 0` (matching the PD `TriangleBending` convention). In that
case the constraint is skipped: grad and hess scalars all write zero.
We implement this cleanly by reading weights unconditionally and
gating the effective stiffness on `n_target > 1e-6`:

```
k_eff = (n_target > 1e-6) ? stiffness[c] : 0
```

Reading weight[base..base+3] outside the if-branch is safe even on
padded slots — multiplying by `k_eff = 0` zeroes everything, no UB.

Bindings (set 0):

  0  StructuredBuffer<float3>   positions    length = N_verts
  1  StructuredBuffer<uint>     idx          length = 4 · N_bend
  2  StructuredBuffer<float>    weight       length = 4 · N_bend
  3  StructuredBuffer<float>    nTarget      length = N_bend
  4  StructuredBuffer<float>    stiffness    length = N_bend
  5  RWStructuredBuffer<float3> grad         length = 4 · N_bend
  6  RWStructuredBuffer<float>  hessScalar   length = 4 · N_bend
-/

namespace Cloth.SlangCodegen.TriangleBendingForce

open LeanSlang

private def f3 : SlangType := .vec .float 3
private def u  : SlangType := .scalar .uint
private def f  : SlangType := .scalar .float

private def bnd (n : Nat) (name : String) (t : SlangType) : SlangBinding :=
  { name := name, type := t, semantic := Semantic.none
  , binding := some n, space := some 0 }

private def body : List SlangStmt :=
  [ .declInit u  "c"    (.member (.var "tid") "x")
  , .declInit f  "n_c"  (.index (.var "nTarget") (.var "c"))
  , .declInit u  "base" (.bin "*" (.var "c") (.litUint 4))
  , .declInit f  "w0"   (.index (.var "weight") (.var "base"))
  , .declInit f  "w1"   (.index (.var "weight") (.bin "+" (.var "base") (.litUint 1)))
  , .declInit f  "w2"   (.index (.var "weight") (.bin "+" (.var "base") (.litUint 2)))
  , .declInit f  "w3"   (.index (.var "weight") (.bin "+" (.var "base") (.litUint 3)))
  , .declInit f  "ex"   (.litFloat 0.0)
  , .declInit f  "ey"   (.litFloat 0.0)
  , .declInit f  "ez"   (.litFloat 0.0)
  , .declInit f  "k_eff" (.litFloat 0.0)
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
      ]
  , .assign (.index (.var "grad") (.var "base"))
      (.call "float3"
        [ .bin "*" (.bin "*" (.var "k_eff") (.var "w0")) (.var "ex")
        , .bin "*" (.bin "*" (.var "k_eff") (.var "w0")) (.var "ey")
        , .bin "*" (.bin "*" (.var "k_eff") (.var "w0")) (.var "ez") ])
  , .assign (.index (.var "grad") (.bin "+" (.var "base") (.litUint 1)))
      (.call "float3"
        [ .bin "*" (.bin "*" (.var "k_eff") (.var "w1")) (.var "ex")
        , .bin "*" (.bin "*" (.var "k_eff") (.var "w1")) (.var "ey")
        , .bin "*" (.bin "*" (.var "k_eff") (.var "w1")) (.var "ez") ])
  , .assign (.index (.var "grad") (.bin "+" (.var "base") (.litUint 2)))
      (.call "float3"
        [ .bin "*" (.bin "*" (.var "k_eff") (.var "w2")) (.var "ex")
        , .bin "*" (.bin "*" (.var "k_eff") (.var "w2")) (.var "ey")
        , .bin "*" (.bin "*" (.var "k_eff") (.var "w2")) (.var "ez") ])
  , .assign (.index (.var "grad") (.bin "+" (.var "base") (.litUint 3)))
      (.call "float3"
        [ .bin "*" (.bin "*" (.var "k_eff") (.var "w3")) (.var "ex")
        , .bin "*" (.bin "*" (.var "k_eff") (.var "w3")) (.var "ey")
        , .bin "*" (.bin "*" (.var "k_eff") (.var "w3")) (.var "ez") ])
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
      , bnd 5 "grad"       (.rwBuf f3)
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
StructuredBuffer<uint> idx;
[[vk::binding(2, 0)]]
StructuredBuffer<float> weight;
[[vk::binding(3, 0)]]
StructuredBuffer<float> nTarget;
[[vk::binding(4, 0)]]
StructuredBuffer<float> stiffness;
[[vk::binding(5, 0)]]
RWStructuredBuffer<float3> grad;
[[vk::binding(6, 0)]]
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
  }
  grad[base] = float3(((k_eff * w0) * ex), ((k_eff * w0) * ey), ((k_eff * w0) * ez));
  grad[(base + 1u)] = float3(((k_eff * w1) * ex), ((k_eff * w1) * ey), ((k_eff * w1) * ez));
  grad[(base + 2u)] = float3(((k_eff * w2) * ex), ((k_eff * w2) * ey), ((k_eff * w2) * ez));
  grad[(base + 3u)] = float3(((k_eff * w3) * ex), ((k_eff * w3) * ey), ((k_eff * w3) * ez));
  hessScalar[base] = (k_eff * (w0 * w0));
  hessScalar[(base + 1u)] = (k_eff * (w1 * w1));
  hessScalar[(base + 2u)] = (k_eff * (w2 * w2));
  hessScalar[(base + 3u)] = (k_eff * (w3 * w3));
}"

example : LeanSlang.emit shader = expected := by native_decide
example : shader.entryPointName = "main" := by native_decide

end Cloth.SlangCodegen.TriangleBendingForce
