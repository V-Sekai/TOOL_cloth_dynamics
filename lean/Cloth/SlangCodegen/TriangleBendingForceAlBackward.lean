import LeanSlang

/-!
# `Cloth.SlangCodegen.TriangleBendingForceAlBackward` — adjoint of triangle_bending_force_al (PR-G / CHI-13)

Reverse-mode VJP for the AL-augmented dihedral bending force kernel.
Forward per bending stencil `c` (4-vertex Laplacian) when `n_c > eps`:

```
s     = w0·p0 + w1·p1 + w2·p2 + w3·p3   -- 3-vec weighted sum
len   = |s|
scale = n_c / len
om    = 1 − scale
e     = s · om                          -- 3-vec residual (s − R·s where R is closest-target dir)
te    = k · e + λ                       -- 3-vec AL-augmented force
grad[4c+r]       = w_r · te
hessScalar[4c+r] = k · w_r²
```

When `n_c ≤ eps`: forward writes zeros, backward propagates zeros.

Per-constraint thread. Cotangents flowing in from
`vbd_gather_bending_backward`:

* `v_grad[4c+r]`      — 3-vec cotangent on grad, per corner r ∈ {0..3}
* `v_hessScalar[4c+r]` — scalar cotangent on hess, per corner r

Define the contracted cotangents:

```
A  = Σ_r  w_r  · v_grad[4c+r]            -- 3-vec
HW = Σ_r  w_r² · v_hessScalar[4c+r]      -- scalar
```

Then the chain rule gives:

```
v_lambda  = A
v_k       = A · e + HW
v_nTarget = −(k · (A · s)) / len
v_p_r     = w_r · (k · om · A + k · n_c · (A · s) / len³ · s)
```

The position cotangent `v_p_r` is emitted PER-CORNER (4 entries per
constraint, slot `4c+r`). The host orchestration accumulates it into
per-vertex `v_positions` via re-use of `vbd_gather_bending`
(with `bendGrad := v_p`, `bendHessScalar := zeros`), which uses the
existing bending-vertex CSR.

Inv-deltaUV / weight cotangents are not emitted — those are
bind-time geometric constants, not optimized by L-BFGS-B (see CHI-13).

Bindings (set 0):

  0  StructuredBuffer<float3>   positions     length = N_verts
  1  StructuredBuffer<uint>     idx           length = 4·N_bend
  2  StructuredBuffer<float>    weight        length = 4·N_bend
  3  StructuredBuffer<float>    nTarget       length = N_bend
  4  StructuredBuffer<float>    stiffness     length = N_bend
  5  StructuredBuffer<float3>   lambda        length = N_bend
  6  StructuredBuffer<float3>   v_grad        length = 4·N_bend  (cotangent)
  7  StructuredBuffer<float>    v_hessScalar  length = 4·N_bend  (cotangent)
  8  RWStructuredBuffer<float3> v_p           length = 4·N_bend  (∂L/∂p per corner)
  9  RWStructuredBuffer<float>  v_nTarget     length = N_bend    (∂L/∂n_c)
 10  RWStructuredBuffer<float>  v_stiffness   length = N_bend    (∂L/∂k)
 11  RWStructuredBuffer<float3> v_lambda      length = N_bend    (∂L/∂λ)
-/

namespace Cloth.SlangCodegen.TriangleBendingForceAlBackward

open LeanSlang

private def f3 : SlangType := .vec .float 3
private def u  : SlangType := .scalar .uint
private def f  : SlangType := .scalar .float

private def bnd (n : Nat) (name : String) (t : SlangType) : SlangBinding :=
  { name := name, type := t, semantic := Semantic.none
  , binding := some n, space := some 0 }

private def body : List SlangStmt :=
  [ .declInit u  "c"     (.member (.var "tid") "x")
  , .declInit u  "base"  (.bin "*" (.var "c") (.litUint 4))
  , .declInit f  "n_c"   (.index (.var "nTarget") (.var "c"))
  , .declInit f  "w0"    (.index (.var "weight") (.var "base"))
  , .declInit f  "w1"    (.index (.var "weight") (.bin "+" (.var "base") (.litUint 1)))
  , .declInit f  "w2"    (.index (.var "weight") (.bin "+" (.var "base") (.litUint 2)))
  , .declInit f  "w3"    (.index (.var "weight") (.bin "+" (.var "base") (.litUint 3)))
  , .declInit f3 "vg0"   (.index (.var "v_grad") (.var "base"))
  , .declInit f3 "vg1"   (.index (.var "v_grad") (.bin "+" (.var "base") (.litUint 1)))
  , .declInit f3 "vg2"   (.index (.var "v_grad") (.bin "+" (.var "base") (.litUint 2)))
  , .declInit f3 "vg3"   (.index (.var "v_grad") (.bin "+" (.var "base") (.litUint 3)))
  , .declInit f  "vh0"   (.index (.var "v_hessScalar") (.var "base"))
  , .declInit f  "vh1"   (.index (.var "v_hessScalar") (.bin "+" (.var "base") (.litUint 1)))
  , .declInit f  "vh2"   (.index (.var "v_hessScalar") (.bin "+" (.var "base") (.litUint 2)))
  , .declInit f  "vh3"   (.index (.var "v_hessScalar") (.bin "+" (.var "base") (.litUint 3)))
  -- A = Σ w_r · vg_r (3-vec)
  , .declInit f  "Ax"
      (.bin "+"
        (.bin "+" (.bin "*" (.var "w0") (.member (.var "vg0") "x"))
                  (.bin "*" (.var "w1") (.member (.var "vg1") "x")))
        (.bin "+" (.bin "*" (.var "w2") (.member (.var "vg2") "x"))
                  (.bin "*" (.var "w3") (.member (.var "vg3") "x"))))
  , .declInit f  "Ay"
      (.bin "+"
        (.bin "+" (.bin "*" (.var "w0") (.member (.var "vg0") "y"))
                  (.bin "*" (.var "w1") (.member (.var "vg1") "y")))
        (.bin "+" (.bin "*" (.var "w2") (.member (.var "vg2") "y"))
                  (.bin "*" (.var "w3") (.member (.var "vg3") "y"))))
  , .declInit f  "Az"
      (.bin "+"
        (.bin "+" (.bin "*" (.var "w0") (.member (.var "vg0") "z"))
                  (.bin "*" (.var "w1") (.member (.var "vg1") "z")))
        (.bin "+" (.bin "*" (.var "w2") (.member (.var "vg2") "z"))
                  (.bin "*" (.var "w3") (.member (.var "vg3") "z"))))
  -- HW = Σ w_r² · vh_r
  , .declInit f  "HW"
      (.bin "+"
        (.bin "+" (.bin "*" (.bin "*" (.var "w0") (.var "w0")) (.var "vh0"))
                  (.bin "*" (.bin "*" (.var "w1") (.var "w1")) (.var "vh1")))
        (.bin "+" (.bin "*" (.bin "*" (.var "w2") (.var "w2")) (.var "vh2"))
                  (.bin "*" (.bin "*" (.var "w3") (.var "w3")) (.var "vh3"))))
  -- Outputs default to zero (degenerate case)
  , .declInit f  "vpx0" (.litFloat 0.0)
  , .declInit f  "vpy0" (.litFloat 0.0)
  , .declInit f  "vpz0" (.litFloat 0.0)
  , .declInit f  "vpx1" (.litFloat 0.0)
  , .declInit f  "vpy1" (.litFloat 0.0)
  , .declInit f  "vpz1" (.litFloat 0.0)
  , .declInit f  "vpx2" (.litFloat 0.0)
  , .declInit f  "vpy2" (.litFloat 0.0)
  , .declInit f  "vpz2" (.litFloat 0.0)
  , .declInit f  "vpx3" (.litFloat 0.0)
  , .declInit f  "vpy3" (.litFloat 0.0)
  , .declInit f  "vpz3" (.litFloat 0.0)
  , .declInit f  "vlx"  (.litFloat 0.0)
  , .declInit f  "vly"  (.litFloat 0.0)
  , .declInit f  "vlz"  (.litFloat 0.0)
  , .declInit f  "vk"   (.litFloat 0.0)
  , .declInit f  "vn"   (.litFloat 0.0)
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
      , .declInit f  "len2"
          (.bin "+"
            (.bin "+" (.bin "*" (.var "sx") (.var "sx"))
                      (.bin "*" (.var "sy") (.var "sy")))
            (.bin "*" (.var "sz") (.var "sz")))
      , .declInit f  "len" (.call "sqrt" [.var "len2"])
      , .declInit f  "scale" (.bin "/" (.var "n_c") (.var "len"))
      , .declInit f  "om"    (.bin "-" (.litFloat 1.0) (.var "scale"))
      , .declInit f  "ex"  (.bin "*" (.var "sx") (.var "om"))
      , .declInit f  "ey"  (.bin "*" (.var "sy") (.var "om"))
      , .declInit f  "ez"  (.bin "*" (.var "sz") (.var "om"))
      , .declInit f  "k"   (.index (.var "stiffness") (.var "c"))
      , .declInit f  "AdotE"
          (.bin "+"
            (.bin "+" (.bin "*" (.var "Ax") (.var "ex"))
                      (.bin "*" (.var "Ay") (.var "ey")))
            (.bin "*" (.var "Az") (.var "ez")))
      , .declInit f  "AdotS"
          (.bin "+"
            (.bin "+" (.bin "*" (.var "Ax") (.var "sx"))
                      (.bin "*" (.var "Ay") (.var "sy")))
            (.bin "*" (.var "Az") (.var "sz")))
      , .declInit f  "len3" (.bin "*" (.var "len") (.var "len2"))
      , .declInit f  "factor"
          (.bin "/" (.bin "*" (.var "k") (.bin "*" (.var "n_c") (.var "AdotS"))) (.var "len3"))
      , .declInit f  "kom" (.bin "*" (.var "k") (.var "om"))
      , .declInit f  "tx"
          (.bin "+" (.bin "*" (.var "kom") (.var "Ax")) (.bin "*" (.var "factor") (.var "sx")))
      , .declInit f  "ty"
          (.bin "+" (.bin "*" (.var "kom") (.var "Ay")) (.bin "*" (.var "factor") (.var "sy")))
      , .declInit f  "tz"
          (.bin "+" (.bin "*" (.var "kom") (.var "Az")) (.bin "*" (.var "factor") (.var "sz")))
      , .assign (.var "vpx0") (.bin "*" (.var "w0") (.var "tx"))
      , .assign (.var "vpy0") (.bin "*" (.var "w0") (.var "ty"))
      , .assign (.var "vpz0") (.bin "*" (.var "w0") (.var "tz"))
      , .assign (.var "vpx1") (.bin "*" (.var "w1") (.var "tx"))
      , .assign (.var "vpy1") (.bin "*" (.var "w1") (.var "ty"))
      , .assign (.var "vpz1") (.bin "*" (.var "w1") (.var "tz"))
      , .assign (.var "vpx2") (.bin "*" (.var "w2") (.var "tx"))
      , .assign (.var "vpy2") (.bin "*" (.var "w2") (.var "ty"))
      , .assign (.var "vpz2") (.bin "*" (.var "w2") (.var "tz"))
      , .assign (.var "vpx3") (.bin "*" (.var "w3") (.var "tx"))
      , .assign (.var "vpy3") (.bin "*" (.var "w3") (.var "ty"))
      , .assign (.var "vpz3") (.bin "*" (.var "w3") (.var "tz"))
      , .assign (.var "vlx") (.var "Ax")
      , .assign (.var "vly") (.var "Ay")
      , .assign (.var "vlz") (.var "Az")
      , .assign (.var "vk")  (.bin "+" (.var "AdotE") (.var "HW"))
      , .assign (.var "vn")
          (.bin "-" (.litFloat 0.0)
            (.bin "/" (.bin "*" (.var "k") (.var "AdotS")) (.var "len")))
      ]
  , .assign (.index (.var "v_p") (.var "base"))
      (.call "float3" [.var "vpx0", .var "vpy0", .var "vpz0"])
  , .assign (.index (.var "v_p") (.bin "+" (.var "base") (.litUint 1)))
      (.call "float3" [.var "vpx1", .var "vpy1", .var "vpz1"])
  , .assign (.index (.var "v_p") (.bin "+" (.var "base") (.litUint 2)))
      (.call "float3" [.var "vpx2", .var "vpy2", .var "vpz2"])
  , .assign (.index (.var "v_p") (.bin "+" (.var "base") (.litUint 3)))
      (.call "float3" [.var "vpx3", .var "vpy3", .var "vpz3"])
  , .assign (.index (.var "v_lambda") (.var "c"))
      (.call "float3" [.var "vlx", .var "vly", .var "vlz"])
  , .assign (.index (.var "v_stiffness") (.var "c")) (.var "vk")
  , .assign (.index (.var "v_nTarget") (.var "c")) (.var "vn")
  ]

def shader : SlangShaderModule :=
  { globals :=
      [ bnd 0  "positions"    (.roBuf f3)
      , bnd 1  "idx"          (.roBuf u)
      , bnd 2  "weight"       (.roBuf f)
      , bnd 3  "nTarget"      (.roBuf f)
      , bnd 4  "stiffness"    (.roBuf f)
      , bnd 5  "lambda"       (.roBuf f3)
      , bnd 6  "v_grad"       (.roBuf f3)
      , bnd 7  "v_hessScalar" (.roBuf f)
      , bnd 8  "v_p"          (.rwBuf f3)
      , bnd 9  "v_nTarget"    (.rwBuf f)
      , bnd 10 "v_stiffness"  (.rwBuf f)
      , bnd 11 "v_lambda"     (.rwBuf f3)
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
StructuredBuffer<float3> v_grad;
[[vk::binding(7, 0)]]
StructuredBuffer<float> v_hessScalar;
[[vk::binding(8, 0)]]
RWStructuredBuffer<float3> v_p;
[[vk::binding(9, 0)]]
RWStructuredBuffer<float> v_nTarget;
[[vk::binding(10, 0)]]
RWStructuredBuffer<float> v_stiffness;
[[vk::binding(11, 0)]]
RWStructuredBuffer<float3> v_lambda;

[shader(\"compute\")] [numthreads(64, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
  uint c = tid.x;
  uint base = (c * 4u);
  float n_c = nTarget[c];
  float w0 = weight[base];
  float w1 = weight[(base + 1u)];
  float w2 = weight[(base + 2u)];
  float w3 = weight[(base + 3u)];
  float3 vg0 = v_grad[base];
  float3 vg1 = v_grad[(base + 1u)];
  float3 vg2 = v_grad[(base + 2u)];
  float3 vg3 = v_grad[(base + 3u)];
  float vh0 = v_hessScalar[base];
  float vh1 = v_hessScalar[(base + 1u)];
  float vh2 = v_hessScalar[(base + 2u)];
  float vh3 = v_hessScalar[(base + 3u)];
  float Ax = (((w0 * vg0.x) + (w1 * vg1.x)) + ((w2 * vg2.x) + (w3 * vg3.x)));
  float Ay = (((w0 * vg0.y) + (w1 * vg1.y)) + ((w2 * vg2.y) + (w3 * vg3.y)));
  float Az = (((w0 * vg0.z) + (w1 * vg1.z)) + ((w2 * vg2.z) + (w3 * vg3.z)));
  float HW = ((((w0 * w0) * vh0) + ((w1 * w1) * vh1)) + (((w2 * w2) * vh2) + ((w3 * w3) * vh3)));
  float vpx0 = 0.000000;
  float vpy0 = 0.000000;
  float vpz0 = 0.000000;
  float vpx1 = 0.000000;
  float vpy1 = 0.000000;
  float vpz1 = 0.000000;
  float vpx2 = 0.000000;
  float vpy2 = 0.000000;
  float vpz2 = 0.000000;
  float vpx3 = 0.000000;
  float vpy3 = 0.000000;
  float vpz3 = 0.000000;
  float vlx = 0.000000;
  float vly = 0.000000;
  float vlz = 0.000000;
  float vk = 0.000000;
  float vn = 0.000000;
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
    float len2 = (((sx * sx) + (sy * sy)) + (sz * sz));
    float len = sqrt(len2);
    float scale = (n_c / len);
    float om = (1.000000 - scale);
    float ex = (sx * om);
    float ey = (sy * om);
    float ez = (sz * om);
    float k = stiffness[c];
    float AdotE = (((Ax * ex) + (Ay * ey)) + (Az * ez));
    float AdotS = (((Ax * sx) + (Ay * sy)) + (Az * sz));
    float len3 = (len * len2);
    float factor = ((k * (n_c * AdotS)) / len3);
    float kom = (k * om);
    float tx = ((kom * Ax) + (factor * sx));
    float ty = ((kom * Ay) + (factor * sy));
    float tz = ((kom * Az) + (factor * sz));
    vpx0 = (w0 * tx);
    vpy0 = (w0 * ty);
    vpz0 = (w0 * tz);
    vpx1 = (w1 * tx);
    vpy1 = (w1 * ty);
    vpz1 = (w1 * tz);
    vpx2 = (w2 * tx);
    vpy2 = (w2 * ty);
    vpz2 = (w2 * tz);
    vpx3 = (w3 * tx);
    vpy3 = (w3 * ty);
    vpz3 = (w3 * tz);
    vlx = Ax;
    vly = Ay;
    vlz = Az;
    vk = (AdotE + HW);
    vn = (0.000000 - ((k * AdotS) / len));
  }
  v_p[base] = float3(vpx0, vpy0, vpz0);
  v_p[(base + 1u)] = float3(vpx1, vpy1, vpz1);
  v_p[(base + 2u)] = float3(vpx2, vpy2, vpz2);
  v_p[(base + 3u)] = float3(vpx3, vpy3, vpz3);
  v_lambda[c] = float3(vlx, vly, vlz);
  v_stiffness[c] = vk;
  v_nTarget[c] = vn;
}"

example : LeanSlang.emit shader = expected := by native_decide
example : shader.entryPointName = "main" := by native_decide

end Cloth.SlangCodegen.TriangleBendingForceAlBackward
