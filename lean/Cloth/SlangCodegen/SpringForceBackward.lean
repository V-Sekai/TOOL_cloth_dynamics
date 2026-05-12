import LeanSlang

/-!
# `Cloth.SlangCodegen.SpringForceBackward` — adjoint of spring_force (PR-G / CHI-13)

Reverse-mode VJP for the spring force kernel. Forward per spring `i`
with endpoints `(a, b)`, rest length `L`, stiffness `k`:

```
d      = p_a − p_b                    -- 3-vec
r      = |d|
cval   = r − L                         -- signed constraint
gScale = k · cval / r
gradA  = gScale · d                    -- 3-vec
h      = k / r²
hess   = h · d ⊗ d                     -- sym 3x3, packed 6-vec
```

Per-constraint thread (one per spring). Cotangents flowing in from
`vbd_gather_spring_backward`:

* `v_gradA[c]`     — 3-vec cotangent on gradA
* `v_springHess[c]` — scalar trace cotangent (gather backward sums
  only the diagonal entries of the per-vertex Hessian cotangent;
  this matches the existing convention in
  `[[Cloth.SlangCodegen.VbdGatherSpringBackward]]`)

The trace of the spring's Hessian is `h · r² = k`, so the trace
cotangent only contributes to `v_k`. Position / rest-length
cotangents come purely from the `v_gradA` path.

Chain rule with `dot_vG_d = v_gradA · d`:

```
v_p_d = gScale · v_gradA + (k · L · dot_vG_d / r³) · d   -- 3-vec
v_restLen   = −(k / r) · dot_vG_d
v_stiffness = ((r − L) / r) · dot_vG_d + v_springHess
```

The position cotangent is emitted PER-SPRING as `v_p_d[c]` (the
gradient w.r.t. `d = p_a − p_b`). The host orchestration accumulates
it into per-vertex `v_positions` via a re-use of `vbd_gather_spring`
(with `springGradA := v_p_d`, `springHess := zeros`), which applies
the role sign flip (+1 for endpoint a, −1 for endpoint b) using the
existing spring-vertex CSR.

Bindings (set 0):

  0  StructuredBuffer<float3>   positions     length = N_verts
  1  StructuredBuffer<uint>     p1Idx         length = N_springs
  2  StructuredBuffer<uint>     p2Idx         length = N_springs
  3  StructuredBuffer<float>    restLen       length = N_springs
  4  StructuredBuffer<float>    stiffness     length = N_springs
  5  StructuredBuffer<float3>   v_gradA       length = N_springs  (cotangent)
  6  StructuredBuffer<float>    v_springHess  length = N_springs  (scalar cotangent)
  7  RWStructuredBuffer<float3> v_p_d         length = N_springs  (∂L/∂d)
  8  RWStructuredBuffer<float>  v_restLen     length = N_springs  (∂L/∂L)
  9  RWStructuredBuffer<float>  v_stiffness   length = N_springs  (∂L/∂k)
-/

namespace Cloth.SlangCodegen.SpringForceBackward

open LeanSlang

private def f3 : SlangType := .vec .float 3
private def u  : SlangType := .scalar .uint
private def f  : SlangType := .scalar .float

private def bnd (n : Nat) (name : String) (t : SlangType) : SlangBinding :=
  { name := name, type := t, semantic := Semantic.none
  , binding := some n, space := some 0 }

private def body : List SlangStmt :=
  [ .declInit u  "i"      (.member (.var "tid") "x")
  , .declInit u  "a"      (.index (.var "p1Idx") (.var "i"))
  , .declInit u  "b"      (.index (.var "p2Idx") (.var "i"))
  , .declInit f3 "p1"     (.index (.var "positions") (.var "a"))
  , .declInit f3 "p2"     (.index (.var "positions") (.var "b"))
  , .declInit f  "dx"     (.bin "-" (.member (.var "p1") "x") (.member (.var "p2") "x"))
  , .declInit f  "dy"     (.bin "-" (.member (.var "p1") "y") (.member (.var "p2") "y"))
  , .declInit f  "dz"     (.bin "-" (.member (.var "p1") "z") (.member (.var "p2") "z"))
  , .declInit f  "len2"
      (.bin "+"
        (.bin "+" (.bin "*" (.var "dx") (.var "dx"))
                  (.bin "*" (.var "dy") (.var "dy")))
        (.bin "*" (.var "dz") (.var "dz")))
  , .declInit f  "len"    (.call "sqrt" [.var "len2"])
  , .declInit f  "k"      (.index (.var "stiffness") (.var "i"))
  , .declInit f  "L"      (.index (.var "restLen")   (.var "i"))
  , .declInit f3 "vG"     (.index (.var "v_gradA")   (.var "i"))
  , .declInit f  "vH"     (.index (.var "v_springHess") (.var "i"))
  , .declInit f  "gScale" (.bin "/" (.bin "*" (.var "k") (.bin "-" (.var "len") (.var "L")))
                                    (.var "len"))
  , .declInit f  "dotVGd"
      (.bin "+"
        (.bin "+" (.bin "*" (.member (.var "vG") "x") (.var "dx"))
                  (.bin "*" (.member (.var "vG") "y") (.var "dy")))
        (.bin "*" (.member (.var "vG") "z") (.var "dz")))
  , .declInit f  "len3"   (.bin "*" (.var "len") (.var "len2"))
  , .declInit f  "kLd"    (.bin "/" (.bin "*" (.var "k") (.bin "*" (.var "L") (.var "dotVGd")))
                                    (.var "len3"))
  , .assign (.index (.var "v_p_d") (.var "i"))
      (.call "float3"
        [ .bin "+" (.bin "*" (.var "gScale") (.member (.var "vG") "x"))
                   (.bin "*" (.var "kLd") (.var "dx"))
        , .bin "+" (.bin "*" (.var "gScale") (.member (.var "vG") "y"))
                   (.bin "*" (.var "kLd") (.var "dy"))
        , .bin "+" (.bin "*" (.var "gScale") (.member (.var "vG") "z"))
                   (.bin "*" (.var "kLd") (.var "dz")) ])
  , .assign (.index (.var "v_restLen") (.var "i"))
      (.bin "-" (.litFloat 0.0)
        (.bin "/" (.bin "*" (.var "k") (.var "dotVGd")) (.var "len")))
  , .assign (.index (.var "v_stiffness") (.var "i"))
      (.bin "+"
        (.bin "/" (.bin "*" (.bin "-" (.var "len") (.var "L")) (.var "dotVGd")) (.var "len"))
        (.var "vH"))
  ]

def shader : SlangShaderModule :=
  { globals :=
      [ bnd 0 "positions"    (.roBuf f3)
      , bnd 1 "p1Idx"        (.roBuf u)
      , bnd 2 "p2Idx"        (.roBuf u)
      , bnd 3 "restLen"      (.roBuf f)
      , bnd 4 "stiffness"    (.roBuf f)
      , bnd 5 "v_gradA"      (.roBuf f3)
      , bnd 6 "v_springHess" (.roBuf f)
      , bnd 7 "v_p_d"        (.rwBuf f3)
      , bnd 8 "v_restLen"    (.rwBuf f)
      , bnd 9 "v_stiffness"  (.rwBuf f)
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
StructuredBuffer<uint> p1Idx;
[[vk::binding(2, 0)]]
StructuredBuffer<uint> p2Idx;
[[vk::binding(3, 0)]]
StructuredBuffer<float> restLen;
[[vk::binding(4, 0)]]
StructuredBuffer<float> stiffness;
[[vk::binding(5, 0)]]
StructuredBuffer<float3> v_gradA;
[[vk::binding(6, 0)]]
StructuredBuffer<float> v_springHess;
[[vk::binding(7, 0)]]
RWStructuredBuffer<float3> v_p_d;
[[vk::binding(8, 0)]]
RWStructuredBuffer<float> v_restLen;
[[vk::binding(9, 0)]]
RWStructuredBuffer<float> v_stiffness;

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
  float k = stiffness[i];
  float L = restLen[i];
  float3 vG = v_gradA[i];
  float vH = v_springHess[i];
  float gScale = ((k * (len - L)) / len);
  float dotVGd = (((vG.x * dx) + (vG.y * dy)) + (vG.z * dz));
  float len3 = (len * len2);
  float kLd = ((k * (L * dotVGd)) / len3);
  v_p_d[i] = float3(((gScale * vG.x) + (kLd * dx)), ((gScale * vG.y) + (kLd * dy)), ((gScale * vG.z) + (kLd * dz)));
  v_restLen[i] = (0.000000 - ((k * dotVGd) / len));
  v_stiffness[i] = ((((len - L) * dotVGd) / len) + vH);
}"

example : LeanSlang.emit shader = expected := by native_decide
example : shader.entryPointName = "main" := by native_decide

end Cloth.SlangCodegen.SpringForceBackward
