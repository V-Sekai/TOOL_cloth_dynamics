import LeanSlang

/-!
# `Cloth.SlangCodegen.SpringForce` — AVBD spring force + GN Hessian

First kernel for the AVBD port (see `todo.md`). For each spring `i`
with endpoints `(a, b)`, rest length `L`, and stiffness `k`, computes:

```
d   = p_a − p_b
len = |d|
c   = len − L                  -- constraint value (signed)
n   = d / len                  -- unit edge direction

gradA[i] = k · c · n           -- ∇_a E for the spring's energy
                                  (vertex b gets −gradA at gather time)

hess[6i..6i+5] = packed sym 3x3  Hess block = k · (d ⊗ d) / len²
                                  (Gauss-Newton approximation; the
                                  diagonal block ∇²_a E = ∇²_b E shares
                                  this matrix, no sign flip needed.)
```

This is the per-constraint primitive AVBD's vertex-block-update kernel
will gather over a vertex's incident springs. One thread per spring,
output indexed by constraint id — same dispatch shape as the
PD-local-step `SpringProject` kernel. The vertex gather (and the role
sign flip on the gradient) is a later PR.

Energy reference:

```
E_i = (1/2) · k · (|p_a − p_b| − L)²
```

Gradient on `p_a`:

```
∂E/∂p_a = k · (len − L) · (d / len) = k · c · n
```

Newton's 3rd: ∂E/∂p_b = −∂E/∂p_a. Vertex-block update reads `gradA[i]`
and negates it if the gathering vertex's role for spring `i` is `b`.

Gauss-Newton Hessian (drops the `c · ∂n/∂p_a` term, which is the
geometric stiffness):

```
∇²_a E ≈ k · (n n^T) = k · (d d^T) / len²
```

The same matrix sits on both diagonal blocks (`∇²_a E = ∇²_b E`).
The off-diagonal block is `−k · n n^T`, but the per-vertex update only
needs the diagonal — that's why we write the GN block once and let the
gather use it for whichever endpoint is being updated.

Bindings (set 0):

  0  StructuredBuffer<float3>   positions   length = N_verts
  1  StructuredBuffer<uint>     p1Idx       length = N_springs
  2  StructuredBuffer<uint>     p2Idx       length = N_springs
  3  StructuredBuffer<float>    restLen     length = N_springs
  4  StructuredBuffer<float>    stiffness   length = N_springs
  5  RWStructuredBuffer<float3> gradA       length = N_springs
  6  RWStructuredBuffer<float>  hess        length = 6 · N_springs

Hess packing order: `[xx, xy, xz, yy, yz, zz]` (upper-triangle row-major
of a symmetric 3x3). Per spring: 6 floats starting at offset `6·i`.
-/

namespace Cloth.SlangCodegen.SpringForce

open LeanSlang

private def f3 : SlangType := .vec .float 3
private def u  : SlangType := .scalar .uint
private def f  : SlangType := .scalar .float

private def bnd (n : Nat) (name : String) (t : SlangType) : SlangBinding :=
  { name := name, type := t, semantic := Semantic.none
  , binding := some n, space := some 0 }

private def body : List SlangStmt :=
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
  , .declInit f  "k"   (.index (.var "stiffness") (.var "i"))
  , .declInit f  "L"   (.index (.var "restLen")   (.var "i"))
  , .declInit f  "c"   (.bin "-" (.var "len") (.var "L"))
  , .declInit f  "gScale"
      (.bin "/" (.bin "*" (.var "k") (.var "c")) (.var "len"))
  , .assign (.index (.var "gradA") (.var "i"))
      (.call "float3"
        [ .bin "*" (.var "gScale") (.var "dx")
        , .bin "*" (.var "gScale") (.var "dy")
        , .bin "*" (.var "gScale") (.var "dz") ])
  , .declInit f  "h" (.bin "/" (.var "k") (.var "len2"))
  , .declInit u  "h0" (.bin "*" (.litUint 6) (.var "i"))
  , .assign (.index (.var "hess") (.var "h0"))
      (.bin "*" (.var "h") (.bin "*" (.var "dx") (.var "dx")))
  , .assign (.index (.var "hess") (.bin "+" (.var "h0") (.litUint 1)))
      (.bin "*" (.var "h") (.bin "*" (.var "dx") (.var "dy")))
  , .assign (.index (.var "hess") (.bin "+" (.var "h0") (.litUint 2)))
      (.bin "*" (.var "h") (.bin "*" (.var "dx") (.var "dz")))
  , .assign (.index (.var "hess") (.bin "+" (.var "h0") (.litUint 3)))
      (.bin "*" (.var "h") (.bin "*" (.var "dy") (.var "dy")))
  , .assign (.index (.var "hess") (.bin "+" (.var "h0") (.litUint 4)))
      (.bin "*" (.var "h") (.bin "*" (.var "dy") (.var "dz")))
  , .assign (.index (.var "hess") (.bin "+" (.var "h0") (.litUint 5)))
      (.bin "*" (.var "h") (.bin "*" (.var "dz") (.var "dz")))
  ]

def shader : SlangShaderModule :=
  { globals :=
      [ bnd 0 "positions"  (.roBuf f3)
      , bnd 1 "p1Idx"      (.roBuf u)
      , bnd 2 "p2Idx"      (.roBuf u)
      , bnd 3 "restLen"    (.roBuf f)
      , bnd 4 "stiffness"  (.roBuf f)
      , bnd 5 "gradA"      (.rwBuf f3)
      , bnd 6 "hess"       (.rwBuf f)
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
RWStructuredBuffer<float3> gradA;
[[vk::binding(6, 0)]]
RWStructuredBuffer<float> hess;

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
  float c = (len - L);
  float gScale = ((k * c) / len);
  gradA[i] = float3((gScale * dx), (gScale * dy), (gScale * dz));
  float h = (k / len2);
  uint h0 = (6u * i);
  hess[h0] = (h * (dx * dx));
  hess[(h0 + 1u)] = (h * (dx * dy));
  hess[(h0 + 2u)] = (h * (dx * dz));
  hess[(h0 + 3u)] = (h * (dy * dy));
  hess[(h0 + 4u)] = (h * (dy * dz));
  hess[(h0 + 5u)] = (h * (dz * dz));
}"

example : LeanSlang.emit shader = expected := by native_decide
example : shader.entryPointName = "main" := by native_decide

end Cloth.SlangCodegen.SpringForce
