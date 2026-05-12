import LeanSlang

/-!
# `Cloth.SlangCodegen.TriangleMembraneDualUpdate` — AVBD AL ramp for triangle membrane

PR-F's empirical finding in #77: AL on attachments alone doesn't fix
the dress's oscillation (\`conv_max ≈ 0.71\` unchanged). The
oscillating vertex is on triangle-mesh dynamics, not waistband pins.
This kernel extends the AL dual ramp to the triangle membrane
constraint — same idea as `AttachmentDualUpdate` (#74), but the
constraint here is the 2-column ARAP residual `F − R` instead of a
single point distance.

Per triangle `c` with vertex indices `(i0, i1, i2)`, per-triangle
inverse rest-material matrix `inv_deltaUV` (row-major 2×2 from
DiffCloth's `Triangle::inv_deltaUV`), and AL penalty `γ_c`:

```
P        = [p1 − p0 | p2 − p0]              -- 3x2 raw current edges
F        = P · inv_deltaUV                  -- 3x2 deformation gradient
R        = closest-rotation(F)              -- same Gram-Schmidt + 2D
                                               polar as membrane force
e0       = F.col(0) − R.col(0)              -- residual column 0
e1       = F.col(1) − R.col(1)              -- residual column 1

λ_c.col(0) ← λ_c.col(0) + γ_c · e0          -- dual ascent on first column
λ_c.col(1) ← λ_c.col(1) + γ_c · e1          -- dual ascent on second column
```

`λ` is stored as two `float3` buffers — one per column of the 3×2
residual matrix — so the future `triangle_membrane_force_al` kernel
can do `gradV[3c+r] += sign_r · λ_c.col(*)` cheaply (with per-vertex
sign `{ -(λ0+λ1), +λ0, +λ1 }` for roles 0, 1, 2).

Bindings (set 0):

  0  StructuredBuffer<float3>   positions   length = N_verts
  1  StructuredBuffer<uint>     idx         length = 3 · N_tri
  2  StructuredBuffer<float>    gamma       length = N_tri
  3  RWStructuredBuffer<float3> lambda0     length = N_tri   (column 0 of λ)
  4  RWStructuredBuffer<float3> lambda1     length = N_tri   (column 1 of λ)
  5  StructuredBuffer<float>    inv_deltaUV length = 4 · N_tri
                                            (row-major 2x2 per tri:
                                             [m00, m01, m10, m11])

Mirrors the closest-rotation math from TriangleMembraneForce. With
`inv_deltaUV = I` reduces bit-exactly to the pre-PR raw-edge form.
No output forces — this kernel only updates λ. Caller dispatches
`triangle_membrane_force_al` separately to consume λ in the force
gradient.
-/

namespace Cloth.SlangCodegen.TriangleMembraneDualUpdate

open LeanSlang

private def f3 : SlangType := .vec .float 3
private def u  : SlangType := .scalar .uint
private def f  : SlangType := .scalar .float

private def bnd (n : Nat) (name : String) (t : SlangType) : SlangBinding :=
  { name := name, type := t, semantic := Semantic.none
  , binding := some n, space := some 0 }

private def pmv (s : String) (field : String) : SlangExpr :=
  .member (.var s) field

private def sum3 (a b c : SlangExpr) : SlangExpr :=
  .bin "+" (.bin "+" a b) c

private def len3 (x y z : SlangExpr) : SlangExpr :=
  .call "sqrt" [ sum3 (.bin "*" x x) (.bin "*" y y) (.bin "*" z z) ]

private def body : List SlangStmt :=
  [ .declInit u  "c"    (.member (.var "tid") "x")
  , .declInit u  "base" (.bin "*" (.var "c") (.litUint 3))
  , .declInit u  "i0"   (.index (.var "idx") (.var "base"))
  , .declInit u  "i1"   (.index (.var "idx") (.bin "+" (.var "base") (.litUint 1)))
  , .declInit u  "i2"   (.index (.var "idx") (.bin "+" (.var "base") (.litUint 2)))
  , .declInit f3 "p0"   (.index (.var "positions") (.var "i0"))
  , .declInit f3 "p1"   (.index (.var "positions") (.var "i1"))
  , .declInit f3 "p2"   (.index (.var "positions") (.var "i2"))
  , .declInit u "iub"   (.bin "*" (.var "c") (.litUint 4))
  , .declInit f "m00"   (.index (.var "inv_deltaUV") (.var "iub"))
  , .declInit f "m01"   (.index (.var "inv_deltaUV") (.bin "+" (.var "iub") (.litUint 1)))
  , .declInit f "m10"   (.index (.var "inv_deltaUV") (.bin "+" (.var "iub") (.litUint 2)))
  , .declInit f "m11"   (.index (.var "inv_deltaUV") (.bin "+" (.var "iub") (.litUint 3)))
  , .declInit f "ex0" (.bin "-" (pmv "p1" "x") (pmv "p0" "x"))
  , .declInit f "ey0" (.bin "-" (pmv "p1" "y") (pmv "p0" "y"))
  , .declInit f "ez0" (.bin "-" (pmv "p1" "z") (pmv "p0" "z"))
  , .declInit f "ex1" (.bin "-" (pmv "p2" "x") (pmv "p0" "x"))
  , .declInit f "ey1" (.bin "-" (pmv "p2" "y") (pmv "p0" "y"))
  , .declInit f "ez1" (.bin "-" (pmv "p2" "z") (pmv "p0" "z"))
  , .declInit f "f0x" (.bin "+" (.bin "*" (.var "ex0") (.var "m00"))
                                (.bin "*" (.var "ex1") (.var "m10")))
  , .declInit f "f0y" (.bin "+" (.bin "*" (.var "ey0") (.var "m00"))
                                (.bin "*" (.var "ey1") (.var "m10")))
  , .declInit f "f0z" (.bin "+" (.bin "*" (.var "ez0") (.var "m00"))
                                (.bin "*" (.var "ez1") (.var "m10")))
  , .declInit f "f1x" (.bin "+" (.bin "*" (.var "ex0") (.var "m01"))
                                (.bin "*" (.var "ex1") (.var "m11")))
  , .declInit f "f1y" (.bin "+" (.bin "*" (.var "ey0") (.var "m01"))
                                (.bin "*" (.var "ey1") (.var "m11")))
  , .declInit f "f1z" (.bin "+" (.bin "*" (.var "ez0") (.var "m01"))
                                (.bin "*" (.var "ez1") (.var "m11")))
  , .declInit f "a"   (len3 (.var "f0x") (.var "f0y") (.var "f0z"))
  , .declInit f "e1x" (.bin "/" (.var "f0x") (.var "a"))
  , .declInit f "e1y" (.bin "/" (.var "f0y") (.var "a"))
  , .declInit f "e1z" (.bin "/" (.var "f0z") (.var "a"))
  , .declInit f "b"
      (sum3 (.bin "*" (.var "e1x") (.var "f1x"))
            (.bin "*" (.var "e1y") (.var "f1y"))
            (.bin "*" (.var "e1z") (.var "f1z")))
  , .declInit f "tx" (.bin "-" (.var "f1x") (.bin "*" (.var "b") (.var "e1x")))
  , .declInit f "ty" (.bin "-" (.var "f1y") (.bin "*" (.var "b") (.var "e1y")))
  , .declInit f "tz" (.bin "-" (.var "f1z") (.bin "*" (.var "b") (.var "e1z")))
  , .declInit f "d"   (len3 (.var "tx") (.var "ty") (.var "tz"))
  , .declInit f "e2x" (.bin "/" (.var "tx") (.var "d"))
  , .declInit f "e2y" (.bin "/" (.var "ty") (.var "d"))
  , .declInit f "e2z" (.bin "/" (.var "tz") (.var "d"))
  , .declInit f "xv" (.bin "+" (.var "a") (.var "d"))
  , .declInit f "yv" (.bin "-" (.litFloat 0.0) (.var "b"))
  , .declInit f "rn"
      (.call "sqrt" [.bin "+" (.bin "*" (.var "xv") (.var "xv"))
                              (.bin "*" (.var "yv") (.var "yv"))])
  , .declInit f "r00" (.bin "/" (.var "xv") (.var "rn"))
  , .declInit f "r01" (.bin "/" (.var "b")  (.var "rn"))
  , .declInit f "r10" (.bin "/" (.var "yv") (.var "rn"))
  , .declInit f "r11" (.bin "/" (.var "xv") (.var "rn"))
  , .declInit f "n0x" (.bin "+" (.bin "*" (.var "e1x") (.var "r00"))
                                (.bin "*" (.var "e2x") (.var "r10")))
  , .declInit f "n0y" (.bin "+" (.bin "*" (.var "e1y") (.var "r00"))
                                (.bin "*" (.var "e2y") (.var "r10")))
  , .declInit f "n0z" (.bin "+" (.bin "*" (.var "e1z") (.var "r00"))
                                (.bin "*" (.var "e2z") (.var "r10")))
  , .declInit f "n1x" (.bin "+" (.bin "*" (.var "e1x") (.var "r01"))
                                (.bin "*" (.var "e2x") (.var "r11")))
  , .declInit f "n1y" (.bin "+" (.bin "*" (.var "e1y") (.var "r01"))
                                (.bin "*" (.var "e2y") (.var "r11")))
  , .declInit f "n1z" (.bin "+" (.bin "*" (.var "e1z") (.var "r01"))
                                (.bin "*" (.var "e2z") (.var "r11")))
  , .declInit f "e0x" (.bin "-" (.var "f0x") (.var "n0x"))
  , .declInit f "e0y" (.bin "-" (.var "f0y") (.var "n0y"))
  , .declInit f "e0z" (.bin "-" (.var "f0z") (.var "n0z"))
  , .declInit f "e1rx" (.bin "-" (.var "f1x") (.var "n1x"))
  , .declInit f "e1ry" (.bin "-" (.var "f1y") (.var "n1y"))
  , .declInit f "e1rz" (.bin "-" (.var "f1z") (.var "n1z"))
  , .declInit f  "g"   (.index (.var "gamma") (.var "c"))
  , .declInit f3 "l0"  (.index (.var "lambda0") (.var "c"))
  , .declInit f3 "l1"  (.index (.var "lambda1") (.var "c"))
  , .assign (.index (.var "lambda0") (.var "c"))
      (.call "float3"
        [ .bin "+" (pmv "l0" "x") (.bin "*" (.var "g") (.var "e0x"))
        , .bin "+" (pmv "l0" "y") (.bin "*" (.var "g") (.var "e0y"))
        , .bin "+" (pmv "l0" "z") (.bin "*" (.var "g") (.var "e0z")) ])
  , .assign (.index (.var "lambda1") (.var "c"))
      (.call "float3"
        [ .bin "+" (pmv "l1" "x") (.bin "*" (.var "g") (.var "e1rx"))
        , .bin "+" (pmv "l1" "y") (.bin "*" (.var "g") (.var "e1ry"))
        , .bin "+" (pmv "l1" "z") (.bin "*" (.var "g") (.var "e1rz")) ])
  ]

def shader : SlangShaderModule :=
  { globals :=
      [ bnd 0 "positions"   (.roBuf f3)
      , bnd 1 "idx"         (.roBuf u)
      , bnd 2 "gamma"       (.roBuf f)
      , bnd 3 "lambda0"     (.rwBuf f3)
      , bnd 4 "lambda1"     (.rwBuf f3)
      , bnd 5 "inv_deltaUV" (.roBuf f)
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
StructuredBuffer<float> gamma;
[[vk::binding(3, 0)]]
RWStructuredBuffer<float3> lambda0;
[[vk::binding(4, 0)]]
RWStructuredBuffer<float3> lambda1;
[[vk::binding(5, 0)]]
StructuredBuffer<float> inv_deltaUV;

[shader(\"compute\")] [numthreads(64, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
  uint c = tid.x;
  uint base = (c * 3u);
  uint i0 = idx[base];
  uint i1 = idx[(base + 1u)];
  uint i2 = idx[(base + 2u)];
  float3 p0 = positions[i0];
  float3 p1 = positions[i1];
  float3 p2 = positions[i2];
  uint iub = (c * 4u);
  float m00 = inv_deltaUV[iub];
  float m01 = inv_deltaUV[(iub + 1u)];
  float m10 = inv_deltaUV[(iub + 2u)];
  float m11 = inv_deltaUV[(iub + 3u)];
  float ex0 = (p1.x - p0.x);
  float ey0 = (p1.y - p0.y);
  float ez0 = (p1.z - p0.z);
  float ex1 = (p2.x - p0.x);
  float ey1 = (p2.y - p0.y);
  float ez1 = (p2.z - p0.z);
  float f0x = ((ex0 * m00) + (ex1 * m10));
  float f0y = ((ey0 * m00) + (ey1 * m10));
  float f0z = ((ez0 * m00) + (ez1 * m10));
  float f1x = ((ex0 * m01) + (ex1 * m11));
  float f1y = ((ey0 * m01) + (ey1 * m11));
  float f1z = ((ez0 * m01) + (ez1 * m11));
  float a = sqrt((((f0x * f0x) + (f0y * f0y)) + (f0z * f0z)));
  float e1x = (f0x / a);
  float e1y = (f0y / a);
  float e1z = (f0z / a);
  float b = (((e1x * f1x) + (e1y * f1y)) + (e1z * f1z));
  float tx = (f1x - (b * e1x));
  float ty = (f1y - (b * e1y));
  float tz = (f1z - (b * e1z));
  float d = sqrt((((tx * tx) + (ty * ty)) + (tz * tz)));
  float e2x = (tx / d);
  float e2y = (ty / d);
  float e2z = (tz / d);
  float xv = (a + d);
  float yv = (0.000000 - b);
  float rn = sqrt(((xv * xv) + (yv * yv)));
  float r00 = (xv / rn);
  float r01 = (b / rn);
  float r10 = (yv / rn);
  float r11 = (xv / rn);
  float n0x = ((e1x * r00) + (e2x * r10));
  float n0y = ((e1y * r00) + (e2y * r10));
  float n0z = ((e1z * r00) + (e2z * r10));
  float n1x = ((e1x * r01) + (e2x * r11));
  float n1y = ((e1y * r01) + (e2y * r11));
  float n1z = ((e1z * r01) + (e2z * r11));
  float e0x = (f0x - n0x);
  float e0y = (f0y - n0y);
  float e0z = (f0z - n0z);
  float e1rx = (f1x - n1x);
  float e1ry = (f1y - n1y);
  float e1rz = (f1z - n1z);
  float g = gamma[c];
  float3 l0 = lambda0[c];
  float3 l1 = lambda1[c];
  lambda0[c] = float3((l0.x + (g * e0x)), (l0.y + (g * e0y)), (l0.z + (g * e0z)));
  lambda1[c] = float3((l1.x + (g * e1rx)), (l1.y + (g * e1ry)), (l1.z + (g * e1rz)));
}"

example : LeanSlang.emit shader = expected := by native_decide
example : shader.entryPointName = "main" := by native_decide

end Cloth.SlangCodegen.TriangleMembraneDualUpdate
