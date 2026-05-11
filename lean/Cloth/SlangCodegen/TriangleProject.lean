import LeanSlang

/-!
# `Cloth.SlangCodegen.TriangleProject` — DiffCloth PD triangle in-plane stretch

Fourth and final constraint-projection kernel. Implements the PD local
step for in-plane triangle stretch, as defined in
`src/code/simulation/Triangle.cpp:projectToManifold + project`:

```cpp
Mat3x2d projectToManifold(x_vec) {
  Mat3x2d F = getDeformationGradient(x_vec);   // 3x2; raw edges when inv_deltaUV = I
  Mat3x2d p = F;
  newdeltaUV.col(0) = p.col(0).normalized();
  newdeltaUV.col(1) = (p.col(1) - p.col(1).dot(newdeltaUV.col(0)) * newdeltaUV.col(0))
                          .normalized();
  Mat2x2d F_2D     = newdeltaUV.transpose() * F;       // upper triangular [a b; 0 d]
  // SVD then U·V^T = closest 2D rotation
  Mat2x2d F_2D_project = svd.matrixU() * svd.matrixV().transpose();
  return newdeltaUV * F_2D_project;
}

VecXd project(x_vec) {
  Mat3x2d newF = projectToManifold(x_vec);
  return constrainWeightSqrt * concat(newF.col(0), newF.col(1));
}
```

**Restriction.** This kernel assumes `inv_deltaUV = identity`, i.e. the
no-UV-mapping path that DiffCloth currently uses (note the `// TODO:
Restore uv mapping.` in `Triangle.cpp:projectToManifold`). With
`inv_deltaUV = I`, `F` equals the raw current edges:

```
F.col(0) = positions[i1] − positions[i0]
F.col(1) = positions[i2] − positions[i0]
```

Adding a per-triangle `inv_deltaUV` buffer + premultiplying `F` is a
trivial follow-up when needed.

**Closed-form 2D polar.** The Gram-Schmidt makes `F_2D` upper
triangular `[a b; 0 d]`. The closest-rotation of any 2x2 matrix
`M = [m00 m01; m10 m11]` (det(M) > 0) is

```
x    = m00 + m11
y    = m10 − m01
norm = sqrt(x² + y²)
R    = (1/norm) · [x  −y; y  x]
```

Specialised to upper-triangular `F_2D`: `x = a + d`, `y = −b`. No SVD
needed.

Per triangle `c`:

```
newF      = newdeltaUV · R
projected[2c + 0] = sqrtWeight[c] · newF.col(0)
projected[2c + 1] = sqrtWeight[c] · newF.col(1)
```

One thread per triangle. Host pads with `sqrtWeight = 0` so unused
slots emit `(0,0,0), (0,0,0)`.

Bindings (set 0):

  0  StructuredBuffer<float3>   positions   length = N_verts
  1  RWStructuredBuffer<float3> projected   length = 2 · N_tri (rounded up)
  2  StructuredBuffer<uint>     idx         length = 3 · N_tri
  3  StructuredBuffer<float>    sqrtWeight  length = N_tri
-/

namespace Cloth.SlangCodegen.TriangleProject

open LeanSlang

/-- PD per-triangle in-plane stretch projection. -/
def shader : SlangShaderModule :=
  let f3 : SlangType := .vec .float 3
  let u  : SlangType := .scalar .uint
  let f  : SlangType := .scalar .float
  let bnd (n : Nat) (name : String) (t : SlangType) : SlangBinding :=
    { name := name, type := t, semantic := Semantic.none
    , binding := some n, space := some 0 }
  -- Helpers to keep the body legible.
  let pmv (s : String) (field : String) : SlangExpr :=
    .member (.var s) field
  let sum3 (a b c : SlangExpr) : SlangExpr :=
    .bin "+" (.bin "+" a b) c
  let len3 (x y z : SlangExpr) : SlangExpr :=
    .call "sqrt" [ sum3 (.bin "*" x x) (.bin "*" y y) (.bin "*" z z) ]
  let body : List SlangStmt :=
    [ .declInit u  "c"    (.member (.var "tid") "x")
    , .declInit u  "base" (.bin "*" (.var "c") (.litUint 3))
    , .declInit u  "i0"   (.index (.var "idx") (.var "base"))
    , .declInit u  "i1"   (.index (.var "idx") (.bin "+" (.var "base") (.litUint 1)))
    , .declInit u  "i2"   (.index (.var "idx") (.bin "+" (.var "base") (.litUint 2)))
    , .declInit f3 "p0"   (.index (.var "positions") (.var "i0"))
    , .declInit f3 "p1"   (.index (.var "positions") (.var "i1"))
    , .declInit f3 "p2"   (.index (.var "positions") (.var "i2"))
    -- raw edges: f0 = p1 - p0, f1 = p2 - p0
    , .declInit f "f0x" (.bin "-" (pmv "p1" "x") (pmv "p0" "x"))
    , .declInit f "f0y" (.bin "-" (pmv "p1" "y") (pmv "p0" "y"))
    , .declInit f "f0z" (.bin "-" (pmv "p1" "z") (pmv "p0" "z"))
    , .declInit f "f1x" (.bin "-" (pmv "p2" "x") (pmv "p0" "x"))
    , .declInit f "f1y" (.bin "-" (pmv "p2" "y") (pmv "p0" "y"))
    , .declInit f "f1z" (.bin "-" (pmv "p2" "z") (pmv "p0" "z"))
    -- a = |f0|, e1 = f0 / a
    , .declInit f "a"   (len3 (.var "f0x") (.var "f0y") (.var "f0z"))
    , .declInit f "e1x" (.bin "/" (.var "f0x") (.var "a"))
    , .declInit f "e1y" (.bin "/" (.var "f0y") (.var "a"))
    , .declInit f "e1z" (.bin "/" (.var "f0z") (.var "a"))
    -- b = dot(e1, f1); tmp = f1 - b * e1
    , .declInit f "b"
        (sum3 (.bin "*" (.var "e1x") (.var "f1x"))
              (.bin "*" (.var "e1y") (.var "f1y"))
              (.bin "*" (.var "e1z") (.var "f1z")))
    , .declInit f "tx" (.bin "-" (.var "f1x") (.bin "*" (.var "b") (.var "e1x")))
    , .declInit f "ty" (.bin "-" (.var "f1y") (.bin "*" (.var "b") (.var "e1y")))
    , .declInit f "tz" (.bin "-" (.var "f1z") (.bin "*" (.var "b") (.var "e1z")))
    -- d = |tmp|, e2 = tmp / d
    , .declInit f "d"   (len3 (.var "tx") (.var "ty") (.var "tz"))
    , .declInit f "e2x" (.bin "/" (.var "tx") (.var "d"))
    , .declInit f "e2y" (.bin "/" (.var "ty") (.var "d"))
    , .declInit f "e2z" (.bin "/" (.var "tz") (.var "d"))
    -- closest-rotation R of F_2D = [a b; 0 d]:
    --   xv = a + d, yv = 0 - b, rn = sqrt(xv*xv + yv*yv)
    --   R = (1/rn) * [xv -yv; yv xv] = (1/rn) * [xv b; -b xv]
    , .declInit f "xv" (.bin "+" (.var "a") (.var "d"))
    , .declInit f "yv" (.bin "-" (.litFloat 0.0) (.var "b"))
    , .declInit f "rn"
        (.call "sqrt" [.bin "+" (.bin "*" (.var "xv") (.var "xv"))
                                (.bin "*" (.var "yv") (.var "yv"))])
    , .declInit f "r00" (.bin "/" (.var "xv") (.var "rn"))
    , .declInit f "r01" (.bin "/" (.var "b")  (.var "rn"))
    , .declInit f "r10" (.bin "/" (.var "yv") (.var "rn"))
    , .declInit f "r11" (.bin "/" (.var "xv") (.var "rn"))
    -- newF = newdeltaUV * R; newdeltaUV = [e1 | e2]
    --   newF.col(0) = e1 * r00 + e2 * r10
    --   newF.col(1) = e1 * r01 + e2 * r11
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
    -- Output: two float3s per triangle, scaled by sqrt(stiffness)
    , .declInit f "sw" (.index (.var "sqrtWeight") (.var "c"))
    , .declInit u "ob" (.bin "*" (.var "c") (.litUint 2))
    , .assign (.index (.var "projected") (.var "ob"))
        (.call "float3"
          [ .bin "*" (.var "sw") (.var "n0x")
          , .bin "*" (.var "sw") (.var "n0y")
          , .bin "*" (.var "sw") (.var "n0z") ])
    , .assign (.index (.var "projected") (.bin "+" (.var "ob") (.litUint 1)))
        (.call "float3"
          [ .bin "*" (.var "sw") (.var "n1x")
          , .bin "*" (.var "sw") (.var "n1y")
          , .bin "*" (.var "sw") (.var "n1z") ])
    ]
  { globals :=
      [ bnd 0 "positions"  (.roBuf f3)
      , bnd 1 "projected"  (.rwBuf f3)
      , bnd 2 "idx"        (.roBuf u)
      , bnd 3 "sqrtWeight" (.roBuf f)
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
    `native_decide` below. -/
def expected : String :=
"[[vk::binding(0, 0)]]
StructuredBuffer<float3> positions;
[[vk::binding(1, 0)]]
RWStructuredBuffer<float3> projected;
[[vk::binding(2, 0)]]
StructuredBuffer<uint> idx;
[[vk::binding(3, 0)]]
StructuredBuffer<float> sqrtWeight;

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
  float f0x = (p1.x - p0.x);
  float f0y = (p1.y - p0.y);
  float f0z = (p1.z - p0.z);
  float f1x = (p2.x - p0.x);
  float f1y = (p2.y - p0.y);
  float f1z = (p2.z - p0.z);
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
  float sw = sqrtWeight[c];
  uint ob = (c * 2u);
  projected[ob] = float3((sw * n0x), (sw * n0y), (sw * n0z));
  projected[(ob + 1u)] = float3((sw * n1x), (sw * n1y), (sw * n1z));
}"

example : LeanSlang.emit shader = expected := by native_decide

example : shader.entryPointName = "main" := by native_decide

end Cloth.SlangCodegen.TriangleProject
