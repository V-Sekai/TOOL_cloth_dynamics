import LeanSlang

/-!
# `Cloth.SlangCodegen.TriangleMembraneForceAl` — AL-augmented ARAP membrane force

Companion to `TriangleMembraneDualUpdate` (#78). Same per-triangle
force + GN Hessian as `triangle_membrane_force` (#45) but with two
`lambda` input columns folded into the per-vertex gradient. With
per-triangle `inv_deltaUV` (rest material matrix), the residual
columns `e0 = F.col(0) − R.col(0)`, `e1 = F.col(1) − R.col(1)` and
the λ columns share the same `∂F/∂p` chain through `inv_deltaUV`.
Define `te0 = k·e0 + λ0`, `te1 = k·e1 + λ1`, then:

```
grad[3c+0] = -(te0·(m00+m10) + te1·(m01+m11))   -- vertex 0
grad[3c+1] = +(te0·m00 + te1·m01)               -- vertex 1
grad[3c+2] = +(te0·m10 + te1·m11)               -- vertex 2

hessScalar[3c+0] = k · ((m00+m10)² + (m01+m11)²)
hessScalar[3c+1] = k · (m00² + m01²)
hessScalar[3c+2] = k · (m10² + m11²)
```

With `inv_deltaUV = I`, this reduces to the canonical AL form:
`grad[3c+0] = -k·(e0+e1) - (λ0+λ1)`, `hessScalar = 2k, k, k`.

`k` here is the effective stiffness `k_phys + γ_al`. The Hessian
shape is identical to the non-AL kernel — AL augmentation only
shifts the gradient by `λ·(∂F/∂p)`.

Bindings (set 0):

  0  StructuredBuffer<float3>   positions   length = N_verts
  1  StructuredBuffer<uint>     idx         length = 3 · N_tri
  2  StructuredBuffer<float>    stiffness   length = N_tri
  3  StructuredBuffer<float3>   lambda0     length = N_tri
  4  StructuredBuffer<float3>   lambda1     length = N_tri
  5  RWStructuredBuffer<float3> grad        length = 3 · N_tri
  6  RWStructuredBuffer<float>  hessScalar  length = 3 · N_tri
  7  StructuredBuffer<float>    inv_deltaUV length = 4 · N_tri
                                            (row-major 2x2 per tri)
-/

namespace Cloth.SlangCodegen.TriangleMembraneForceAl

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
  , .declInit f  "k"   (.index (.var "stiffness") (.var "c"))
  , .declInit f3 "l0"  (.index (.var "lambda0")   (.var "c"))
  , .declInit f3 "l1"  (.index (.var "lambda1")   (.var "c"))
  -- te0 = k·e0 + λ0, te1 = k·e1r + λ1 (3-vector, combined physics+AL)
  , .declInit f "te0x" (.bin "+" (.bin "*" (.var "k") (.var "e0x"))  (pmv "l0" "x"))
  , .declInit f "te0y" (.bin "+" (.bin "*" (.var "k") (.var "e0y"))  (pmv "l0" "y"))
  , .declInit f "te0z" (.bin "+" (.bin "*" (.var "k") (.var "e0z"))  (pmv "l0" "z"))
  , .declInit f "te1x" (.bin "+" (.bin "*" (.var "k") (.var "e1rx")) (pmv "l1" "x"))
  , .declInit f "te1y" (.bin "+" (.bin "*" (.var "k") (.var "e1ry")) (pmv "l1" "y"))
  , .declInit f "te1z" (.bin "+" (.bin "*" (.var "k") (.var "e1rz")) (pmv "l1" "z"))
  , .declInit f "w0a" (.bin "+" (.var "m00") (.var "m10"))
  , .declInit f "w0b" (.bin "+" (.var "m01") (.var "m11"))
  , .declInit u "gb"  (.bin "*" (.var "c") (.litUint 3))
  -- grad_p0 = -(te0 * w0a + te1 * w0b)
  , .assign (.index (.var "grad") (.var "gb"))
      (.call "float3"
        [ .bin "-" (.litFloat 0.0)
            (.bin "+" (.bin "*" (.var "te0x") (.var "w0a"))
                      (.bin "*" (.var "te1x") (.var "w0b")))
        , .bin "-" (.litFloat 0.0)
            (.bin "+" (.bin "*" (.var "te0y") (.var "w0a"))
                      (.bin "*" (.var "te1y") (.var "w0b")))
        , .bin "-" (.litFloat 0.0)
            (.bin "+" (.bin "*" (.var "te0z") (.var "w0a"))
                      (.bin "*" (.var "te1z") (.var "w0b"))) ])
  -- grad_p1 = te0 * m00 + te1 * m01
  , .assign (.index (.var "grad") (.bin "+" (.var "gb") (.litUint 1)))
      (.call "float3"
        [ .bin "+" (.bin "*" (.var "te0x") (.var "m00"))
                   (.bin "*" (.var "te1x") (.var "m01"))
        , .bin "+" (.bin "*" (.var "te0y") (.var "m00"))
                   (.bin "*" (.var "te1y") (.var "m01"))
        , .bin "+" (.bin "*" (.var "te0z") (.var "m00"))
                   (.bin "*" (.var "te1z") (.var "m01")) ])
  -- grad_p2 = te0 * m10 + te1 * m11
  , .assign (.index (.var "grad") (.bin "+" (.var "gb") (.litUint 2)))
      (.call "float3"
        [ .bin "+" (.bin "*" (.var "te0x") (.var "m10"))
                   (.bin "*" (.var "te1x") (.var "m11"))
        , .bin "+" (.bin "*" (.var "te0y") (.var "m10"))
                   (.bin "*" (.var "te1y") (.var "m11"))
        , .bin "+" (.bin "*" (.var "te0z") (.var "m10"))
                   (.bin "*" (.var "te1z") (.var "m11")) ])
  , .assign (.index (.var "hessScalar") (.var "gb"))
      (.bin "*" (.var "k")
        (.bin "+" (.bin "*" (.var "w0a") (.var "w0a"))
                  (.bin "*" (.var "w0b") (.var "w0b"))))
  , .assign (.index (.var "hessScalar") (.bin "+" (.var "gb") (.litUint 1)))
      (.bin "*" (.var "k")
        (.bin "+" (.bin "*" (.var "m00") (.var "m00"))
                  (.bin "*" (.var "m01") (.var "m01"))))
  , .assign (.index (.var "hessScalar") (.bin "+" (.var "gb") (.litUint 2)))
      (.bin "*" (.var "k")
        (.bin "+" (.bin "*" (.var "m10") (.var "m10"))
                  (.bin "*" (.var "m11") (.var "m11"))))
  ]

def shader : SlangShaderModule :=
  { globals :=
      [ bnd 0 "positions"   (.roBuf f3)
      , bnd 1 "idx"         (.roBuf u)
      , bnd 2 "stiffness"   (.roBuf f)
      , bnd 3 "lambda0"     (.roBuf f3)
      , bnd 4 "lambda1"     (.roBuf f3)
      , bnd 5 "grad"        (.rwBuf f3)
      , bnd 6 "hessScalar"  (.rwBuf f)
      , bnd 7 "inv_deltaUV" (.roBuf f)
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
StructuredBuffer<float> stiffness;
[[vk::binding(3, 0)]]
StructuredBuffer<float3> lambda0;
[[vk::binding(4, 0)]]
StructuredBuffer<float3> lambda1;
[[vk::binding(5, 0)]]
RWStructuredBuffer<float3> grad;
[[vk::binding(6, 0)]]
RWStructuredBuffer<float> hessScalar;
[[vk::binding(7, 0)]]
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
  float k = stiffness[c];
  float3 l0 = lambda0[c];
  float3 l1 = lambda1[c];
  float te0x = ((k * e0x) + l0.x);
  float te0y = ((k * e0y) + l0.y);
  float te0z = ((k * e0z) + l0.z);
  float te1x = ((k * e1rx) + l1.x);
  float te1y = ((k * e1ry) + l1.y);
  float te1z = ((k * e1rz) + l1.z);
  float w0a = (m00 + m10);
  float w0b = (m01 + m11);
  uint gb = (c * 3u);
  grad[gb] = float3((0.000000 - ((te0x * w0a) + (te1x * w0b))), (0.000000 - ((te0y * w0a) + (te1y * w0b))), (0.000000 - ((te0z * w0a) + (te1z * w0b))));
  grad[(gb + 1u)] = float3(((te0x * m00) + (te1x * m01)), ((te0y * m00) + (te1y * m01)), ((te0z * m00) + (te1z * m01)));
  grad[(gb + 2u)] = float3(((te0x * m10) + (te1x * m11)), ((te0y * m10) + (te1y * m11)), ((te0z * m10) + (te1z * m11)));
  hessScalar[gb] = (k * ((w0a * w0a) + (w0b * w0b)));
  hessScalar[(gb + 1u)] = (k * ((m00 * m00) + (m01 * m01)));
  hessScalar[(gb + 2u)] = (k * ((m10 * m10) + (m11 * m11)));
}"

example : LeanSlang.emit shader = expected := by native_decide
example : shader.entryPointName = "main" := by native_decide

end Cloth.SlangCodegen.TriangleMembraneForceAl
