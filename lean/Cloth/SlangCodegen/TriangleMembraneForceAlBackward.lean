import LeanSlang

/-!
# `Cloth.SlangCodegen.TriangleMembraneForceAlBackward` — adjoint of triangle_membrane_force_al (PR-G / CHI-13)

Reverse-mode VJP for the AL-augmented ARAP membrane force kernel.
This is the research-grade piece of PR-G. The forward (see
`[[Cloth.SlangCodegen.TriangleMembraneForceAl]]`) computes a 2D
polar decomposition of `F = P · inv_deltaUV` and then forms the
augmented Lagrangian gradient and Hessian; the backward needs the
differential of that polar decomposition.

## Forward recap

```
ex0,ex1     = p1−p0, p2−p0                    -- material differences
f0          = m00·ex0 + m10·ex1               -- F.col0 (3-vec)
f1          = m01·ex0 + m11·ex1               -- F.col1 (3-vec)
-- Gram-Schmidt of F (3x2) into upper-triangular F_2d = [[a, b], [0, d]]
a           = |f0|
e1          = f0/a
b           = e1·f1
t           = f1 − b·e1
d           = |t|
e2          = t/d
-- 2D polar of F_2d
xv, rn      = a+d, sqrt((a+d)² + b²)
r00 = r11   = xv/rn       (cos θ)
r01         = b/rn        (sin θ)
r10         = −b/rn       (−sin θ)
-- Project R back to 3D: R.col0 = e1·r00 + e2·r10, R.col1 = e1·r01 + e2·r11
n0          = e1·r00 + e2·r10
n1          = e1·r01 + e2·r11
e0, e1r     = f0 − n0, f1 − n1                -- residuals
te0, te1    = k·e0 + λ0, k·e1r + λ1
w0a, w0b    = m00 + m10, m01 + m11
grad[3c+0]  = −(te0·w0a + te1·w0b)
grad[3c+1]  =  te0·m00 + te1·m01
grad[3c+2]  =  te0·m10 + te1·m11
hessScalar  = k·(w0a²+w0b², m00²+m01², m10²+m11²)
```

## Adjoint with full 2D polar derivative

Cotangents in (per corner r ∈ {0,1,2}):
- `v_grad[3c+r]`     — 3-vec
- `v_hessScalar[3c+r]` — scalar

Contracted F-cotangents:
```
B = −w0a·v_grad[3c+0] + m00·v_grad[3c+1] + m10·v_grad[3c+2]    -- ∂L/∂te0
C = −w0b·v_grad[3c+0] + m01·v_grad[3c+1] + m11·v_grad[3c+2]    -- ∂L/∂te1
H = v_h[3c+0]·(w0a²+w0b²) + v_h[3c+1]·(m00²+m01²) + v_h[3c+2]·(m10²+m11²)
```

Easy outputs (linear in te → linear in λ; only B, C appear):
```
v_lambda0 = B
v_lambda1 = C
v_k       = B·e0 + C·e1r + H
```

Position cotangents: full chain through Gram-Schmidt + 2D polar.

**Step 1 — seed:** `v_te0 = B`, `v_te1 = C`. Since `te = k·e + λ`,
`v_e0 = k·B`, `v_e1r = k·C`. From `e0 = f0 − n0`, `e1r = f1 − n1`:
```
v_f0 := k·B,     v_n0 := −k·B
v_f1 := k·C,     v_n1 := −k·C
```

**Step 2 — project-back adjoint** (`n0 = e1·r00 + e2·r10`, `n1 = e1·r01 + e2·r11`):
```
v_e1   = v_n0·r00 + v_n1·r01
v_e2   = v_n0·r10 + v_n1·r11
v_r00  = v_n0 · e1     (dot)
v_r01  = v_n1 · e1
v_r10  = v_n0 · e2
v_r11  = v_n1 · e2
```

**Step 3 — 2D polar adjoint.** With `rn = sqrt(xv² + b²)`,
`rn³ = rn·rn·rn`:
```
v_xv      = (v_r00 + v_r11)·b²/rn³ + (v_r10 − v_r01)·xv·b/rn³
v_b_polar = −(v_r00 + v_r11)·xv·b/rn³ + (v_r01 − v_r10)·xv²/rn³
v_a = v_xv,  v_d = v_xv                       (since xv = a + d)
```

**Step 4 — Gram-Schmidt adjoint.**
* `e2 = t/d`: `v_t = (v_e2 − (v_e2·e2)·e2)/d + v_d·e2`
* `t = f1 − b·e1`: `v_f1 += v_t`, `v_e1 += −b·v_t`, `v_b_gs = −(v_t·e1)`
* `b = e1·f1`: with `v_b = v_b_polar + v_b_gs`, `v_e1 += v_b·f1`, `v_f1 += v_b·e1`
* `e1 = f0/a`: `v_f0 += (v_e1 − (v_e1·e1)·e1)/a`
* `a = |f0|`: `v_f0 += v_a·e1`

**Step 5 — F → positions** (linear):
```
v_p0 = −(w0a·v_f0 + w0b·v_f1)
v_p1 =  m00·v_f0 + m01·v_f1
v_p2 =  m10·v_f0 + m11·v_f1
```

The adjoint is exact to fp32 precision — no Sifakis-cookie /
frozen-R approximation. FD-validated against the forward kernel to
1e-2 relative on all paths (positions, stiffness, lambda).

Bindings (set 0):

  0  StructuredBuffer<float3>   positions     length = N_verts
  1  StructuredBuffer<uint>     idx           length = 3·N_tri
  2  StructuredBuffer<float>    stiffness     length = N_tri
  3  StructuredBuffer<float3>   lambda0       length = N_tri
  4  StructuredBuffer<float3>   lambda1       length = N_tri
  5  StructuredBuffer<float>    inv_deltaUV   length = 4·N_tri
  6  StructuredBuffer<float3>   v_grad        length = 3·N_tri  (cotangent)
  7  StructuredBuffer<float>    v_hessScalar  length = 3·N_tri  (cotangent)
  8  RWStructuredBuffer<float3> v_p           length = 3·N_tri  (∂L/∂p per corner)
  9  RWStructuredBuffer<float>  v_stiffness   length = N_tri    (∂L/∂k)
 10  RWStructuredBuffer<float3> v_lambda0     length = N_tri    (∂L/∂λ0)
 11  RWStructuredBuffer<float3> v_lambda1     length = N_tri    (∂L/∂λ1)
-/

namespace Cloth.SlangCodegen.TriangleMembraneForceAlBackward

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
  , .declInit u  "iub"  (.bin "*" (.var "c") (.litUint 4))
  , .declInit f  "m00"  (.index (.var "inv_deltaUV") (.var "iub"))
  , .declInit f  "m01"  (.index (.var "inv_deltaUV") (.bin "+" (.var "iub") (.litUint 1)))
  , .declInit f  "m10"  (.index (.var "inv_deltaUV") (.bin "+" (.var "iub") (.litUint 2)))
  , .declInit f  "m11"  (.index (.var "inv_deltaUV") (.bin "+" (.var "iub") (.litUint 3)))
  -- Recompute forward F, polar R, residuals e0, e1r
  , .declInit f  "ex0" (.bin "-" (pmv "p1" "x") (pmv "p0" "x"))
  , .declInit f  "ey0" (.bin "-" (pmv "p1" "y") (pmv "p0" "y"))
  , .declInit f  "ez0" (.bin "-" (pmv "p1" "z") (pmv "p0" "z"))
  , .declInit f  "ex1" (.bin "-" (pmv "p2" "x") (pmv "p0" "x"))
  , .declInit f  "ey1" (.bin "-" (pmv "p2" "y") (pmv "p0" "y"))
  , .declInit f  "ez1" (.bin "-" (pmv "p2" "z") (pmv "p0" "z"))
  , .declInit f  "f0x" (.bin "+" (.bin "*" (.var "ex0") (.var "m00"))
                                 (.bin "*" (.var "ex1") (.var "m10")))
  , .declInit f  "f0y" (.bin "+" (.bin "*" (.var "ey0") (.var "m00"))
                                 (.bin "*" (.var "ey1") (.var "m10")))
  , .declInit f  "f0z" (.bin "+" (.bin "*" (.var "ez0") (.var "m00"))
                                 (.bin "*" (.var "ez1") (.var "m10")))
  , .declInit f  "f1x" (.bin "+" (.bin "*" (.var "ex0") (.var "m01"))
                                 (.bin "*" (.var "ex1") (.var "m11")))
  , .declInit f  "f1y" (.bin "+" (.bin "*" (.var "ey0") (.var "m01"))
                                 (.bin "*" (.var "ey1") (.var "m11")))
  , .declInit f  "f1z" (.bin "+" (.bin "*" (.var "ez0") (.var "m01"))
                                 (.bin "*" (.var "ez1") (.var "m11")))
  , .declInit f  "a"   (len3 (.var "f0x") (.var "f0y") (.var "f0z"))
  , .declInit f  "e1x" (.bin "/" (.var "f0x") (.var "a"))
  , .declInit f  "e1y" (.bin "/" (.var "f0y") (.var "a"))
  , .declInit f  "e1z" (.bin "/" (.var "f0z") (.var "a"))
  , .declInit f  "b"
      (sum3 (.bin "*" (.var "e1x") (.var "f1x"))
            (.bin "*" (.var "e1y") (.var "f1y"))
            (.bin "*" (.var "e1z") (.var "f1z")))
  , .declInit f  "tx" (.bin "-" (.var "f1x") (.bin "*" (.var "b") (.var "e1x")))
  , .declInit f  "ty" (.bin "-" (.var "f1y") (.bin "*" (.var "b") (.var "e1y")))
  , .declInit f  "tz" (.bin "-" (.var "f1z") (.bin "*" (.var "b") (.var "e1z")))
  , .declInit f  "d"   (len3 (.var "tx") (.var "ty") (.var "tz"))
  , .declInit f  "e2x" (.bin "/" (.var "tx") (.var "d"))
  , .declInit f  "e2y" (.bin "/" (.var "ty") (.var "d"))
  , .declInit f  "e2z" (.bin "/" (.var "tz") (.var "d"))
  , .declInit f  "xv" (.bin "+" (.var "a") (.var "d"))
  , .declInit f  "rn"
      (.call "sqrt" [.bin "+" (.bin "*" (.var "xv") (.var "xv"))
                              (.bin "*" (.var "b") (.var "b"))])
  , .declInit f  "r00" (.bin "/" (.var "xv") (.var "rn"))
  , .declInit f  "r01" (.bin "/" (.var "b")  (.var "rn"))
  , .declInit f  "r10" (.bin "-" (.litFloat 0.0) (.bin "/" (.var "b") (.var "rn")))
  , .declInit f  "r11" (.bin "/" (.var "xv") (.var "rn"))
  , .declInit f  "n0x" (.bin "+" (.bin "*" (.var "e1x") (.var "r00"))
                                 (.bin "*" (.var "e2x") (.var "r10")))
  , .declInit f  "n0y" (.bin "+" (.bin "*" (.var "e1y") (.var "r00"))
                                 (.bin "*" (.var "e2y") (.var "r10")))
  , .declInit f  "n0z" (.bin "+" (.bin "*" (.var "e1z") (.var "r00"))
                                 (.bin "*" (.var "e2z") (.var "r10")))
  , .declInit f  "n1x" (.bin "+" (.bin "*" (.var "e1x") (.var "r01"))
                                 (.bin "*" (.var "e2x") (.var "r11")))
  , .declInit f  "n1y" (.bin "+" (.bin "*" (.var "e1y") (.var "r01"))
                                 (.bin "*" (.var "e2y") (.var "r11")))
  , .declInit f  "n1z" (.bin "+" (.bin "*" (.var "e1z") (.var "r01"))
                                 (.bin "*" (.var "e2z") (.var "r11")))
  , .declInit f  "e0x" (.bin "-" (.var "f0x") (.var "n0x"))
  , .declInit f  "e0y" (.bin "-" (.var "f0y") (.var "n0y"))
  , .declInit f  "e0z" (.bin "-" (.var "f0z") (.var "n0z"))
  , .declInit f  "e1rx" (.bin "-" (.var "f1x") (.var "n1x"))
  , .declInit f  "e1ry" (.bin "-" (.var "f1y") (.var "n1y"))
  , .declInit f  "e1rz" (.bin "-" (.var "f1z") (.var "n1z"))
  -- w0a, w0b
  , .declInit f "w0a" (.bin "+" (.var "m00") (.var "m10"))
  , .declInit f "w0b" (.bin "+" (.var "m01") (.var "m11"))
  , .declInit f  "k"  (.index (.var "stiffness") (.var "c"))
  -- Read input cotangents
  , .declInit f3 "vg0" (.index (.var "v_grad") (.var "base"))
  , .declInit f3 "vg1" (.index (.var "v_grad") (.bin "+" (.var "base") (.litUint 1)))
  , .declInit f3 "vg2" (.index (.var "v_grad") (.bin "+" (.var "base") (.litUint 2)))
  , .declInit f  "vh0" (.index (.var "v_hessScalar") (.var "base"))
  , .declInit f  "vh1" (.index (.var "v_hessScalar") (.bin "+" (.var "base") (.litUint 1)))
  , .declInit f  "vh2" (.index (.var "v_hessScalar") (.bin "+" (.var "base") (.litUint 2)))
  -- B = -w0a·vg0 + m00·vg1 + m10·vg2   (3-vec, ∂L/∂te0)
  , .declInit f "Bx"
      (.bin "+"
        (.bin "+" (.bin "*" (.bin "-" (.litFloat 0.0) (.var "w0a")) (pmv "vg0" "x"))
                  (.bin "*" (.var "m00") (pmv "vg1" "x")))
        (.bin "*" (.var "m10") (pmv "vg2" "x")))
  , .declInit f "By"
      (.bin "+"
        (.bin "+" (.bin "*" (.bin "-" (.litFloat 0.0) (.var "w0a")) (pmv "vg0" "y"))
                  (.bin "*" (.var "m00") (pmv "vg1" "y")))
        (.bin "*" (.var "m10") (pmv "vg2" "y")))
  , .declInit f "Bz"
      (.bin "+"
        (.bin "+" (.bin "*" (.bin "-" (.litFloat 0.0) (.var "w0a")) (pmv "vg0" "z"))
                  (.bin "*" (.var "m00") (pmv "vg1" "z")))
        (.bin "*" (.var "m10") (pmv "vg2" "z")))
  -- C = -w0b·vg0 + m01·vg1 + m11·vg2   (3-vec, ∂L/∂te1)
  , .declInit f "Cx"
      (.bin "+"
        (.bin "+" (.bin "*" (.bin "-" (.litFloat 0.0) (.var "w0b")) (pmv "vg0" "x"))
                  (.bin "*" (.var "m01") (pmv "vg1" "x")))
        (.bin "*" (.var "m11") (pmv "vg2" "x")))
  , .declInit f "Cy"
      (.bin "+"
        (.bin "+" (.bin "*" (.bin "-" (.litFloat 0.0) (.var "w0b")) (pmv "vg0" "y"))
                  (.bin "*" (.var "m01") (pmv "vg1" "y")))
        (.bin "*" (.var "m11") (pmv "vg2" "y")))
  , .declInit f "Cz"
      (.bin "+"
        (.bin "+" (.bin "*" (.bin "-" (.litFloat 0.0) (.var "w0b")) (pmv "vg0" "z"))
                  (.bin "*" (.var "m01") (pmv "vg1" "z")))
        (.bin "*" (.var "m11") (pmv "vg2" "z")))
  -- H = v_h[0]·(w0a² + w0b²) + v_h[1]·(m00² + m01²) + v_h[2]·(m10² + m11²)
  , .declInit f "H"
      (.bin "+"
        (.bin "+"
          (.bin "*" (.var "vh0")
            (.bin "+" (.bin "*" (.var "w0a") (.var "w0a"))
                      (.bin "*" (.var "w0b") (.var "w0b"))))
          (.bin "*" (.var "vh1")
            (.bin "+" (.bin "*" (.var "m00") (.var "m00"))
                      (.bin "*" (.var "m01") (.var "m01")))))
        (.bin "*" (.var "vh2")
          (.bin "+" (.bin "*" (.var "m10") (.var "m10"))
                    (.bin "*" (.var "m11") (.var "m11")))))
  -- B·e0
  , .declInit f "BdotE0"
      (.bin "+"
        (.bin "+" (.bin "*" (.var "Bx") (.var "e0x"))
                  (.bin "*" (.var "By") (.var "e0y")))
        (.bin "*" (.var "Bz") (.var "e0z")))
  -- C·e1r
  , .declInit f "CdotE1"
      (.bin "+"
        (.bin "+" (.bin "*" (.var "Cx") (.var "e1rx"))
                  (.bin "*" (.var "Cy") (.var "e1ry")))
        (.bin "*" (.var "Cz") (.var "e1rz")))
  -- Outputs for lambda, k (exact, no polar dependence)
  , .assign (.index (.var "v_lambda0") (.var "c"))
      (.call "float3" [.var "Bx", .var "By", .var "Bz"])
  , .assign (.index (.var "v_lambda1") (.var "c"))
      (.call "float3" [.var "Cx", .var "Cy", .var "Cz"])
  , .assign (.index (.var "v_stiffness") (.var "c"))
      (.bin "+" (.bin "+" (.var "BdotE0") (.var "CdotE1")) (.var "H"))
  -- Full polar adjoint for position cotangents.
  -- Step 1: seed v_f0, v_f1 from v_e0 = k·B, v_e1r = k·C.
  -- Also v_n0 = -k·B, v_n1 = -k·C.
  , .declInit f "vf0x" (.bin "*" (.var "k") (.var "Bx"))
  , .declInit f "vf0y" (.bin "*" (.var "k") (.var "By"))
  , .declInit f "vf0z" (.bin "*" (.var "k") (.var "Bz"))
  , .declInit f "vf1x" (.bin "*" (.var "k") (.var "Cx"))
  , .declInit f "vf1y" (.bin "*" (.var "k") (.var "Cy"))
  , .declInit f "vf1z" (.bin "*" (.var "k") (.var "Cz"))
  , .declInit f "vn0x" (.bin "-" (.litFloat 0.0) (.var "vf0x"))
  , .declInit f "vn0y" (.bin "-" (.litFloat 0.0) (.var "vf0y"))
  , .declInit f "vn0z" (.bin "-" (.litFloat 0.0) (.var "vf0z"))
  , .declInit f "vn1x" (.bin "-" (.litFloat 0.0) (.var "vf1x"))
  , .declInit f "vn1y" (.bin "-" (.litFloat 0.0) (.var "vf1y"))
  , .declInit f "vn1z" (.bin "-" (.litFloat 0.0) (.var "vf1z"))
  -- Step 2: project-back adjoint.
  --   v_e1 = vn0·r00 + vn1·r01
  --   v_e2 = vn0·r10 + vn1·r11
  --   v_r{00,01,10,11} = dots with e1/e2
  , .declInit f "ve1x"
      (.bin "+" (.bin "*" (.var "vn0x") (.var "r00")) (.bin "*" (.var "vn1x") (.var "r01")))
  , .declInit f "ve1y"
      (.bin "+" (.bin "*" (.var "vn0y") (.var "r00")) (.bin "*" (.var "vn1y") (.var "r01")))
  , .declInit f "ve1z"
      (.bin "+" (.bin "*" (.var "vn0z") (.var "r00")) (.bin "*" (.var "vn1z") (.var "r01")))
  , .declInit f "ve2x"
      (.bin "+" (.bin "*" (.var "vn0x") (.var "r10")) (.bin "*" (.var "vn1x") (.var "r11")))
  , .declInit f "ve2y"
      (.bin "+" (.bin "*" (.var "vn0y") (.var "r10")) (.bin "*" (.var "vn1y") (.var "r11")))
  , .declInit f "ve2z"
      (.bin "+" (.bin "*" (.var "vn0z") (.var "r10")) (.bin "*" (.var "vn1z") (.var "r11")))
  , .declInit f "vr00"
      (sum3 (.bin "*" (.var "vn0x") (.var "e1x"))
            (.bin "*" (.var "vn0y") (.var "e1y"))
            (.bin "*" (.var "vn0z") (.var "e1z")))
  , .declInit f "vr01"
      (sum3 (.bin "*" (.var "vn1x") (.var "e1x"))
            (.bin "*" (.var "vn1y") (.var "e1y"))
            (.bin "*" (.var "vn1z") (.var "e1z")))
  , .declInit f "vr10"
      (sum3 (.bin "*" (.var "vn0x") (.var "e2x"))
            (.bin "*" (.var "vn0y") (.var "e2y"))
            (.bin "*" (.var "vn0z") (.var "e2z")))
  , .declInit f "vr11"
      (sum3 (.bin "*" (.var "vn1x") (.var "e2x"))
            (.bin "*" (.var "vn1y") (.var "e2y"))
            (.bin "*" (.var "vn1z") (.var "e2z")))
  -- Step 3: 2D polar adjoint.
  -- rn3 = rn·rn·rn (rn is already the polar normalisation factor)
  , .declInit f "rn3" (.bin "*" (.var "rn") (.bin "*" (.var "rn") (.var "rn")))
  , .declInit f "trR" (.bin "+" (.var "vr00") (.var "vr11"))    -- v_r00 + v_r11
  , .declInit f "antR" (.bin "-" (.var "vr10") (.var "vr01"))   -- v_r10 - v_r01
  , .declInit f "antR2" (.bin "-" (.var "vr01") (.var "vr10"))  -- v_r01 - v_r10
  -- v_xv = trR·b²/rn³ + antR·xv·b/rn³
  , .declInit f "vxv"
      (.bin "+"
        (.bin "/" (.bin "*" (.var "trR") (.bin "*" (.var "b") (.var "b"))) (.var "rn3"))
        (.bin "/" (.bin "*" (.var "antR") (.bin "*" (.var "xv") (.var "b"))) (.var "rn3")))
  -- v_b_polar = -trR·xv·b/rn³ + antR2·xv²/rn³
  , .declInit f "vb_polar"
      (.bin "+"
        (.bin "-" (.litFloat 0.0)
          (.bin "/" (.bin "*" (.var "trR") (.bin "*" (.var "xv") (.var "b"))) (.var "rn3")))
        (.bin "/" (.bin "*" (.var "antR2") (.bin "*" (.var "xv") (.var "xv"))) (.var "rn3")))
  , .declInit f "va" (.var "vxv")
  , .declInit f "vd" (.var "vxv")
  -- Step 4a: e2 = t/d adjoint. v_t = (v_e2 − (v_e2·e2)·e2)/d + v_d·e2.
  , .declInit f "ve2dote2"
      (sum3 (.bin "*" (.var "ve2x") (.var "e2x"))
            (.bin "*" (.var "ve2y") (.var "e2y"))
            (.bin "*" (.var "ve2z") (.var "e2z")))
  , .declInit f "vtx"
      (.bin "+"
        (.bin "/" (.bin "-" (.var "ve2x") (.bin "*" (.var "ve2dote2") (.var "e2x"))) (.var "d"))
        (.bin "*" (.var "vd") (.var "e2x")))
  , .declInit f "vty"
      (.bin "+"
        (.bin "/" (.bin "-" (.var "ve2y") (.bin "*" (.var "ve2dote2") (.var "e2y"))) (.var "d"))
        (.bin "*" (.var "vd") (.var "e2y")))
  , .declInit f "vtz"
      (.bin "+"
        (.bin "/" (.bin "-" (.var "ve2z") (.bin "*" (.var "ve2dote2") (.var "e2z"))) (.var "d"))
        (.bin "*" (.var "vd") (.var "e2z")))
  -- Step 4b: t = f1 − b·e1 adjoint.
  -- v_f1 += v_t
  , .assign (.var "vf1x") (.bin "+" (.var "vf1x") (.var "vtx"))
  , .assign (.var "vf1y") (.bin "+" (.var "vf1y") (.var "vty"))
  , .assign (.var "vf1z") (.bin "+" (.var "vf1z") (.var "vtz"))
  -- v_e1 += -b·v_t
  , .assign (.var "ve1x") (.bin "-" (.var "ve1x") (.bin "*" (.var "b") (.var "vtx")))
  , .assign (.var "ve1y") (.bin "-" (.var "ve1y") (.bin "*" (.var "b") (.var "vty")))
  , .assign (.var "ve1z") (.bin "-" (.var "ve1z") (.bin "*" (.var "b") (.var "vtz")))
  -- v_b_gs = -(v_t·e1)
  , .declInit f "vt_dot_e1"
      (sum3 (.bin "*" (.var "vtx") (.var "e1x"))
            (.bin "*" (.var "vty") (.var "e1y"))
            (.bin "*" (.var "vtz") (.var "e1z")))
  , .declInit f "vb" (.bin "+" (.var "vb_polar") (.bin "-" (.litFloat 0.0) (.var "vt_dot_e1")))
  -- Step 4c: b = e1·f1 adjoint.
  -- v_e1 += vb·f1, v_f1 += vb·e1
  , .assign (.var "ve1x") (.bin "+" (.var "ve1x") (.bin "*" (.var "vb") (.var "f1x")))
  , .assign (.var "ve1y") (.bin "+" (.var "ve1y") (.bin "*" (.var "vb") (.var "f1y")))
  , .assign (.var "ve1z") (.bin "+" (.var "ve1z") (.bin "*" (.var "vb") (.var "f1z")))
  , .assign (.var "vf1x") (.bin "+" (.var "vf1x") (.bin "*" (.var "vb") (.var "e1x")))
  , .assign (.var "vf1y") (.bin "+" (.var "vf1y") (.bin "*" (.var "vb") (.var "e1y")))
  , .assign (.var "vf1z") (.bin "+" (.var "vf1z") (.bin "*" (.var "vb") (.var "e1z")))
  -- Step 4d: e1 = f0/a adjoint.
  -- v_f0 += (v_e1 − (v_e1·e1)·e1)/a
  , .declInit f "ve1dote1"
      (sum3 (.bin "*" (.var "ve1x") (.var "e1x"))
            (.bin "*" (.var "ve1y") (.var "e1y"))
            (.bin "*" (.var "ve1z") (.var "e1z")))
  , .assign (.var "vf0x")
      (.bin "+" (.var "vf0x")
        (.bin "/" (.bin "-" (.var "ve1x") (.bin "*" (.var "ve1dote1") (.var "e1x"))) (.var "a")))
  , .assign (.var "vf0y")
      (.bin "+" (.var "vf0y")
        (.bin "/" (.bin "-" (.var "ve1y") (.bin "*" (.var "ve1dote1") (.var "e1y"))) (.var "a")))
  , .assign (.var "vf0z")
      (.bin "+" (.var "vf0z")
        (.bin "/" (.bin "-" (.var "ve1z") (.bin "*" (.var "ve1dote1") (.var "e1z"))) (.var "a")))
  -- Step 4e: a = |f0| adjoint. v_f0 += va·e1.
  , .assign (.var "vf0x") (.bin "+" (.var "vf0x") (.bin "*" (.var "va") (.var "e1x")))
  , .assign (.var "vf0y") (.bin "+" (.var "vf0y") (.bin "*" (.var "va") (.var "e1y")))
  , .assign (.var "vf0z") (.bin "+" (.var "vf0z") (.bin "*" (.var "va") (.var "e1z")))
  -- Step 5: F → positions.
  --   v_p0 = -(w0a·v_f0 + w0b·v_f1)
  --   v_p1 = m00·v_f0 + m01·v_f1
  --   v_p2 = m10·v_f0 + m11·v_f1
  , .assign (.index (.var "v_p") (.var "base"))
      (.call "float3"
        [ .bin "-" (.litFloat 0.0)
            (.bin "+" (.bin "*" (.var "w0a") (.var "vf0x"))
                      (.bin "*" (.var "w0b") (.var "vf1x")))
        , .bin "-" (.litFloat 0.0)
            (.bin "+" (.bin "*" (.var "w0a") (.var "vf0y"))
                      (.bin "*" (.var "w0b") (.var "vf1y")))
        , .bin "-" (.litFloat 0.0)
            (.bin "+" (.bin "*" (.var "w0a") (.var "vf0z"))
                      (.bin "*" (.var "w0b") (.var "vf1z"))) ])
  , .assign (.index (.var "v_p") (.bin "+" (.var "base") (.litUint 1)))
      (.call "float3"
        [ .bin "+" (.bin "*" (.var "m00") (.var "vf0x")) (.bin "*" (.var "m01") (.var "vf1x"))
        , .bin "+" (.bin "*" (.var "m00") (.var "vf0y")) (.bin "*" (.var "m01") (.var "vf1y"))
        , .bin "+" (.bin "*" (.var "m00") (.var "vf0z")) (.bin "*" (.var "m01") (.var "vf1z")) ])
  , .assign (.index (.var "v_p") (.bin "+" (.var "base") (.litUint 2)))
      (.call "float3"
        [ .bin "+" (.bin "*" (.var "m10") (.var "vf0x")) (.bin "*" (.var "m11") (.var "vf1x"))
        , .bin "+" (.bin "*" (.var "m10") (.var "vf0y")) (.bin "*" (.var "m11") (.var "vf1y"))
        , .bin "+" (.bin "*" (.var "m10") (.var "vf0z")) (.bin "*" (.var "m11") (.var "vf1z")) ])
  ]

def shader : SlangShaderModule :=
  { globals :=
      [ bnd 0  "positions"    (.roBuf f3)
      , bnd 1  "idx"          (.roBuf u)
      , bnd 2  "stiffness"    (.roBuf f)
      , bnd 3  "lambda0"      (.roBuf f3)
      , bnd 4  "lambda1"      (.roBuf f3)
      , bnd 5  "inv_deltaUV"  (.roBuf f)
      , bnd 6  "v_grad"       (.roBuf f3)
      , bnd 7  "v_hessScalar" (.roBuf f)
      , bnd 8  "v_p"          (.rwBuf f3)
      , bnd 9  "v_stiffness"  (.rwBuf f)
      , bnd 10 "v_lambda0"    (.rwBuf f3)
      , bnd 11 "v_lambda1"    (.rwBuf f3)
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
StructuredBuffer<float> inv_deltaUV;
[[vk::binding(6, 0)]]
StructuredBuffer<float3> v_grad;
[[vk::binding(7, 0)]]
StructuredBuffer<float> v_hessScalar;
[[vk::binding(8, 0)]]
RWStructuredBuffer<float3> v_p;
[[vk::binding(9, 0)]]
RWStructuredBuffer<float> v_stiffness;
[[vk::binding(10, 0)]]
RWStructuredBuffer<float3> v_lambda0;
[[vk::binding(11, 0)]]
RWStructuredBuffer<float3> v_lambda1;

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
  float rn = sqrt(((xv * xv) + (b * b)));
  float r00 = (xv / rn);
  float r01 = (b / rn);
  float r10 = (0.000000 - (b / rn));
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
  float w0a = (m00 + m10);
  float w0b = (m01 + m11);
  float k = stiffness[c];
  float3 vg0 = v_grad[base];
  float3 vg1 = v_grad[(base + 1u)];
  float3 vg2 = v_grad[(base + 2u)];
  float vh0 = v_hessScalar[base];
  float vh1 = v_hessScalar[(base + 1u)];
  float vh2 = v_hessScalar[(base + 2u)];
  float Bx = ((((0.000000 - w0a) * vg0.x) + (m00 * vg1.x)) + (m10 * vg2.x));
  float By = ((((0.000000 - w0a) * vg0.y) + (m00 * vg1.y)) + (m10 * vg2.y));
  float Bz = ((((0.000000 - w0a) * vg0.z) + (m00 * vg1.z)) + (m10 * vg2.z));
  float Cx = ((((0.000000 - w0b) * vg0.x) + (m01 * vg1.x)) + (m11 * vg2.x));
  float Cy = ((((0.000000 - w0b) * vg0.y) + (m01 * vg1.y)) + (m11 * vg2.y));
  float Cz = ((((0.000000 - w0b) * vg0.z) + (m01 * vg1.z)) + (m11 * vg2.z));
  float H = (((vh0 * ((w0a * w0a) + (w0b * w0b))) + (vh1 * ((m00 * m00) + (m01 * m01)))) + (vh2 * ((m10 * m10) + (m11 * m11))));
  float BdotE0 = (((Bx * e0x) + (By * e0y)) + (Bz * e0z));
  float CdotE1 = (((Cx * e1rx) + (Cy * e1ry)) + (Cz * e1rz));
  v_lambda0[c] = float3(Bx, By, Bz);
  v_lambda1[c] = float3(Cx, Cy, Cz);
  v_stiffness[c] = ((BdotE0 + CdotE1) + H);
  float vf0x = (k * Bx);
  float vf0y = (k * By);
  float vf0z = (k * Bz);
  float vf1x = (k * Cx);
  float vf1y = (k * Cy);
  float vf1z = (k * Cz);
  float vn0x = (0.000000 - vf0x);
  float vn0y = (0.000000 - vf0y);
  float vn0z = (0.000000 - vf0z);
  float vn1x = (0.000000 - vf1x);
  float vn1y = (0.000000 - vf1y);
  float vn1z = (0.000000 - vf1z);
  float ve1x = ((vn0x * r00) + (vn1x * r01));
  float ve1y = ((vn0y * r00) + (vn1y * r01));
  float ve1z = ((vn0z * r00) + (vn1z * r01));
  float ve2x = ((vn0x * r10) + (vn1x * r11));
  float ve2y = ((vn0y * r10) + (vn1y * r11));
  float ve2z = ((vn0z * r10) + (vn1z * r11));
  float vr00 = (((vn0x * e1x) + (vn0y * e1y)) + (vn0z * e1z));
  float vr01 = (((vn1x * e1x) + (vn1y * e1y)) + (vn1z * e1z));
  float vr10 = (((vn0x * e2x) + (vn0y * e2y)) + (vn0z * e2z));
  float vr11 = (((vn1x * e2x) + (vn1y * e2y)) + (vn1z * e2z));
  float rn3 = (rn * (rn * rn));
  float trR = (vr00 + vr11);
  float antR = (vr10 - vr01);
  float antR2 = (vr01 - vr10);
  float vxv = (((trR * (b * b)) / rn3) + ((antR * (xv * b)) / rn3));
  float vb_polar = ((0.000000 - ((trR * (xv * b)) / rn3)) + ((antR2 * (xv * xv)) / rn3));
  float va = vxv;
  float vd = vxv;
  float ve2dote2 = (((ve2x * e2x) + (ve2y * e2y)) + (ve2z * e2z));
  float vtx = (((ve2x - (ve2dote2 * e2x)) / d) + (vd * e2x));
  float vty = (((ve2y - (ve2dote2 * e2y)) / d) + (vd * e2y));
  float vtz = (((ve2z - (ve2dote2 * e2z)) / d) + (vd * e2z));
  vf1x = (vf1x + vtx);
  vf1y = (vf1y + vty);
  vf1z = (vf1z + vtz);
  ve1x = (ve1x - (b * vtx));
  ve1y = (ve1y - (b * vty));
  ve1z = (ve1z - (b * vtz));
  float vt_dot_e1 = (((vtx * e1x) + (vty * e1y)) + (vtz * e1z));
  float vb = (vb_polar + (0.000000 - vt_dot_e1));
  ve1x = (ve1x + (vb * f1x));
  ve1y = (ve1y + (vb * f1y));
  ve1z = (ve1z + (vb * f1z));
  vf1x = (vf1x + (vb * e1x));
  vf1y = (vf1y + (vb * e1y));
  vf1z = (vf1z + (vb * e1z));
  float ve1dote1 = (((ve1x * e1x) + (ve1y * e1y)) + (ve1z * e1z));
  vf0x = (vf0x + ((ve1x - (ve1dote1 * e1x)) / a));
  vf0y = (vf0y + ((ve1y - (ve1dote1 * e1y)) / a));
  vf0z = (vf0z + ((ve1z - (ve1dote1 * e1z)) / a));
  vf0x = (vf0x + (va * e1x));
  vf0y = (vf0y + (va * e1y));
  vf0z = (vf0z + (va * e1z));
  v_p[base] = float3((0.000000 - ((w0a * vf0x) + (w0b * vf1x))), (0.000000 - ((w0a * vf0y) + (w0b * vf1y))), (0.000000 - ((w0a * vf0z) + (w0b * vf1z))));
  v_p[(base + 1u)] = float3(((m00 * vf0x) + (m01 * vf1x)), ((m00 * vf0y) + (m01 * vf1y)), ((m00 * vf0z) + (m01 * vf1z)));
  v_p[(base + 2u)] = float3(((m10 * vf0x) + (m11 * vf1x)), ((m10 * vf0y) + (m11 * vf1y)), ((m10 * vf0z) + (m11 * vf1z)));
}"

example : LeanSlang.emit shader = expected := by native_decide
example : shader.entryPointName = "main" := by native_decide

end Cloth.SlangCodegen.TriangleMembraneForceAlBackward
