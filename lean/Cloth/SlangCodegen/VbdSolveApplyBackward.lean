import LeanSlang

/-!
# `Cloth.SlangCodegen.VbdSolveApplyBackward` — adjoint of vbd_solve_apply (PR-G start)

Backward (reverse-mode) kernel for `vbd_solve_apply`. Computes the
vector-Jacobian product (VJP) for the per-vertex 3x3 solve.

Forward (per vertex `v`):
  Δx = −H⁻¹ g                     // H sym 3x3, g 3-vec
  x_out = x_in + Δx

For an output cotangent `v_out` (the gradient of the loss w.r.t. x_out),
the chain rule gives:

  ∂x_out/∂x_in     = I             ⇒ v_in = v_out             (identity)
  ∂Δx/∂g           = -H⁻¹          ⇒ v_g  = -H⁻¹ · v_out
  ∂Δx/∂H_ij        = -H⁻¹_·i · Δx_j ⇒ v_H_ij = -(H⁻¹ v_out)_i · Δx_j

For symmetric H storage (6 floats: Hxx, Hxy, Hxz, Hyy, Hyz, Hzz):

  let y = H⁻¹ · v_out   (same 3x3 adjugate solve as the forward kernel)
  v_g = -y
  v_H (full) = -y · Δxᵀ      // 3x3 outer product
  v_H (sym  with off-diag doubled to match d(L)/d(H_off)):
       v_H_xx = -y.x · Δx.x
       v_H_yy = -y.y · Δx.y
       v_H_zz = -y.z · Δx.z
       v_H_xy = -(y.x · Δx.y + y.y · Δx.x)
       v_H_xz = -(y.x · Δx.z + y.z · Δx.x)
       v_H_yz = -(y.y · Δx.z + y.z · Δx.y)

The off-diagonal entries pair (H_ij and H_ji) because the sym storage
counts each off-diagonal once but the function depends on both —
the d/d(H_off) gradient is the sum of the d/d(H_ij) and d/d(H_ji)
gradients. This matches PyTorch's `torch.linalg.solve` adjoint when
H is stored symmetric.

The forward kernel reads positions to add Δx; this backward kernel
takes Δx as input (caller can recompute it as -(H⁻¹ g) or cache the
forward-pass output). Caller-cached avoids redundant 3x3 inversion.

Bindings (set 0):

  0  StructuredBuffer<float3>   v_out      length = N_verts   // cotangent of x_out
  1  StructuredBuffer<float>    hScratch   length = 6 · N_verts  // forward H
  2  StructuredBuffer<float3>   deltaX     length = N_verts   // forward Δx
  3  RWStructuredBuffer<float3> v_g        length = N_verts   // output: ∂L/∂g
  4  RWStructuredBuffer<float>  v_H        length = 6 · N_verts  // output: ∂L/∂H (sym)
-/

namespace Cloth.SlangCodegen.VbdSolveApplyBackward

open LeanSlang

private def f3 : SlangType := .vec .float 3
private def u  : SlangType := .scalar .uint
private def f  : SlangType := .scalar .float

private def bnd (n : Nat) (name : String) (t : SlangType) : SlangBinding :=
  { name := name, type := t, semantic := Semantic.none
  , binding := some n, space := some 0 }

private def body : List SlangStmt :=
  [ .declInit u  "v"   (.member (.var "tid") "x")
  , .declInit u  "hb"  (.bin "*" (.litUint 6) (.var "v"))
  , .declInit f3 "vo"  (.index (.var "v_out") (.var "v"))
  , .declInit f  "Hxx" (.index (.var "hScratch") (.var "hb"))
  , .declInit f  "Hxy" (.index (.var "hScratch") (.bin "+" (.var "hb") (.litUint 1)))
  , .declInit f  "Hxz" (.index (.var "hScratch") (.bin "+" (.var "hb") (.litUint 2)))
  , .declInit f  "Hyy" (.index (.var "hScratch") (.bin "+" (.var "hb") (.litUint 3)))
  , .declInit f  "Hyz" (.index (.var "hScratch") (.bin "+" (.var "hb") (.litUint 4)))
  , .declInit f  "Hzz" (.index (.var "hScratch") (.bin "+" (.var "hb") (.litUint 5)))
  -- adjugate of sym 3x3 H (identical pattern to vbd_solve_apply)
  , .declInit f  "a00"
      (.bin "-" (.bin "*" (.var "Hyy") (.var "Hzz"))
                (.bin "*" (.var "Hyz") (.var "Hyz")))
  , .declInit f  "a11"
      (.bin "-" (.bin "*" (.var "Hxx") (.var "Hzz"))
                (.bin "*" (.var "Hxz") (.var "Hxz")))
  , .declInit f  "a22"
      (.bin "-" (.bin "*" (.var "Hxx") (.var "Hyy"))
                (.bin "*" (.var "Hxy") (.var "Hxy")))
  , .declInit f  "a01"
      (.bin "-" (.bin "*" (.var "Hxz") (.var "Hyz"))
                (.bin "*" (.var "Hxy") (.var "Hzz")))
  , .declInit f  "a02"
      (.bin "-" (.bin "*" (.var "Hxy") (.var "Hyz"))
                (.bin "*" (.var "Hxz") (.var "Hyy")))
  , .declInit f  "a12"
      (.bin "-" (.bin "*" (.var "Hxy") (.var "Hxz"))
                (.bin "*" (.var "Hxx") (.var "Hyz")))
  , .declInit f  "det"
      (.bin "+"
        (.bin "+" (.bin "*" (.var "Hxx") (.var "a00"))
                  (.bin "*" (.var "Hxy") (.var "a01")))
        (.bin "*" (.var "Hxz") (.var "a02")))
  , .declInit f  "invDet" (.bin "/" (.litFloat 1.0) (.var "det"))
  -- y = H⁻¹ v_out  (same adjugate as forward, applied to v_out instead of g)
  , .declInit f  "ux" (.member (.var "vo") "x")
  , .declInit f  "uy" (.member (.var "vo") "y")
  , .declInit f  "uz" (.member (.var "vo") "z")
  , .declInit f  "yx"
      (.bin "*" (.var "invDet")
                (.bin "+"
                  (.bin "+" (.bin "*" (.var "a00") (.var "ux"))
                            (.bin "*" (.var "a01") (.var "uy")))
                  (.bin "*" (.var "a02") (.var "uz"))))
  , .declInit f  "yy"
      (.bin "*" (.var "invDet")
                (.bin "+"
                  (.bin "+" (.bin "*" (.var "a01") (.var "ux"))
                            (.bin "*" (.var "a11") (.var "uy")))
                  (.bin "*" (.var "a12") (.var "uz"))))
  , .declInit f  "yz"
      (.bin "*" (.var "invDet")
                (.bin "+"
                  (.bin "+" (.bin "*" (.var "a02") (.var "ux"))
                            (.bin "*" (.var "a12") (.var "uy")))
                  (.bin "*" (.var "a22") (.var "uz"))))
  -- v_g = -y
  , .assign (.index (.var "v_g") (.var "v"))
      (.call "float3"
        [ .bin "-" (.litFloat 0.0) (.var "yx")
        , .bin "-" (.litFloat 0.0) (.var "yy")
        , .bin "-" (.litFloat 0.0) (.var "yz") ])
  -- Read Δx
  , .declInit f3 "dx" (.index (.var "deltaX") (.var "v"))
  , .declInit f  "dxx" (.member (.var "dx") "x")
  , .declInit f  "dxy" (.member (.var "dx") "y")
  , .declInit f  "dxz" (.member (.var "dx") "z")
  -- v_H sym = -(y · Δxᵀ) symmetrized:
  --   diag: v_H_ii = -y_i · Δx_i
  --   off:  v_H_ij = -(y_i · Δx_j + y_j · Δx_i)
  , .assign (.index (.var "v_H") (.var "hb"))
      (.bin "-" (.litFloat 0.0) (.bin "*" (.var "yx") (.var "dxx")))
  , .assign (.index (.var "v_H") (.bin "+" (.var "hb") (.litUint 1)))
      (.bin "-" (.litFloat 0.0)
        (.bin "+" (.bin "*" (.var "yx") (.var "dxy"))
                  (.bin "*" (.var "yy") (.var "dxx"))))
  , .assign (.index (.var "v_H") (.bin "+" (.var "hb") (.litUint 2)))
      (.bin "-" (.litFloat 0.0)
        (.bin "+" (.bin "*" (.var "yx") (.var "dxz"))
                  (.bin "*" (.var "yz") (.var "dxx"))))
  , .assign (.index (.var "v_H") (.bin "+" (.var "hb") (.litUint 3)))
      (.bin "-" (.litFloat 0.0) (.bin "*" (.var "yy") (.var "dxy")))
  , .assign (.index (.var "v_H") (.bin "+" (.var "hb") (.litUint 4)))
      (.bin "-" (.litFloat 0.0)
        (.bin "+" (.bin "*" (.var "yy") (.var "dxz"))
                  (.bin "*" (.var "yz") (.var "dxy"))))
  , .assign (.index (.var "v_H") (.bin "+" (.var "hb") (.litUint 5)))
      (.bin "-" (.litFloat 0.0) (.bin "*" (.var "yz") (.var "dxz")))
  ]

def shader : SlangShaderModule :=
  { globals :=
      [ bnd 0 "v_out"    (.roBuf f3)
      , bnd 1 "hScratch" (.roBuf f)
      , bnd 2 "deltaX"   (.roBuf f3)
      , bnd 3 "v_g"      (.rwBuf f3)
      , bnd 4 "v_H"      (.rwBuf f)
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
StructuredBuffer<float3> v_out;
[[vk::binding(1, 0)]]
StructuredBuffer<float> hScratch;
[[vk::binding(2, 0)]]
StructuredBuffer<float3> deltaX;
[[vk::binding(3, 0)]]
RWStructuredBuffer<float3> v_g;
[[vk::binding(4, 0)]]
RWStructuredBuffer<float> v_H;

[shader(\"compute\")] [numthreads(64, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
  uint v = tid.x;
  uint hb = (6u * v);
  float3 vo = v_out[v];
  float Hxx = hScratch[hb];
  float Hxy = hScratch[(hb + 1u)];
  float Hxz = hScratch[(hb + 2u)];
  float Hyy = hScratch[(hb + 3u)];
  float Hyz = hScratch[(hb + 4u)];
  float Hzz = hScratch[(hb + 5u)];
  float a00 = ((Hyy * Hzz) - (Hyz * Hyz));
  float a11 = ((Hxx * Hzz) - (Hxz * Hxz));
  float a22 = ((Hxx * Hyy) - (Hxy * Hxy));
  float a01 = ((Hxz * Hyz) - (Hxy * Hzz));
  float a02 = ((Hxy * Hyz) - (Hxz * Hyy));
  float a12 = ((Hxy * Hxz) - (Hxx * Hyz));
  float det = (((Hxx * a00) + (Hxy * a01)) + (Hxz * a02));
  float invDet = (1.000000 / det);
  float ux = vo.x;
  float uy = vo.y;
  float uz = vo.z;
  float yx = (invDet * (((a00 * ux) + (a01 * uy)) + (a02 * uz)));
  float yy = (invDet * (((a01 * ux) + (a11 * uy)) + (a12 * uz)));
  float yz = (invDet * (((a02 * ux) + (a12 * uy)) + (a22 * uz)));
  v_g[v] = float3((0.000000 - yx), (0.000000 - yy), (0.000000 - yz));
  float3 dx = deltaX[v];
  float dxx = dx.x;
  float dxy = dx.y;
  float dxz = dx.z;
  v_H[hb] = (0.000000 - (yx * dxx));
  v_H[(hb + 1u)] = (0.000000 - ((yx * dxy) + (yy * dxx)));
  v_H[(hb + 2u)] = (0.000000 - ((yx * dxz) + (yz * dxx)));
  v_H[(hb + 3u)] = (0.000000 - (yy * dxy));
  v_H[(hb + 4u)] = (0.000000 - ((yy * dxz) + (yz * dxy)));
  v_H[(hb + 5u)] = (0.000000 - (yz * dxz));
}"

example : LeanSlang.emit shader = expected := by native_decide
example : shader.entryPointName = "main" := by native_decide

end Cloth.SlangCodegen.VbdSolveApplyBackward
