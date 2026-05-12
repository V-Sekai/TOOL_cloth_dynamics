import LeanSlang

/-!
# `Cloth.SlangCodegen.VbdGatherAttachmentBackward` — adjoint of vbd_gather_attachment (PR-G continued)

Backward kernel for `vbd_gather_attachment`. The forward gathers
per-attachment outputs into per-vertex scratch:

  for v in verts:
    for c in vertAttach-CSR[v]:
      g_v        += attachGradV[c]
      H_v_diag   += attachHessScalar[c]      (added to all 3 diagonals)

Attachments have K=1 vertex per constraint and no role flip, so the
adjoint is a straight fan-out keyed by the constraint side:

  per c:
    v = attachVertIdx[c]
    v_attachGradV[c]      = v_g[v]
    v_attachHessScalar[c] = v_H_xx[v] + v_H_yy[v] + v_H_zz[v]

Off-diagonal entries of v_H do not flow back — the forward only
touches diagonals.

Bindings (set 0):

  0  StructuredBuffer<uint>     attachVertIdx         length = N_attach
  1  StructuredBuffer<float3>   v_g                   length = N_verts (cotangent)
  2  StructuredBuffer<float>    v_H                   length = 6·N_verts (cotangent, sym 3x3)
  3  RWStructuredBuffer<float3> v_attachGradV         length = N_attach (output)
  4  RWStructuredBuffer<float>  v_attachHessScalar    length = N_attach (output)
-/

namespace Cloth.SlangCodegen.VbdGatherAttachmentBackward

open LeanSlang

private def f3 : SlangType := .vec .float 3
private def u  : SlangType := .scalar .uint
private def f  : SlangType := .scalar .float

private def bnd (n : Nat) (name : String) (t : SlangType) : SlangBinding :=
  { name := name, type := t, semantic := Semantic.none
  , binding := some n, space := some 0 }

private def body : List SlangStmt :=
  [ .declInit u  "c"    (.member (.var "tid") "x")
  , .declInit u  "v"    (.index (.var "attachVertIdx") (.var "c"))
  , .declInit u  "hb"   (.bin "*" (.litUint 6) (.var "v"))
  , .declInit f3 "vg"   (.index (.var "v_g") (.var "v"))
  , .declInit f  "vHxx" (.index (.var "v_H") (.var "hb"))
  , .declInit f  "vHyy" (.index (.var "v_H") (.bin "+" (.var "hb") (.litUint 3)))
  , .declInit f  "vHzz" (.index (.var "v_H") (.bin "+" (.var "hb") (.litUint 5)))
  , .assign (.index (.var "v_attachGradV") (.var "c"))
      (.call "float3"
        [ .member (.var "vg") "x"
        , .member (.var "vg") "y"
        , .member (.var "vg") "z" ])
  , .assign (.index (.var "v_attachHessScalar") (.var "c"))
      (.bin "+" (.bin "+" (.var "vHxx") (.var "vHyy")) (.var "vHzz"))
  ]

def shader : SlangShaderModule :=
  { globals :=
      [ bnd 0 "attachVertIdx"      (.roBuf u)
      , bnd 1 "v_g"                (.roBuf f3)
      , bnd 2 "v_H"                (.roBuf f)
      , bnd 3 "v_attachGradV"      (.rwBuf f3)
      , bnd 4 "v_attachHessScalar" (.rwBuf f)
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
StructuredBuffer<uint> attachVertIdx;
[[vk::binding(1, 0)]]
StructuredBuffer<float3> v_g;
[[vk::binding(2, 0)]]
StructuredBuffer<float> v_H;
[[vk::binding(3, 0)]]
RWStructuredBuffer<float3> v_attachGradV;
[[vk::binding(4, 0)]]
RWStructuredBuffer<float> v_attachHessScalar;

[shader(\"compute\")] [numthreads(64, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
  uint c = tid.x;
  uint v = attachVertIdx[c];
  uint hb = (6u * v);
  float3 vg = v_g[v];
  float vHxx = v_H[hb];
  float vHyy = v_H[(hb + 3u)];
  float vHzz = v_H[(hb + 5u)];
  v_attachGradV[c] = float3(vg.x, vg.y, vg.z);
  v_attachHessScalar[c] = ((vHxx + vHyy) + vHzz);
}"

example : LeanSlang.emit shader = expected := by native_decide
example : shader.entryPointName = "main" := by native_decide

end Cloth.SlangCodegen.VbdGatherAttachmentBackward
