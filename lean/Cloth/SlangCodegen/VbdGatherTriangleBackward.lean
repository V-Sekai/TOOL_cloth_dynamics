import LeanSlang

/-!
# `Cloth.SlangCodegen.VbdGatherTriangleBackward` — adjoint of vbd_gather_triangle (PR-G continued)

Backward kernel for `vbd_gather_triangle`. The forward gathers
per-(constraint, role) outputs into per-vertex scratch:

  for v in verts:
    for (c, r) in vertTri-CSR[v]:
      g_v        += triGrad[3c+r]               (3-vec accumulation)
      H_v_diag   += triHessScalar[3c+r]         (added to all 3 diagonals)

The adjoint of an additive accumulation is fan-out: each (c, r)
gets the cotangent of the vertex it contributed to. Indexed by
the constraint-side (3·nTri threads, role implicit by thread id):

  per (c, r):
    v = triIdx[3c+r]
    v_triGrad[3c+r] = v_g[v]                                    (3-vec)
    v_triHessScalar[3c+r] = v_H_xx[v] + v_H_yy[v] + v_H_zz[v]   (scalar
                          fan-out from the three diagonal entries hess
                          was written to in the forward)

Off-diagonal entries of v_H don't flow back through the gather —
the forward only touches diagonals. The validator's expected
matches this exactly.

Bindings (set 0):

  0  StructuredBuffer<uint>     triIdx              length = 3·N_tri
  1  StructuredBuffer<float3>   v_g                 length = N_verts (cotangent)
  2  StructuredBuffer<float>    v_H                 length = 6·N_verts (cotangent, sym 3x3)
  3  RWStructuredBuffer<float3> v_triGrad           length = 3·N_tri (output)
  4  RWStructuredBuffer<float>  v_triHessScalar     length = 3·N_tri (output)
-/

namespace Cloth.SlangCodegen.VbdGatherTriangleBackward

open LeanSlang

private def f3 : SlangType := .vec .float 3
private def u  : SlangType := .scalar .uint
private def f  : SlangType := .scalar .float

private def bnd (n : Nat) (name : String) (t : SlangType) : SlangBinding :=
  { name := name, type := t, semantic := Semantic.none
  , binding := some n, space := some 0 }

private def body : List SlangStmt :=
  [ .declInit u  "slot" (.member (.var "tid") "x")
  , .declInit u  "v"    (.index (.var "triIdx") (.var "slot"))
  , .declInit u  "hb"   (.bin "*" (.litUint 6) (.var "v"))
  , .declInit f3 "vg"   (.index (.var "v_g") (.var "v"))
  , .declInit f  "vHxx" (.index (.var "v_H") (.var "hb"))
  , .declInit f  "vHyy" (.index (.var "v_H") (.bin "+" (.var "hb") (.litUint 3)))
  , .declInit f  "vHzz" (.index (.var "v_H") (.bin "+" (.var "hb") (.litUint 5)))
  , .assign (.index (.var "v_triGrad") (.var "slot"))
      (.call "float3"
        [ .member (.var "vg") "x"
        , .member (.var "vg") "y"
        , .member (.var "vg") "z" ])
  , .assign (.index (.var "v_triHessScalar") (.var "slot"))
      (.bin "+" (.bin "+" (.var "vHxx") (.var "vHyy")) (.var "vHzz"))
  ]

def shader : SlangShaderModule :=
  { globals :=
      [ bnd 0 "triIdx"          (.roBuf u)
      , bnd 1 "v_g"             (.roBuf f3)
      , bnd 2 "v_H"             (.roBuf f)
      , bnd 3 "v_triGrad"       (.rwBuf f3)
      , bnd 4 "v_triHessScalar" (.rwBuf f)
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
StructuredBuffer<uint> triIdx;
[[vk::binding(1, 0)]]
StructuredBuffer<float3> v_g;
[[vk::binding(2, 0)]]
StructuredBuffer<float> v_H;
[[vk::binding(3, 0)]]
RWStructuredBuffer<float3> v_triGrad;
[[vk::binding(4, 0)]]
RWStructuredBuffer<float> v_triHessScalar;

[shader(\"compute\")] [numthreads(64, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
  uint slot = tid.x;
  uint v = triIdx[slot];
  uint hb = (6u * v);
  float3 vg = v_g[v];
  float vHxx = v_H[hb];
  float vHyy = v_H[(hb + 3u)];
  float vHzz = v_H[(hb + 5u)];
  v_triGrad[slot] = float3(vg.x, vg.y, vg.z);
  v_triHessScalar[slot] = ((vHxx + vHyy) + vHzz);
}"

example : LeanSlang.emit shader = expected := by native_decide
example : shader.entryPointName = "main" := by native_decide

end Cloth.SlangCodegen.VbdGatherTriangleBackward
