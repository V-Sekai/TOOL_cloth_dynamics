import LeanSlang

/-!
# `Cloth.SlangCodegen.VbdGatherBendingBackward` — adjoint of vbd_gather_bending (PR-G continued)

Backward kernel for `vbd_gather_bending`. The forward gathers
per-(constraint, role) outputs into per-vertex scratch:

  for v in verts:
    for (c, r) in vertBend-CSR[v]:
      g_v        += bendGrad[4c+r]
      H_v_diag   += bendHessScalar[4c+r]

Bending has K=4 (the dihedral stencil), so the slot index is
`4·c + r` with role ∈ {0,1,2,3}. Adjoint dispatches one thread per
(c, r) slot:

  per slot s = 4c+r:
    v = bendIdx[s]
    v_bendGrad[s]       = v_g[v]
    v_bendHessScalar[s] = v_H_xx[v] + v_H_yy[v] + v_H_zz[v]

Off-diagonal entries of v_H don't flow back through the gather —
the forward only touches diagonals.

Bindings (set 0):

  0  StructuredBuffer<uint>     bendIdx             length = 4·N_bend
  1  StructuredBuffer<float3>   v_g                 length = N_verts (cotangent)
  2  StructuredBuffer<float>    v_H                 length = 6·N_verts (cotangent, sym 3x3)
  3  RWStructuredBuffer<float3> v_bendGrad          length = 4·N_bend (output)
  4  RWStructuredBuffer<float>  v_bendHessScalar    length = 4·N_bend (output)
-/

namespace Cloth.SlangCodegen.VbdGatherBendingBackward

open LeanSlang

private def f3 : SlangType := .vec .float 3
private def u  : SlangType := .scalar .uint
private def f  : SlangType := .scalar .float

private def bnd (n : Nat) (name : String) (t : SlangType) : SlangBinding :=
  { name := name, type := t, semantic := Semantic.none
  , binding := some n, space := some 0 }

private def body : List SlangStmt :=
  [ .declInit u  "slot" (.member (.var "tid") "x")
  , .declInit u  "v"    (.index (.var "bendIdx") (.var "slot"))
  , .declInit u  "hb"   (.bin "*" (.litUint 6) (.var "v"))
  , .declInit f3 "vg"   (.index (.var "v_g") (.var "v"))
  , .declInit f  "vHxx" (.index (.var "v_H") (.var "hb"))
  , .declInit f  "vHyy" (.index (.var "v_H") (.bin "+" (.var "hb") (.litUint 3)))
  , .declInit f  "vHzz" (.index (.var "v_H") (.bin "+" (.var "hb") (.litUint 5)))
  , .assign (.index (.var "v_bendGrad") (.var "slot"))
      (.call "float3"
        [ .member (.var "vg") "x"
        , .member (.var "vg") "y"
        , .member (.var "vg") "z" ])
  , .assign (.index (.var "v_bendHessScalar") (.var "slot"))
      (.bin "+" (.bin "+" (.var "vHxx") (.var "vHyy")) (.var "vHzz"))
  ]

def shader : SlangShaderModule :=
  { globals :=
      [ bnd 0 "bendIdx"          (.roBuf u)
      , bnd 1 "v_g"              (.roBuf f3)
      , bnd 2 "v_H"              (.roBuf f)
      , bnd 3 "v_bendGrad"       (.rwBuf f3)
      , bnd 4 "v_bendHessScalar" (.rwBuf f)
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
StructuredBuffer<uint> bendIdx;
[[vk::binding(1, 0)]]
StructuredBuffer<float3> v_g;
[[vk::binding(2, 0)]]
StructuredBuffer<float> v_H;
[[vk::binding(3, 0)]]
RWStructuredBuffer<float3> v_bendGrad;
[[vk::binding(4, 0)]]
RWStructuredBuffer<float> v_bendHessScalar;

[shader(\"compute\")] [numthreads(64, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
  uint slot = tid.x;
  uint v = bendIdx[slot];
  uint hb = (6u * v);
  float3 vg = v_g[v];
  float vHxx = v_H[hb];
  float vHyy = v_H[(hb + 3u)];
  float vHzz = v_H[(hb + 5u)];
  v_bendGrad[slot] = float3(vg.x, vg.y, vg.z);
  v_bendHessScalar[slot] = ((vHxx + vHyy) + vHzz);
}"

example : LeanSlang.emit shader = expected := by native_decide
example : shader.entryPointName = "main" := by native_decide

end Cloth.SlangCodegen.VbdGatherBendingBackward
