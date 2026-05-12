import LeanSlang

/-!
# `Cloth.SlangCodegen.VbdGatherSpringBackward` — adjoint of vbd_gather_spring (PR-G continued)

Backward kernel for `vbd_gather_spring`. The forward gathers
per-spring outputs into per-vertex scratch with a sign flip on
endpoint b:

  for v in verts:
    for (c, r) in vertSpring-CSR[v]:
      sign = 1 − 2·r                            -- +1 (p1) or −1 (p2)
      g_v        += sign · springGradA[c]
      H_v_diag   += springHessScalar[c]         -- no sign flip

Springs have K=2 with sign flip on b. The adjoint dispatches one
thread per spring c. Each thread reads both endpoint cotangents and
fans them out with the correct signs:

  per c:
    p1 = springP1Idx[c],  p2 = springP2Idx[c]
    vg_p1 = v_g[p1],      vg_p2 = v_g[p2]
    v_springGradA[c] = vg_p1 − vg_p2           -- +1·p1  +  (−1)·p2

  Hessian fan-out: the forward writes springHessScalar[c] into the
  three diagonal entries of BOTH endpoints (no sign flip on the
  Hessian), so the adjoint sums the trace contributions from both:

    v_springHess[c] = (v_H_xx[p1] + v_H_yy[p1] + v_H_zz[p1])
                    + (v_H_xx[p2] + v_H_yy[p2] + v_H_zz[p2])

Off-diagonal entries of v_H don't flow back through the gather —
the forward only touches diagonals.

Bindings (set 0):

  0  StructuredBuffer<uint>     springP1Idx       length = N_springs
  1  StructuredBuffer<uint>     springP2Idx       length = N_springs
  2  StructuredBuffer<float3>   v_g               length = N_verts (cotangent)
  3  StructuredBuffer<float>    v_H               length = 6·N_verts (cotangent, sym 3x3)
  4  RWStructuredBuffer<float3> v_springGradA     length = N_springs (output)
  5  RWStructuredBuffer<float>  v_springHess      length = N_springs (output)
-/

namespace Cloth.SlangCodegen.VbdGatherSpringBackward

open LeanSlang

private def f3 : SlangType := .vec .float 3
private def u  : SlangType := .scalar .uint
private def f  : SlangType := .scalar .float

private def bnd (n : Nat) (name : String) (t : SlangType) : SlangBinding :=
  { name := name, type := t, semantic := Semantic.none
  , binding := some n, space := some 0 }

private def body : List SlangStmt :=
  [ .declInit u  "c"      (.member (.var "tid") "x")
  , .declInit u  "p1"     (.index (.var "springP1Idx") (.var "c"))
  , .declInit u  "p2"     (.index (.var "springP2Idx") (.var "c"))
  , .declInit u  "hb1"    (.bin "*" (.litUint 6) (.var "p1"))
  , .declInit u  "hb2"    (.bin "*" (.litUint 6) (.var "p2"))
  , .declInit f3 "vg1"    (.index (.var "v_g") (.var "p1"))
  , .declInit f3 "vg2"    (.index (.var "v_g") (.var "p2"))
  , .declInit f  "vH1xx"  (.index (.var "v_H") (.var "hb1"))
  , .declInit f  "vH1yy"  (.index (.var "v_H") (.bin "+" (.var "hb1") (.litUint 3)))
  , .declInit f  "vH1zz"  (.index (.var "v_H") (.bin "+" (.var "hb1") (.litUint 5)))
  , .declInit f  "vH2xx"  (.index (.var "v_H") (.var "hb2"))
  , .declInit f  "vH2yy"  (.index (.var "v_H") (.bin "+" (.var "hb2") (.litUint 3)))
  , .declInit f  "vH2zz"  (.index (.var "v_H") (.bin "+" (.var "hb2") (.litUint 5)))
  , .assign (.index (.var "v_springGradA") (.var "c"))
      (.call "float3"
        [ .bin "-" (.member (.var "vg1") "x") (.member (.var "vg2") "x")
        , .bin "-" (.member (.var "vg1") "y") (.member (.var "vg2") "y")
        , .bin "-" (.member (.var "vg1") "z") (.member (.var "vg2") "z") ])
  , .assign (.index (.var "v_springHess") (.var "c"))
      (.bin "+"
        (.bin "+" (.bin "+" (.var "vH1xx") (.var "vH1yy")) (.var "vH1zz"))
        (.bin "+" (.bin "+" (.var "vH2xx") (.var "vH2yy")) (.var "vH2zz")))
  ]

def shader : SlangShaderModule :=
  { globals :=
      [ bnd 0 "springP1Idx"   (.roBuf u)
      , bnd 1 "springP2Idx"   (.roBuf u)
      , bnd 2 "v_g"           (.roBuf f3)
      , bnd 3 "v_H"           (.roBuf f)
      , bnd 4 "v_springGradA" (.rwBuf f3)
      , bnd 5 "v_springHess"  (.rwBuf f)
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
StructuredBuffer<uint> springP1Idx;
[[vk::binding(1, 0)]]
StructuredBuffer<uint> springP2Idx;
[[vk::binding(2, 0)]]
StructuredBuffer<float3> v_g;
[[vk::binding(3, 0)]]
StructuredBuffer<float> v_H;
[[vk::binding(4, 0)]]
RWStructuredBuffer<float3> v_springGradA;
[[vk::binding(5, 0)]]
RWStructuredBuffer<float> v_springHess;

[shader(\"compute\")] [numthreads(64, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
  uint c = tid.x;
  uint p1 = springP1Idx[c];
  uint p2 = springP2Idx[c];
  uint hb1 = (6u * p1);
  uint hb2 = (6u * p2);
  float3 vg1 = v_g[p1];
  float3 vg2 = v_g[p2];
  float vH1xx = v_H[hb1];
  float vH1yy = v_H[(hb1 + 3u)];
  float vH1zz = v_H[(hb1 + 5u)];
  float vH2xx = v_H[hb2];
  float vH2yy = v_H[(hb2 + 3u)];
  float vH2zz = v_H[(hb2 + 5u)];
  v_springGradA[c] = float3((vg1.x - vg2.x), (vg1.y - vg2.y), (vg1.z - vg2.z));
  v_springHess[c] = (((vH1xx + vH1yy) + vH1zz) + ((vH2xx + vH2yy) + vH2zz));
}"

example : LeanSlang.emit shader = expected := by native_decide
example : shader.entryPointName = "main" := by native_decide

end Cloth.SlangCodegen.VbdGatherSpringBackward
