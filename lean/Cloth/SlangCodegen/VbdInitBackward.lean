import LeanSlang

/-!
# `Cloth.SlangCodegen.VbdInitBackward` — adjoint of vbd_init (PR-G continued)

Backward kernel for `vbd_init`. The forward writes per-vertex
inertial seed values:

  w = m_v · invHSq                            (scalar per vertex)
  g_v = w · (x_v − y_v)                       (3-vec into gScratch)
  H_v = diag(w, w, w)                         (sym 3x3 into hScratch:
                                                Hxx=Hyy=Hzz=w, off=0)

Output cotangents v_g (3-vec) and v_H (sym 3x3 stored as 6 floats:
Hxx, Hxy, Hxz, Hyy, Hyz, Hzz). The adjoint is pure multiplication —
no inversion, no spatial coupling, fully parallel per vertex:

  v_x_v   = (m_v · invHSq) · v_g_v
  v_y_v   = -(m_v · invHSq) · v_g_v
  v_mass  = invHSq · ((x_v − y_v) · v_g_v   + v_H_xx + v_H_yy + v_H_zz)

The mass gradient sums:
  - dot product of x_v−y_v with v_g_v (linear g term)
  - trace of v_H (diagonal-only H means the off-diagonal cotangents
    don't contribute to v_mass; we still read them but they collapse)

This is the easiest backward kernel in the AVBD chain — no rotation
projection, no constraint adjacency. It pairs with VbdSolveApplyBackward
(#106) which handles the per-vertex H⁻¹ adjoint.

Bindings (set 0):

  0  StructuredBuffer<float3>   positions   length = N_verts   (forward x)
  1  StructuredBuffer<float3>   predicted   length = N_verts   (forward y)
  2  StructuredBuffer<float>    mass        length = N_verts
  3  StructuredBuffer<float3>   v_g         length = N_verts   (cotangent)
  4  StructuredBuffer<float>    v_H         length = 6·N_verts (cotangent)
  5  RWStructuredBuffer<float3> v_x         length = N_verts   (output)
  6  RWStructuredBuffer<float3> v_y         length = N_verts   (output)
  7  RWStructuredBuffer<float>  v_mass      length = N_verts   (output)
  8  ConstantBuffer<VbdInitBackwardParams> { invHSquared }
-/

namespace Cloth.SlangCodegen.VbdInitBackward

open LeanSlang

private def f3 : SlangType := .vec .float 3
private def u  : SlangType := .scalar .uint
private def f  : SlangType := .scalar .float

private def body : List SlangStmt :=
  [ .declInit u  "v"   (.member (.var "tid") "x")
  , .declInit u  "hb"  (.bin "*" (.litUint 6) (.var "v"))
  , .declInit f3 "x"   (.index (.var "positions") (.var "v"))
  , .declInit f3 "y"   (.index (.var "predicted") (.var "v"))
  , .declInit f  "m"   (.index (.var "mass") (.var "v"))
  , .declInit f3 "vg"  (.index (.var "v_g") (.var "v"))
  , .declInit f  "w"   (.bin "*" (.var "m") (.member (.var "params") "invHSquared"))
  -- v_x = w · v_g
  , .assign (.index (.var "v_x") (.var "v"))
      (.call "float3"
        [ .bin "*" (.var "w") (.member (.var "vg") "x")
        , .bin "*" (.var "w") (.member (.var "vg") "y")
        , .bin "*" (.var "w") (.member (.var "vg") "z") ])
  -- v_y = -w · v_g
  , .assign (.index (.var "v_y") (.var "v"))
      (.call "float3"
        [ .bin "*" (.bin "-" (.litFloat 0.0) (.var "w")) (.member (.var "vg") "x")
        , .bin "*" (.bin "-" (.litFloat 0.0) (.var "w")) (.member (.var "vg") "y")
        , .bin "*" (.bin "-" (.litFloat 0.0) (.var "w")) (.member (.var "vg") "z") ])
  -- v_mass = invHSq · ((x−y) · v_g + Hxx + Hyy + Hzz)
  , .declInit f "dot_dx_vg"
      (.bin "+"
        (.bin "+"
          (.bin "*" (.bin "-" (.member (.var "x") "x") (.member (.var "y") "x"))
                    (.member (.var "vg") "x"))
          (.bin "*" (.bin "-" (.member (.var "x") "y") (.member (.var "y") "y"))
                    (.member (.var "vg") "y")))
        (.bin "*" (.bin "-" (.member (.var "x") "z") (.member (.var "y") "z"))
                  (.member (.var "vg") "z")))
  , .declInit f "vHxx" (.index (.var "v_H") (.var "hb"))
  , .declInit f "vHyy" (.index (.var "v_H") (.bin "+" (.var "hb") (.litUint 3)))
  , .declInit f "vHzz" (.index (.var "v_H") (.bin "+" (.var "hb") (.litUint 5)))
  , .declInit f "trH"
      (.bin "+" (.bin "+" (.var "vHxx") (.var "vHyy")) (.var "vHzz"))
  , .assign (.index (.var "v_mass") (.var "v"))
      (.bin "*" (.member (.var "params") "invHSquared")
                (.bin "+" (.var "dot_dx_vg") (.var "trH")))
  ]

def shader : SlangShaderModule :=
  { structs :=
      [ { name := "VbdInitBackwardParams"
        , fields := [⟨"invHSquared", f, Semantic.none, none, none, .qIn⟩] } ]
  , globals :=
      [ ⟨"positions", .roBuf f3,                       Semantic.none, some 0, some 0, .qIn⟩
      , ⟨"predicted", .roBuf f3,                       Semantic.none, some 1, some 0, .qIn⟩
      , ⟨"mass",      .roBuf f,                        Semantic.none, some 2, some 0, .qIn⟩
      , ⟨"v_g",       .roBuf f3,                       Semantic.none, some 3, some 0, .qIn⟩
      , ⟨"v_H",       .roBuf f,                        Semantic.none, some 4, some 0, .qIn⟩
      , ⟨"v_x",       .rwBuf f3,                       Semantic.none, some 5, some 0, .qIn⟩
      , ⟨"v_y",       .rwBuf f3,                       Semantic.none, some 6, some 0, .qIn⟩
      , ⟨"v_mass",    .rwBuf f,                        Semantic.none, some 7, some 0, .qIn⟩
      , ⟨"params",    .const "VbdInitBackwardParams", Semantic.none, some 8, some 0, .qIn⟩
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
"struct VbdInitBackwardParams {
  float invHSquared;
};

[[vk::binding(0, 0)]]
StructuredBuffer<float3> positions;
[[vk::binding(1, 0)]]
StructuredBuffer<float3> predicted;
[[vk::binding(2, 0)]]
StructuredBuffer<float> mass;
[[vk::binding(3, 0)]]
StructuredBuffer<float3> v_g;
[[vk::binding(4, 0)]]
StructuredBuffer<float> v_H;
[[vk::binding(5, 0)]]
RWStructuredBuffer<float3> v_x;
[[vk::binding(6, 0)]]
RWStructuredBuffer<float3> v_y;
[[vk::binding(7, 0)]]
RWStructuredBuffer<float> v_mass;
[[vk::binding(8, 0)]]
ConstantBuffer<VbdInitBackwardParams> params;

[shader(\"compute\")] [numthreads(64, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
  uint v = tid.x;
  uint hb = (6u * v);
  float3 x = positions[v];
  float3 y = predicted[v];
  float m = mass[v];
  float3 vg = v_g[v];
  float w = (m * params.invHSquared);
  v_x[v] = float3((w * vg.x), (w * vg.y), (w * vg.z));
  v_y[v] = float3(((0.000000 - w) * vg.x), ((0.000000 - w) * vg.y), ((0.000000 - w) * vg.z));
  float dot_dx_vg = ((((x.x - y.x) * vg.x) + ((x.y - y.y) * vg.y)) + ((x.z - y.z) * vg.z));
  float vHxx = v_H[hb];
  float vHyy = v_H[(hb + 3u)];
  float vHzz = v_H[(hb + 5u)];
  float trH = ((vHxx + vHyy) + vHzz);
  v_mass[v] = (params.invHSquared * (dot_dx_vg + trH));
}"

example : LeanSlang.emit shader = expected := by native_decide
example : shader.entryPointName = "main" := by native_decide

end Cloth.SlangCodegen.VbdInitBackward
