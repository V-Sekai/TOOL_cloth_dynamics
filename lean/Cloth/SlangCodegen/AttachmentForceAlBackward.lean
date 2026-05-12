import LeanSlang

/-!
# `Cloth.SlangCodegen.AttachmentForceAlBackward` — adjoint of attachment_force_al (PR-G / CHI-13)

Reverse-mode VJP for the AL-augmented attachment force kernel. Forward
per attachment c attached to vertex v = vertIdx[c]:

  C(x_v)        = x_v − fixedPos[c]
  gradV[c]      = k_c · C(x_v) + λ_c           // 3-vec
  hessScalar[c] = k_c                          // diag-scalar Hessian

The function is linear in every input (the only nonlinearity λ adds
is the affine shift). All partials are constants — no closest-rotation
projection, no eigen-decomp, no special-casing.

Per-attachment thread (one per c). Adjoint:

  v_p_v       += k_c · v_gradV[c]              // scatter into positions
  v_fixedPos[c] = -k_c · v_gradV[c]
  v_lambda[c]   = v_gradV[c]                   // identity through λ
  v_k_c         = C(x_v) · v_gradV[c] + v_hessScalar[c]

For the dress and the existing demos each attachment binds a unique
vertex (1:1 incidence), so `v_p_v` writes have no contention and we
emit a plain store. If two attachments ever share a vertex, the
host needs to either dispatch atomic-add via a vert-side gather kernel
or pre-sort attachments to colors — neither is currently needed.

Bindings (set 0):

  0  StructuredBuffer<float3>   positions     length = N_verts
  1  StructuredBuffer<uint>     vertIdx       length = N_attach
  2  StructuredBuffer<float3>   fixedPos      length = N_attach
  3  StructuredBuffer<float>    stiffness     length = N_attach
  4  StructuredBuffer<float3>   v_gradV       length = N_attach   (cotangent)
  5  StructuredBuffer<float>    v_hessScalar  length = N_attach   (cotangent)
  6  RWStructuredBuffer<float3> v_positions   length = N_verts    (∂L/∂x_v)
  7  RWStructuredBuffer<float3> v_fixedPos    length = N_attach   (∂L/∂anchor)
  8  RWStructuredBuffer<float3> v_lambda      length = N_attach   (∂L/∂λ)
  9  RWStructuredBuffer<float>  v_stiffness   length = N_attach   (∂L/∂k)
-/

namespace Cloth.SlangCodegen.AttachmentForceAlBackward

open LeanSlang

private def f3 : SlangType := .vec .float 3
private def u  : SlangType := .scalar .uint
private def f  : SlangType := .scalar .float

private def bnd (n : Nat) (name : String) (t : SlangType) : SlangBinding :=
  { name := name, type := t, semantic := Semantic.none
  , binding := some n, space := some 0 }

private def body : List SlangStmt :=
  [ .declInit u  "c"      (.member (.var "tid") "x")
  , .declInit u  "v"      (.index (.var "vertIdx") (.var "c"))
  , .declInit f3 "p"      (.index (.var "positions") (.var "v"))
  , .declInit f3 "fp"     (.index (.var "fixedPos") (.var "c"))
  , .declInit f  "k"      (.index (.var "stiffness") (.var "c"))
  , .declInit f3 "vG"     (.index (.var "v_gradV") (.var "c"))
  , .declInit f  "vH"     (.index (.var "v_hessScalar") (.var "c"))
  -- v_p_v = k · v_gradV (full write — caller guarantees attachments
  --        do not share vertices on the demos this kernel is built
  --        for; see header).
  , .assign (.index (.var "v_positions") (.var "v"))
      (.call "float3"
        [ .bin "*" (.var "k") (.member (.var "vG") "x")
        , .bin "*" (.var "k") (.member (.var "vG") "y")
        , .bin "*" (.var "k") (.member (.var "vG") "z") ])
  -- v_fixedPos = -k · v_gradV
  , .assign (.index (.var "v_fixedPos") (.var "c"))
      (.call "float3"
        [ .bin "*" (.bin "-" (.litFloat 0.0) (.var "k")) (.member (.var "vG") "x")
        , .bin "*" (.bin "-" (.litFloat 0.0) (.var "k")) (.member (.var "vG") "y")
        , .bin "*" (.bin "-" (.litFloat 0.0) (.var "k")) (.member (.var "vG") "z") ])
  -- v_lambda = v_gradV (identity)
  , .assign (.index (.var "v_lambda") (.var "c"))
      (.call "float3"
        [ .member (.var "vG") "x"
        , .member (.var "vG") "y"
        , .member (.var "vG") "z" ])
  -- v_k = C(x_v) · v_gradV + v_hessScalar
  , .declInit f "dotCv"
      (.bin "+"
        (.bin "+"
          (.bin "*" (.bin "-" (.member (.var "p") "x") (.member (.var "fp") "x"))
                    (.member (.var "vG") "x"))
          (.bin "*" (.bin "-" (.member (.var "p") "y") (.member (.var "fp") "y"))
                    (.member (.var "vG") "y")))
        (.bin "*" (.bin "-" (.member (.var "p") "z") (.member (.var "fp") "z"))
                  (.member (.var "vG") "z")))
  , .assign (.index (.var "v_stiffness") (.var "c"))
      (.bin "+" (.var "dotCv") (.var "vH"))
  ]

def shader : SlangShaderModule :=
  { globals :=
      [ bnd 0 "positions"     (.roBuf f3)
      , bnd 1 "vertIdx"       (.roBuf u)
      , bnd 2 "fixedPos"      (.roBuf f3)
      , bnd 3 "stiffness"     (.roBuf f)
      , bnd 4 "v_gradV"       (.roBuf f3)
      , bnd 5 "v_hessScalar"  (.roBuf f)
      , bnd 6 "v_positions"   (.rwBuf f3)
      , bnd 7 "v_fixedPos"    (.rwBuf f3)
      , bnd 8 "v_lambda"      (.rwBuf f3)
      , bnd 9 "v_stiffness"   (.rwBuf f)
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
StructuredBuffer<uint> vertIdx;
[[vk::binding(2, 0)]]
StructuredBuffer<float3> fixedPos;
[[vk::binding(3, 0)]]
StructuredBuffer<float> stiffness;
[[vk::binding(4, 0)]]
StructuredBuffer<float3> v_gradV;
[[vk::binding(5, 0)]]
StructuredBuffer<float> v_hessScalar;
[[vk::binding(6, 0)]]
RWStructuredBuffer<float3> v_positions;
[[vk::binding(7, 0)]]
RWStructuredBuffer<float3> v_fixedPos;
[[vk::binding(8, 0)]]
RWStructuredBuffer<float3> v_lambda;
[[vk::binding(9, 0)]]
RWStructuredBuffer<float> v_stiffness;

[shader(\"compute\")] [numthreads(64, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
  uint c = tid.x;
  uint v = vertIdx[c];
  float3 p = positions[v];
  float3 fp = fixedPos[c];
  float k = stiffness[c];
  float3 vG = v_gradV[c];
  float vH = v_hessScalar[c];
  v_positions[v] = float3((k * vG.x), (k * vG.y), (k * vG.z));
  v_fixedPos[c] = float3(((0.000000 - k) * vG.x), ((0.000000 - k) * vG.y), ((0.000000 - k) * vG.z));
  v_lambda[c] = float3(vG.x, vG.y, vG.z);
  float dotCv = ((((p.x - fp.x) * vG.x) + ((p.y - fp.y) * vG.y)) + ((p.z - fp.z) * vG.z));
  v_stiffness[c] = (dotCv + vH);
}"

example : LeanSlang.emit shader = expected := by native_decide
example : shader.entryPointName = "main" := by native_decide

end Cloth.SlangCodegen.AttachmentForceAlBackward
