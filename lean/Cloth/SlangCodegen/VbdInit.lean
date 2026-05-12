import LeanSlang

/-!
# `Cloth.SlangCodegen.VbdInit` — AVBD vertex-update inertial init (PR-D part 1)

First kernel in the AVBD vertex-block-update pipeline (`todo.md` PR-D).
Initializes per-vertex scratch buffers with the implicit-Euler
inertial term before the per-constraint gather kernels add their
contributions.

Per vertex `v` with mass `m`, current position `x`, predictor `x̃`, and
time-step `h`:

```
w        = m / h²                               -- inertial weight

gScratch[v]            = w · (x − x̃)            -- ∂E_inertia/∂x_v
hScratch[6v..6v+5]     = [w, 0, 0, w, 0, w]     -- packed sym 3x3
                                                   (w · I₃, the
                                                   inertial Hessian)
```

The packed sym 3x3 layout is `[Hxx, Hxy, Hxz, Hyy, Hyz, Hzz]`, the
same order spring_force uses. Per-constraint gather kernels (PR-D
part 2: vbd_gather_spring, etc.) read this layout and accumulate
their own contributions on top before the solve+apply kernel (PR-D
part 3) inverts the 3x3 and writes `x ← x − H⁻¹ g`.

`invHSquared = 1/h²` is uniform across vertices and lives in a small
constant buffer (`VbdInitParams`).

Bindings (set 0):

  0  ConstantBuffer<VbdInitParams> { float invHSquared; }
  1  StructuredBuffer<float3>      positions    length = N_verts
  2  StructuredBuffer<float3>      predicted    length = N_verts
  3  StructuredBuffer<float>       mass         length = N_verts
  4  RWStructuredBuffer<float3>    gScratch     length = N_verts
  5  RWStructuredBuffer<float>     hScratch     length = 6 · N_verts
-/

namespace Cloth.SlangCodegen.VbdInit

open LeanSlang

private def f3 : SlangType := .vec .float 3
private def u  : SlangType := .scalar .uint
private def f  : SlangType := .scalar .float

private def body : List SlangStmt :=
  [ .declInit u  "lane" (.member (.var "tid") "x")
  , .declInit u  "v"
      (.index (.var "vertPerm")
        (.bin "+" (.var "lane")
                  (.member (.var "params") "colorOffset")))
  , .declInit f3 "x"   (.index (.var "positions") (.var "v"))
  , .declInit f3 "px"  (.index (.var "predicted") (.var "v"))
  , .declInit f  "m"   (.index (.var "mass") (.var "v"))
  , .declInit f  "w"   (.bin "*" (.var "m")
                                 (.member (.var "params") "invHSquared"))
  , .assign (.index (.var "gScratch") (.var "v"))
      (.call "float3"
        [ .bin "*" (.var "w")
            (.bin "-" (.member (.var "x") "x") (.member (.var "px") "x"))
        , .bin "*" (.var "w")
            (.bin "-" (.member (.var "x") "y") (.member (.var "px") "y"))
        , .bin "*" (.var "w")
            (.bin "-" (.member (.var "x") "z") (.member (.var "px") "z")) ])
  , .declInit u "hb" (.bin "*" (.litUint 6) (.var "v"))
  , .assign (.index (.var "hScratch") (.var "hb")) (.var "w")
  , .assign (.index (.var "hScratch") (.bin "+" (.var "hb") (.litUint 1)))
      (.litFloat 0.0)
  , .assign (.index (.var "hScratch") (.bin "+" (.var "hb") (.litUint 2)))
      (.litFloat 0.0)
  , .assign (.index (.var "hScratch") (.bin "+" (.var "hb") (.litUint 3)))
      (.var "w")
  , .assign (.index (.var "hScratch") (.bin "+" (.var "hb") (.litUint 4)))
      (.litFloat 0.0)
  , .assign (.index (.var "hScratch") (.bin "+" (.var "hb") (.litUint 5)))
      (.var "w")
  ]

def shader : SlangShaderModule :=
  { structs :=
      [ { name := "VbdInitParams"
        , fields :=
            [ ⟨"invHSquared", f, Semantic.none, none, none, .qIn⟩
            , ⟨"colorOffset", u, Semantic.none, none, none, .qIn⟩ ] } ]
  , globals :=
      [ ⟨"params",    .const "VbdInitParams", Semantic.none, some 0, some 0, .qIn⟩
      , ⟨"positions", .roBuf f3,              Semantic.none, some 1, some 0, .qIn⟩
      , ⟨"predicted", .roBuf f3,              Semantic.none, some 2, some 0, .qIn⟩
      , ⟨"mass",      .roBuf f,               Semantic.none, some 3, some 0, .qIn⟩
      , ⟨"gScratch",  .rwBuf f3,              Semantic.none, some 4, some 0, .qIn⟩
      , ⟨"hScratch",  .rwBuf f,               Semantic.none, some 5, some 0, .qIn⟩
      , ⟨"vertPerm",  .roBuf u,               Semantic.none, some 6, some 0, .qIn⟩ ]
  , functions := [{
      attrs  := [.shaderCompute, .numthreads 64 1 1]
      name   := "main"
      params := [{ name := "tid", type := .vec .uint 3
                 , semantic := Semantic.svDispatchThreadId
                 , binding := none, space := none }]
      body   := body
    }] }

def expected : String :=
"struct VbdInitParams {
  float invHSquared;
  uint colorOffset;
};

[[vk::binding(0, 0)]]
ConstantBuffer<VbdInitParams> params;
[[vk::binding(1, 0)]]
StructuredBuffer<float3> positions;
[[vk::binding(2, 0)]]
StructuredBuffer<float3> predicted;
[[vk::binding(3, 0)]]
StructuredBuffer<float> mass;
[[vk::binding(4, 0)]]
RWStructuredBuffer<float3> gScratch;
[[vk::binding(5, 0)]]
RWStructuredBuffer<float> hScratch;
[[vk::binding(6, 0)]]
StructuredBuffer<uint> vertPerm;

[shader(\"compute\")] [numthreads(64, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
  uint lane = tid.x;
  uint v = vertPerm[(lane + params.colorOffset)];
  float3 x = positions[v];
  float3 px = predicted[v];
  float m = mass[v];
  float w = (m * params.invHSquared);
  gScratch[v] = float3((w * (x.x - px.x)), (w * (x.y - px.y)), (w * (x.z - px.z)));
  uint hb = (6u * v);
  hScratch[hb] = w;
  hScratch[(hb + 1u)] = 0.000000;
  hScratch[(hb + 2u)] = 0.000000;
  hScratch[(hb + 3u)] = w;
  hScratch[(hb + 4u)] = 0.000000;
  hScratch[(hb + 5u)] = w;
}"

example : LeanSlang.emit shader = expected := by native_decide
example : shader.entryPointName = "main" := by native_decide

end Cloth.SlangCodegen.VbdInit
