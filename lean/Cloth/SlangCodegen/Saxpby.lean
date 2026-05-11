import LeanSlang

/-!
# `Cloth.SlangCodegen.Saxpby` — generic linear combination

First of the global-solve building blocks. Computes the standard
linear-combination

  dst[i] = alpha * x[i] + beta * y[i]

over a flat scalar vector — `i ∈ [0, n)`. Together with `spmv` and
`dot_reduce` (follow-up PRs) this is everything a CG outer loop
needs to consume the per-element constraint projections that the
spring / bending / attachment / triangle local-step kernels emit.

The PD position state is stored stacked as a flat `float[3·V]`,
matching Eigen's `VecXd` layout in `Simulation.cpp`. Saxpby
operates on those flat layouts blindly — one thread per scalar
component, no awareness of vertex stride.

Mirrors V-Sekai-fire/Curvenet's `Curvenet.SlangCodegen.Saxpby`
verbatim (same buffer layout, same `[numthreads(256, 1, 1)]`,
same fused `fma(α, x[i], β · y[i])`).

Bindings (set 0):

  0  ConstantBuffer<SaxpbyParams> { uint n; float alpha; float beta; }
  1  StructuredBuffer<float>      x
  2  StructuredBuffer<float>      y
  3  RWStructuredBuffer<float>    dst

The pinned reference is asserted by `native_decide` below.
-/

namespace Cloth.SlangCodegen.Saxpby

open LeanSlang

def shader : SlangShaderModule :=
  { structs :=
      [ { name := "SaxpbyParams"
        , fields :=
            [ ⟨"n",     .scalar .uint,  Semantic.none, none, none, .qIn⟩
            , ⟨"alpha", .scalar .float, Semantic.none, none, none, .qIn⟩
            , ⟨"beta",  .scalar .float, Semantic.none, none, none, .qIn⟩ ] } ]
  , globals :=
      [ ⟨"params", .const "SaxpbyParams",     Semantic.none, some 0, some 0, .qIn⟩
      , ⟨"x",      .roBuf (.scalar .float),   Semantic.none, some 1, some 0, .qIn⟩
      , ⟨"y",      .roBuf (.scalar .float),   Semantic.none, some 2, some 0, .qIn⟩
      , ⟨"dst",    .rwBuf (.scalar .float),   Semantic.none, some 3, some 0, .qIn⟩ ]
  , functions := [{
      attrs  := [.shaderCompute, .numthreads 256 1 1]
      name   := "main"
      params := [⟨"tid", .vec .uint 3, .svDispatchThreadId, none, none, .qIn⟩]
      body   :=
        [ .declInit (.scalar .uint) "i" (.member (.var "tid") "x")
        , .ifNoElse (.bin ">=" (.var "i") (.member (.var "params") "n"))
            [ .ret none ]
        , .assign (.index (.var "dst") (.var "i"))
            (.call "fma"
              [ .member (.var "params") "alpha"
              , .index (.var "x") (.var "i")
              , .bin "*" (.member (.var "params") "beta")
                         (.index (.var "y") (.var "i")) ])
        ] }] }

def expected : String :=
"struct SaxpbyParams {
  uint n;
  float alpha;
  float beta;
};

[[vk::binding(0, 0)]]
ConstantBuffer<SaxpbyParams> params;
[[vk::binding(1, 0)]]
StructuredBuffer<float> x;
[[vk::binding(2, 0)]]
StructuredBuffer<float> y;
[[vk::binding(3, 0)]]
RWStructuredBuffer<float> dst;

[shader(\"compute\")] [numthreads(256, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
  uint i = tid.x;
  if ((i >= params.n)) {
    return;
  }
  dst[i] = fma(params.alpha, x[i], (params.beta * y[i]));
}"

example : LeanSlang.emit shader = expected := by native_decide
example : shader.entryPointName = "main" := by native_decide

end Cloth.SlangCodegen.Saxpby
