import LeanSlang

/-!
# `Cloth.SlangCodegen.SaxpbyIndirect` — saxpby with α/β from buffer

Variant of `Cloth.SlangCodegen.Saxpby` that reads `α` and `β` from
1-element scalar buffers instead of a `ConstantBuffer<SaxpbyParams>`.
Computes the same thing:

```
dst[i] = α · x[i] + β · y[i]    for i ∈ [0, n)
```

Why this exists: in the CG inner loop, α and β come from
`cg_alpha` / `cg_beta` (GPU-side computation). With the original
`Saxpby` kernel they'd have to be written into a constant struct
*on the CPU* between dispatches — which means commit + wait per
saxpby. With this variant, the CG iter chain stays on-GPU:

```
dot(p,q) → cg_alpha → saxpby_indirect (β buffer = cg_alpha out[0])
                                       (β buffer = cg_alpha out[1])
dot(r,r) → cg_beta  → saxpby_indirect (β buffer = cg_beta out)
```

α is hardcoded to come from its own buffer too, even though the CG
pattern always uses α'=1 — a separate 1-float "ones" buffer (set
once at bind time) keeps the kernel generic.

Bindings (set 0):

  0  ConstantBuffer<SaxpbyIndirectParams> { uint n; }
  1  StructuredBuffer<float>   alpha   length ≥ 1 (reads [0])
  2  StructuredBuffer<float>   beta    length ≥ 1 (reads [0])
  3  StructuredBuffer<float>   x       length = n
  4  StructuredBuffer<float>   y       length = n
  5  RWStructuredBuffer<float> dst     length = n

Same `[numthreads(256, 1, 1)]` as `Saxpby`. Per-element work is one
`fma`; the buffer index for α/β reads is constant so the GPU should
broadcast it via the L1 cache. No precision change from `Saxpby`.
-/

namespace Cloth.SlangCodegen.SaxpbyIndirect

open LeanSlang

def shader : SlangShaderModule :=
  { structs :=
      [ { name := "SaxpbyIndirectParams"
        , fields := [⟨"n", .scalar .uint, Semantic.none, none, none, .qIn⟩] } ]
  , globals :=
      [ ⟨"params", .const "SaxpbyIndirectParams", Semantic.none, some 0, some 0, .qIn⟩
      , ⟨"alpha",  .roBuf (.scalar .float),       Semantic.none, some 1, some 0, .qIn⟩
      , ⟨"beta",   .roBuf (.scalar .float),       Semantic.none, some 2, some 0, .qIn⟩
      , ⟨"x",      .roBuf (.scalar .float),       Semantic.none, some 3, some 0, .qIn⟩
      , ⟨"y",      .roBuf (.scalar .float),       Semantic.none, some 4, some 0, .qIn⟩
      , ⟨"dst",    .rwBuf (.scalar .float),       Semantic.none, some 5, some 0, .qIn⟩ ]
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
              [ .index (.var "alpha") (.litUint 0)
              , .index (.var "x") (.var "i")
              , .bin "*" (.index (.var "beta") (.litUint 0))
                         (.index (.var "y") (.var "i")) ])
        ] }] }

def expected : String :=
"struct SaxpbyIndirectParams {
  uint n;
};

[[vk::binding(0, 0)]]
ConstantBuffer<SaxpbyIndirectParams> params;
[[vk::binding(1, 0)]]
StructuredBuffer<float> alpha;
[[vk::binding(2, 0)]]
StructuredBuffer<float> beta;
[[vk::binding(3, 0)]]
StructuredBuffer<float> x;
[[vk::binding(4, 0)]]
StructuredBuffer<float> y;
[[vk::binding(5, 0)]]
RWStructuredBuffer<float> dst;

[shader(\"compute\")] [numthreads(256, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
  uint i = tid.x;
  if ((i >= params.n)) {
    return;
  }
  dst[i] = fma(alpha[0u], x[i], (beta[0u] * y[i]));
}"

example : LeanSlang.emit shader = expected := by native_decide
example : shader.entryPointName = "main" := by native_decide

end Cloth.SlangCodegen.SaxpbyIndirect
