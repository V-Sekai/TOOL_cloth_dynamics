import LeanSlang

/-!
# `Cloth.SlangCodegen.CGBeta` — GPU-side β for the CG inner loop

Sibling to `cg_alpha`. Single-thread kernel that consumes two df32 dot
products (the previous and current `dot(r, r)`) and emits β plus
overwrites `δ_old` so the next iter doesn't need a CPU-side copy:

```
β        = (dotNew[0] + dotNew[1]) / (dotOld[0] + dotOld[1])
betaOut  = +β
dotOld  := dotNew       (in-place; next iter's "old" is this iter's "new")
```

After this dispatch the orchestrator can rebind `dotOld` directly as
the new `dot(r, r)` target for the next iter — no CPU ping-pong of
buffers required.

Bindings (set 0):

  0  StructuredBuffer<float>   dotNew   length ≥ 2 (df32 hi, lo; current iter)
  1  RWStructuredBuffer<float> dotOld   length ≥ 2 (read THEN written in place)
  2  RWStructuredBuffer<float> betaOut  length ≥ 1 ([0] = β)

`[numthreads(1, 1, 1)]` — one thread. The read-before-write on
`dotOld` is sequentially safe inside a single thread.
-/

namespace Cloth.SlangCodegen.CGBeta

open LeanSlang

private def floatTy : SlangType := .scalar .float
private def uintTy  : SlangType := .scalar .uint

def shader : SlangShaderModule :=
  { globals :=
      [ ⟨"dotNew",  .roBuf floatTy, Semantic.none, some 0, some 0, .qIn⟩
      , ⟨"dotOld",  .rwBuf floatTy, Semantic.none, some 1, some 0, .qIn⟩
      , ⟨"betaOut", .rwBuf floatTy, Semantic.none, some 2, some 0, .qIn⟩ ]
  , functions := [{
      attrs  := [.shaderCompute, .numthreads 1 1 1]
      name   := "main"
      params := [⟨"tid", .vec .uint 3, .svDispatchThreadId, none, none, .qIn⟩]
      body   :=
        [ .declInit floatTy "dnew"
            (.bin "+" (.index (.var "dotNew") (.litUint 0))
                      (.index (.var "dotNew") (.litUint 1)))
        , .declInit floatTy "dold"
            (.bin "+" (.index (.var "dotOld") (.litUint 0))
                      (.index (.var "dotOld") (.litUint 1)))
        , .declInit floatTy "b" (.bin "/" (.var "dnew") (.var "dold"))
        , .assign (.index (.var "betaOut") (.litUint 0)) (.var "b")
        , .assign (.index (.var "dotOld") (.litUint 0))
            (.index (.var "dotNew") (.litUint 0))
        , .assign (.index (.var "dotOld") (.litUint 1))
            (.index (.var "dotNew") (.litUint 1))
        ] }] }

def expected : String :=
"[[vk::binding(0, 0)]]
StructuredBuffer<float> dotNew;
[[vk::binding(1, 0)]]
RWStructuredBuffer<float> dotOld;
[[vk::binding(2, 0)]]
RWStructuredBuffer<float> betaOut;

[shader(\"compute\")] [numthreads(1, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
  float dnew = (dotNew[0u] + dotNew[1u]);
  float dold = (dotOld[0u] + dotOld[1u]);
  float b = (dnew / dold);
  betaOut[0u] = b;
  dotOld[0u] = dotNew[0u];
  dotOld[1u] = dotNew[1u];
}"

example : LeanSlang.emit shader = expected := by native_decide
example : shader.entryPointName = "main" := by native_decide

end Cloth.SlangCodegen.CGBeta
