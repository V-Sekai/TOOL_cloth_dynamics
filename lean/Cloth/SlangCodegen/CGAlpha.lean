import LeanSlang

/-!
# `Cloth.SlangCodegen.CGAlpha` — GPU-side α / −α for the CG inner loop

Single-thread kernel that consumes two df32 dot products and emits
both `+α` and `−α` to a 2-element output buffer:

```
α   = dot(r, r) / dot(p, q)     (reconstructed from df32 (hi, lo) pairs)
out[0] = +α
out[1] = −α
```

Why: without this kernel, the CG inner loop must round-trip dot
results to the CPU between every dispatch (commit + waitUntilCompleted)
in order to compute α and use it in the next saxpby. With the kernel,
`dot → cg_alpha → saxpby_x → saxpby_r` is a single command buffer
with no CPU intervention — the saxpby reads `±α` from the output
buffer of this kernel.

The CG inner loop needs both `+α` (for `x += α·p`) and `−α` (for
`r −= α·q`), so we emit both. A saxpby kernel that takes scalar α/β
from an indexed buffer will live in a follow-up PR; this one is just
the scalar arithmetic.

Bindings (set 0):

  0  StructuredBuffer<float>   dotPQ      length ≥ 2 (df32 hi, lo)
  1  StructuredBuffer<float>   dotDelta   length ≥ 2 (df32 hi, lo)
  2  RWStructuredBuffer<float> alphaOut   length ≥ 2 ([0]=+α, [1]=−α)

`[numthreads(1, 1, 1)]` — exactly one thread; the work is trivially
serial and the kernel exists to break the CPU-sync dependency chain,
not to be parallel.
-/

namespace Cloth.SlangCodegen.CGAlpha

open LeanSlang

private def floatTy : SlangType := .scalar .float
private def uintTy  : SlangType := .scalar .uint

def shader : SlangShaderModule :=
  { globals :=
      [ ⟨"dotPQ",    .roBuf floatTy, Semantic.none, some 0, some 0, .qIn⟩
      , ⟨"dotDelta", .roBuf floatTy, Semantic.none, some 1, some 0, .qIn⟩
      , ⟨"alphaOut", .rwBuf floatTy, Semantic.none, some 2, some 0, .qIn⟩ ]
  , functions := [{
      attrs  := [.shaderCompute, .numthreads 1 1 1]
      name   := "main"
      params := [⟨"tid", .vec .uint 3, .svDispatchThreadId, none, none, .qIn⟩]
      body   :=
        [ .declInit floatTy "pq"
            (.bin "+" (.index (.var "dotPQ")    (.litUint 0))
                      (.index (.var "dotPQ")    (.litUint 1)))
        , .declInit floatTy "dn"
            (.bin "+" (.index (.var "dotDelta") (.litUint 0))
                      (.index (.var "dotDelta") (.litUint 1)))
        , .declInit floatTy "a" (.bin "/" (.var "dn") (.var "pq"))
        , .assign (.index (.var "alphaOut") (.litUint 0)) (.var "a")
        , .assign (.index (.var "alphaOut") (.litUint 1))
            (.bin "-" (.litFloat 0.0) (.var "a"))
        ] }] }

def expected : String :=
"[[vk::binding(0, 0)]]
StructuredBuffer<float> dotPQ;
[[vk::binding(1, 0)]]
StructuredBuffer<float> dotDelta;
[[vk::binding(2, 0)]]
RWStructuredBuffer<float> alphaOut;

[shader(\"compute\")] [numthreads(1, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
  float pq = (dotPQ[0u] + dotPQ[1u]);
  float dn = (dotDelta[0u] + dotDelta[1u]);
  float a = (dn / pq);
  alphaOut[0u] = a;
  alphaOut[1u] = (0.000000 - a);
}"

example : LeanSlang.emit shader = expected := by native_decide
example : shader.entryPointName = "main" := by native_decide

end Cloth.SlangCodegen.CGAlpha
