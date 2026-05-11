import LeanSlang

/-!
# `Cloth.SlangCodegen.DotReduceSerial` — single-thread df32 dot product

Sibling kernel to `Cloth.SlangCodegen.DotReduce`. Computes the same
quantity

```
Σ_{i=0..n}  a[i] · b[i]
```

with the **same df32 EFT helpers** (`two_sum`, `quick_two_sum`,
`two_prod`, `df_add`) and the **same numerical result** — but as a
single-thread serial fold:

  `[numthreads(1, 1, 1)]` + plain for-loop, no `groupshared`, no
  `GroupMemoryBarrierWithGroupSync()`.

**Why this kernel exists.** The parallel `dot_reduce` uses a
groupshared tree reduce with workgroup barriers; `slangc -target cpp`
rejects the barrier (E36107) because its sequential per-thread
dispatch can't honour cross-thread synchronisation. We isolated the
blocker by probing the two features individually — `groupshared`
alone compiles fine; `GroupMemoryBarrierWithGroupSync()` is the wall.

For GPU dispatch, `DotReduce` (parallel) is the production kernel —
it gets one thread per element, log₂(n) reduction depth. For
`slang_validate` host-diff and any single-threaded CPU dispatch,
`DotReduceSerial` is what we run — it produces the same df32 result
through the same EFTs, just folded sequentially.

Both are validated:

  * `DotReduce`         — Layer 0 (native_decide) + Layer 1 (SPIR-V).
  * `DotReduceSerial`   — Layer 0 + Layer 1 + Layer 2 (slang_validate
                          host-diff against an fp64 reference).

Bindings are identical to `DotReduce`:

  0  ConstantBuffer<DotReduceSerialParams> { uint n; }
  1  StructuredBuffer<float>         a
  2  StructuredBuffer<float>         b
  3  RWStructuredBuffer<float>       dst   (length ≥ 2; dst[0]=hi, dst[1]=lo)
-/

namespace Cloth.SlangCodegen.DotReduceSerial

open LeanSlang

private def floatTy : SlangType := .scalar .float
private def uintTy  : SlangType := .scalar .uint

private def fIn  (name : String) : SlangBinding :=
  ⟨name, floatTy, Semantic.none, none, none, .qIn⟩
private def fOut (name : String) : SlangBinding :=
  ⟨name, floatTy, Semantic.none, none, none, .qOut⟩

private def two_sum : SlangFunctionDecl :=
  { attrs   := []
  , retType := .named "void"
  , name    := "two_sum"
  , params  := [fIn "a", fIn "b", fOut "hi", fOut "lo"]
  , body    :=
      [ .declInit floatTy "h"    (.bin "+" (.var "a") (.var "b"))
      , .declInit floatTy "bb"   (.bin "-" (.var "h") (.var "a"))
      , .declInit floatTy "ah"   (.bin "-" (.var "h") (.var "bb"))
      , .declInit floatTy "lo_a" (.bin "-" (.var "a") (.var "ah"))
      , .declInit floatTy "lo_b" (.bin "-" (.var "b") (.var "bb"))
      , .assign (.var "hi") (.var "h")
      , .assign (.var "lo") (.bin "+" (.var "lo_a") (.var "lo_b"))
      , .ret none ] }

private def quick_two_sum : SlangFunctionDecl :=
  { attrs   := []
  , retType := .named "void"
  , name    := "quick_two_sum"
  , params  := [fIn "a", fIn "b", fOut "hi", fOut "lo"]
  , body    :=
      [ .declInit floatTy "h" (.bin "+" (.var "a") (.var "b"))
      , .declInit floatTy "t" (.bin "-" (.var "h") (.var "a"))
      , .assign (.var "hi") (.var "h")
      , .assign (.var "lo") (.bin "-" (.var "b") (.var "t"))
      , .ret none ] }

private def two_prod : SlangFunctionDecl :=
  { attrs   := []
  , retType := .named "void"
  , name    := "two_prod"
  , params  := [fIn "a", fIn "b", fOut "hi", fOut "lo"]
  , body    :=
      [ .declInit floatTy "h" (.bin "*" (.var "a") (.var "b"))
      , .assign (.var "hi") (.var "h")
      , .assign (.var "lo")
          (.call "fma" [.var "a", .var "b", .un "-" (.var "h")])
      , .ret none ] }

private def df_add : SlangFunctionDecl :=
  { attrs   := []
  , retType := .named "void"
  , name    := "df_add"
  , params  :=
      [ fIn "x_hi", fIn "x_lo", fIn "y_hi", fIn "y_lo"
      , fOut "z_hi", fOut "z_lo" ]
  , body    :=
      [ .declare floatTy "sh" none
      , .declare floatTy "sl" none
      , .expr (.call "two_sum"
          [.var "x_hi", .var "y_hi", .var "sh", .var "sl"])
      , .declInit floatTy "xy_lo" (.bin "+" (.var "x_lo") (.var "y_lo"))
      , .declInit floatTy "sl2"   (.bin "+" (.var "sl")  (.var "xy_lo"))
      , .expr (.call "quick_two_sum"
          [.var "sh", .var "sl2", .var "z_hi", .var "z_lo"])
      , .ret none ] }

private def mainEntry : SlangFunctionDecl :=
  { attrs  := [.shaderCompute, .numthreads 1 1 1]
  , name   := "main"
  , params := [⟨"tid", .vec .uint 3, .svDispatchThreadId, none, none, .qIn⟩]
  , body   :=
      [ .declInit floatTy "acc_hi" (.litFloat 0.0)
      , .declInit floatTy "acc_lo" (.litFloat 0.0)
      , .forCount "i" (.litUint 0) (.member (.var "params") "n")
          [ .declare floatTy "p_hi" none
          , .declare floatTy "p_lo" none
          , .expr (.call "two_prod"
              [ .index (.var "a") (.var "i")
              , .index (.var "b") (.var "i")
              , .var "p_hi", .var "p_lo" ])
          , .declare floatTy "new_hi" none
          , .declare floatTy "new_lo" none
          , .expr (.call "df_add"
              [ .var "acc_hi", .var "acc_lo", .var "p_hi", .var "p_lo"
              , .var "new_hi", .var "new_lo" ])
          , .assign (.var "acc_hi") (.var "new_hi")
          , .assign (.var "acc_lo") (.var "new_lo") ]
      , .assign (.index (.var "dst") (.litUint 0)) (.var "acc_hi")
      , .assign (.index (.var "dst") (.litUint 1)) (.var "acc_lo")
      , .ret none ] }

def shader : SlangShaderModule :=
  { structs :=
      [ { name := "DotReduceSerialParams"
        , fields := [⟨"n", .scalar .uint, Semantic.none, none, none, .qIn⟩] } ]
  , globals :=
      [ ⟨"params", .const "DotReduceSerialParams", Semantic.none, some 0, some 0, .qIn⟩
      , ⟨"a",      .roBuf (.scalar .float),        Semantic.none, some 1, some 0, .qIn⟩
      , ⟨"b",      .roBuf (.scalar .float),        Semantic.none, some 2, some 0, .qIn⟩
      , ⟨"dst",    .rwBuf (.scalar .float),        Semantic.none, some 3, some 0, .qIn⟩ ]
  , functions := [two_sum, quick_two_sum, two_prod, df_add, mainEntry] }

def expected : String :=
"struct DotReduceSerialParams {
  uint n;
};

[[vk::binding(0, 0)]]
ConstantBuffer<DotReduceSerialParams> params;
[[vk::binding(1, 0)]]
StructuredBuffer<float> a;
[[vk::binding(2, 0)]]
StructuredBuffer<float> b;
[[vk::binding(3, 0)]]
RWStructuredBuffer<float> dst;

void two_sum(float a, float b, out float hi, out float lo) {
  float h = (a + b);
  float bb = (h - a);
  float ah = (h - bb);
  float lo_a = (a - ah);
  float lo_b = (b - bb);
  hi = h;
  lo = (lo_a + lo_b);
  return;
}

void quick_two_sum(float a, float b, out float hi, out float lo) {
  float h = (a + b);
  float t = (h - a);
  hi = h;
  lo = (b - t);
  return;
}

void two_prod(float a, float b, out float hi, out float lo) {
  float h = (a * b);
  hi = h;
  lo = fma(a, b, (-h));
  return;
}

void df_add(float x_hi, float x_lo, float y_hi, float y_lo, out float z_hi, out float z_lo) {
  float sh;
  float sl;
  two_sum(x_hi, y_hi, sh, sl);
  float xy_lo = (x_lo + y_lo);
  float sl2 = (sl + xy_lo);
  quick_two_sum(sh, sl2, z_hi, z_lo);
  return;
}

[shader(\"compute\")] [numthreads(1, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
  float acc_hi = 0.000000;
  float acc_lo = 0.000000;
  for (uint i = 0u; i < params.n; ++i) {
    float p_hi;
    float p_lo;
    two_prod(a[i], b[i], p_hi, p_lo);
    float new_hi;
    float new_lo;
    df_add(acc_hi, acc_lo, p_hi, p_lo, new_hi, new_lo);
    acc_hi = new_hi;
    acc_lo = new_lo;
  }
  dst[0u] = acc_hi;
  dst[1u] = acc_lo;
  return;
}"

example : LeanSlang.emit shader = expected := by native_decide
example : shader.entryPointName = "main" := by native_decide

end Cloth.SlangCodegen.DotReduceSerial
