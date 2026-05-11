import LeanSlang

/-!
# `Cloth.SlangCodegen.SaxpbyIndirectDf32` — df32 saxpby with α/β from buffer

Second kernel attacking the fp32 precision floor from PR #38, paired
with `SpmvDf32` from PR #39. Same ABI as `SaxpbyIndirect` so it's a
drop-in replacement in `MetalCGSolver`.

```
dst[i] = α · x[i] + β · y[i]    for i ∈ [0, n)
```

The computation accumulates via the Knuth/Dekker EFTs (`two_prod`,
`df_add`) so each per-element output has df32 precision before being
collapsed to fp32 for storage. Per-element cost: 2 `two_prod` +
1 `df_add` ≈ 8 extra fp ops vs the original 1 fma. saxpby is
bandwidth-bound so the wall impact is small relative to the
precision win.

Bindings identical to `SaxpbyIndirect`:

  0  ConstantBuffer<SaxpbyIndirectDf32Params> { uint n; }
  1  StructuredBuffer<float>   alpha   length ≥ 1 (reads [0])
  2  StructuredBuffer<float>   beta    length ≥ 1 (reads [0])
  3  StructuredBuffer<float>   x       length = n
  4  StructuredBuffer<float>   y       length = n
  5  RWStructuredBuffer<float> dst     length = n

EFT helpers (two_sum, quick_two_sum, two_prod, df_add) inlined as
private fn decls — same pattern as `SpmvDf32` and `DotReduce`.
-/

namespace Cloth.SlangCodegen.SaxpbyIndirectDf32

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
  { attrs  := [.shaderCompute, .numthreads 256 1 1]
  , name   := "main"
  , params := [⟨"tid", .vec .uint 3, .svDispatchThreadId, none, none, .qIn⟩]
  , body   :=
      [ .declInit uintTy "i" (.member (.var "tid") "x")
      , .ifNoElse (.bin ">=" (.var "i") (.member (.var "params") "n"))
          [ .ret none ]
      , .declInit floatTy "a"  (.index (.var "alpha") (.litUint 0))
      , .declInit floatTy "b"  (.index (.var "beta")  (.litUint 0))
      , .declInit floatTy "xi" (.index (.var "x")     (.var "i"))
      , .declInit floatTy "yi" (.index (.var "y")     (.var "i"))
      , .declare floatTy "a_hi" none
      , .declare floatTy "a_lo" none
      , .expr (.call "two_prod"
          [.var "a", .var "xi", .var "a_hi", .var "a_lo"])
      , .declare floatTy "b_hi" none
      , .declare floatTy "b_lo" none
      , .expr (.call "two_prod"
          [.var "b", .var "yi", .var "b_hi", .var "b_lo"])
      , .declare floatTy "s_hi" none
      , .declare floatTy "s_lo" none
      , .expr (.call "df_add"
          [.var "a_hi", .var "a_lo"
          , .var "b_hi", .var "b_lo"
          , .var "s_hi", .var "s_lo"])
      , .assign (.index (.var "dst") (.var "i"))
          (.bin "+" (.var "s_hi") (.var "s_lo"))
      , .ret none ] }

def shader : SlangShaderModule :=
  { structs :=
      [ { name := "SaxpbyIndirectDf32Params"
        , fields := [⟨"n", .scalar .uint, Semantic.none, none, none, .qIn⟩] } ]
  , globals :=
      [ ⟨"params", .const "SaxpbyIndirectDf32Params", Semantic.none, some 0, some 0, .qIn⟩
      , ⟨"alpha",  .roBuf floatTy,                    Semantic.none, some 1, some 0, .qIn⟩
      , ⟨"beta",   .roBuf floatTy,                    Semantic.none, some 2, some 0, .qIn⟩
      , ⟨"x",      .roBuf floatTy,                    Semantic.none, some 3, some 0, .qIn⟩
      , ⟨"y",      .roBuf floatTy,                    Semantic.none, some 4, some 0, .qIn⟩
      , ⟨"dst",    .rwBuf floatTy,                    Semantic.none, some 5, some 0, .qIn⟩ ]
  , functions := [two_sum, quick_two_sum, two_prod, df_add, mainEntry] }

def expected : String :=
"struct SaxpbyIndirectDf32Params {
  uint n;
};

[[vk::binding(0, 0)]]
ConstantBuffer<SaxpbyIndirectDf32Params> params;
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

[shader(\"compute\")] [numthreads(256, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
  uint i = tid.x;
  if ((i >= params.n)) {
    return;
  }
  float a = alpha[0u];
  float b = beta[0u];
  float xi = x[i];
  float yi = y[i];
  float a_hi;
  float a_lo;
  two_prod(a, xi, a_hi, a_lo);
  float b_hi;
  float b_lo;
  two_prod(b, yi, b_hi, b_lo);
  float s_hi;
  float s_lo;
  df_add(a_hi, a_lo, b_hi, b_lo, s_hi, s_lo);
  dst[i] = (s_hi + s_lo);
  return;
}"

example : LeanSlang.emit shader = expected := by native_decide
example : shader.entryPointName = "main" := by native_decide

end Cloth.SlangCodegen.SaxpbyIndirectDf32
