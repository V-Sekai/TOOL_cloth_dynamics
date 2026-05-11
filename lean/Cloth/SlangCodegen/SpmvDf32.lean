import LeanSlang

/-!
# `Cloth.SlangCodegen.SpmvDf32` — sparse CSR matrix-vector with df32 accumulation

Like `Cloth.SlangCodegen.Spmv` but accumulates each row's sum as a
df32 pair `(s_hi, s_lo)` via the Knuth/Dekker EFT helpers
(`two_prod`, `df_add`). Output is fp32 — `y[i] = s_hi + s_lo` —
matching the original `Spmv` ABI exactly, so it's a drop-in
replacement for `MetalCGSolver`'s spmv pipeline.

Why this exists: PR #38 found that the CG-iter-count amplification
hits an fp32 precision floor around `tol ≈ 1e-4`. The dominant source
of fp32 error inside a CG iter is the per-row spmv sum — for a
typical Laplacian row with ~7 nnz, the fp32 accumulation builds up
~7·ε of error per row, which compounds over CG iters into a per-iter
residual floor that DiffCloth's PD outer convergence test can't
push past.

df32 accumulation drops the per-row error from ~7·ε to ~7·ε² which
is negligible. The output is still fp32 (the (hi, lo) pair is
collapsed at the end), so downstream `saxpby` etc. doesn't need to
change. Cost per row: one extra `two_prod` + one `df_add` per nnz
≈ 5 extra fp ops per nnz vs the original `fma`. Per element this is
~5× the kernel arithmetic; on M-series GPU this is fully bandwidth-
bound for spmv so the wall impact should be small.

Bindings are identical to `Spmv`:

  0  ConstantBuffer<SpmvDf32Params> { uint rows; }
  1  StructuredBuffer<int>       rowPtr   length = rows + 1
  2  StructuredBuffer<int>       colIdx   length = nnz
  3  StructuredBuffer<float>     values   length = nnz
  4  StructuredBuffer<float>     x        length = cols
  5  RWStructuredBuffer<float>   y        length = rows

EFT helpers (two_prod, df_add, two_sum, quick_two_sum) inlined as
private fn decls — same pattern as DotReduce.
-/

namespace Cloth.SlangCodegen.SpmvDf32

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
      , .ifNoElse (.bin ">=" (.var "i") (.member (.var "params") "rows"))
          [ .ret none ]
      , .declInit uintTy "rs"
          (.call "uint" [.index (.var "rowPtr") (.var "i")])
      , .declInit uintTy "re"
          (.call "uint" [.index (.var "rowPtr")
            (.bin "+" (.var "i") (.litUint 1))])
      , .declInit floatTy "s_hi" (.litFloat 0.0)
      , .declInit floatTy "s_lo" (.litFloat 0.0)
      , .forCount "p" (.var "rs") (.var "re")
          [ .declInit floatTy "v"  (.index (.var "values") (.var "p"))
          , .declInit floatTy "xc"
              (.index (.var "x")
                (.call "uint" [.index (.var "colIdx") (.var "p")]))
          , .declare floatTy "t_hi" none
          , .declare floatTy "t_lo" none
          , .expr (.call "two_prod"
              [.var "v", .var "xc", .var "t_hi", .var "t_lo"])
          , .declare floatTy "n_hi" none
          , .declare floatTy "n_lo" none
          , .expr (.call "df_add"
              [.var "s_hi", .var "s_lo"
              , .var "t_hi", .var "t_lo"
              , .var "n_hi", .var "n_lo"])
          , .assign (.var "s_hi") (.var "n_hi")
          , .assign (.var "s_lo") (.var "n_lo") ]
      , .assign (.index (.var "y") (.var "i"))
          (.bin "+" (.var "s_hi") (.var "s_lo"))
      , .ret none ] }

def shader : SlangShaderModule :=
  { structs :=
      [ { name := "SpmvDf32Params"
        , fields := [⟨"rows", .scalar .uint, Semantic.none, none, none, .qIn⟩] } ]
  , globals :=
      [ ⟨"params", .const "SpmvDf32Params",  Semantic.none, some 0, some 0, .qIn⟩
      , ⟨"rowPtr", .roBuf (.scalar .int),    Semantic.none, some 1, some 0, .qIn⟩
      , ⟨"colIdx", .roBuf (.scalar .int),    Semantic.none, some 2, some 0, .qIn⟩
      , ⟨"values", .roBuf (.scalar .float),  Semantic.none, some 3, some 0, .qIn⟩
      , ⟨"x",      .roBuf (.scalar .float),  Semantic.none, some 4, some 0, .qIn⟩
      , ⟨"y",      .rwBuf (.scalar .float),  Semantic.none, some 5, some 0, .qIn⟩ ]
  , functions := [two_sum, quick_two_sum, two_prod, df_add, mainEntry] }

def expected : String :=
"struct SpmvDf32Params {
  uint rows;
};

[[vk::binding(0, 0)]]
ConstantBuffer<SpmvDf32Params> params;
[[vk::binding(1, 0)]]
StructuredBuffer<int> rowPtr;
[[vk::binding(2, 0)]]
StructuredBuffer<int> colIdx;
[[vk::binding(3, 0)]]
StructuredBuffer<float> values;
[[vk::binding(4, 0)]]
StructuredBuffer<float> x;
[[vk::binding(5, 0)]]
RWStructuredBuffer<float> y;

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
  if ((i >= params.rows)) {
    return;
  }
  uint rs = uint(rowPtr[i]);
  uint re = uint(rowPtr[(i + 1u)]);
  float s_hi = 0.000000;
  float s_lo = 0.000000;
  for (uint p = rs; p < re; ++p) {
    float v = values[p];
    float xc = x[uint(colIdx[p])];
    float t_hi;
    float t_lo;
    two_prod(v, xc, t_hi, t_lo);
    float n_hi;
    float n_lo;
    df_add(s_hi, s_lo, t_hi, t_lo, n_hi, n_lo);
    s_hi = n_hi;
    s_lo = n_lo;
  }
  y[i] = (s_hi + s_lo);
  return;
}"

example : LeanSlang.emit shader = expected := by native_decide
example : shader.entryPointName = "main" := by native_decide

end Cloth.SlangCodegen.SpmvDf32
