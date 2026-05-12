import LeanSlang

/-!
# `Cloth.SlangCodegen.VbdSolveApply` — AVBD per-vertex 3x3 solve + apply (PR-D part 2)

Final kernel of the AVBD vertex-block-update pipeline. After `vbd_init`
seeds inertial scratch and the per-constraint gather kernels accumulate
their contributions (one per constraint type in PR-D part 3+), this
kernel closes the loop: for each vertex `v`, invert the local 3x3
Hessian, compute `Δx = −H⁻¹ g`, and update the position in place.

```
Read   g = gScratch[v]                        (float3)
Read   H = hScratch[6v..6v+5]                 (packed sym 3x3)

Compute adjugate of H (symmetric, so adj is also symmetric):
  adj00 = Hyy·Hzz − Hyz²
  adj11 = Hxx·Hzz − Hxz²
  adj22 = Hxx·Hyy − Hxy²
  adj01 = Hxz·Hyz − Hxy·Hzz
  adj02 = Hxy·Hyz − Hxz·Hyy
  adj12 = Hxy·Hxz − Hxx·Hyz

det     = Hxx·adj00 + Hxy·adj01 + Hxz·adj02
invDet  = 1.0 / det

Δx_i    = −invDet · (adj_i0 · g.x + adj_i1 · g.y + adj_i2 · g.z)
        = −invDet · adj · g                  -- standard adj/det
                                                inverse, symmetric adj

x_new   = positions[v] + Δx
```

Caller responsibility: ensure `H` is positive definite (well-defined
in the AVBD step from the inertial term `m/h²·I` plus PSD constraint
Hessians). If a constraint contributes an indefinite block, the host
must regularize (clamp eigenvalues or add `ε·I`) before this kernel
runs.

Bindings (set 0):

  0  StructuredBuffer<float3>   gScratch    length = N_verts
  1  StructuredBuffer<float>    hScratch    length = 6 · N_verts
  2  RWStructuredBuffer<float3> positions   length = N_verts

This kernel takes no parameter buffer — the dispatcher's launch size
covers the active-vertex range. For coloring-aware dispatch the host
slices the launch over `vertPerm[colorOffsets[k]..k+1)` per color.
-/

namespace Cloth.SlangCodegen.VbdSolveApply

open LeanSlang

private def f3 : SlangType := .vec .float 3
private def u  : SlangType := .scalar .uint
private def f  : SlangType := .scalar .float

private def body : List SlangStmt :=
  [ .declInit u  "v"   (.member (.var "tid") "x")
  , .declInit u  "hb"  (.bin "*" (.litUint 6) (.var "v"))
  , .declInit f3 "g"   (.index (.var "gScratch") (.var "v"))
  , .declInit f  "Hxx" (.index (.var "hScratch") (.var "hb"))
  , .declInit f  "Hxy" (.index (.var "hScratch") (.bin "+" (.var "hb") (.litUint 1)))
  , .declInit f  "Hxz" (.index (.var "hScratch") (.bin "+" (.var "hb") (.litUint 2)))
  , .declInit f  "Hyy" (.index (.var "hScratch") (.bin "+" (.var "hb") (.litUint 3)))
  , .declInit f  "Hyz" (.index (.var "hScratch") (.bin "+" (.var "hb") (.litUint 4)))
  , .declInit f  "Hzz" (.index (.var "hScratch") (.bin "+" (.var "hb") (.litUint 5)))
  , .declInit f  "a00"
      (.bin "-" (.bin "*" (.var "Hyy") (.var "Hzz"))
                (.bin "*" (.var "Hyz") (.var "Hyz")))
  , .declInit f  "a11"
      (.bin "-" (.bin "*" (.var "Hxx") (.var "Hzz"))
                (.bin "*" (.var "Hxz") (.var "Hxz")))
  , .declInit f  "a22"
      (.bin "-" (.bin "*" (.var "Hxx") (.var "Hyy"))
                (.bin "*" (.var "Hxy") (.var "Hxy")))
  , .declInit f  "a01"
      (.bin "-" (.bin "*" (.var "Hxz") (.var "Hyz"))
                (.bin "*" (.var "Hxy") (.var "Hzz")))
  , .declInit f  "a02"
      (.bin "-" (.bin "*" (.var "Hxy") (.var "Hyz"))
                (.bin "*" (.var "Hxz") (.var "Hyy")))
  , .declInit f  "a12"
      (.bin "-" (.bin "*" (.var "Hxy") (.var "Hxz"))
                (.bin "*" (.var "Hxx") (.var "Hyz")))
  , .declInit f  "det"
      (.bin "+"
        (.bin "+" (.bin "*" (.var "Hxx") (.var "a00"))
                  (.bin "*" (.var "Hxy") (.var "a01")))
        (.bin "*" (.var "Hxz") (.var "a02")))
  , .declInit f  "invDet" (.bin "/" (.litFloat 1.0) (.var "det"))
  , .declInit f  "gx" (.member (.var "g") "x")
  , .declInit f  "gy" (.member (.var "g") "y")
  , .declInit f  "gz" (.member (.var "g") "z")
  , .declInit f  "dx"
      (.bin "*" (.bin "-" (.litFloat 0.0) (.var "invDet"))
                (.bin "+"
                  (.bin "+" (.bin "*" (.var "a00") (.var "gx"))
                            (.bin "*" (.var "a01") (.var "gy")))
                  (.bin "*" (.var "a02") (.var "gz"))))
  , .declInit f  "dy"
      (.bin "*" (.bin "-" (.litFloat 0.0) (.var "invDet"))
                (.bin "+"
                  (.bin "+" (.bin "*" (.var "a01") (.var "gx"))
                            (.bin "*" (.var "a11") (.var "gy")))
                  (.bin "*" (.var "a12") (.var "gz"))))
  , .declInit f  "dz"
      (.bin "*" (.bin "-" (.litFloat 0.0) (.var "invDet"))
                (.bin "+"
                  (.bin "+" (.bin "*" (.var "a02") (.var "gx"))
                            (.bin "*" (.var "a12") (.var "gy")))
                  (.bin "*" (.var "a22") (.var "gz"))))
  , .declInit f3 "p" (.index (.var "positions") (.var "v"))
  , .assign (.index (.var "positions") (.var "v"))
      (.call "float3"
        [ .bin "+" (.member (.var "p") "x") (.var "dx")
        , .bin "+" (.member (.var "p") "y") (.var "dy")
        , .bin "+" (.member (.var "p") "z") (.var "dz") ])
  ]

private def bnd (n : Nat) (name : String) (t : SlangType) : SlangBinding :=
  { name := name, type := t, semantic := Semantic.none
  , binding := some n, space := some 0 }

def shader : SlangShaderModule :=
  { globals :=
      [ bnd 0 "gScratch"  (.roBuf f3)
      , bnd 1 "hScratch"  (.roBuf f)
      , bnd 2 "positions" (.rwBuf f3)
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
StructuredBuffer<float3> gScratch;
[[vk::binding(1, 0)]]
StructuredBuffer<float> hScratch;
[[vk::binding(2, 0)]]
RWStructuredBuffer<float3> positions;

[shader(\"compute\")] [numthreads(64, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
  uint v = tid.x;
  uint hb = (6u * v);
  float3 g = gScratch[v];
  float Hxx = hScratch[hb];
  float Hxy = hScratch[(hb + 1u)];
  float Hxz = hScratch[(hb + 2u)];
  float Hyy = hScratch[(hb + 3u)];
  float Hyz = hScratch[(hb + 4u)];
  float Hzz = hScratch[(hb + 5u)];
  float a00 = ((Hyy * Hzz) - (Hyz * Hyz));
  float a11 = ((Hxx * Hzz) - (Hxz * Hxz));
  float a22 = ((Hxx * Hyy) - (Hxy * Hxy));
  float a01 = ((Hxz * Hyz) - (Hxy * Hzz));
  float a02 = ((Hxy * Hyz) - (Hxz * Hyy));
  float a12 = ((Hxy * Hxz) - (Hxx * Hyz));
  float det = (((Hxx * a00) + (Hxy * a01)) + (Hxz * a02));
  float invDet = (1.000000 / det);
  float gx = g.x;
  float gy = g.y;
  float gz = g.z;
  float dx = ((0.000000 - invDet) * (((a00 * gx) + (a01 * gy)) + (a02 * gz)));
  float dy = ((0.000000 - invDet) * (((a01 * gx) + (a11 * gy)) + (a12 * gz)));
  float dz = ((0.000000 - invDet) * (((a02 * gx) + (a12 * gy)) + (a22 * gz)));
  float3 p = positions[v];
  positions[v] = float3((p.x + dx), (p.y + dy), (p.z + dz));
}"

example : LeanSlang.emit shader = expected := by native_decide
example : shader.entryPointName = "main" := by native_decide

end Cloth.SlangCodegen.VbdSolveApply
