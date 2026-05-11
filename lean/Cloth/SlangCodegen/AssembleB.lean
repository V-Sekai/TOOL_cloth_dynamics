import LeanSlang

/-!
# `Cloth.SlangCodegen.AssembleB` — PD right-hand-side assembly

The bridge kernel between the per-constraint local-step projections
(spring / bending / attachment / triangle) and the CG inner loop.
Computes the Projective-Dynamics right-hand side

```
b[3v + d]  =  mass[v] * s[3v + d]  +  Σ_{(slot, w) in incident[v]}  w * projections[3*slot + d]
```

for each vertex `v` and dimension `d ∈ {0, 1, 2}`. One thread per
vertex; no atomics. The sum is a CSR-style gather over each vertex's
incident-constraint list.

The constraint-output buffer `projections` is treated as a flat array
of `slot_count` 3-vectors, one per constraint output (spring → 1 slot,
bending → 1 slot, attachment → 1 slot, triangle in-plane → 2 slots
for col0/col1). The bind-time-built CSR `(ctxStart, ctxSlot, ctxWeight)`
encodes the linear map `S^T` from constraint outputs back to per-vertex
contributions:

  ctxStart   : rowPtr, length = numVerts + 1
  ctxSlot    : which projection slot this incidence reads from
  ctxWeight  : ±sqrtConstraintWeight (the entry in S^T at this row/col)

For a spring connecting verts (a, b) with sqrt-weight `w`, the bind
emits two incidences:
  vertex a: slot = spring_slot, weight = +w
  vertex b: slot = spring_slot, weight = −w
And similar for bending (4 entries with cotangent weights), attachment
(1 entry, +w), and triangle in-plane (more complex — selects col0/col1
slots via the rest_inv_uv coefficients).

The mass term `M·s` is baked in directly: lumped (diagonal) PD mass is
just a per-vertex scalar, so `mass[v] * s[3v + d]` is the M-contribution.
Callers wanting the `(M/h² + …)` form scale `mass` before binding.

Bindings (set 0):

  0  ConstantBuffer<AssembleBParams> { uint numVerts; }
  1  StructuredBuffer<float>     s             length = 3 · numVerts
  2  StructuredBuffer<float>     mass          length = numVerts
  3  StructuredBuffer<float>     projections   length = 3 · slot_count
  4  StructuredBuffer<uint>      ctxStart      length = numVerts + 1
  5  StructuredBuffer<uint>      ctxSlot       length = total_incidences
  6  StructuredBuffer<float>     ctxWeight     length = total_incidences
  7  RWStructuredBuffer<float>   b             length = 3 · numVerts
-/

namespace Cloth.SlangCodegen.AssembleB

open LeanSlang

private def floatTy : SlangType := .scalar .float
private def uintTy  : SlangType := .scalar .uint

def shader : SlangShaderModule :=
  { structs :=
      [ { name := "AssembleBParams"
        , fields := [⟨"numVerts", uintTy, Semantic.none, none, none, .qIn⟩] } ]
  , globals :=
      [ ⟨"params",      .const "AssembleBParams",  Semantic.none, some 0, some 0, .qIn⟩
      , ⟨"s",           .roBuf floatTy,            Semantic.none, some 1, some 0, .qIn⟩
      , ⟨"mass",        .roBuf floatTy,            Semantic.none, some 2, some 0, .qIn⟩
      , ⟨"projections", .roBuf floatTy,            Semantic.none, some 3, some 0, .qIn⟩
      , ⟨"ctxStart",    .roBuf uintTy,             Semantic.none, some 4, some 0, .qIn⟩
      , ⟨"ctxSlot",     .roBuf uintTy,             Semantic.none, some 5, some 0, .qIn⟩
      , ⟨"ctxWeight",   .roBuf floatTy,            Semantic.none, some 6, some 0, .qIn⟩
      , ⟨"b",           .rwBuf floatTy,            Semantic.none, some 7, some 0, .qIn⟩ ]
  , functions := [{
      attrs  := [.shaderCompute, .numthreads 64 1 1]
      name   := "main"
      params := [⟨"tid", .vec .uint 3, .svDispatchThreadId, none, none, .qIn⟩]
      body   :=
        [ .declInit uintTy "v" (.member (.var "tid") "x")
        , .ifNoElse (.bin ">=" (.var "v") (.member (.var "params") "numVerts"))
            [ .ret none ]
        , .declInit uintTy  "v3" (.bin "*" (.litUint 3) (.var "v"))
        , .declInit uintTy  "cs" (.index (.var "ctxStart") (.var "v"))
        , .declInit uintTy  "ce" (.index (.var "ctxStart")
                                   (.bin "+" (.var "v") (.litUint 1)))
        , .declInit floatTy "m"  (.index (.var "mass") (.var "v"))
        , .declInit floatTy "bx" (.bin "*" (.var "m") (.index (.var "s") (.var "v3")))
        , .declInit floatTy "by" (.bin "*" (.var "m")
                                   (.index (.var "s")
                                     (.bin "+" (.var "v3") (.litUint 1))))
        , .declInit floatTy "bz" (.bin "*" (.var "m")
                                   (.index (.var "s")
                                     (.bin "+" (.var "v3") (.litUint 2))))
        , .forCount "k" (.var "cs") (.var "ce")
            [ .declInit floatTy "w"    (.index (.var "ctxWeight") (.var "k"))
            , .declInit uintTy  "slot" (.index (.var "ctxSlot") (.var "k"))
            , .declInit uintTy  "s3"   (.bin "*" (.litUint 3) (.var "slot"))
            , .assign (.var "bx")
                (.bin "+" (.var "bx")
                  (.bin "*" (.var "w")
                    (.index (.var "projections") (.var "s3"))))
            , .assign (.var "by")
                (.bin "+" (.var "by")
                  (.bin "*" (.var "w")
                    (.index (.var "projections")
                      (.bin "+" (.var "s3") (.litUint 1)))))
            , .assign (.var "bz")
                (.bin "+" (.var "bz")
                  (.bin "*" (.var "w")
                    (.index (.var "projections")
                      (.bin "+" (.var "s3") (.litUint 2)))))
            ]
        , .assign (.index (.var "b") (.var "v3")) (.var "bx")
        , .assign (.index (.var "b") (.bin "+" (.var "v3") (.litUint 1))) (.var "by")
        , .assign (.index (.var "b") (.bin "+" (.var "v3") (.litUint 2))) (.var "bz")
        ] }] }

def expected : String :=
"struct AssembleBParams {
  uint numVerts;
};

[[vk::binding(0, 0)]]
ConstantBuffer<AssembleBParams> params;
[[vk::binding(1, 0)]]
StructuredBuffer<float> s;
[[vk::binding(2, 0)]]
StructuredBuffer<float> mass;
[[vk::binding(3, 0)]]
StructuredBuffer<float> projections;
[[vk::binding(4, 0)]]
StructuredBuffer<uint> ctxStart;
[[vk::binding(5, 0)]]
StructuredBuffer<uint> ctxSlot;
[[vk::binding(6, 0)]]
StructuredBuffer<float> ctxWeight;
[[vk::binding(7, 0)]]
RWStructuredBuffer<float> b;

[shader(\"compute\")] [numthreads(64, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
  uint v = tid.x;
  if ((v >= params.numVerts)) {
    return;
  }
  uint v3 = (3u * v);
  uint cs = ctxStart[v];
  uint ce = ctxStart[(v + 1u)];
  float m = mass[v];
  float bx = (m * s[v3]);
  float by = (m * s[(v3 + 1u)]);
  float bz = (m * s[(v3 + 2u)]);
  for (uint k = cs; k < ce; ++k) {
    float w = ctxWeight[k];
    uint slot = ctxSlot[k];
    uint s3 = (3u * slot);
    bx = (bx + (w * projections[s3]));
    by = (by + (w * projections[(s3 + 1u)]));
    bz = (bz + (w * projections[(s3 + 2u)]));
  }
  b[v3] = bx;
  b[(v3 + 1u)] = by;
  b[(v3 + 2u)] = bz;
}"

example : LeanSlang.emit shader = expected := by native_decide
example : shader.entryPointName = "main" := by native_decide

end Cloth.SlangCodegen.AssembleB
