import LeanSlang

/-!
# `Cloth.SlangCodegen.SelfCollisionScan` — GPU brute-force vertex-vertex self-collision scan

Replaces DiffCloth's CPU OpenMP spatial-hash detection for the
AVBD post-step self-collision resolution pass (#102). The CPU
path costs ~5 ms on the dress (~55% of per-step wall under the
full AVBD chain). GPU brute-force O(n²) at ~1 TFLOP gets it
under ~50 µs for n=3634 — ~100× faster.

Per vertex `i` (one thread per vertex):

```
init row neighbors[i*K .. i*K+K) to UINT_MAX sentinels
pi = positions[i],  ri = radii[i]
count = 0
for j in 0..nVerts:
  if (j != i) AND (count < K):
    d² = |pi - positions[j]|²
    thresh = ri + radii[j]
    if d² < thresh²:
      neighbors[i*K + count] = j
      count = count + 1
```

K caps the number of overlapping neighbors recorded per vertex
(default K=16 in the host wrapper). Any extra penetrations beyond
K are dropped — chosen so a single pair-resolution pass plus a
second AVBD outer step picks up the residual.

Sentinel value `UINT_MAX = (0u - 1u)` is emitted as the subtraction
so the literal stays portable across Slang backends.

Bindings (set 0):

  0  StructuredBuffer<float3>   positions   length = N_verts
  1  StructuredBuffer<float>    radii       length = N_verts
  2  RWStructuredBuffer<uint>   neighbors   length = N_verts · K
  3  ConstantBuffer<SelfCollisionScanParams>  { nVerts, K }
-/

namespace Cloth.SlangCodegen.SelfCollisionScan

open LeanSlang

private def f3 : SlangType := .vec .float 3
private def u  : SlangType := .scalar .uint
private def f  : SlangType := .scalar .float

private def uintMax : SlangExpr :=
  .bin "-" (.litUint 0) (.litUint 1)

private def initRow : List SlangStmt :=
  [ .forCount "k" (.litUint 0) (.member (.var "params") "K")
      [ .assign
          (.index (.var "neighbors")
            (.bin "+" (.bin "*" (.var "i") (.member (.var "params") "K"))
                      (.var "k")))
          uintMax
      ]
  ]

private def scanBody : List SlangStmt :=
  [ .declInit f3 "dj" (.index (.var "positions") (.var "j"))
  , .declInit f  "dx" (.bin "-" (.member (.var "pi") "x") (.member (.var "dj") "x"))
  , .declInit f  "dy" (.bin "-" (.member (.var "pi") "y") (.member (.var "dj") "y"))
  , .declInit f  "dz" (.bin "-" (.member (.var "pi") "z") (.member (.var "dj") "z"))
  , .declInit f  "d2"
      (.bin "+" (.bin "+" (.bin "*" (.var "dx") (.var "dx"))
                          (.bin "*" (.var "dy") (.var "dy")))
                (.bin "*" (.var "dz") (.var "dz")))
  , .declInit f  "rj" (.index (.var "radii") (.var "j"))
  , .declInit f  "tr" (.bin "+" (.var "ri") (.var "rj"))
  , .declInit f  "t2" (.bin "*" (.var "tr") (.var "tr"))
  , .ifThen
      (.bin "&&"
        (.bin "&&"
          (.bin "!=" (.var "j") (.var "i"))
          (.bin "<" (.var "count") (.member (.var "params") "K")))
        (.bin "<" (.var "d2") (.var "t2")))
      [ .assign
          (.index (.var "neighbors")
            (.bin "+" (.bin "*" (.var "i") (.member (.var "params") "K"))
                      (.var "count")))
          (.var "j")
      , .assign (.var "count") (.bin "+" (.var "count") (.litUint 1))
      ]
      []
  ]

private def body : List SlangStmt :=
  [ .declInit u  "i"     (.member (.var "tid") "x")
  , .ifThen
      (.bin ">=" (.var "i") (.member (.var "params") "nVerts"))
      [ .ret none ]
      []
  ]
  ++ initRow ++
  [ .declInit f3 "pi"    (.index (.var "positions") (.var "i"))
  , .declInit f  "ri"    (.index (.var "radii") (.var "i"))
  , .declInit u  "count" (.litUint 0)
  , .forCount "j" (.litUint 0) (.member (.var "params") "nVerts") scanBody
  ]

def shader : SlangShaderModule :=
  { structs :=
      [ { name := "SelfCollisionScanParams"
        , fields := [ ⟨"nVerts", u, Semantic.none, none, none, .qIn⟩
                    , ⟨"K",      u, Semantic.none, none, none, .qIn⟩ ] } ]
  , globals :=
      [ ⟨"positions",  .roBuf f3,                       Semantic.none, some 0, some 0, .qIn⟩
      , ⟨"radii",      .roBuf f,                        Semantic.none, some 1, some 0, .qIn⟩
      , ⟨"neighbors",  .rwBuf u,                        Semantic.none, some 2, some 0, .qIn⟩
      , ⟨"params",     .const "SelfCollisionScanParams", Semantic.none, some 3, some 0, .qIn⟩
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
"struct SelfCollisionScanParams {
  uint nVerts;
  uint K;
};

[[vk::binding(0, 0)]]
StructuredBuffer<float3> positions;
[[vk::binding(1, 0)]]
StructuredBuffer<float> radii;
[[vk::binding(2, 0)]]
RWStructuredBuffer<uint> neighbors;
[[vk::binding(3, 0)]]
ConstantBuffer<SelfCollisionScanParams> params;

[shader(\"compute\")] [numthreads(64, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
  uint i = tid.x;
  if ((i >= params.nVerts)) {
    return;
  }
  for (uint k = 0u; k < params.K; ++k) {
    neighbors[((i * params.K) + k)] = (0u - 1u);
  }
  float3 pi = positions[i];
  float ri = radii[i];
  uint count = 0u;
  for (uint j = 0u; j < params.nVerts; ++j) {
    float3 dj = positions[j];
    float dx = (pi.x - dj.x);
    float dy = (pi.y - dj.y);
    float dz = (pi.z - dj.z);
    float d2 = (((dx * dx) + (dy * dy)) + (dz * dz));
    float rj = radii[j];
    float tr = (ri + rj);
    float t2 = (tr * tr);
    if ((((j != i) && (count < params.K)) && (d2 < t2))) {
      neighbors[((i * params.K) + count)] = j;
      count = (count + 1u);
    }
  }
}"

example : LeanSlang.emit shader = expected := by native_decide
example : shader.entryPointName = "main" := by native_decide

end Cloth.SlangCodegen.SelfCollisionScan
