import LeanSlang

/-!
# `Cloth.SlangCodegen.VbdGatherTriangle` — AVBD vertex-update triangle membrane gather

Third gather kernel of the AVBD vertex-block-update pipeline. For each
vertex v, loops over the CSR adjacency slice and accumulates every
incident triangle membrane constraint's force + scalar Hessian into
per-vertex scratch.

`triangle_membrane_force` writes per-corner outputs indexed as
`[3·c + r]` for triangle c, role r ∈ {0, 1, 2}. The gather reads the
right slot via the role buffer:

```
for k in [vertTriOffset[v], vertTriOffset[v+1]):
  c    = vertTriIdx[k]
  r    = vertTriRole[k]                          -- 0, 1, or 2
  gi   = 3·c + r
  gScratch[v]    += triGrad[gi]
  hScratch[6v+0] += triHessScalar[gi]            -- Hxx
  hScratch[6v+3] += triHessScalar[gi]            -- Hyy
  hScratch[6v+5] += triHessScalar[gi]            -- Hzz
```

Same scalar-on-diagonal Hessian as attachment_force — the GN
approximation for the ARAP membrane energy reduces each per-vertex
block to `scalar · I₃`. Off-diagonal entries are unchanged.

Bindings (set 0):

  0  StructuredBuffer<float3>   triGrad         length = 3 · N_tri
  1  StructuredBuffer<float>    triHessScalar   length = 3 · N_tri
  2  StructuredBuffer<uint>     vertTriOffset   length = N_verts + 1
  3  StructuredBuffer<uint>     vertTriIdx      length = total incidences
  4  StructuredBuffer<uint>     vertTriRole     length = total incidences
  5  RWStructuredBuffer<float3> gScratch        length = N_verts
  6  RWStructuredBuffer<float>  hScratch        length = 6 · N_verts
-/

namespace Cloth.SlangCodegen.VbdGatherTriangle

open LeanSlang

private def f3 : SlangType := .vec .float 3
private def u  : SlangType := .scalar .uint
private def f  : SlangType := .scalar .float

private def bnd (n : Nat) (name : String) (t : SlangType) : SlangBinding :=
  { name := name, type := t, semantic := Semantic.none
  , binding := some n, space := some 0 }

private def loopBody : List SlangStmt :=
  [ .declInit u  "c"  (.index (.var "vertTriIdx") (.var "k"))
  , .declInit u  "r"  (.index (.var "vertTriRole") (.var "k"))
  , .declInit u  "gi" (.bin "+" (.bin "*" (.litUint 3) (.var "c")) (.var "r"))
  , .declInit f3 "ga" (.index (.var "triGrad") (.var "gi"))
  , .declInit f  "hs" (.index (.var "triHessScalar") (.var "gi"))
  , .assign (.var "gx") (.bin "+" (.var "gx") (.member (.var "ga") "x"))
  , .assign (.var "gy") (.bin "+" (.var "gy") (.member (.var "ga") "y"))
  , .assign (.var "gz") (.bin "+" (.var "gz") (.member (.var "ga") "z"))
  , .assign (.var "Hxx") (.bin "+" (.var "Hxx") (.var "hs"))
  , .assign (.var "Hyy") (.bin "+" (.var "Hyy") (.var "hs"))
  , .assign (.var "Hzz") (.bin "+" (.var "Hzz") (.var "hs"))
  ]

private def body : List SlangStmt :=
  [ .declInit u  "v"      (.member (.var "tid") "x")
  , .declInit u  "sStart" (.index (.var "vertTriOffset") (.var "v"))
  , .declInit u  "sEnd"   (.index (.var "vertTriOffset")
                            (.bin "+" (.var "v") (.litUint 1)))
  , .declInit u  "hb"     (.bin "*" (.litUint 6) (.var "v"))
  , .declInit f3 "g"      (.index (.var "gScratch") (.var "v"))
  , .declInit f  "gx"     (.member (.var "g") "x")
  , .declInit f  "gy"     (.member (.var "g") "y")
  , .declInit f  "gz"     (.member (.var "g") "z")
  , .declInit f  "Hxx"    (.index (.var "hScratch") (.var "hb"))
  , .declInit f  "Hyy"    (.index (.var "hScratch") (.bin "+" (.var "hb") (.litUint 3)))
  , .declInit f  "Hzz"    (.index (.var "hScratch") (.bin "+" (.var "hb") (.litUint 5)))
  , .forCount "k" (.var "sStart") (.var "sEnd") loopBody
  , .assign (.index (.var "gScratch") (.var "v"))
      (.call "float3" [.var "gx", .var "gy", .var "gz"])
  , .assign (.index (.var "hScratch") (.var "hb")) (.var "Hxx")
  , .assign (.index (.var "hScratch") (.bin "+" (.var "hb") (.litUint 3))) (.var "Hyy")
  , .assign (.index (.var "hScratch") (.bin "+" (.var "hb") (.litUint 5))) (.var "Hzz")
  ]

def shader : SlangShaderModule :=
  { globals :=
      [ bnd 0 "triGrad"        (.roBuf f3)
      , bnd 1 "triHessScalar"  (.roBuf f)
      , bnd 2 "vertTriOffset"  (.roBuf u)
      , bnd 3 "vertTriIdx"     (.roBuf u)
      , bnd 4 "vertTriRole"    (.roBuf u)
      , bnd 5 "gScratch"       (.rwBuf f3)
      , bnd 6 "hScratch"       (.rwBuf f)
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
StructuredBuffer<float3> triGrad;
[[vk::binding(1, 0)]]
StructuredBuffer<float> triHessScalar;
[[vk::binding(2, 0)]]
StructuredBuffer<uint> vertTriOffset;
[[vk::binding(3, 0)]]
StructuredBuffer<uint> vertTriIdx;
[[vk::binding(4, 0)]]
StructuredBuffer<uint> vertTriRole;
[[vk::binding(5, 0)]]
RWStructuredBuffer<float3> gScratch;
[[vk::binding(6, 0)]]
RWStructuredBuffer<float> hScratch;

[shader(\"compute\")] [numthreads(64, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
  uint v = tid.x;
  uint sStart = vertTriOffset[v];
  uint sEnd = vertTriOffset[(v + 1u)];
  uint hb = (6u * v);
  float3 g = gScratch[v];
  float gx = g.x;
  float gy = g.y;
  float gz = g.z;
  float Hxx = hScratch[hb];
  float Hyy = hScratch[(hb + 3u)];
  float Hzz = hScratch[(hb + 5u)];
  for (uint k = sStart; k < sEnd; ++k) {
    uint c = vertTriIdx[k];
    uint r = vertTriRole[k];
    uint gi = ((3u * c) + r);
    float3 ga = triGrad[gi];
    float hs = triHessScalar[gi];
    gx = (gx + ga.x);
    gy = (gy + ga.y);
    gz = (gz + ga.z);
    Hxx = (Hxx + hs);
    Hyy = (Hyy + hs);
    Hzz = (Hzz + hs);
  }
  gScratch[v] = float3(gx, gy, gz);
  hScratch[hb] = Hxx;
  hScratch[(hb + 3u)] = Hyy;
  hScratch[(hb + 5u)] = Hzz;
}"

example : LeanSlang.emit shader = expected := by native_decide
example : shader.entryPointName = "main" := by native_decide

end Cloth.SlangCodegen.VbdGatherTriangle
