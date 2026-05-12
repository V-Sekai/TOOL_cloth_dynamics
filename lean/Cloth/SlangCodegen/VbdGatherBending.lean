import LeanSlang

/-!
# `Cloth.SlangCodegen.VbdGatherBending` — AVBD vertex-update dihedral bending gather

Fourth and final gather kernel of the AVBD vertex-block-update
pipeline. Same shape as `vbd_gather_triangle` (#54) but K=4 — the
dihedral bending constraint has a 4-vertex stencil and
`triangle_bending_force` writes its per-vertex outputs indexed as
`[4·c + r]` for r ∈ {0, 1, 2, 3}.

```
for k in [vertBendOffset[v], vertBendOffset[v+1]):
  c    = vertBendIdx[k]
  r    = vertBendRole[k]                         -- 0, 1, 2, or 3
  gi   = 4·c + r
  gScratch[v]    += bendGrad[gi]
  hScratch[6v+0] += bendHessScalar[gi]           -- Hxx
  hScratch[6v+3] += bendHessScalar[gi]           -- Hyy
  hScratch[6v+5] += bendHessScalar[gi]           -- Hzz
```

Diagonal-scalar Hessian, same as attachment/triangle. With this gather
landed, the AVBD vertex-block-update pipeline covers every constraint
type DiffCloth uses: spring, attachment, triangle membrane, and dihedral
bending. Full composition:

  vbd_init →
    vbd_gather_spring → vbd_gather_attachment →
    vbd_gather_triangle → vbd_gather_bending →
    vbd_solve_apply

Bindings (set 0):

  0  StructuredBuffer<float3>   bendGrad        length = 4 · N_bend
  1  StructuredBuffer<float>    bendHessScalar  length = 4 · N_bend
  2  StructuredBuffer<uint>     vertBendOffset  length = N_verts + 1
  3  StructuredBuffer<uint>     vertBendIdx     length = total incidences
  4  StructuredBuffer<uint>     vertBendRole    length = total incidences
  5  RWStructuredBuffer<float3> gScratch        length = N_verts
  6  RWStructuredBuffer<float>  hScratch        length = 6 · N_verts
-/

namespace Cloth.SlangCodegen.VbdGatherBending

open LeanSlang

private def f3 : SlangType := .vec .float 3
private def u  : SlangType := .scalar .uint
private def f  : SlangType := .scalar .float

private def bnd (n : Nat) (name : String) (t : SlangType) : SlangBinding :=
  { name := name, type := t, semantic := Semantic.none
  , binding := some n, space := some 0 }

private def loopBody : List SlangStmt :=
  [ .declInit u  "c"  (.index (.var "vertBendIdx") (.var "k"))
  , .declInit u  "r"  (.index (.var "vertBendRole") (.var "k"))
  , .declInit u  "gi" (.bin "+" (.bin "*" (.litUint 4) (.var "c")) (.var "r"))
  , .declInit f3 "ga" (.index (.var "bendGrad") (.var "gi"))
  , .declInit f  "hs" (.index (.var "bendHessScalar") (.var "gi"))
  , .assign (.var "gx") (.bin "+" (.var "gx") (.member (.var "ga") "x"))
  , .assign (.var "gy") (.bin "+" (.var "gy") (.member (.var "ga") "y"))
  , .assign (.var "gz") (.bin "+" (.var "gz") (.member (.var "ga") "z"))
  , .assign (.var "Hxx") (.bin "+" (.var "Hxx") (.var "hs"))
  , .assign (.var "Hyy") (.bin "+" (.var "Hyy") (.var "hs"))
  , .assign (.var "Hzz") (.bin "+" (.var "Hzz") (.var "hs"))
  ]

private def body : List SlangStmt :=
  [ .declInit u  "lane"   (.member (.var "tid") "x")
  , .declInit u  "v"
      (.index (.var "vertPerm")
        (.bin "+" (.var "lane")
                  (.member (.var "params") "colorOffset")))
  , .declInit u  "sStart" (.index (.var "vertBendOffset") (.var "v"))
  , .declInit u  "sEnd"   (.index (.var "vertBendOffset")
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
  { structs :=
      [ { name := "VbdGatherBendingParams"
        , fields := [⟨"colorOffset", u, Semantic.none, none, none, .qIn⟩] } ]
  , globals :=
      [ bnd 0 "bendGrad"        (.roBuf f3)
      , bnd 1 "bendHessScalar"  (.roBuf f)
      , bnd 2 "vertBendOffset"  (.roBuf u)
      , bnd 3 "vertBendIdx"     (.roBuf u)
      , bnd 4 "vertBendRole"    (.roBuf u)
      , bnd 5 "gScratch"        (.rwBuf f3)
      , bnd 6 "hScratch"        (.rwBuf f)
      , bnd 7 "vertPerm"        (.roBuf u)
      , ⟨"params", .const "VbdGatherBendingParams", Semantic.none, some 8, some 0, .qIn⟩
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
"struct VbdGatherBendingParams {
  uint colorOffset;
};

[[vk::binding(0, 0)]]
StructuredBuffer<float3> bendGrad;
[[vk::binding(1, 0)]]
StructuredBuffer<float> bendHessScalar;
[[vk::binding(2, 0)]]
StructuredBuffer<uint> vertBendOffset;
[[vk::binding(3, 0)]]
StructuredBuffer<uint> vertBendIdx;
[[vk::binding(4, 0)]]
StructuredBuffer<uint> vertBendRole;
[[vk::binding(5, 0)]]
RWStructuredBuffer<float3> gScratch;
[[vk::binding(6, 0)]]
RWStructuredBuffer<float> hScratch;
[[vk::binding(7, 0)]]
StructuredBuffer<uint> vertPerm;
[[vk::binding(8, 0)]]
ConstantBuffer<VbdGatherBendingParams> params;

[shader(\"compute\")] [numthreads(64, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
  uint lane = tid.x;
  uint v = vertPerm[(lane + params.colorOffset)];
  uint sStart = vertBendOffset[v];
  uint sEnd = vertBendOffset[(v + 1u)];
  uint hb = (6u * v);
  float3 g = gScratch[v];
  float gx = g.x;
  float gy = g.y;
  float gz = g.z;
  float Hxx = hScratch[hb];
  float Hyy = hScratch[(hb + 3u)];
  float Hzz = hScratch[(hb + 5u)];
  for (uint k = sStart; k < sEnd; ++k) {
    uint c = vertBendIdx[k];
    uint r = vertBendRole[k];
    uint gi = ((4u * c) + r);
    float3 ga = bendGrad[gi];
    float hs = bendHessScalar[gi];
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

end Cloth.SlangCodegen.VbdGatherBending
