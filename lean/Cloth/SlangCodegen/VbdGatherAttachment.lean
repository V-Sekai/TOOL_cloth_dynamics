import LeanSlang

/-!
# `Cloth.SlangCodegen.VbdGatherAttachment` — AVBD vertex-update attachment gather

Second gather kernel in the AVBD vertex-block-update pipeline, after
`vbd_gather_spring`. Accumulates each incident attachment's force +
Hessian into per-vertex scratch.

Attachments are the simplest constraint geometry — one vertex per
constraint, no role question, Hessian is a scalar multiple of the
identity. The gather therefore omits the role buffer and only adds the
scalar to the three diagonal entries of `hScratch[6v..6v+5]`:

```
for k in [vertAttachOffset[v], vertAttachOffset[v+1]):
  c = vertAttachIdx[k]
  gScratch[v]      += attachGradV[c]                -- ∇_v E_c
  hScratch[6v]     += attachHessScalar[c]           -- Hxx += k
  hScratch[6v+3]   += attachHessScalar[c]           -- Hyy += k
  hScratch[6v+5]   += attachHessScalar[c]           -- Hzz += k
  -- off-diagonal entries Hxy, Hxz, Hyz are unchanged because
  -- attachment Hessian is k · I₃
```

Bindings (set 0):

  0  StructuredBuffer<float3>   attachGradV       length = N_attach
  1  StructuredBuffer<float>    attachHessScalar  length = N_attach
  2  StructuredBuffer<uint>     vertAttachOffset  length = N_verts + 1
  3  StructuredBuffer<uint>     vertAttachIdx     length = total incidences
  4  RWStructuredBuffer<float3> gScratch          length = N_verts
  5  RWStructuredBuffer<float>  hScratch          length = 6 · N_verts

Note the absence of a `vertAttachRole` buffer — attachment always
has K=1, so role is implicitly 0 and contributes no sign flip.
-/

namespace Cloth.SlangCodegen.VbdGatherAttachment

open LeanSlang

private def f3 : SlangType := .vec .float 3
private def u  : SlangType := .scalar .uint
private def f  : SlangType := .scalar .float

private def bnd (n : Nat) (name : String) (t : SlangType) : SlangBinding :=
  { name := name, type := t, semantic := Semantic.none
  , binding := some n, space := some 0 }

private def loopBody : List SlangStmt :=
  [ .declInit u  "c"  (.index (.var "vertAttachIdx") (.var "k"))
  , .declInit f3 "ga" (.index (.var "attachGradV") (.var "c"))
  , .declInit f  "hs" (.index (.var "attachHessScalar") (.var "c"))
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
  , .declInit u  "sStart" (.index (.var "vertAttachOffset") (.var "v"))
  , .declInit u  "sEnd"   (.index (.var "vertAttachOffset")
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
      [ { name := "VbdGatherAttachmentParams"
        , fields := [⟨"colorOffset", u, Semantic.none, none, none, .qIn⟩] } ]
  , globals :=
      [ bnd 0 "attachGradV"      (.roBuf f3)
      , bnd 1 "attachHessScalar" (.roBuf f)
      , bnd 2 "vertAttachOffset" (.roBuf u)
      , bnd 3 "vertAttachIdx"    (.roBuf u)
      , bnd 4 "gScratch"         (.rwBuf f3)
      , bnd 5 "hScratch"         (.rwBuf f)
      , bnd 6 "vertPerm"         (.roBuf u)
      , ⟨"params", .const "VbdGatherAttachmentParams", Semantic.none, some 7, some 0, .qIn⟩
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
"struct VbdGatherAttachmentParams {
  uint colorOffset;
};

[[vk::binding(0, 0)]]
StructuredBuffer<float3> attachGradV;
[[vk::binding(1, 0)]]
StructuredBuffer<float> attachHessScalar;
[[vk::binding(2, 0)]]
StructuredBuffer<uint> vertAttachOffset;
[[vk::binding(3, 0)]]
StructuredBuffer<uint> vertAttachIdx;
[[vk::binding(4, 0)]]
RWStructuredBuffer<float3> gScratch;
[[vk::binding(5, 0)]]
RWStructuredBuffer<float> hScratch;
[[vk::binding(6, 0)]]
StructuredBuffer<uint> vertPerm;
[[vk::binding(7, 0)]]
ConstantBuffer<VbdGatherAttachmentParams> params;

[shader(\"compute\")] [numthreads(64, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
  uint lane = tid.x;
  uint v = vertPerm[(lane + params.colorOffset)];
  uint sStart = vertAttachOffset[v];
  uint sEnd = vertAttachOffset[(v + 1u)];
  uint hb = (6u * v);
  float3 g = gScratch[v];
  float gx = g.x;
  float gy = g.y;
  float gz = g.z;
  float Hxx = hScratch[hb];
  float Hyy = hScratch[(hb + 3u)];
  float Hzz = hScratch[(hb + 5u)];
  for (uint k = sStart; k < sEnd; ++k) {
    uint c = vertAttachIdx[k];
    float3 ga = attachGradV[c];
    float hs = attachHessScalar[c];
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

end Cloth.SlangCodegen.VbdGatherAttachment
