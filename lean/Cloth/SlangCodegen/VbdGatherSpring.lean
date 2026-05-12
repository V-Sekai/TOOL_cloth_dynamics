import LeanSlang

/-!
# `Cloth.SlangCodegen.VbdGatherSpring` — AVBD vertex-update spring gather (PR-D part 3)

Gather kernel between `vbd_init` (PR #50) and `vbd_solve_apply` (PR #51).
For each vertex `v`, loops over the CSR adjacency slice
`[vertSpringOffset[v], vertSpringOffset[v+1])` to accumulate every
incident spring's force and Hessian contribution into the per-vertex
scratch buffers.

Per incidence `k` in the slice:

```
c     = vertSpringIdx[k]                     -- spring id
r     = vertSpringRole[k]                    -- 0 if v is endpoint a,
                                                1 if endpoint b
sign  = 1 − 2·float(r)                       -- +1 or −1

g     += sign · springGradA[c]               -- gradient gather
                                                (sign flip for endpoint b
                                                via Newton's 3rd)

hScratch[6v..6v+5] += springHess[6c..6c+5]   -- Hessian gather
                                                (GN diagonal block is
                                                the same matrix for
                                                both endpoints; no sign)
```

This kernel reads scratch from `vbd_init` and writes back the updated
scratch, so the three-stage pipeline composes:

  vbd_init → vbd_gather_spring → vbd_solve_apply

Bindings (set 0):

  0  StructuredBuffer<float3>   springGradA       length = N_springs
  1  StructuredBuffer<float>    springHess        length = 6 · N_springs
  2  StructuredBuffer<uint>     vertSpringOffset  length = N_verts + 1
  3  StructuredBuffer<uint>     vertSpringIdx     length = total incidences
  4  StructuredBuffer<uint>     vertSpringRole    length = total incidences
  5  RWStructuredBuffer<float3> gScratch          length = N_verts
  6  RWStructuredBuffer<float>  hScratch          length = 6 · N_verts
-/

namespace Cloth.SlangCodegen.VbdGatherSpring

open LeanSlang

private def f3 : SlangType := .vec .float 3
private def u  : SlangType := .scalar .uint
private def f  : SlangType := .scalar .float

private def bnd (n : Nat) (name : String) (t : SlangType) : SlangBinding :=
  { name := name, type := t, semantic := Semantic.none
  , binding := some n, space := some 0 }

private def loopBody : List SlangStmt :=
  [ .declInit u  "c"    (.index (.var "vertSpringIdx") (.var "k"))
  , .declInit u  "r"    (.index (.var "vertSpringRole") (.var "k"))
  , .declInit f  "sign" (.bin "-" (.litFloat 1.0)
                          (.bin "*" (.litFloat 2.0)
                            (.call "float" [.var "r"])))
  , .declInit f3 "ga"   (.index (.var "springGradA") (.var "c"))
  , .assign (.var "gx")
      (.bin "+" (.var "gx")
        (.bin "*" (.var "sign") (.member (.var "ga") "x")))
  , .assign (.var "gy")
      (.bin "+" (.var "gy")
        (.bin "*" (.var "sign") (.member (.var "ga") "y")))
  , .assign (.var "gz")
      (.bin "+" (.var "gz")
        (.bin "*" (.var "sign") (.member (.var "ga") "z")))
  , .declInit u  "shb"  (.bin "*" (.litUint 6) (.var "c"))
  , .assign (.var "Hxx")
      (.bin "+" (.var "Hxx") (.index (.var "springHess") (.var "shb")))
  , .assign (.var "Hxy")
      (.bin "+" (.var "Hxy")
        (.index (.var "springHess") (.bin "+" (.var "shb") (.litUint 1))))
  , .assign (.var "Hxz")
      (.bin "+" (.var "Hxz")
        (.index (.var "springHess") (.bin "+" (.var "shb") (.litUint 2))))
  , .assign (.var "Hyy")
      (.bin "+" (.var "Hyy")
        (.index (.var "springHess") (.bin "+" (.var "shb") (.litUint 3))))
  , .assign (.var "Hyz")
      (.bin "+" (.var "Hyz")
        (.index (.var "springHess") (.bin "+" (.var "shb") (.litUint 4))))
  , .assign (.var "Hzz")
      (.bin "+" (.var "Hzz")
        (.index (.var "springHess") (.bin "+" (.var "shb") (.litUint 5))))
  ]

private def body : List SlangStmt :=
  [ .declInit u  "lane"   (.member (.var "tid") "x")
  , .declInit u  "v"
      (.index (.var "vertPerm")
        (.bin "+" (.var "lane")
                  (.member (.var "params") "colorOffset")))
  , .declInit u  "sStart" (.index (.var "vertSpringOffset") (.var "v"))
  , .declInit u  "sEnd"   (.index (.var "vertSpringOffset")
                            (.bin "+" (.var "v") (.litUint 1)))
  , .declInit u  "hb"     (.bin "*" (.litUint 6) (.var "v"))
  , .declInit f3 "g"      (.index (.var "gScratch") (.var "v"))
  , .declInit f  "gx"     (.member (.var "g") "x")
  , .declInit f  "gy"     (.member (.var "g") "y")
  , .declInit f  "gz"     (.member (.var "g") "z")
  , .declInit f  "Hxx"    (.index (.var "hScratch") (.var "hb"))
  , .declInit f  "Hxy"    (.index (.var "hScratch") (.bin "+" (.var "hb") (.litUint 1)))
  , .declInit f  "Hxz"    (.index (.var "hScratch") (.bin "+" (.var "hb") (.litUint 2)))
  , .declInit f  "Hyy"    (.index (.var "hScratch") (.bin "+" (.var "hb") (.litUint 3)))
  , .declInit f  "Hyz"    (.index (.var "hScratch") (.bin "+" (.var "hb") (.litUint 4)))
  , .declInit f  "Hzz"    (.index (.var "hScratch") (.bin "+" (.var "hb") (.litUint 5)))
  , .forCount "k" (.var "sStart") (.var "sEnd") loopBody
  , .assign (.index (.var "gScratch") (.var "v"))
      (.call "float3" [.var "gx", .var "gy", .var "gz"])
  , .assign (.index (.var "hScratch") (.var "hb")) (.var "Hxx")
  , .assign (.index (.var "hScratch") (.bin "+" (.var "hb") (.litUint 1))) (.var "Hxy")
  , .assign (.index (.var "hScratch") (.bin "+" (.var "hb") (.litUint 2))) (.var "Hxz")
  , .assign (.index (.var "hScratch") (.bin "+" (.var "hb") (.litUint 3))) (.var "Hyy")
  , .assign (.index (.var "hScratch") (.bin "+" (.var "hb") (.litUint 4))) (.var "Hyz")
  , .assign (.index (.var "hScratch") (.bin "+" (.var "hb") (.litUint 5))) (.var "Hzz")
  ]

def shader : SlangShaderModule :=
  { structs :=
      [ { name := "VbdGatherSpringParams"
        , fields := [⟨"colorOffset", u, Semantic.none, none, none, .qIn⟩] } ]
  , globals :=
      [ bnd 0 "springGradA"      (.roBuf f3)
      , bnd 1 "springHess"       (.roBuf f)
      , bnd 2 "vertSpringOffset" (.roBuf u)
      , bnd 3 "vertSpringIdx"    (.roBuf u)
      , bnd 4 "vertSpringRole"   (.roBuf u)
      , bnd 5 "gScratch"         (.rwBuf f3)
      , bnd 6 "hScratch"         (.rwBuf f)
      , bnd 7 "vertPerm"         (.roBuf u)
      , ⟨"params", .const "VbdGatherSpringParams", Semantic.none, some 8, some 0, .qIn⟩
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
"struct VbdGatherSpringParams {
  uint colorOffset;
};

[[vk::binding(0, 0)]]
StructuredBuffer<float3> springGradA;
[[vk::binding(1, 0)]]
StructuredBuffer<float> springHess;
[[vk::binding(2, 0)]]
StructuredBuffer<uint> vertSpringOffset;
[[vk::binding(3, 0)]]
StructuredBuffer<uint> vertSpringIdx;
[[vk::binding(4, 0)]]
StructuredBuffer<uint> vertSpringRole;
[[vk::binding(5, 0)]]
RWStructuredBuffer<float3> gScratch;
[[vk::binding(6, 0)]]
RWStructuredBuffer<float> hScratch;
[[vk::binding(7, 0)]]
StructuredBuffer<uint> vertPerm;
[[vk::binding(8, 0)]]
ConstantBuffer<VbdGatherSpringParams> params;

[shader(\"compute\")] [numthreads(64, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
  uint lane = tid.x;
  uint v = vertPerm[(lane + params.colorOffset)];
  uint sStart = vertSpringOffset[v];
  uint sEnd = vertSpringOffset[(v + 1u)];
  uint hb = (6u * v);
  float3 g = gScratch[v];
  float gx = g.x;
  float gy = g.y;
  float gz = g.z;
  float Hxx = hScratch[hb];
  float Hxy = hScratch[(hb + 1u)];
  float Hxz = hScratch[(hb + 2u)];
  float Hyy = hScratch[(hb + 3u)];
  float Hyz = hScratch[(hb + 4u)];
  float Hzz = hScratch[(hb + 5u)];
  for (uint k = sStart; k < sEnd; ++k) {
    uint c = vertSpringIdx[k];
    uint r = vertSpringRole[k];
    float sign = (1.000000 - (2.000000 * float(r)));
    float3 ga = springGradA[c];
    gx = (gx + (sign * ga.x));
    gy = (gy + (sign * ga.y));
    gz = (gz + (sign * ga.z));
    uint shb = (6u * c);
    Hxx = (Hxx + springHess[shb]);
    Hxy = (Hxy + springHess[(shb + 1u)]);
    Hxz = (Hxz + springHess[(shb + 2u)]);
    Hyy = (Hyy + springHess[(shb + 3u)]);
    Hyz = (Hyz + springHess[(shb + 4u)]);
    Hzz = (Hzz + springHess[(shb + 5u)]);
  }
  gScratch[v] = float3(gx, gy, gz);
  hScratch[hb] = Hxx;
  hScratch[(hb + 1u)] = Hxy;
  hScratch[(hb + 2u)] = Hxz;
  hScratch[(hb + 3u)] = Hyy;
  hScratch[(hb + 4u)] = Hyz;
  hScratch[(hb + 5u)] = Hzz;
}"

example : LeanSlang.emit shader = expected := by native_decide
example : shader.entryPointName = "main" := by native_decide

end Cloth.SlangCodegen.VbdGatherSpring
