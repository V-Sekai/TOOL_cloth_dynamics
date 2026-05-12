/-!
# `Cloth.Avbd.AdjacencySpring` — vertex → incident-spring CSR

First half of PR-B in the AVBD port (see `todo.md`). Given a list of
spring constraints (each as a pair of endpoint vertex indices) and the
number of vertices, produces a CSR-style adjacency that the
vertex-block-update kernel (PR-D) will gather over.

```
offsets[v..v+1)   slice into `springIdx` / `role` for vertex v's
                  incident springs
springIdx[k]      spring id of the k-th incidence
role[k]           0 if vertex is endpoint a, 1 if endpoint b
                  (used by the gather kernel to negate the gradient
                  for endpoint b — see SpringForce's per-endpoint
                  sign convention)
```

Per spring, both endpoints get an entry — so total incidences =
`2 · N_springs`. Memory: `offsets` is `N_verts + 1`, `springIdx` and
`role` are each `2 · N_springs`. All `UInt32` to match the GPU
buffer ABI we'll feed at runtime.

Pure-functional build (no `IO`). `native_decide` example pins the
output on a 3-vertex 2-spring fixture so any regression in
`SpringCsr.build` trips immediately.
-/

namespace Cloth.Avbd.AdjacencySpring

/-- CSR layout: `offsets` is monotonically increasing length
`N_verts + 1`; `springIdx[offsets[v]..offsets[v+1])` lists the spring
ids touching vertex `v`; `role` is the parallel role per incidence. -/
structure SpringCsr where
  offsets   : Array Nat
  springIdx : Array Nat
  role      : Array Nat
  deriving DecidableEq, Repr

/-- Build the CSR adjacency from a list of spring endpoint pairs.

`nVerts` is the total vertex count (must exceed every index that
appears in `springs`; we do not check, the caller is expected to
own the topology).

`springs[i] = (a, b)` declares that spring `i` connects vertex `a`
(role 0) and vertex `b` (role 1). -/
def SpringCsr.build (nVerts : Nat) (springs : List (Nat × Nat)) : SpringCsr := Id.run do
  -- Pass 1: count incidences per vertex.
  let mut counts : Array Nat := Array.replicate nVerts 0
  for (a, b) in springs do
    counts := counts.modify a (· + 1)
    counts := counts.modify b (· + 1)

  -- Prefix sum into offsets[0..nVerts+1].
  let mut offsets : Array Nat := Array.replicate (nVerts + 1) 0
  for v in [0:nVerts] do
    offsets := offsets.set! (v + 1) (offsets[v]! + counts[v]!)

  -- Pass 2: scatter spring ids + roles into CSR slots, tracking the
  -- write cursor per vertex.
  let total := offsets[nVerts]!
  let mut springIdx : Array Nat := Array.replicate total 0
  let mut role      : Array Nat := Array.replicate total 0
  let mut cursor    : Array Nat := Array.replicate nVerts 0
  let mut i := 0
  for (a, b) in springs do
    let pa := offsets[a]! + cursor[a]!
    springIdx := springIdx.set! pa i
    role      := role.set!      pa 0
    cursor    := cursor.modify a (· + 1)
    let pb := offsets[b]! + cursor[b]!
    springIdx := springIdx.set! pb i
    role      := role.set!      pb 1
    cursor    := cursor.modify b (· + 1)
    i := i + 1

  return ⟨offsets, springIdx, role⟩

/-! ## Fixture: 3 verts, 2 springs

Springs: `[(0, 1), (0, 2)]`. Edges go 0—1 and 0—2 (a star centred on
v0). Each spring has one endpoint at v0 (role 0) and one at the leaf.

Expected:
  v0: 2 incidences — spring 0 role 0, spring 1 role 0
  v1: 1 incidence  — spring 0 role 1
  v2: 1 incidence  — spring 1 role 1

  offsets   = [0, 2, 3, 4]
  springIdx = [0, 1, 0, 1]
  role      = [0, 0, 1, 1]
-/

def starFixture : SpringCsr := SpringCsr.build 3 [(0, 1), (0, 2)]

def starExpected : SpringCsr :=
  { offsets   := #[0, 2, 3, 4]
  , springIdx := #[0, 1, 0, 1]
  , role      := #[0, 0, 1, 1] }

example : starFixture = starExpected := by native_decide

/-! ## Fixture: 4 verts in a chain

Springs: `[(0, 1), (1, 2), (2, 3)]`. Linear chain.
  v0: spring 0 role 0
  v1: spring 0 role 1, spring 1 role 0
  v2: spring 1 role 1, spring 2 role 0
  v3: spring 2 role 1

  offsets   = [0, 1, 3, 5, 6]
  springIdx = [0, 0, 1, 1, 2, 2]
  role      = [0, 1, 0, 1, 0, 1]
-/

def chainFixture : SpringCsr := SpringCsr.build 4 [(0, 1), (1, 2), (2, 3)]

def chainExpected : SpringCsr :=
  { offsets   := #[0, 1, 3, 5, 6]
  , springIdx := #[0, 0, 1, 1, 2, 2]
  , role      := #[0, 1, 0, 1, 0, 1] }

example : chainFixture = chainExpected := by native_decide

end Cloth.Avbd.AdjacencySpring
