/-!
# `Cloth.Avbd.Coloring` — vertex graph coloring (PR-C)

Greedy first-fit vertex coloring on the constraint hypergraph, so
the PR-D vertex-block-update kernel can dispatch one color batch at
a time — vertices of the same color are guaranteed to not share any
constraint, so their independent block updates are race-free.

Adjacency definition: vertex `u ~ v` iff some constraint lists both
`u` and `v` in its incidence list. For springs this means edge-connected;
for triangles, every pair of corners is adjacent; for bending stencils,
every pair within the 4-vertex window is adjacent.

Algorithm (Welsh–Powell / first-fit): for each vertex `v` in order
`[0..nVerts)`, scan its already-colored neighbors and assign the
smallest color not in the used set. Worst-case `Δ+1` colors where
`Δ` is the max degree.

```
vertColor[v]                        color of v (0..numColors)
vertPerm[colorOffsets[k]..k+1)      vertices in color k
```

The kernel dispatches one color at a time: for color k, run threads
over `vertPerm[colorOffsets[k]..colorOffsets[k+1])`, which gives one
thread per same-colored vertex. No two of these threads can write to
overlapping constraint state, so the per-vertex block update is safe
under parallel execution.

`native_decide`-pinned on three fixtures: disconnected, two triangles
sharing an edge, pentagon-of-springs cycle.
-/

namespace Cloth.Avbd.Coloring

/-- Output of `assignColors`. -/
structure ColorAssignment where
  vertColor    : Array Nat   -- length nVerts
  colorOffsets : Array Nat   -- length numColors + 1
  vertPerm     : Array Nat   -- length nVerts; sorted by color
  numColors    : Nat
  deriving DecidableEq, Repr

/-- Build adjacency `adj[v]` = list of vertices sharing any constraint
with `v`. Each pair within an incidence list contributes one entry to
both endpoints (may produce duplicates if the same pair appears in
multiple constraints — duplicates don't affect coloring correctness). -/
private def buildAdj (nVerts : Nat) (constraints : List (List Nat))
    : Array (Array Nat) := Id.run do
  let mut adj : Array (Array Nat) := Array.replicate nVerts #[]
  for inc in constraints do
    -- All ordered (i, j) pairs with i ≠ j within this constraint's
    -- incidence list.
    let arr := inc.toArray
    for i in [0:arr.size] do
      for j in [0:arr.size] do
        if i != j then
          let u := arr[i]!
          let v := arr[j]!
          adj := adj.modify u (·.push v)
  return adj

/-- Greedy first-fit coloring on a pre-built adjacency. -/
def assignColors (nVerts : Nat)
    (constraints : List (List Nat)) : ColorAssignment := Id.run do
  let adj := buildAdj nVerts constraints
  let mut color : Array Nat := Array.replicate nVerts 0
  let mut numColors : Nat := 0
  for v in [0:nVerts] do
    -- Mark colors already used by colored neighbors.
    let mut used : Array Bool := Array.replicate (nVerts + 1) false
    for nb in adj[v]! do
      if nb < v then
        used := used.set! color[nb]! true
    -- Find smallest unused color.
    let mut k : Nat := 0
    let mut found : Bool := false
    for cand in [0:nVerts + 1] do
      if !found && !used[cand]! then
        k := cand
        found := true
    color := color.set! v k
    if k + 1 > numColors then numColors := k + 1

  -- Build colorOffsets and vertPerm. Two-pass:
  --   pass A: count verts per color
  --   pass B: prefix sum → offsets, scatter into perm
  let mut counts : Array Nat := Array.replicate numColors 0
  for v in [0:nVerts] do
    counts := counts.modify color[v]! (· + 1)
  let mut offsets : Array Nat := Array.replicate (numColors + 1) 0
  for k in [0:numColors] do
    offsets := offsets.set! (k + 1) (offsets[k]! + counts[k]!)
  let mut perm : Array Nat := Array.replicate nVerts 0
  let mut cursor : Array Nat := Array.replicate numColors 0
  for v in [0:nVerts] do
    let c := color[v]!
    let p := offsets[c]! + cursor[c]!
    perm := perm.set! p v
    cursor := cursor.modify c (· + 1)
  return ⟨color, offsets, perm, numColors⟩

/-! ## Fixture: 3 isolated verts, no constraints

Trivial — all verts get color 0.
-/

def isolatedFixture : ColorAssignment := assignColors 3 []
def isolatedExpected : ColorAssignment :=
  { vertColor    := #[0, 0, 0]
  , colorOffsets := #[0, 3]
  , vertPerm     := #[0, 1, 2]
  , numColors    := 1 }

example : isolatedFixture = isolatedExpected := by native_decide

/-! ## Fixture: 4 verts, two triangles sharing edge (1, 2)

T0 = (0, 1, 2), T1 = (3, 2, 1). Adjacency:
  v0 ↔ {v1, v2}
  v1 ↔ {v0, v2, v3}
  v2 ↔ {v0, v1, v3}
  v3 ↔ {v1, v2}

Greedy in order 0..3:
  v0 → 0 (no colored neighbors)
  v1 → 1 (neighbor v0=0)
  v2 → 2 (neighbors v0=0, v1=1)
  v3 → 0 (neighbors v1=1, v2=2; color 0 free)
-/

def twoTriFixture : ColorAssignment := assignColors 4 [[0, 1, 2], [3, 2, 1]]
def twoTriExpected : ColorAssignment :=
  { vertColor    := #[0, 1, 2, 0]
  , colorOffsets := #[0, 2, 3, 4]
  , vertPerm     := #[0, 3, 1, 2]
  , numColors    := 3 }

example : twoTriFixture = twoTriExpected := by native_decide

/-! ## Fixture: 5-cycle of springs (pentagon)

Springs: (0,1) (1,2) (2,3) (3,4) (4,0). Odd cycle, needs 3 colors.

Greedy in order 0..4:
  v0 → 0
  v1 → 1 (neighbor v0=0)
  v2 → 0 (neighbor v1=1)
  v3 → 1 (neighbor v2=0)
  v4 → 2 (neighbors v3=1, v0=0)
-/

def pentagonFixture : ColorAssignment :=
  assignColors 5 [[0, 1], [1, 2], [2, 3], [3, 4], [4, 0]]
def pentagonExpected : ColorAssignment :=
  { vertColor    := #[0, 1, 0, 1, 2]
  , colorOffsets := #[0, 2, 4, 5]
  , vertPerm     := #[0, 2, 1, 3, 4]
  , numColors    := 3 }

example : pentagonFixture = pentagonExpected := by native_decide

end Cloth.Avbd.Coloring
