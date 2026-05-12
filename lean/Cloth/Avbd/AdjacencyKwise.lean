/-!
# `Cloth.Avbd.AdjacencyKwise` — vertex → incident-constraint CSR (generic arity)

Second half of PR-B in the AVBD port (`todo.md`). Generic K-arity
extension of `AdjacencySpring` that covers the three remaining
constraint shapes the AVBD vertex-block-update kernel (PR-D) needs:

- **K = 1**  attachment constraint (one vertex pinned to a target)
- **K = 3**  triangle membrane stretch (three corners)
- **K = 4**  triangle bending (four-vertex weighted stencil)

Each constraint contributes one entry per incidence; the gather kernel
uses `role` to pick the right per-vertex slice of the corresponding
`*_force` kernel's per-constraint output arrays (which themselves are
indexed as `[K · c + role]`).

```
offsets[v..v+1)   slice into cIdx / role for vertex v
cIdx[k]           constraint id of the k-th incidence at this vertex
role[k]           index of the vertex within constraint cIdx[k]'s
                  incidence list (0 .. K−1)
```

The build is generic in K: each constraint is a `List Nat` of vertex
indices, and the role of vertex `v` in constraint `c` is its position
in that list (first occurrence if there are duplicates — degenerate
but handled deterministically). Springs (K=2) live in
`Cloth.Avbd.AdjacencySpring` as their own typed wrapper (already
shipped, no change needed).
-/

namespace Cloth.Avbd.AdjacencyKwise

/-- Generic CSR adjacency. Same shape as `AdjacencySpring.SpringCsr`
but the role values range over `0 .. K−1` rather than just `{0, 1}`. -/
structure CsrAdj where
  offsets : Array Nat
  cIdx    : Array Nat
  role    : Array Nat
  deriving DecidableEq, Repr

/-- Build the CSR from `constraints[c] : List Nat` listing the K vertex
indices touching constraint `c`. All constraints may have different K
in principle — the role for each entry is its position in that
constraint's incidence list. In practice each kernel type fixes K. -/
def CsrAdj.build (nVerts : Nat)
    (constraints : List (List Nat)) : CsrAdj := Id.run do
  -- Pass 1: count incidences per vertex.
  let mut counts : Array Nat := Array.replicate nVerts 0
  for inc in constraints do
    for v in inc do
      counts := counts.modify v (· + 1)

  -- Prefix sum into offsets.
  let mut offsets : Array Nat := Array.replicate (nVerts + 1) 0
  for v in [0:nVerts] do
    offsets := offsets.set! (v + 1) (offsets[v]! + counts[v]!)

  -- Pass 2: scatter constraint ids + roles into CSR slots.
  let total := offsets[nVerts]!
  let mut cIdx   : Array Nat := Array.replicate total 0
  let mut role   : Array Nat := Array.replicate total 0
  let mut cursor : Array Nat := Array.replicate nVerts 0
  let mut c := 0
  for inc in constraints do
    let mut r := 0
    for v in inc do
      let p := offsets[v]! + cursor[v]!
      cIdx := cIdx.set! p c
      role := role.set! p r
      cursor := cursor.modify v (· + 1)
      r := r + 1
    c := c + 1

  return ⟨offsets, cIdx, role⟩

/-! ## Fixture: K = 1, attachments

Three vertices, two attachments: attachment 0 pins v0, attachment 1
pins v2.

  v0: 1 incidence — attachment 0, role 0
  v1: 0 incidences
  v2: 1 incidence — attachment 1, role 0

  offsets = [0, 1, 1, 2]
  cIdx    = [0, 1]
  role    = [0, 0]
-/

def attachFixture : CsrAdj :=
  CsrAdj.build 3 [[0], [2]]

def attachExpected : CsrAdj :=
  { offsets := #[0, 1, 1, 2]
  , cIdx    := #[0, 1]
  , role    := #[0, 0] }

example : attachFixture = attachExpected := by native_decide

/-! ## Fixture: K = 3, triangles

Four vertices, two triangles sharing edge (1, 2):
  T0 = (0, 1, 2)
  T1 = (3, 2, 1)

  v0: T0 role 0
  v1: T0 role 1, T1 role 2
  v2: T0 role 2, T1 role 1
  v3: T1 role 0

  offsets = [0, 1, 3, 5, 6]
  cIdx    = [0, 0, 1, 0, 1, 1]
  role    = [0, 1, 2, 2, 1, 0]
-/

def triFixture : CsrAdj :=
  CsrAdj.build 4 [[0, 1, 2], [3, 2, 1]]

def triExpected : CsrAdj :=
  { offsets := #[0, 1, 3, 5, 6]
  , cIdx    := #[0, 0, 1, 0, 1, 1]
  , role    := #[0, 1, 2, 2, 1, 0] }

example : triFixture = triExpected := by native_decide

/-! ## Fixture: K = 4, dihedral bending

Five vertices, two bending stencils:
  B0 = (0, 1, 2, 3)
  B1 = (1, 2, 3, 4)

  v0: B0 role 0
  v1: B0 role 1, B1 role 0
  v2: B0 role 2, B1 role 1
  v3: B0 role 3, B1 role 2
  v4: B1 role 3

  offsets = [0, 1, 3, 5, 7, 8]
  cIdx    = [0, 0, 1, 0, 1, 0, 1, 1]
  role    = [0, 1, 0, 2, 1, 3, 2, 3]
-/

def bendFixture : CsrAdj :=
  CsrAdj.build 5 [[0, 1, 2, 3], [1, 2, 3, 4]]

def bendExpected : CsrAdj :=
  { offsets := #[0, 1, 3, 5, 7, 8]
  , cIdx    := #[0, 0, 1, 0, 1, 0, 1, 1]
  , role    := #[0, 1, 0, 2, 1, 3, 2, 3] }

example : bendFixture = bendExpected := by native_decide

end Cloth.Avbd.AdjacencyKwise
