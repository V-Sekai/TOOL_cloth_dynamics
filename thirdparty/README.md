# Third-party dependencies

## FCL (Flexible Collision Library)

- **Location:** `thirdparty/fcl` (clone from https://github.com/flexible-collision-library/fcl, tag `0.7.0`).
- **libccd:** `thirdparty/libccd` (required by FCL; clone from https://github.com/danfis/libccd).

### Building with FCL

- Configure with `-DUSE_FCL=ON`. The project will build libccd and FCL as subdirectories and link the simulation to `fcl`.
- **Eigen compatibility:** FCL 0.7 was written for an older Eigen API. When using a newer Eigen (e.g. from `external/eigen`), you may need to patch FCL:
  - In `thirdparty/fcl/include/fcl/broadphase/broadphase_SSaP-inl.h`, AABB coefficient access: use `(aabb.min_)(i)` and `(aabb.max_)(i)` instead of `aabb.min_[i]` / `aabb.max_[i]` if you see "no match for 'operator[]'" errors.
  - Similar changes may be needed in `broadphase_SaP-inl.h` and other broadphase headers if your Eigen does not provide `operator[]` for const vectors.

### Using FCL collision

When `USE_FCL=ON`, the simulation uses `collisionDetectionFCL` (cloth BVH, primitive shapes, self-collision, and the same layering as the custom path via `contactSorting`). Without FCL, the default custom collision path is used.
