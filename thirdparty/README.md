# Third-party dependencies

## FCL and libccd (git-subrepo)

FCL and libccd are brought in via **git subrepo**. From the project root (`TOOL_cloth_dynamics`), install [git-subrepo](https://github.com/ingydotnet/git-subrepo) if needed. If `thirdparty/libccd` or `thirdparty/fcl` already exist (e.g. from a manual clone), remove them first, then run:

```bash
# Clone libccd first (FCL depends on it)
git subrepo clone https://github.com/danfis/libccd thirdparty/libccd

# Clone FCL (use tag 0.7.0 for stability)
git subrepo clone https://github.com/flexible-collision-library/fcl thirdparty/fcl -- --branch 0.7.0
```

Or use the justfile target: `just thirdparty-fetch`.

- **FCL:** https://github.com/flexible-collision-library/fcl (tag `0.7.0`).
- **libccd:** https://github.com/danfis/libccd.

### Building with FCL

- FCL is **on by default** (`USE_FCL=ON`). The project builds libccd and FCL as subdirectories and links the simulation to `fcl`. To disable: `-DUSE_FCL=OFF`.
- **Eigen compatibility:** FCL 0.7 was written for an older Eigen API. When using a newer Eigen (e.g. from `external/eigen`), you may need to patch FCL:
  - In `thirdparty/fcl/include/fcl/broadphase/broadphase_SSaP-inl.h`, AABB coefficient access: use `(aabb.min_)(i)` and `(aabb.max_)(i)` instead of `aabb.min_[i]` / `aabb.max_[i]` if you see "no match for 'operator[]'" errors.
  - Similar changes may be needed in `broadphase_SaP-inl.h` and other broadphase headers if your Eigen does not provide `operator[]` for const vectors.

### Using FCL collision

When `USE_FCL=ON`, the simulation uses `collisionDetectionFCL` (cloth BVH, primitive shapes, self-collision, and the same layering as the custom path via `contactSorting`). Without FCL, the default custom collision path is used.
