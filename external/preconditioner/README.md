# Schwarz preconditioner (Option B: copied sources)

Sources copied from `preconditioner-for-cloth-and-deformable-body-simulation` for use as the backward-pass preconditioner in TOOL_cloth_dynamics. No submodule; full control over patches and updates.

## Contents

- **Core:** `SeSchwarzPreconditioner.h/.cpp`, `SeOmp.h/.cpp`
- **Types & math:** `SePreDefine.h`, `SeVector.h`, `SeVectorSimd.h`, `SeMatrix.h`, `SeMath.h`, `SeMathSimd.h`, `SeUtility.h`, `SeArray2D.h`, `SeCsr.h`, `SeIntrinsic.h`
- **Collision & geometry:** `SeCollisionElements.h`, `SeAabb.h`, `SeAabbSimd.h`, `SeMorton.h`

## Patches applied (Linux/macOS)

- **SePreDefine.h:** `SE_ALIGN` / `SE_INLINE` for non-MSVC (GCC/Clang).
- **SeOmp.h:** OpenMP pragmas via `_Pragma()` on non-MSVC.
- **SeOmp.cpp:** `CPU_THREAD_NUM` uses `omp_get_num_procs()` on all platforms.
- **SeIntrinsic.h:** `<intrin.h>` only on Windows; `<strings.h>` for `ffs()` on POSIX.
- **SeVectorSimd.h:** SSE path enabled for `__SSE__`/`__x86_64__`; union uses `m128_f32[4]` and `struct { float x,y,z,w; }` for portability; scalar fallback when SSE unavailable.
- **SeUtility.h:** `#include <cstring>` for `std::memcpy`/`std::memset`.
- **SeVector.h:** `rhs.values[i]` → `rhs.m_data[i]`.
- **SeMath.h / SeMathSimd.h:** `std::isnan` / `std::isinf`, `#include <cmath>`.
- **SeAabb.h / SeAabbSimd.h:** Removed `static` from explicit template specializations (C++ compliance).

## Design (from upstream)

The preconditioner computes the **inverse** of each domain sub-matrix (rather than a Cholesky/LDLT factorization) so that one thread per domain can apply the preconditioner without repeated global-memory reads; triangular solves would parallelize poorly at typical mesh sizes. Single precision is used by default; stability in simulation is attributed to small sub-matrices and better conditioning after dropping inter-domain links. See **Design rationale** in `docs/PRECONDITIONER_INVESTIGATION.md`.

## Usage

The executable links these sources; no separate library. To wire the preconditioner into the backward solve, see `docs/PRECONDITIONER_INVESTIGATION.md` and the integration plan.

## Updating from upstream

Re-copy from the source repo and re-apply the patches above (or keep a patch set and re-apply after copy).
