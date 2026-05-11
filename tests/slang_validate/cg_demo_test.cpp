// Composition test for spmv + saxpby — runs a textbook Conjugate Gradient
// solve to convergence on a 3x3 SPD system, dispatching only those two
// slangc-emitted kernels per iteration (dot products are computed on the
// host because slangc-cpp target can't link the dot_reduce kernel — see
// Cloth.SlangCodegen.DotReduce docstring).
//
// System:
//
//   A = [ 4 1 0 ]      b = [5, 7, 6]
//       [ 1 3 0 ]      analytic x = A⁻¹·b = [8/11, 23/11, 3]
//       [ 0 0 2 ]                          ≈ [0.7272727, 2.0909091, 3.0]
//
//   CSR:
//     rowPtr = [0, 2, 4, 5]
//     colIdx = [0, 1, 0, 1, 2]
//     values = [4, 1, 1, 3, 2]
//
// CG (textbook, Shewchuk § B.2):
//
//   r₀ = b − A·x₀    (= b when x₀ = 0)
//   p₀ = r₀
//   δ_new = dot(r₀, r₀)
//   loop:
//     q     = A·p              ← spmv kernel
//     α     = δ_new / dot(p,q)
//     x    := 1·x + α·p        ← saxpby kernel (in-place on x)
//     r    := 1·r + (−α)·q     ← saxpby kernel (in-place on r)
//     δ_old = δ_new
//     δ_new = dot(r, r)
//     if δ_new < tol²·δ₀ break
//     β     = δ_new / δ_old
//     p    := 1·r + β·p        ← saxpby kernel (in-place on p)
//
// CG on a 3×3 SPD matrix converges in ≤ 3 iterations in exact
// arithmetic; with fp32 storage we accept 1e-5 absolute tolerance and
// allow up to 20 iterations for safety.
//
// **How two kernels co-exist in one TU.** slangc-cpp emits each kernel
// as a translation unit with `main_0` and `GlobalParams_0` at file
// scope. `main_0` is wrapped in SLANG_PRELUDE_EXPORT (= `extern "C"`),
// so naive namespace wrapping leaks the symbol to global scope.
//
// We work around this by `#undef`-ing the macro inside each namespace
// before `#include`-ing the emit.cpp, so the kernel functions become
// regular C++ functions that respect the surrounding namespace. The
// `<cstdint>` / Slang prelude headers are pulled in once at file scope
// before the namespaces (their include guards make the per-namespace
// `#include`s of the prelude no-ops).

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>

#include "slang-cpp-prelude.h"

namespace cg_spmv {
    #pragma push_macro("SLANG_PRELUDE_EXTERN_C")
    #pragma push_macro("SLANG_PRELUDE_EXTERN_C_START")
    #pragma push_macro("SLANG_PRELUDE_EXTERN_C_END")
    #undef  SLANG_PRELUDE_EXTERN_C
    #undef  SLANG_PRELUDE_EXTERN_C_START
    #undef  SLANG_PRELUDE_EXTERN_C_END
    #define SLANG_PRELUDE_EXTERN_C
    #define SLANG_PRELUDE_EXTERN_C_START
    #define SLANG_PRELUDE_EXTERN_C_END
    #include "spmv_emit.cpp"
    #pragma pop_macro("SLANG_PRELUDE_EXTERN_C")
    #pragma pop_macro("SLANG_PRELUDE_EXTERN_C_START")
    #pragma pop_macro("SLANG_PRELUDE_EXTERN_C_END")
}

namespace cg_saxpby {
    #pragma push_macro("SLANG_PRELUDE_EXTERN_C")
    #pragma push_macro("SLANG_PRELUDE_EXTERN_C_START")
    #pragma push_macro("SLANG_PRELUDE_EXTERN_C_END")
    #undef  SLANG_PRELUDE_EXTERN_C
    #undef  SLANG_PRELUDE_EXTERN_C_START
    #undef  SLANG_PRELUDE_EXTERN_C_END
    #define SLANG_PRELUDE_EXTERN_C
    #define SLANG_PRELUDE_EXTERN_C_START
    #define SLANG_PRELUDE_EXTERN_C_END
    #include "saxpby_emit.cpp"
    #pragma pop_macro("SLANG_PRELUDE_EXTERN_C")
    #pragma pop_macro("SLANG_PRELUDE_EXTERN_C_START")
    #pragma pop_macro("SLANG_PRELUDE_EXTERN_C_END")
}

static constexpr double kBudgetSeconds = 5.0;

static double hostDot(const float* a, const float* b, uint32_t n) {
    // Plain fp64 accumulation. For n = 3 there's no precision concern;
    // for larger problems the matching GPU dispatch would use the
    // dot_reduce kernel (df32) instead.
    double s = 0.0;
    for (uint32_t i = 0; i < n; ++i) s += double(a[i]) * double(b[i]);
    return s;
}

int main() {
    const auto t0 = std::chrono::steady_clock::now();

    constexpr uint32_t N    = 3;
    constexpr uint32_t NNZ  = 5;
    constexpr int MAX_ITER  = 20;
    constexpr double TOL    = 1e-6;

    // SPD system.
    int32_t rowPtr[N + 1] = {0, 2, 4, 5};
    int32_t colIdx[NNZ]   = {0, 1, 0, 1, 2};
    float   values[NNZ]   = {4.0f, 1.0f, 1.0f, 3.0f, 2.0f};
    float   b[N]          = {5.0f, 7.0f, 6.0f};

    // CG workspace.
    float x[N] = {0.0f, 0.0f, 0.0f};   // initial guess: zero → r₀ = b
    float r[N] = {b[0], b[1], b[2]};
    float p[N] = {b[0], b[1], b[2]};
    float q[N] = {0.0f, 0.0f, 0.0f};   // A·p workspace

    double delta_new = hostDot(r, r, N);
    const double delta_0 = delta_new;

    // Persistent parameter buffers (one per kernel).
    cg_spmv::SpmvParams_0     sp{N};
    cg_spmv::GlobalParams_0   gp_spmv{};
    gp_spmv.params_0      = &sp;
    gp_spmv.rowPtr_0.data = rowPtr; gp_spmv.rowPtr_0.count = N + 1;
    gp_spmv.colIdx_0.data = colIdx; gp_spmv.colIdx_0.count = NNZ;
    gp_spmv.values_0.data = values; gp_spmv.values_0.count = NNZ;
    gp_spmv.x_0.data      = p;      gp_spmv.x_0.count      = N;
    gp_spmv.y_0.data      = q;      gp_spmv.y_0.count      = N;

    cg_saxpby::SaxpbyParams_0 spy{N, 0.0f, 0.0f};
    cg_saxpby::GlobalParams_0 gp_sax{};
    gp_sax.params_0 = &spy;

    ComputeVaryingInput vi{};
    vi.startGroupID = uint3(0, 0, 0);
    vi.endGroupID   = uint3(1, 1, 1);

    int iter = 0;
    for (; iter < MAX_ITER; ++iter) {
        // q = A · p
        gp_spmv.x_0.data = p; gp_spmv.x_0.count = N;
        gp_spmv.y_0.data = q; gp_spmv.y_0.count = N;
        cg_spmv::main_0(&vi, nullptr, &gp_spmv);

        const double pq    = hostDot(p, q, N);
        const double alpha = delta_new / pq;

        // x := 1·x + α·p     (saxpby with src x aliased to dst)
        spy.alpha_0 = 1.0f; spy.beta_0 = float(alpha);
        gp_sax.x_0.data = x; gp_sax.x_0.count = N;
        gp_sax.y_0.data = p; gp_sax.y_0.count = N;
        gp_sax.dst_0.data = x; gp_sax.dst_0.count = N;
        cg_saxpby::main_0(&vi, nullptr, &gp_sax);

        // r := 1·r + (−α)·q
        spy.alpha_0 = 1.0f; spy.beta_0 = float(-alpha);
        gp_sax.x_0.data = r; gp_sax.x_0.count = N;
        gp_sax.y_0.data = q; gp_sax.y_0.count = N;
        gp_sax.dst_0.data = r; gp_sax.dst_0.count = N;
        cg_saxpby::main_0(&vi, nullptr, &gp_sax);

        const double delta_old = delta_new;
        delta_new = hostDot(r, r, N);
        if (delta_new < TOL * TOL * delta_0) { ++iter; break; }

        const double beta = delta_new / delta_old;

        // p := 1·r + β·p
        spy.alpha_0 = 1.0f; spy.beta_0 = float(beta);
        gp_sax.x_0.data = r; gp_sax.x_0.count = N;
        gp_sax.y_0.data = p; gp_sax.y_0.count = N;
        gp_sax.dst_0.data = p; gp_sax.dst_0.count = N;
        cg_saxpby::main_0(&vi, nullptr, &gp_sax);
    }

    // Compare against analytic A⁻¹·b = [8/11, 23/11, 3].
    const float expected[N] = {8.0f/11.0f, 23.0f/11.0f, 3.0f};
    int   fails        = 0;
    float max_abs_diff = 0.0f;
    for (uint32_t i = 0; i < N; ++i) {
        const float d = std::fabs(x[i] - expected[i]);
        if (d > max_abs_diff) max_abs_diff = d;
        if (d > 1e-5f) {
            if (fails < 5) {
                std::fprintf(stderr,
                    "cg_demo mismatch at i=%u: got %g, expected %g (diff %g)\n",
                    i, x[i], expected[i], d);
            }
            ++fails;
        }
    }

    const auto t1 = std::chrono::steady_clock::now();
    const double elapsed = std::chrono::duration<double>(t1 - t0).count();
    if (elapsed > kBudgetSeconds) {
        std::fprintf(stderr, "cg_demo: TIMEOUT — %.3fs > %.1fs\n",
                     elapsed, kBudgetSeconds);
        return 2;
    }
    if (fails == 0) {
        std::printf("cg_demo: %u/%u OK (iters=%d, max_abs_diff=%g, residual=%g, %.1fms)\n",
                    N, N, iter, max_abs_diff, std::sqrt(delta_new / delta_0),
                    elapsed * 1000.0);
        return 0;
    }
    std::fprintf(stderr,
        "cg_demo: FAIL (iters=%d, max_abs_diff=%g, x=[%g %g %g])\n",
        iter, max_abs_diff, x[0], x[1], x[2]);
    return 1;
}
