// Real-input validator for Cloth.SlangCodegen.SaxpbyIndirect.
//   dst[i] = alpha[0] * x[i] + beta[0] * y[i]
//
// Same arithmetic as the original Saxpby (#12); only the source of
// alpha and beta differs. Test fixture deliberately matches the original
// saxpby_test fixture (alpha=3, beta=4) so the result must be bit-identical.

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>

#include "saxpby_indirect_emit.cpp"

static constexpr double kBudgetSeconds = 5.0;

int main() {
    const auto t0 = std::chrono::steady_clock::now();

    constexpr uint32_t N = 5;
    SaxpbyIndirectParams_0 params{N};

    float alpha[1] = {3.0f};
    float beta[1]  = {4.0f};
    float x[N]   = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
    float y[N]   = {0.5f, 1.0f, 1.5f, 2.0f, 2.5f};
    float dst[N] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    const float expected[N] = {
        3.0f * 1.0f + 4.0f * 0.5f,    // 5
        3.0f * 2.0f + 4.0f * 1.0f,    // 10
        3.0f * 3.0f + 4.0f * 1.5f,    // 15
        3.0f * 4.0f + 4.0f * 2.0f,    // 20
        3.0f * 5.0f + 4.0f * 2.5f,    // 25
    };

    GlobalParams_0 gp{};
    gp.params_0    = &params;
    gp.alpha_0.data = alpha;  gp.alpha_0.count = 1;
    gp.beta_0.data  = beta;   gp.beta_0.count  = 1;
    gp.x_0.data    = x;       gp.x_0.count     = N;
    gp.y_0.data    = y;       gp.y_0.count     = N;
    gp.dst_0.data  = dst;     gp.dst_0.count   = N;

    ComputeVaryingInput vi{};
    vi.startGroupID = uint3(0, 0, 0);
    vi.endGroupID   = uint3(1, 1, 1);
    main_0(&vi, nullptr, &gp);

    int   fails        = 0;
    float max_abs_diff = 0.0f;
    for (uint32_t i = 0; i < N; ++i) {
        const float d = std::fabs(dst[i] - expected[i]);
        if (d > max_abs_diff) max_abs_diff = d;
        if (d > 1e-6f) {
            std::fprintf(stderr,
                "saxpby_indirect mismatch at i=%u: got %g, expected %g (diff %g)\n",
                i, dst[i], expected[i], d);
            ++fails;
        }
    }

    const auto t1 = std::chrono::steady_clock::now();
    const double elapsed = std::chrono::duration<double>(t1 - t0).count();
    if (elapsed > kBudgetSeconds) {
        std::fprintf(stderr, "saxpby_indirect: TIMEOUT — %.3fs > %.1fs\n",
                     elapsed, kBudgetSeconds);
        return 2;
    }
    if (fails == 0) {
        std::printf("saxpby_indirect: %u/%u OK (alpha=%g, beta=%g, "
                    "max_abs_diff=%g, %.1fms)\n",
                    N, N, alpha[0], beta[0], max_abs_diff, elapsed * 1000.0);
        return 0;
    }
    std::fprintf(stderr, "saxpby_indirect: %d FAIL out of %u\n", fails, N);
    return 1;
}
