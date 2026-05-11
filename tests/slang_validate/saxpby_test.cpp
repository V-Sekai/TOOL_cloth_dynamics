// Real-input validator for Cloth.SlangCodegen.Saxpby.
//   dst[i] = alpha * x[i] + beta * y[i]
// Mirrors curvenet/saxpby_test.cpp for ABI compatibility.

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>

#include "saxpby_emit.cpp"

static constexpr double kBudgetSeconds = 5.0;

int main() {
    const auto t0 = std::chrono::steady_clock::now();

    constexpr uint32_t N = 5;
    SaxpbyParams_0 params{N, 3.0f, 4.0f};

    // dst[i] = 3 * x[i] + 4 * y[i]
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
    gp.params_0   = &params;
    gp.x_0.data   = x;     gp.x_0.count   = N;
    gp.y_0.data   = y;     gp.y_0.count   = N;
    gp.dst_0.data = dst;   gp.dst_0.count = N;

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
            if (fails < 5) {
                std::fprintf(stderr,
                    "saxpby mismatch at i=%u: got %g, expected %g (diff %g)\n",
                    i, dst[i], expected[i], d);
            }
            ++fails;
        }
    }

    const auto t1 = std::chrono::steady_clock::now();
    const double elapsed = std::chrono::duration<double>(t1 - t0).count();
    if (elapsed > kBudgetSeconds) {
        std::fprintf(stderr, "saxpby: TIMEOUT — %.3fs > %.1fs\n",
                     elapsed, kBudgetSeconds);
        return 2;
    }
    if (fails == 0) {
        std::printf("saxpby: %u/%u OK (alpha=%g, beta=%g, max_abs_diff=%g, %.1fms)\n",
                    N, N, params.alpha_0, params.beta_0, max_abs_diff, elapsed * 1000.0);
        return 0;
    }
    std::fprintf(stderr, "saxpby: %d FAIL out of %u\n", fails, N);
    return 1;
}
