// Real-input validator for Cloth.SlangCodegen.CGBeta — the
// single-thread GPU-side β kernel + in-place dotOld update for the
// CG inner loop.
//
//   β        = dot_new / dot_old   (each reconstructed from df32 hi+lo)
//   betaOut  = +β
//   dotOld  := dotNew              (in-place; sets up next iter)
//
// Test fixture: dotNew = (50, 0), dotOld = (110, 0).
//   expected β = 50/110 ≈ 0.4545454...
//   after the dispatch, dotOld should equal dotNew = (50, 0).

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>

#include "cg_beta_emit.cpp"

static constexpr double kBudgetSeconds = 5.0;

int main() {
    const auto t0 = std::chrono::steady_clock::now();

    float dotNew[2] = { 50.0f, 0.0f};
    float dotOld[2] = {110.0f, 0.0f};
    float betaOut[1] = {0.0f};

    GlobalParams_0 gp{};
    gp.dotNew_0.data  = dotNew;   gp.dotNew_0.count  = 2;
    gp.dotOld_0.data  = dotOld;   gp.dotOld_0.count  = 2;
    gp.betaOut_0.data = betaOut;  gp.betaOut_0.count = 1;

    ComputeVaryingInput vi{};
    vi.startGroupID = uint3(0, 0, 0);
    vi.endGroupID   = uint3(1, 1, 1);
    main_0(&vi, nullptr, &gp);

    const float expected_b = 50.0f / 110.0f;
    int   fails        = 0;
    float max_abs_diff = 0.0f;

    auto check = [&](const char* what, float got, float ref) {
        const float d = std::fabs(got - ref);
        if (d > max_abs_diff) max_abs_diff = d;
        if (d > 1e-6f) {
            std::fprintf(stderr,
                "cg_beta mismatch (%s): got %g, expected %g (diff %g)\n",
                what, got, ref, d);
            ++fails;
        }
    };

    check("betaOut[0]", betaOut[0], expected_b);
    check("dotOld[0]",  dotOld[0],  50.0f);
    check("dotOld[1]",  dotOld[1],   0.0f);

    const auto t1 = std::chrono::steady_clock::now();
    const double elapsed = std::chrono::duration<double>(t1 - t0).count();
    if (elapsed > kBudgetSeconds) {
        std::fprintf(stderr, "cg_beta: TIMEOUT — %.3fs > %.1fs\n",
                     elapsed, kBudgetSeconds);
        return 2;
    }
    if (fails == 0) {
        std::printf("cg_beta: 3/3 OK (β=%.6f, dotOld now=[%.1f, %.1f], "
                    "max_abs_diff=%g, %.1fms)\n",
                    betaOut[0], dotOld[0], dotOld[1], max_abs_diff,
                    elapsed * 1000.0);
        return 0;
    }
    std::fprintf(stderr, "cg_beta: %d FAIL\n", fails);
    return 1;
}
