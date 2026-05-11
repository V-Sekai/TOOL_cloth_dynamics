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

    // -------- Normal case: nonzero dold ----------------------------------
    {
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
        check("normal.betaOut[0]", betaOut[0], expected_b);
        check("normal.dotOld[0]",  dotOld[0],  50.0f);
        check("normal.dotOld[1]",  dotOld[1],   0.0f);
    }

    // -------- Underflow case: dold == 0 (CG over-convergence) ------------
    // Without the ternary clamp this would give NaN from 0/0. With it,
    // β should be 0 — making the downstream saxpby `p = r + 0·p = r`,
    // which is a soft CG restart.
    {
        float dotNew[2] = {1e-12f, 0.0f};
        float dotOld[2] = {0.0f,   0.0f};
        float betaOut[1] = {0.0f};

        GlobalParams_0 gp{};
        gp.dotNew_0.data  = dotNew;   gp.dotNew_0.count  = 2;
        gp.dotOld_0.data  = dotOld;   gp.dotOld_0.count  = 2;
        gp.betaOut_0.data = betaOut;  gp.betaOut_0.count = 1;

        ComputeVaryingInput vi{};
        vi.startGroupID = uint3(0, 0, 0);
        vi.endGroupID   = uint3(1, 1, 1);
        main_0(&vi, nullptr, &gp);

        if (std::isnan(betaOut[0])) {
            std::fprintf(stderr, "cg_beta underflow: betaOut[0] is NaN (clamp not effective)\n");
            ++fails;
        }
        check("underflow.betaOut[0]", betaOut[0], 0.0f);
        check("underflow.dotOld[0]",  dotOld[0],  1e-12f);
        check("underflow.dotOld[1]",  dotOld[1],  0.0f);
    }

    const auto t1 = std::chrono::steady_clock::now();
    const double elapsed = std::chrono::duration<double>(t1 - t0).count();
    if (elapsed > kBudgetSeconds) {
        std::fprintf(stderr, "cg_beta: TIMEOUT — %.3fs > %.1fs\n",
                     elapsed, kBudgetSeconds);
        return 2;
    }
    if (fails == 0) {
        std::printf("cg_beta: 6/6 OK (normal + underflow clamp, "
                    "max_abs_diff=%g, %.1fms)\n",
                    max_abs_diff, elapsed * 1000.0);
        return 0;
    }
    std::fprintf(stderr, "cg_beta: %d FAIL\n", fails);
    return 1;
}
