// Real-input validator for Cloth.SlangCodegen.AttachmentDualUpdate —
// AVBD augmented-Lagrangian dual ramp per attachment.
//
// Per attachment c pinning vertex v=vertIdx[c] to fixedPos[c] with
// AL penalty gamma[c], updates the multiplier:
//   λ_c ← λ_c + γ_c · (p_v − fixedPos[c])
//
// Test fixture (2 attachments over 3 verts):
//   v0 = (1, 2, 3)   v1 = (4, 5, 6)   v2 = (0, 0, 0)
//   c0: pin v0 to (1, 0, 3), γ=2, λ_init=(10, 20, 30)
//       C = (0, 2, 0); update += 2·C = (0, 4, 0)
//       λ_new = (10, 24, 30)
//   c1: pin v1 to (4, 5, 0), γ=1, λ_init=(-1, -2, -3)
//       C = (0, 0, 6); update += 1·C = (0, 0, 6)
//       λ_new = (-1, -2, 3)
//
// Padding (slots 2..63): vertIdx=2 (v2=zero), fixedPos=zero, γ=0 →
// no-op update.

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <vector>

#include "attachment_dual_update_emit.cpp"

static constexpr double kBudgetSeconds = 5.0;

int main() {
    const auto t0 = std::chrono::steady_clock::now();

    constexpr uint32_t N_ATTACH = 2;
    constexpr uint32_t GROUP_SIZE = 64;

    std::vector<Vector<float, 3>> positions = {
        Vector<float, 3>(1.0f, 2.0f, 3.0f),
        Vector<float, 3>(4.0f, 5.0f, 6.0f),
        Vector<float, 3>(0.0f, 0.0f, 0.0f),
    };

    std::vector<uint32_t>         vertIdx(GROUP_SIZE, 2u);
    std::vector<Vector<float, 3>> fixedPos(GROUP_SIZE, Vector<float, 3>(0.0f));
    std::vector<float>            gamma(GROUP_SIZE, 0.0f);
    std::vector<Vector<float, 3>> lambda(GROUP_SIZE, Vector<float, 3>(0.0f));

    vertIdx[0] = 0u;
    fixedPos[0] = Vector<float, 3>(1.0f, 0.0f, 3.0f);
    gamma[0] = 2.0f;
    lambda[0] = Vector<float, 3>(10.0f, 20.0f, 30.0f);

    vertIdx[1] = 1u;
    fixedPos[1] = Vector<float, 3>(4.0f, 5.0f, 0.0f);
    gamma[1] = 1.0f;
    lambda[1] = Vector<float, 3>(-1.0f, -2.0f, -3.0f);

    GlobalParams_0 gp{};
    gp.positions_0.data = positions.data();  gp.positions_0.count = positions.size();
    gp.vertIdx_0.data   = vertIdx.data();    gp.vertIdx_0.count   = vertIdx.size();
    gp.fixedPos_0.data  = fixedPos.data();   gp.fixedPos_0.count  = fixedPos.size();
    gp.gamma_0.data     = gamma.data();      gp.gamma_0.count     = gamma.size();
    gp.lambda_0.data    = lambda.data();     gp.lambda_0.count    = lambda.size();

    ComputeVaryingInput vi{};
    vi.startGroupID = uint3(0, 0, 0);
    vi.endGroupID   = uint3(1, 1, 1);
    main_0(&vi, nullptr, &gp);

    const float expected_lambda[N_ATTACH][3] = {
        {10.0f, 24.0f, 30.0f},
        {-1.0f, -2.0f,  3.0f},
    };

    int fails = 0;
    float max_abs_diff = 0.0f;
    auto check = [&](float got, float ref, uint32_t c, int axis) {
        const float d = std::fabs(got - ref);
        if (d > max_abs_diff) max_abs_diff = d;
        if (d > 1e-6f) {
            if (fails < 5) {
                std::fprintf(stderr,
                    "lambda mismatch at c=%u axis=%d: got %g, expected %g\n",
                    c, axis, got, ref);
            }
            ++fails;
        }
    };
    for (uint32_t c = 0; c < N_ATTACH; ++c) {
        for (int axis = 0; axis < 3; ++axis) {
            check(lambda[c][axis], expected_lambda[c][axis], c, axis);
        }
    }

    const auto t1 = std::chrono::steady_clock::now();
    const double elapsed = std::chrono::duration<double>(t1 - t0).count();
    if (elapsed > kBudgetSeconds) {
        std::fprintf(stderr, "attachment_dual_update: TIMEOUT — %.3fs > %.1fs\n",
                     elapsed, kBudgetSeconds);
        return 2;
    }
    if (fails == 0) {
        std::printf("attachment_dual_update: %u/%u OK (max_abs_diff=%g, %.1fms)\n",
                    N_ATTACH, N_ATTACH, max_abs_diff, elapsed * 1000.0);
        return 0;
    }
    std::fprintf(stderr, "attachment_dual_update: %d FAIL\n", fails);
    return 1;
}
