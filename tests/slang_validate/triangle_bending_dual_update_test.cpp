// Real-input validator for Cloth.SlangCodegen.TriangleBendingDualUpdate —
// AVBD AL dual ramp for dihedral bending.
//
// Per constraint c (4-vertex weighted sum, rest magnitude n_target, AL γ):
//   s    = Σ_r w[4c+r] · p[idx[4c+r]]
//   C    = s · (1 − n_target/|s|)   if n_target > eps, else 0
//   λ_c ← λ_c + γ_c · C
//
// Test fixture (2 bending stencils):
//   c0 non-degenerate:
//     v0..v3 with weights (1,1,-1,-1), n_target=1, γ=1
//     positions: v0=v1=v2=(0,0,0), v3=(0,0,2)
//     s = (0,0,-2), |s|=2, scale=0.5, om=0.5
//     C = s · 0.5 = (0, 0, -1)
//     λ_init=(10,20,30) → λ_new = (10, 20, 29)
//
//   c1 degenerate (n_target=0):
//     Same weights/positions, γ=1
//     C=(0,0,0); λ unchanged at (5, 6, 7)

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <vector>

#include "triangle_bending_dual_update_emit.cpp"

static constexpr double kBudgetSeconds = 5.0;

int main() {
    const auto t0 = std::chrono::steady_clock::now();

    constexpr uint32_t N_BEND = 2;
    constexpr uint32_t GROUP_SIZE = 64;

    std::vector<Vector<float, 3>> positions = {
        Vector<float, 3>(0.0f, 0.0f, 0.0f),
        Vector<float, 3>(0.0f, 0.0f, 0.0f),
        Vector<float, 3>(0.0f, 0.0f, 0.0f),
        Vector<float, 3>(0.0f, 0.0f, 2.0f),
    };

    std::vector<uint32_t>         idx(4 * GROUP_SIZE, 0u);
    std::vector<float>            weight(4 * GROUP_SIZE, 0.0f);
    std::vector<float>            nTarget(GROUP_SIZE, 0.0f);
    std::vector<float>            gamma(GROUP_SIZE, 0.0f);
    std::vector<Vector<float, 3>> lambda(GROUP_SIZE, Vector<float, 3>(0.0f));

    // c0 — non-degenerate
    for (int r = 0; r < 4; ++r) idx[4*0 + r] = uint32_t(r);
    weight[0]=1.0f; weight[1]=1.0f; weight[2]=-1.0f; weight[3]=-1.0f;
    nTarget[0] = 1.0f;  gamma[0] = 1.0f;
    lambda[0] = Vector<float, 3>(10.0f, 20.0f, 30.0f);

    // c1 — degenerate (n_target=0)
    for (int r = 0; r < 4; ++r) idx[4*1 + r] = uint32_t(r);
    weight[4]=1.0f; weight[5]=1.0f; weight[6]=-1.0f; weight[7]=-1.0f;
    nTarget[1] = 0.0f;  gamma[1] = 1.0f;
    lambda[1] = Vector<float, 3>(5.0f, 6.0f, 7.0f);

    GlobalParams_0 gp{};
    gp.positions_0.data = positions.data(); gp.positions_0.count = positions.size();
    gp.idx_0.data       = idx.data();       gp.idx_0.count       = idx.size();
    gp.weight_0.data    = weight.data();    gp.weight_0.count    = weight.size();
    gp.nTarget_0.data   = nTarget.data();   gp.nTarget_0.count   = nTarget.size();
    gp.gamma_0.data     = gamma.data();     gp.gamma_0.count     = gamma.size();
    gp.lambda_0.data    = lambda.data();    gp.lambda_0.count    = lambda.size();

    ComputeVaryingInput vi{};
    vi.startGroupID = uint3(0, 0, 0);
    vi.endGroupID   = uint3(1, 1, 1);
    main_0(&vi, nullptr, &gp);

    const float expected[N_BEND][3] = {
        {10.0f, 20.0f, 29.0f},   // c0: +γ·(0,0,-1)
        { 5.0f,  6.0f,  7.0f},   // c1: unchanged (degenerate)
    };

    int fails = 0;
    float max_abs_diff = 0.0f;
    auto check = [&](float got, float ref, uint32_t c, int axis) {
        const float d = std::fabs(got - ref);
        if (d > max_abs_diff) max_abs_diff = d;
        if (d > 1e-5f) {
            std::fprintf(stderr, "λ mismatch c=%u axis=%d: got %g, expected %g\n",
                         c, axis, got, ref);
            ++fails;
        }
    };
    for (uint32_t c = 0; c < N_BEND; ++c)
        for (int axis = 0; axis < 3; ++axis)
            check(lambda[c][axis], expected[c][axis], c, axis);

    const auto t1 = std::chrono::steady_clock::now();
    const double elapsed = std::chrono::duration<double>(t1 - t0).count();
    if (elapsed > kBudgetSeconds) {
        std::fprintf(stderr, "triangle_bending_dual_update: TIMEOUT\n");
        return 2;
    }
    if (fails == 0) {
        std::printf("triangle_bending_dual_update: %u/%u OK (max_abs_diff=%g, %.1fms)\n",
                    N_BEND, N_BEND, max_abs_diff, elapsed * 1000.0);
        return 0;
    }
    std::fprintf(stderr, "triangle_bending_dual_update: %d FAIL\n", fails);
    return 1;
}
