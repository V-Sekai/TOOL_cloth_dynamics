// Real-input validator for Cloth.SlangCodegen.TriangleMembraneDualUpdate —
// AVBD AL dual ramp for triangle membrane constraints.
//
// Per triangle c (corners i0,i1,i2), with γ=gamma[c]:
//   F.col0 = p1-p0, F.col1 = p2-p0
//   R = closest-rotation(F) via Gram-Schmidt + 2D polar
//   e0 = F.col0 - R.col0,  e1 = F.col1 - R.col1
//   λ0[c] += γ · e0
//   λ1[c] += γ · e1
//
// Test fixture (3 verts, 1 triangle, deformed 2x in x and y):
//   v0=(0,0,0), v1=(2,0,0), v2=(0,2,0)
//   T0 = (0,1,2), γ=1
//   F.col0=(2,0,0), F.col1=(0,2,0); 2D form [2,0;0,2] → R=I
//   R.col0=(1,0,0), R.col1=(0,1,0)
//   e0=(1,0,0), e1=(0,1,0)
//   λ0_init=(10,20,30) → λ0_new=(11,20,30)
//   λ1_init=(-1,-2,-3) → λ1_new=(-1,-1,-3)

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <vector>

#include "triangle_membrane_dual_update_emit.cpp"

static constexpr double kBudgetSeconds = 5.0;

int main() {
    const auto t0 = std::chrono::steady_clock::now();

    constexpr uint32_t N_TRI = 1;
    constexpr uint32_t GROUP_SIZE = 64;

    std::vector<Vector<float, 3>> positions = {
        Vector<float, 3>(0.0f, 0.0f, 0.0f),
        Vector<float, 3>(2.0f, 0.0f, 0.0f),
        Vector<float, 3>(0.0f, 2.0f, 0.0f),
    };

    std::vector<uint32_t>         idx(3 * GROUP_SIZE, 0u);
    std::vector<float>            gamma(GROUP_SIZE, 0.0f);
    std::vector<Vector<float, 3>> lambda0(GROUP_SIZE, Vector<float, 3>(0.0f));
    std::vector<Vector<float, 3>> lambda1(GROUP_SIZE, Vector<float, 3>(0.0f));
    // M = identity for every triangle so F = P · I = P, keeping the
    // pre-IUV closest-rotation math bit-exact.
    std::vector<float>            inv_deltaUV(4 * GROUP_SIZE, 0.0f);
    for (uint32_t c = 0; c < GROUP_SIZE; ++c) {
        inv_deltaUV[4*c + 0] = 1.0f; // m00
        inv_deltaUV[4*c + 1] = 0.0f; // m01
        inv_deltaUV[4*c + 2] = 0.0f; // m10
        inv_deltaUV[4*c + 3] = 1.0f; // m11
    }

    for (uint32_t c = 0; c < GROUP_SIZE; ++c) {
        idx[3*c+0] = 0u; idx[3*c+1] = 1u; idx[3*c+2] = 2u;
    }
    idx[0] = 0u; idx[1] = 1u; idx[2] = 2u;
    gamma[0] = 1.0f;
    lambda0[0] = Vector<float, 3>(10.0f, 20.0f, 30.0f);
    lambda1[0] = Vector<float, 3>(-1.0f, -2.0f, -3.0f);

    GlobalParams_0 gp{};
    gp.positions_0.data    = positions.data();   gp.positions_0.count    = positions.size();
    gp.idx_0.data          = idx.data();         gp.idx_0.count          = idx.size();
    gp.gamma_0.data        = gamma.data();       gp.gamma_0.count        = gamma.size();
    gp.lambda0_0.data      = lambda0.data();     gp.lambda0_0.count      = lambda0.size();
    gp.lambda1_0.data      = lambda1.data();     gp.lambda1_0.count      = lambda1.size();
    gp.inv_deltaUV_0.data  = inv_deltaUV.data(); gp.inv_deltaUV_0.count  = inv_deltaUV.size();

    ComputeVaryingInput vi{};
    vi.startGroupID = uint3(0, 0, 0);
    vi.endGroupID   = uint3(1, 1, 1);
    main_0(&vi, nullptr, &gp);

    const float expected_l0[3] = {11.0f, 20.0f, 30.0f};
    const float expected_l1[3] = {-1.0f, -1.0f, -3.0f};

    int fails = 0;
    float max_abs_diff = 0.0f;
    auto check = [&](float got, float ref, const char* label, int axis) {
        const float d = std::fabs(got - ref);
        if (d > max_abs_diff) max_abs_diff = d;
        if (d > 1e-5f) {
            std::fprintf(stderr, "%s axis=%d: got %g, expected %g (diff %g)\n",
                         label, axis, got, ref, d);
            ++fails;
        }
    };
    for (int a = 0; a < 3; ++a) check(lambda0[0][a], expected_l0[a], "λ0", a);
    for (int a = 0; a < 3; ++a) check(lambda1[0][a], expected_l1[a], "λ1", a);

    const auto t1 = std::chrono::steady_clock::now();
    const double elapsed = std::chrono::duration<double>(t1 - t0).count();
    if (elapsed > kBudgetSeconds) {
        std::fprintf(stderr, "triangle_membrane_dual_update: TIMEOUT\n");
        return 2;
    }
    if (fails == 0) {
        std::printf("triangle_membrane_dual_update: %u/%u OK (max_abs_diff=%g, %.1fms)\n",
                    N_TRI, N_TRI, max_abs_diff, elapsed * 1000.0);
        return 0;
    }
    std::fprintf(stderr, "triangle_membrane_dual_update: %d FAIL\n", fails);
    return 1;
}
