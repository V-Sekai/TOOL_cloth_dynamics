// Real-input validator for Cloth.SlangCodegen.SpringForce — first AVBD
// constraint-force primitive (PR-A of the AVBD port; see todo.md).
//
// Per spring i with endpoints (a, b), rest length L, stiffness k:
//   d   = p_a − p_b
//   len = |d|
//   c   = len − L
//   gradA[i] = (k · c / len) · d     -- ∇_a E
//   hess[6i..6i+5] packed sym 3x3 = k · (d ⊗ d) / len²
//
// Test fixture (2 real springs over 3 verts):
//   v0 = (0,0,0), v1 = (3,0,0), v2 = (0,4,0)
//   spring 0: a=1, b=0, L=2.0, k=1.0
//     d=(3,0,0), len=3, c=1, gScale=1/3 → gradA=(1,0,0)
//     hess = (1/9) · [9,0,0,0,0,0] = [1,0,0,0,0,0]
//   spring 1: a=2, b=0, L=2.0, k=1.0
//     d=(0,4,0), len=4, c=2, gScale=1/2 → gradA=(0,2,0)
//     hess = (1/16) · [0,0,0,16,0,0] = [0,0,0,1,0,0]
//
// Padding (slots 2..63): a=1, b=0 (distinct positions → nonzero len),
// L=0, k=0 → gradA=(0,0,0), hess=zeros. Safe filler.

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <vector>

#include "spring_force_emit.cpp"

static constexpr double kBudgetSeconds = 5.0;

int main() {
    const auto t0 = std::chrono::steady_clock::now();

    constexpr uint32_t N_SPRINGS = 2;
    constexpr uint32_t GROUP_SIZE = 64;

    std::vector<Vector<float, 3>> positions = {
        Vector<float, 3>(0.0f, 0.0f, 0.0f),
        Vector<float, 3>(3.0f, 0.0f, 0.0f),
        Vector<float, 3>(0.0f, 4.0f, 0.0f),
    };

    std::vector<Vector<float, 3>> gradA(GROUP_SIZE, Vector<float, 3>(0.0f));
    std::vector<float>            hess(6 * GROUP_SIZE, 0.0f);

    std::vector<uint32_t> p1Idx(GROUP_SIZE, 1u);
    std::vector<uint32_t> p2Idx(GROUP_SIZE, 0u);
    std::vector<float>    restLen(GROUP_SIZE, 0.0f);
    std::vector<float>    stiffness(GROUP_SIZE, 0.0f);

    p1Idx[0] = 1u; p2Idx[0] = 0u; restLen[0] = 2.0f; stiffness[0] = 1.0f;
    p1Idx[1] = 2u; p2Idx[1] = 0u; restLen[1] = 2.0f; stiffness[1] = 1.0f;

    GlobalParams_0 gp{};
    gp.positions_0.data  = positions.data();  gp.positions_0.count  = positions.size();
    gp.p1Idx_0.data      = p1Idx.data();      gp.p1Idx_0.count      = p1Idx.size();
    gp.p2Idx_0.data      = p2Idx.data();      gp.p2Idx_0.count      = p2Idx.size();
    gp.restLen_0.data    = restLen.data();    gp.restLen_0.count    = restLen.size();
    gp.stiffness_0.data  = stiffness.data();  gp.stiffness_0.count  = stiffness.size();
    gp.gradA_0.data      = gradA.data();      gp.gradA_0.count      = gradA.size();
    gp.hess_0.data       = hess.data();       gp.hess_0.count       = hess.size();

    ComputeVaryingInput vi{};
    vi.startGroupID = uint3(0, 0, 0);
    vi.endGroupID   = uint3(1, 1, 1);
    main_0(&vi, nullptr, &gp);

    // Reference: independent CPU evaluation.
    const float expected_grad[N_SPRINGS][3] = {
        {1.0f, 0.0f, 0.0f},
        {0.0f, 2.0f, 0.0f},
    };
    const float expected_hess[N_SPRINGS][6] = {
        // spring 0: d=(3,0,0), k=1, len2=9 → k/len2 * d⊗d = (1/9)*[9,0,0,0,0,0]
        {1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f},
        // spring 1: d=(0,4,0), k=1, len2=16 → (1/16)*[0,0,0,16,0,0]
        {0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f},
    };

    int   fails        = 0;
    float max_abs_diff = 0.0f;
    auto check = [&](float got, float ref, const char* label, uint32_t s, int c) {
        const float d = std::fabs(got - ref);
        if (d > max_abs_diff) max_abs_diff = d;
        if (d > 1e-6f) {
            if (fails < 5) {
                std::fprintf(stderr,
                    "spring_force %s mismatch at s=%u, c=%d: got %g, expected %g (diff %g)\n",
                    label, s, c, got, ref, d);
            }
            ++fails;
        }
    };
    for (uint32_t s = 0; s < N_SPRINGS; ++s) {
        for (int c = 0; c < 3; ++c) check(gradA[s][c], expected_grad[s][c], "gradA", s, c);
        for (int c = 0; c < 6; ++c) check(hess[6*s + c], expected_hess[s][c], "hess", s, c);
    }

    const auto t1 = std::chrono::steady_clock::now();
    const double elapsed = std::chrono::duration<double>(t1 - t0).count();
    if (elapsed > kBudgetSeconds) {
        std::fprintf(stderr, "spring_force: TIMEOUT — %.3fs > %.1fs\n",
                     elapsed, kBudgetSeconds);
        return 2;
    }
    if (fails == 0) {
        std::printf("spring_force: %u/%u springs OK (max_abs_diff=%g, %.1fms)\n",
                    N_SPRINGS, N_SPRINGS, max_abs_diff, elapsed * 1000.0);
        return 0;
    }
    std::fprintf(stderr, "spring_force: %d FAIL out of %u checks\n",
                 fails, N_SPRINGS * 9u);
    return 1;
}
