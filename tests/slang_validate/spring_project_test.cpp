// Real-input validator for Cloth.SlangCodegen.SpringProject — the
// DiffCloth PD spring local-step, the first kernel ported through the
// Lean → LeanSlang → slangc-cpp → executable chain.
//
// Reference: src/code/simulation/Spring.cpp:108–113. For each spring,
//   projected[i] = (sqrtWeight[i] * restLen[i] / |p_a − p_b|) · (p_a − p_b)
// where (a, b) = (p1Idx[i], p2Idx[i]).
//
// Test fixture (2 real springs over 3 verts):
//   v0 = (0,0,0), v1 = (3,0,0), v2 = (0,4,0)
//   spring 0: a=1 b=0, restLen=1.0, sqrtWeight=1.0
//     diff=(3,0,0), len=3, scale=1/3 → out=(1,0,0)
//   spring 1: a=2 b=0, restLen=2.0, sqrtWeight=1.0
//     diff=(0,4,0), len=4, scale=1/2 → out=(0,2,0)
//
// One thread group of 64 (matches the kernel's [numthreads(64,1,1)]).
// Padded slots 2..63 use a=1, b=0 (distinct positions → nonzero len,
// no divide-by-zero) with restLen=0 and sqrtWeight=0, so scale=0 and
// the slot output is (0,0,0). Only slots 0..1 are checked.

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <vector>

#include "spring_project_emit.cpp"

static constexpr double kBudgetSeconds = 5.0;

int main() {
    const auto t0 = std::chrono::steady_clock::now();

    constexpr uint32_t N_SPRINGS = 2;
    constexpr uint32_t GROUP_SIZE = 64;   // numthreads(64,1,1) — pad to one group

    // ---- Positions: 3 verts, no padding needed (only referenced by index).
    std::vector<Vector<float, 3>> positions = {
        Vector<float, 3>(0.0f, 0.0f, 0.0f),
        Vector<float, 3>(3.0f, 0.0f, 0.0f),
        Vector<float, 3>(0.0f, 4.0f, 0.0f),
    };

    std::vector<Vector<float, 3>> projected(GROUP_SIZE, Vector<float, 3>(0.0f));

    // ---- Spring buffers padded to GROUP_SIZE.
    // Real springs:                    spring 0,   spring 1
    // Padded slots (2..63):  a=1, b=0 (distinct → len>0), restLen=0, sqrtWeight=0.
    std::vector<uint32_t> p1Idx(GROUP_SIZE, 1u);
    std::vector<uint32_t> p2Idx(GROUP_SIZE, 0u);
    std::vector<float>    restLen(GROUP_SIZE, 0.0f);
    std::vector<float>    sqrtWeight(GROUP_SIZE, 0.0f);

    p1Idx[0] = 1u; p2Idx[0] = 0u; restLen[0] = 1.0f; sqrtWeight[0] = 1.0f;
    p1Idx[1] = 2u; p2Idx[1] = 0u; restLen[1] = 2.0f; sqrtWeight[1] = 1.0f;

    GlobalParams_0 gp{};
    gp.positions_0.data   = positions.data();   gp.positions_0.count   = positions.size();
    gp.projected_0.data   = projected.data();   gp.projected_0.count   = projected.size();
    gp.p1Idx_0.data       = p1Idx.data();       gp.p1Idx_0.count       = p1Idx.size();
    gp.p2Idx_0.data       = p2Idx.data();       gp.p2Idx_0.count       = p2Idx.size();
    gp.restLen_0.data     = restLen.data();     gp.restLen_0.count     = restLen.size();
    gp.sqrtWeight_0.data  = sqrtWeight.data();  gp.sqrtWeight_0.count  = sqrtWeight.size();

    // One thread group of 64 covers both real springs plus the padding.
    ComputeVaryingInput vi{};
    vi.startGroupID = uint3(0, 0, 0);
    vi.endGroupID   = uint3(1, 1, 1);
    main_0(&vi, nullptr, &gp);

    // ---- Reference: independent CPU evaluation of the same kernel.
    const float expected[N_SPRINGS][3] = {
        {1.0f, 0.0f, 0.0f},
        {0.0f, 2.0f, 0.0f},
    };

    int   fails        = 0;
    float max_abs_diff = 0.0f;
    for (uint32_t s = 0; s < N_SPRINGS; ++s) {
        for (int c = 0; c < 3; ++c) {
            const float got = projected[s][c];
            const float ref = expected[s][c];
            const float d   = std::fabs(got - ref);
            if (d > max_abs_diff) max_abs_diff = d;
            if (d > 1e-6f) {
                if (fails < 5) {
                    std::fprintf(stderr,
                        "spring_project mismatch at s=%u, c=%d: got %g, expected %g (diff %g)\n",
                        s, c, got, ref, d);
                }
                ++fails;
            }
        }
    }

    const auto t1 = std::chrono::steady_clock::now();
    const double elapsed = std::chrono::duration<double>(t1 - t0).count();
    if (elapsed > kBudgetSeconds) {
        std::fprintf(stderr,
            "spring_project: TIMEOUT — %.3fs > budget %.1fs\n",
            elapsed, kBudgetSeconds);
        return 2;
    }
    if (fails == 0) {
        std::printf(
            "spring_project: %u/%u springs OK (max_abs_diff=%g, %.1fms)\n",
            N_SPRINGS, N_SPRINGS, max_abs_diff, elapsed * 1000.0);
        return 0;
    }
    std::fprintf(stderr, "spring_project: %d FAIL out of %u (%u springs × 3 components)\n",
                 fails, N_SPRINGS * 3u, N_SPRINGS);
    return 1;
}
