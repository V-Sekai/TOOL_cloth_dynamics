// Real-input validator for Cloth.SlangCodegen.TriangleBending — the
// DiffCloth PD dihedral-bending local step.
//
// Reference: src/code/simulation/TriangleBending.cpp:project. For each
// constraint, with 4-vertex stencil (idx[4c..4c+3]) and cotangent
// weights (weight[4c..4c+3]):
//
//   if (nTarget[c] > 1e-6) {
//     e_c   = sum_j weight[4c+j] * positions[idx[4c+j]]
//     e_c   = (e_c / |e_c|) * nTarget[c]
//   } else {
//     e_c   = (0, 0, 0)
//   }
//   projected[c] = sqrtWeight[c] * e_c
//
// Test fixture (2 constraints sharing 4 vertices in a tetrahedral pose):
//   v0 = (0, 0, 0),  v1 = (1, 0, 0),  v2 = (0, 1, 0),  v3 = (0, 0, 1)
//
//   constraint 0 (active): weights = (-1, 1, 1, -1), nTarget = sqrt(3),
//                          sqrtWeight = 1
//       sum = (-1*v0 + 1*v1 + 1*v2 + -1*v3) = (1, 1, -1), |sum| = sqrt(3)
//       e   = sum / sqrt(3) * sqrt(3)     = (1, 1, -1)
//       out = 1 * (1, 1, -1)              = (1, 1, -1)
//
//   constraint 1 (degenerate): nTarget = 0 → short-circuit
//       out = (0, 0, 0) regardless of weights / positions
//
// Padding (slots 2..63): nTarget = 0 makes the kernel skip the
// idx/weight loads entirely, so those buffers only need the 8 entries
// covering the 2 real constraints.

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <vector>

#include "triangle_bending_emit.cpp"

static constexpr double kBudgetSeconds = 5.0;

int main() {
    const auto t0 = std::chrono::steady_clock::now();

    constexpr uint32_t N_CONSTRAINTS_REAL = 2;
    constexpr uint32_t N_CONSTRAINTS_TOTAL = 64;  // = GROUP_SIZE
    constexpr uint32_t N_REAL_INDICES = 4u * N_CONSTRAINTS_REAL;

    // 4 vertices forming a corner of a unit cube.
    std::vector<Vector<float, 3>> positions = {
        Vector<float, 3>(0.0f, 0.0f, 0.0f),
        Vector<float, 3>(1.0f, 0.0f, 0.0f),
        Vector<float, 3>(0.0f, 1.0f, 0.0f),
        Vector<float, 3>(0.0f, 0.0f, 1.0f),
    };

    std::vector<Vector<float, 3>> projected(N_CONSTRAINTS_TOTAL,
                                            Vector<float, 3>(0.0f));

    // Stencil indices for the 2 real constraints — both reference the
    // same 4 verts in the same order.
    std::vector<uint32_t> idx(N_REAL_INDICES);
    for (uint32_t c = 0; c < N_CONSTRAINTS_REAL; ++c) {
        idx[4*c + 0] = 0u;
        idx[4*c + 1] = 1u;
        idx[4*c + 2] = 2u;
        idx[4*c + 3] = 3u;
    }

    // Weights: constraint 0 uses an asymmetric (−1,1,1,−1) stencil to
    // give a non-axis-aligned, easy-to-eyeball expected output.
    std::vector<float> weight(N_REAL_INDICES);
    weight[0] = -1.0f; weight[1] = 1.0f; weight[2] = 1.0f; weight[3] = -1.0f;
    weight[4] =  0.0f; weight[5] = 0.0f; weight[6] = 0.0f; weight[7] =  0.0f;

    // nTarget = 0 short-circuits to (0,0,0); all padded slots stay 0.
    std::vector<float> nTarget(N_CONSTRAINTS_TOTAL, 0.0f);
    nTarget[0] = std::sqrt(3.0f);   // active
    // nTarget[1] = 0 (degenerate)

    std::vector<float> sqrtWeight(N_CONSTRAINTS_TOTAL, 1.0f);

    GlobalParams_0 gp{};
    gp.positions_0.data   = positions.data();    gp.positions_0.count   = positions.size();
    gp.projected_0.data   = projected.data();    gp.projected_0.count   = projected.size();
    gp.idx_0.data         = idx.data();          gp.idx_0.count         = idx.size();
    gp.weight_0.data      = weight.data();       gp.weight_0.count      = weight.size();
    gp.nTarget_0.data     = nTarget.data();      gp.nTarget_0.count     = nTarget.size();
    gp.sqrtWeight_0.data  = sqrtWeight.data();   gp.sqrtWeight_0.count  = sqrtWeight.size();

    ComputeVaryingInput vi{};
    vi.startGroupID = uint3(0, 0, 0);
    vi.endGroupID   = uint3(1, 1, 1);
    main_0(&vi, nullptr, &gp);

    const float expected[N_CONSTRAINTS_REAL][3] = {
        {1.0f, 1.0f, -1.0f},   // active branch
        {0.0f, 0.0f,  0.0f},   // short-circuit branch
    };

    int   fails        = 0;
    float max_abs_diff = 0.0f;
    for (uint32_t c = 0; c < N_CONSTRAINTS_REAL; ++c) {
        for (int k = 0; k < 3; ++k) {
            const float got = projected[c][k];
            const float ref = expected[c][k];
            const float d   = std::fabs(got - ref);
            if (d > max_abs_diff) max_abs_diff = d;
            if (d > 1e-6f) {
                if (fails < 5) {
                    std::fprintf(stderr,
                        "triangle_bending mismatch at c=%u, k=%d: got %g, expected %g (diff %g)\n",
                        c, k, got, ref, d);
                }
                ++fails;
            }
        }
    }

    const auto t1 = std::chrono::steady_clock::now();
    const double elapsed = std::chrono::duration<double>(t1 - t0).count();
    if (elapsed > kBudgetSeconds) {
        std::fprintf(stderr,
            "triangle_bending: TIMEOUT — %.3fs > budget %.1fs\n",
            elapsed, kBudgetSeconds);
        return 2;
    }
    if (fails == 0) {
        std::printf(
            "triangle_bending: %u/%u constraints OK (max_abs_diff=%g, %.1fms)\n",
            N_CONSTRAINTS_REAL, N_CONSTRAINTS_REAL, max_abs_diff, elapsed * 1000.0);
        return 0;
    }
    std::fprintf(stderr, "triangle_bending: %d FAIL out of %u (%u constraints × 3 components)\n",
                 fails, N_CONSTRAINTS_REAL * 3u, N_CONSTRAINTS_REAL);
    return 1;
}
