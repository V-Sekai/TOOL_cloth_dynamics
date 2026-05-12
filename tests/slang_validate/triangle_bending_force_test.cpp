// Real-input validator for Cloth.SlangCodegen.TriangleBendingForce —
// AVBD per-constraint dihedral bending force (PR-A cont. of AVBD port).
//
// Per bending constraint c with 4-vertex weighted-sum stencil:
//   s        = Σ_r w[4c+r] · positions[idx[4c+r]]
//   target   = (n_target/|s|) · s          if |s| > 0 and n_target > eps
//   resid    = s − target = s · (1 − n_target/|s|)
//   grad[4c+r]       = k · w[4c+r] · resid
//   hessScalar[4c+r] = k · w[4c+r]²
//
// When n_target ≤ 1e-6 (degenerate stencil from bind time), k_eff = 0
// → all grad and hess outputs are zero.
//
// Test fixture (2 constraints over 4 verts):
//
// c0 (non-degenerate, k=1, n_target=1):
//   v0=(0,0,0), v1=(0,0,0), v2=(0,0,0), v3=(0,0,2)
//   w = (1, 1, −1, −1)
//   s = 1·(0,0,0) + 1·(0,0,0) − 1·(0,0,0) − 1·(0,0,2) = (0,0,−2)
//   |s| = 2, scale = 1/2 = 0.5, om = 1 − 0.5 = 0.5
//   resid = (0,0,−1)
//   grad = [(0,0,−1), (0,0,−1), (0,0,1), (0,0,1)]
//   hess = [1, 1, 1, 1]
//
// c1 (degenerate, n_target=0, k=1):
//   grad = [(0,0,0)] × 4
//   hess = [0, 0, 0, 0]

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <vector>

#include "triangle_bending_force_emit.cpp"

static constexpr double kBudgetSeconds = 5.0;

int main() {
    const auto t0 = std::chrono::steady_clock::now();

    constexpr uint32_t N_BEND     = 2;
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
    std::vector<float>            stiffness(GROUP_SIZE, 0.0f);
    std::vector<Vector<float, 3>> grad(4 * GROUP_SIZE, Vector<float, 3>(0.0f));
    std::vector<float>            hessScalar(4 * GROUP_SIZE, 0.0f);

    // c0: non-degenerate
    for (int r = 0; r < 4; ++r) idx[4*0 + r] = uint32_t(r);
    weight[0] = 1.0f; weight[1] = 1.0f; weight[2] = -1.0f; weight[3] = -1.0f;
    nTarget[0] = 1.0f; stiffness[0] = 1.0f;

    // c1: degenerate (nTarget=0)
    for (int r = 0; r < 4; ++r) idx[4*1 + r] = uint32_t(r);
    weight[4] = 1.0f; weight[5] = 1.0f; weight[6] = -1.0f; weight[7] = -1.0f;
    nTarget[1] = 0.0f; stiffness[1] = 1.0f;

    GlobalParams_0 gp{};
    gp.positions_0.data  = positions.data();  gp.positions_0.count  = positions.size();
    gp.idx_0.data        = idx.data();        gp.idx_0.count        = idx.size();
    gp.weight_0.data     = weight.data();     gp.weight_0.count     = weight.size();
    gp.nTarget_0.data    = nTarget.data();    gp.nTarget_0.count    = nTarget.size();
    gp.stiffness_0.data  = stiffness.data();  gp.stiffness_0.count  = stiffness.size();
    gp.grad_0.data       = grad.data();       gp.grad_0.count       = grad.size();
    gp.hessScalar_0.data = hessScalar.data(); gp.hessScalar_0.count = hessScalar.size();

    ComputeVaryingInput vi{};
    vi.startGroupID = uint3(0, 0, 0);
    vi.endGroupID   = uint3(1, 1, 1);
    main_0(&vi, nullptr, &gp);

    const float expected_grad[N_BEND][4][3] = {
        { {0.0f, 0.0f, -1.0f},
          {0.0f, 0.0f, -1.0f},
          {0.0f, 0.0f,  1.0f},
          {0.0f, 0.0f,  1.0f} },
        { {0.0f, 0.0f, 0.0f},
          {0.0f, 0.0f, 0.0f},
          {0.0f, 0.0f, 0.0f},
          {0.0f, 0.0f, 0.0f} },
    };
    const float expected_hess[N_BEND][4] = {
        {1.0f, 1.0f, 1.0f, 1.0f},
        {0.0f, 0.0f, 0.0f, 0.0f},
    };

    int   fails        = 0;
    float max_abs_diff = 0.0f;
    auto check = [&](float got, float ref, const char* label, uint32_t c, int r, int axis) {
        const float d = std::fabs(got - ref);
        if (d > max_abs_diff) max_abs_diff = d;
        if (d > 1e-5f) {
            if (fails < 5) {
                std::fprintf(stderr,
                    "triangle_bending_force %s mismatch at c=%u, r=%d, axis=%d: got %g, expected %g (diff %g)\n",
                    label, c, r, axis, got, ref, d);
            }
            ++fails;
        }
    };
    for (uint32_t c = 0; c < N_BEND; ++c) {
        for (int r = 0; r < 4; ++r) {
            for (int axis = 0; axis < 3; ++axis) {
                check(grad[4*c + r][axis], expected_grad[c][r][axis], "grad", c, r, axis);
            }
            check(hessScalar[4*c + r], expected_hess[c][r], "hess", c, r, 0);
        }
    }

    const auto t1 = std::chrono::steady_clock::now();
    const double elapsed = std::chrono::duration<double>(t1 - t0).count();
    if (elapsed > kBudgetSeconds) {
        std::fprintf(stderr, "triangle_bending_force: TIMEOUT — %.3fs > %.1fs\n",
                     elapsed, kBudgetSeconds);
        return 2;
    }
    if (fails == 0) {
        std::printf("triangle_bending_force: %u/%u constraints OK (max_abs_diff=%g, %.1fms)\n",
                    N_BEND, N_BEND, max_abs_diff, elapsed * 1000.0);
        return 0;
    }
    std::fprintf(stderr, "triangle_bending_force: %d FAIL out of %u checks\n",
                 fails, N_BEND * 16u);
    return 1;
}
