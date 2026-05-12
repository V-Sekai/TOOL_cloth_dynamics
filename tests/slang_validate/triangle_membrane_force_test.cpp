// Real-input validator for Cloth.SlangCodegen.TriangleMembraneForce —
// AVBD per-triangle in-plane stretch force (PR-A of AVBD port).
//
// Per triangle c with vertices (i0,i1,i2) and stiffness k:
//   F = [p1-p0 | p2-p0],  R = closest-rotation-in-image-plane(F)
//   e0 = F.col(0) − R.col(0)
//   e1 = F.col(1) − R.col(1)
//   grad[3c+0] = −k · (e0 + e1)
//   grad[3c+1] = +k · e0
//   grad[3c+2] = +k · e1
//   hessScalar[3c+0] = 2·k
//   hessScalar[3c+1] = k
//   hessScalar[3c+2] = k
//
// Test fixture (2 triangles, 6 verts):
//
// Triangle 0 — stretched 2x in x and y, k=1:
//   v0=(0,0,0), v1=(2,0,0), v2=(0,2,0)
//   F = [(2,0,0)|(0,2,0)], 2D form [2,0;0,2], closest rot = I
//   R = [(1,0,0)|(0,1,0)]
//   e0=(1,0,0), e1=(0,1,0)
//   grad[0] = (−1, −1, 0)
//   grad[1] = (1, 0, 0)
//   grad[2] = (0, 1, 0)
//   hess = [2, 1, 1]
//
// Triangle 1 — unit rest triangle, k=3:
//   v3=(10,10,10), v4=(11,10,10), v5=(10,11,10)
//   F = [(1,0,0)|(0,1,0)] = R, so e0=e1=(0,0,0)
//   grad = [(0,0,0)] × 3
//   hess = [6, 3, 3]
//
// Padding (slots 2..63): idx=(0,1,2), k=0 → safe rotation math + zero grad/hess.

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <vector>

#include "triangle_membrane_force_emit.cpp"

static constexpr double kBudgetSeconds = 5.0;

int main() {
    const auto t0 = std::chrono::steady_clock::now();

    constexpr uint32_t N_TRI = 2;
    constexpr uint32_t GROUP_SIZE = 64;

    std::vector<Vector<float, 3>> positions = {
        Vector<float, 3>(0.0f, 0.0f, 0.0f),    // v0
        Vector<float, 3>(2.0f, 0.0f, 0.0f),    // v1
        Vector<float, 3>(0.0f, 2.0f, 0.0f),    // v2
        Vector<float, 3>(10.0f, 10.0f, 10.0f), // v3
        Vector<float, 3>(11.0f, 10.0f, 10.0f), // v4
        Vector<float, 3>(10.0f, 11.0f, 10.0f), // v5
    };

    std::vector<uint32_t>         idx(3 * GROUP_SIZE, 0u);
    std::vector<float>            stiffness(GROUP_SIZE, 0.0f);
    std::vector<Vector<float, 3>> grad(3 * GROUP_SIZE, Vector<float, 3>(0.0f));
    std::vector<float>            hessScalar(3 * GROUP_SIZE, 0.0f);

    // Default padding: idx = (0, 1, 2) (well-defined positions), k = 0
    for (uint32_t c = 0; c < GROUP_SIZE; ++c) {
        idx[3*c + 0] = 0u; idx[3*c + 1] = 1u; idx[3*c + 2] = 2u;
    }

    idx[0] = 0u; idx[1] = 1u; idx[2] = 2u; stiffness[0] = 1.0f;
    idx[3] = 3u; idx[4] = 4u; idx[5] = 5u; stiffness[1] = 3.0f;

    GlobalParams_0 gp{};
    gp.positions_0.data  = positions.data();  gp.positions_0.count  = positions.size();
    gp.idx_0.data        = idx.data();        gp.idx_0.count        = idx.size();
    gp.stiffness_0.data  = stiffness.data();  gp.stiffness_0.count  = stiffness.size();
    gp.grad_0.data       = grad.data();       gp.grad_0.count       = grad.size();
    gp.hessScalar_0.data = hessScalar.data(); gp.hessScalar_0.count = hessScalar.size();

    ComputeVaryingInput vi{};
    vi.startGroupID = uint3(0, 0, 0);
    vi.endGroupID   = uint3(1, 1, 1);
    main_0(&vi, nullptr, &gp);

    const float expected_grad[N_TRI][3][3] = {
        { {-1.0f, -1.0f, 0.0f}, {1.0f, 0.0f, 0.0f}, {0.0f, 1.0f, 0.0f} },
        { { 0.0f,  0.0f, 0.0f}, {0.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 0.0f} },
    };
    const float expected_hess[N_TRI][3] = {
        {2.0f, 1.0f, 1.0f},
        {6.0f, 3.0f, 3.0f},
    };

    int   fails        = 0;
    float max_abs_diff = 0.0f;
    auto check = [&](float got, float ref, const char* label, uint32_t c, int r, int axis) {
        const float d = std::fabs(got - ref);
        if (d > max_abs_diff) max_abs_diff = d;
        if (d > 1e-5f) {
            if (fails < 5) {
                std::fprintf(stderr,
                    "triangle_membrane_force %s mismatch at c=%u, r=%d, axis=%d: got %g, expected %g (diff %g)\n",
                    label, c, r, axis, got, ref, d);
            }
            ++fails;
        }
    };
    for (uint32_t c = 0; c < N_TRI; ++c) {
        for (int r = 0; r < 3; ++r) {
            for (int axis = 0; axis < 3; ++axis) {
                check(grad[3*c + r][axis], expected_grad[c][r][axis], "grad", c, r, axis);
            }
            check(hessScalar[3*c + r], expected_hess[c][r], "hess", c, r, 0);
        }
    }

    const auto t1 = std::chrono::steady_clock::now();
    const double elapsed = std::chrono::duration<double>(t1 - t0).count();
    if (elapsed > kBudgetSeconds) {
        std::fprintf(stderr, "triangle_membrane_force: TIMEOUT — %.3fs > %.1fs\n",
                     elapsed, kBudgetSeconds);
        return 2;
    }
    if (fails == 0) {
        std::printf("triangle_membrane_force: %u/%u tris OK (max_abs_diff=%g, %.1fms)\n",
                    N_TRI, N_TRI, max_abs_diff, elapsed * 1000.0);
        return 0;
    }
    std::fprintf(stderr, "triangle_membrane_force: %d FAIL out of %u checks\n",
                 fails, N_TRI * 12u);
    return 1;
}
