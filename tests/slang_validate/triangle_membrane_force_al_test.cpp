// Real-input validator for Cloth.SlangCodegen.TriangleMembraneForceAl —
// AL-augmented ARAP membrane force with per-triangle inv_deltaUV.
//
// Define te0 = k·e0 + λ0, te1 = k·e1r + λ1 (3-vectors). Per triangle:
//   grad[3c+0] = -(te0·(m00+m10) + te1·(m01+m11))
//   grad[3c+1] = +(te0·m00       + te1·m01)
//   grad[3c+2] = +(te0·m10       + te1·m11)
//   hessScalar[3c+0] = k · ((m00+m10)² + (m01+m11)²)
//   hessScalar[3c+1] = k · (m00² + m01²)
//   hessScalar[3c+2] = k · (m10² + m11²)
//
// Test fixture (3 verts, 3 triangles, 2× stretched both axes):
//   v0=(0,0,0), v1=(2,0,0), v2=(0,2,0)
//
// T0 (M=I, λ=0, k=1): bit-exact pre-IUV result.
//   F=[(2,0,0)|(0,2,0)], R=[(1,0,0)|(0,1,0)]
//   e0=(1,0,0), e1=(0,1,0); te0=e0, te1=e1
//   grad[0]=(-1,-1,0), grad[1]=(1,0,0), grad[2]=(0,1,0)
//   hess=[2, 1, 1]
//
// T1 (M=I, λ0=(1,2,3), λ1=(4,5,6), k=1):
//   te0 = (1,0,0)+(1,2,3) = (2,2,3)
//   te1 = (0,1,0)+(4,5,6) = (4,6,6)
//   grad[0] = -(te0+te1)         = (-6,-8,-9)
//   grad[1] = te0                = (2,2,3)
//   grad[2] = te1                = (4,6,6)
//   hess=[2, 1, 1]
//
// T2 (M=2·I, λ0=(1,0,0), λ1=(0,1,0), k=1):
//   F = P · 2I = [(4,0,0)|(0,4,0)], R=[(1,0,0)|(0,1,0)]
//   e0=(3,0,0), e1=(0,3,0); te0=(4,0,0), te1=(0,4,0)
//   (m00+m10)=2, (m01+m11)=2
//   grad[0] = -((4,0,0)·2 + (0,4,0)·2) = (-8,-8,0)
//   grad[1] = (4,0,0)·2 + (0,4,0)·0    = (8,0,0)
//   grad[2] = (4,0,0)·0 + (0,4,0)·2    = (0,8,0)
//   hess=[8, 4, 4]

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <vector>

#include "triangle_membrane_force_al_emit.cpp"

static constexpr double kBudgetSeconds = 5.0;

int main() {
    const auto t0 = std::chrono::steady_clock::now();

    constexpr uint32_t N_TRI = 3;
    constexpr uint32_t GROUP_SIZE = 64;

    std::vector<Vector<float, 3>> positions = {
        Vector<float, 3>(0.0f, 0.0f, 0.0f),
        Vector<float, 3>(2.0f, 0.0f, 0.0f),
        Vector<float, 3>(0.0f, 2.0f, 0.0f),
    };

    std::vector<uint32_t>         idx(3 * GROUP_SIZE, 0u);
    std::vector<float>            stiffness(GROUP_SIZE, 0.0f);
    std::vector<Vector<float, 3>> lambda0(GROUP_SIZE, Vector<float, 3>(0.0f));
    std::vector<Vector<float, 3>> lambda1(GROUP_SIZE, Vector<float, 3>(0.0f));
    std::vector<Vector<float, 3>> grad(3 * GROUP_SIZE, Vector<float, 3>(0.0f));
    std::vector<float>            hessScalar(3 * GROUP_SIZE, 0.0f);
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
    stiffness[0] = 1.0f;  // T0: λ=0, M=I
    stiffness[1] = 1.0f;  // T1: λ≠0, M=I
    stiffness[2] = 1.0f;  // T2: λ≠0, M=2I
    lambda0[1] = Vector<float, 3>(1.0f, 2.0f, 3.0f);
    lambda1[1] = Vector<float, 3>(4.0f, 5.0f, 6.0f);
    lambda0[2] = Vector<float, 3>(1.0f, 0.0f, 0.0f);
    lambda1[2] = Vector<float, 3>(0.0f, 1.0f, 0.0f);
    inv_deltaUV[4*2 + 0] = 2.0f;
    inv_deltaUV[4*2 + 3] = 2.0f;

    GlobalParams_0 gp{};
    gp.positions_0.data    = positions.data();    gp.positions_0.count    = positions.size();
    gp.idx_0.data          = idx.data();          gp.idx_0.count          = idx.size();
    gp.stiffness_0.data    = stiffness.data();    gp.stiffness_0.count    = stiffness.size();
    gp.lambda0_0.data      = lambda0.data();      gp.lambda0_0.count      = lambda0.size();
    gp.lambda1_0.data      = lambda1.data();      gp.lambda1_0.count      = lambda1.size();
    gp.grad_0.data         = grad.data();         gp.grad_0.count         = grad.size();
    gp.hessScalar_0.data   = hessScalar.data();   gp.hessScalar_0.count   = hessScalar.size();
    gp.inv_deltaUV_0.data  = inv_deltaUV.data();  gp.inv_deltaUV_0.count  = inv_deltaUV.size();

    ComputeVaryingInput vi{};
    vi.startGroupID = uint3(0, 0, 0);
    vi.endGroupID   = uint3(1, 1, 1);
    main_0(&vi, nullptr, &gp);

    const float expected_grad[N_TRI][3][3] = {
        { {-1.0f, -1.0f,  0.0f}, {1.0f, 0.0f, 0.0f}, {0.0f, 1.0f, 0.0f} },
        { {-6.0f, -8.0f, -9.0f}, {2.0f, 2.0f, 3.0f}, {4.0f, 6.0f, 6.0f} },
        { {-8.0f, -8.0f,  0.0f}, {8.0f, 0.0f, 0.0f}, {0.0f, 8.0f, 0.0f} },
    };
    const float expected_hess[N_TRI][3] = {
        {2.0f, 1.0f, 1.0f},
        {2.0f, 1.0f, 1.0f},
        {8.0f, 4.0f, 4.0f},
    };

    int fails = 0;
    float max_abs_diff = 0.0f;
    auto check = [&](float got, float ref, const char* label, uint32_t c, int r, int axis) {
        const float d = std::fabs(got - ref);
        if (d > max_abs_diff) max_abs_diff = d;
        if (d > 1e-5f) {
            if (fails < 5) {
                std::fprintf(stderr, "%s mismatch c=%u r=%d axis=%d: got %g, expected %g\n",
                             label, c, r, axis, got, ref);
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
        std::fprintf(stderr, "triangle_membrane_force_al: TIMEOUT\n");
        return 2;
    }
    if (fails == 0) {
        std::printf("triangle_membrane_force_al: %u/%u tris OK (max_abs_diff=%g, %.1fms)\n",
                    N_TRI, N_TRI, max_abs_diff, elapsed * 1000.0);
        return 0;
    }
    std::fprintf(stderr, "triangle_membrane_force_al: %d FAIL\n", fails);
    return 1;
}
