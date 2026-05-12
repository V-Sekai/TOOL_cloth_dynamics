// Real-input validator for Cloth.SlangCodegen.TriangleBendingForceAl —
// AL-augmented dihedral bending force.
//
// Same per-constraint bending force + GN Hessian as
// triangle_bending_force (#46) but adds λ_c · w_r to each per-vertex
// gradient (where w_r is the cotangent Laplacian weight for vertex r).
//
//   grad[4c+r]       = k · w_r · resid  +  w_r · λ
//   hessScalar[4c+r] = k · w_r²
//
// Test fixture (2 constraints over 4 verts):
//   v0=v1=v2=(0,0,0), v3=(0,0,2), weights (1,1,-1,-1)
//   s=(0,0,-2), om=0.5, resid=(0,0,-1)
//
// c0 — λ=0 backward compat: identical to triangle_bending_force
//   grad=[(0,0,-1),(0,0,-1),(0,0,1),(0,0,1)]; hess=[1,1,1,1]
//
// c1 — λ=(2,3,5):
//   v0 role 0 (w=+1):  grad = (0,0,-1) + 1·(2,3,5) = (2, 3, 4)
//   v1 role 1 (w=+1):  same: (2, 3, 4)
//   v2 role 2 (w=-1):  grad = (0,0,1) + (-1)·(2,3,5) = (-2, -3, -4)
//   v3 role 3 (w=-1):  same: (-2, -3, -4)
//   hess unchanged: [1, 1, 1, 1]

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <vector>

#include "triangle_bending_force_al_emit.cpp"

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
    std::vector<Vector<float, 3>> lambda(GROUP_SIZE, Vector<float, 3>(0.0f));
    std::vector<Vector<float, 3>> grad(4 * GROUP_SIZE, Vector<float, 3>(0.0f));
    std::vector<float>            hessScalar(4 * GROUP_SIZE, 0.0f);

    // c0 — λ=0 backward compat case
    for (int r = 0; r < 4; ++r) idx[4*0 + r] = uint32_t(r);
    weight[0]=1.0f; weight[1]=1.0f; weight[2]=-1.0f; weight[3]=-1.0f;
    nTarget[0] = 1.0f;  stiffness[0] = 1.0f;
    // lambda[0] stays zero

    // c1 — λ≠0 case
    for (int r = 0; r < 4; ++r) idx[4*1 + r] = uint32_t(r);
    weight[4]=1.0f; weight[5]=1.0f; weight[6]=-1.0f; weight[7]=-1.0f;
    nTarget[1] = 1.0f;  stiffness[1] = 1.0f;
    lambda[1] = Vector<float, 3>(2.0f, 3.0f, 5.0f);

    GlobalParams_0 gp{};
    gp.positions_0.data  = positions.data();  gp.positions_0.count  = positions.size();
    gp.idx_0.data        = idx.data();        gp.idx_0.count        = idx.size();
    gp.weight_0.data     = weight.data();     gp.weight_0.count     = weight.size();
    gp.nTarget_0.data    = nTarget.data();    gp.nTarget_0.count    = nTarget.size();
    gp.stiffness_0.data  = stiffness.data();  gp.stiffness_0.count  = stiffness.size();
    gp.lambda_0.data     = lambda.data();     gp.lambda_0.count     = lambda.size();
    gp.grad_0.data       = grad.data();       gp.grad_0.count       = grad.size();
    gp.hessScalar_0.data = hessScalar.data(); gp.hessScalar_0.count = hessScalar.size();

    ComputeVaryingInput vi{};
    vi.startGroupID = uint3(0, 0, 0);
    vi.endGroupID   = uint3(1, 1, 1);
    main_0(&vi, nullptr, &gp);

    const float expected_grad[N_BEND][4][3] = {
        // c0 λ=0: matches triangle_bending_force
        { {0.0f, 0.0f, -1.0f},
          {0.0f, 0.0f, -1.0f},
          {0.0f, 0.0f,  1.0f},
          {0.0f, 0.0f,  1.0f} },
        // c1 λ=(2,3,5)
        { { 2.0f,  3.0f,  4.0f},
          { 2.0f,  3.0f,  4.0f},
          {-2.0f, -3.0f, -4.0f},
          {-2.0f, -3.0f, -4.0f} },
    };
    const float expected_hess[N_BEND][4] = {
        {1.0f, 1.0f, 1.0f, 1.0f},
        {1.0f, 1.0f, 1.0f, 1.0f},
    };

    int fails = 0;
    float max_abs_diff = 0.0f;
    auto check = [&](float got, float ref, const char* label, uint32_t c, int r, int axis) {
        const float d = std::fabs(got - ref);
        if (d > max_abs_diff) max_abs_diff = d;
        if (d > 1e-5f) {
            if (fails < 5) {
                std::fprintf(stderr,
                    "%s mismatch c=%u r=%d axis=%d: got %g, expected %g\n",
                    label, c, r, axis, got, ref);
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
        std::fprintf(stderr, "triangle_bending_force_al: TIMEOUT\n");
        return 2;
    }
    if (fails == 0) {
        std::printf("triangle_bending_force_al: %u/%u OK (max_abs_diff=%g, %.1fms)\n",
                    N_BEND, N_BEND, max_abs_diff, elapsed * 1000.0);
        return 0;
    }
    std::fprintf(stderr, "triangle_bending_force_al: %d FAIL\n", fails);
    return 1;
}
