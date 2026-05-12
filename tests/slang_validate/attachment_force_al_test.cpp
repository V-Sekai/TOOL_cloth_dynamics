// Real-input validator for Cloth.SlangCodegen.AttachmentForceAl —
// augmented-Lagrangian attachment force.
//
// Per attachment c, with C(x_v) = x_v − fixedPos[c]:
//   gradV[c]      = stiffness[c] · C  +  lambda[c]
//   hessScalar[c] = stiffness[c]
//
// Test fixture (2 attachments over 3 verts):
//   v0 = (1, 2, 3)   v1 = (4, 5, 5)   v2 = (0, 0, 0)
//   c0: pin v0 to (1, 0, 3), k=2, λ=(0, 0, 0)
//       C = (0, 2, 0); grad = 2·(0,2,0) + 0 = (0, 4, 0)
//       hess = 2
//   c1: pin v1 to (4, 5, 0), k=1, λ=(10, -20, 30)
//       C = (0, 0, 5); grad = 1·(0,0,5) + (10,-20,30) = (10, -20, 35)
//       hess = 1
//
// Padding: vertIdx=2, fixedPos=zero, stiffness=0, lambda=zero → all
// outputs zero.

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <vector>

#include "attachment_force_al_emit.cpp"

static constexpr double kBudgetSeconds = 5.0;

int main() {
    const auto t0 = std::chrono::steady_clock::now();

    constexpr uint32_t N_ATTACH = 2;
    constexpr uint32_t GROUP_SIZE = 64;

    std::vector<Vector<float, 3>> positions = {
        Vector<float, 3>(1.0f, 2.0f, 3.0f),
        Vector<float, 3>(4.0f, 5.0f, 5.0f),
        Vector<float, 3>(0.0f, 0.0f, 0.0f),
    };

    std::vector<uint32_t>         vertIdx(GROUP_SIZE, 2u);
    std::vector<Vector<float, 3>> fixedPos(GROUP_SIZE, Vector<float, 3>(0.0f));
    std::vector<float>            stiffness(GROUP_SIZE, 0.0f);
    std::vector<Vector<float, 3>> lambda(GROUP_SIZE, Vector<float, 3>(0.0f));
    std::vector<Vector<float, 3>> gradV(GROUP_SIZE, Vector<float, 3>(0.0f));
    std::vector<float>            hessScalar(GROUP_SIZE, 0.0f);

    vertIdx[0] = 0u; fixedPos[0] = Vector<float, 3>(1.0f, 0.0f, 3.0f);
    stiffness[0] = 2.0f;
    // λ_0 stays zero

    vertIdx[1] = 1u; fixedPos[1] = Vector<float, 3>(4.0f, 5.0f, 0.0f);
    stiffness[1] = 1.0f;
    lambda[1] = Vector<float, 3>(10.0f, -20.0f, 30.0f);

    GlobalParams_0 gp{};
    gp.positions_0.data  = positions.data();  gp.positions_0.count  = positions.size();
    gp.vertIdx_0.data    = vertIdx.data();    gp.vertIdx_0.count    = vertIdx.size();
    gp.fixedPos_0.data   = fixedPos.data();   gp.fixedPos_0.count   = fixedPos.size();
    gp.stiffness_0.data  = stiffness.data();  gp.stiffness_0.count  = stiffness.size();
    gp.lambda_0.data     = lambda.data();     gp.lambda_0.count     = lambda.size();
    gp.gradV_0.data      = gradV.data();      gp.gradV_0.count      = gradV.size();
    gp.hessScalar_0.data = hessScalar.data(); gp.hessScalar_0.count = hessScalar.size();

    ComputeVaryingInput vi{};
    vi.startGroupID = uint3(0, 0, 0);
    vi.endGroupID   = uint3(1, 1, 1);
    main_0(&vi, nullptr, &gp);

    const float expected_grad[N_ATTACH][3] = {
        { 0.0f,  4.0f,  0.0f},
        {10.0f, -20.0f, 35.0f},
    };
    const float expected_hess[N_ATTACH] = {2.0f, 1.0f};

    int fails = 0;
    float max_abs_diff = 0.0f;
    auto check = [&](float got, float ref, const char* label, uint32_t c, int axis) {
        const float d = std::fabs(got - ref);
        if (d > max_abs_diff) max_abs_diff = d;
        if (d > 1e-6f) {
            if (fails < 5) {
                std::fprintf(stderr,
                    "%s mismatch at c=%u axis=%d: got %g, expected %g\n",
                    label, c, axis, got, ref);
            }
            ++fails;
        }
    };
    for (uint32_t c = 0; c < N_ATTACH; ++c) {
        for (int axis = 0; axis < 3; ++axis) check(gradV[c][axis], expected_grad[c][axis], "gradV", c, axis);
        check(hessScalar[c], expected_hess[c], "hessScalar", c, 0);
    }

    const auto t1 = std::chrono::steady_clock::now();
    const double elapsed = std::chrono::duration<double>(t1 - t0).count();
    if (elapsed > kBudgetSeconds) {
        std::fprintf(stderr, "attachment_force_al: TIMEOUT — %.3fs > %.1fs\n",
                     elapsed, kBudgetSeconds);
        return 2;
    }
    if (fails == 0) {
        std::printf("attachment_force_al: %u/%u OK (max_abs_diff=%g, %.1fms)\n",
                    N_ATTACH, N_ATTACH, max_abs_diff, elapsed * 1000.0);
        return 0;
    }
    std::fprintf(stderr, "attachment_force_al: %d FAIL\n", fails);
    return 1;
}
