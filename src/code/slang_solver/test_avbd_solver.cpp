// Smoke test for AvbdSolver — verifies the AVBD pipeline dispatches
// vbd_init + spring_force on real Metal hardware and produces the
// expected outputs.
//
// Test fixture (3 verts, 2 springs):
//
// Vertices for vbd_init:
//   v0:  x=(0,0,0)  pred=(1,2,3)  m=2  → w=4
//                                   expected g=(-4,-8,-12), h_diag=4
//   v1:  x=(5,5,5)  pred=(5,5,5)  m=1  → w=2
//                                   expected g=(0,0,0),     h_diag=2
//   v2:  x=(3,0,0)  pred=(3,0,0)  m=1  → w=2
//                                   expected g=(0,0,0),     h_diag=2
//   invH²=2
//
// Springs for spring_force:
//   s0: a=2, b=0, L=2, k=1  d=(3,0,0), len=3, c=1, scale=1/3
//                            expected gradA=(1,0,0), hess=[1,0,0,0,0,0]
//   s1: a=2, b=1, L=2, k=1  d=(-2,-5,-5), len=√54
//                            (kept as a second test — full numeric check
//                            on s0 only; s1 just sanity-checks no crash)

#include "AvbdSolver.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s <metallib_dir>\n", argv[0]);
        return 2;
    }

    cloth::AvbdSolver solver(argv[1]);
    if (!solver.ok()) {
        std::fprintf(stderr, "test_avbd_solver: construction failed\n");
        return 1;
    }

    constexpr uint32_t N = 3;
    const float positions[3 * N] = {
        0.0f, 0.0f, 0.0f,
        5.0f, 5.0f, 5.0f,
        3.0f, 0.0f, 0.0f,
    };
    const float predicted[3 * N] = {
        1.0f, 2.0f, 3.0f,
        5.0f, 5.0f, 5.0f,
        3.0f, 0.0f, 0.0f,
    };
    const float mass[N] = {2.0f, 1.0f, 1.0f};
    constexpr float invHSquared = 2.0f;
    solver.setupMesh(N, positions, predicted, mass, invHSquared);

    constexpr uint32_t N_S = 2;
    const uint32_t p1Idx[N_S]    = {2u, 2u};
    const uint32_t p2Idx[N_S]    = {0u, 1u};
    const float restLen[N_S]     = {2.0f, 2.0f};
    const float stiffness[N_S]   = {1.0f, 1.0f};
    solver.uploadSprings(N_S, p1Idx, p2Idx, restLen, stiffness);

    const int rc = solver.step();
    if (rc != 0) {
        std::fprintf(stderr, "test_avbd_solver: step() returned %d\n", rc);
        return 1;
    }

    std::vector<float> gOut, hOut;
    solver.readScratch(gOut, hOut);

    const float expected_g[3 * N] = {
        -4.0f, -8.0f, -12.0f,
         0.0f,  0.0f,   0.0f,
         0.0f,  0.0f,   0.0f,
    };
    const float expected_h[6 * N] = {
        4.0f, 0.0f, 0.0f, 4.0f, 0.0f, 4.0f,
        2.0f, 0.0f, 0.0f, 2.0f, 0.0f, 2.0f,
        2.0f, 0.0f, 0.0f, 2.0f, 0.0f, 2.0f,
    };

    int fails = 0;
    float max_abs_diff = 0.0f;
    auto check = [&](float got, float ref, const char* label, uint32_t i) {
        const float d = std::fabs(got - ref);
        if (d > max_abs_diff) max_abs_diff = d;
        if (d > 1e-5f) {
            if (fails < 5) {
                std::fprintf(stderr,
                    "%s mismatch at i=%u: got %g, expected %g\n",
                    label, i, got, ref);
            }
            ++fails;
        }
    };
    for (uint32_t i = 0; i < 3 * N; ++i) check(gOut[i], expected_g[i], "g", i);
    for (uint32_t i = 0; i < 6 * N; ++i) check(hOut[i], expected_h[i], "h", i);

    // spring_force check on s0 (a=2, b=0, L=2, k=1, d=(3,0,0), len=3, c=1)
    std::vector<float> springGradA, springHess;
    solver.readSpringForce(springGradA, springHess);
    const float expected_s0_grad[3] = {1.0f, 0.0f, 0.0f};
    const float expected_s0_hess[6] = {1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    for (int i = 0; i < 3; ++i) check(springGradA[i], expected_s0_grad[i], "spring0.gradA", i);
    for (int i = 0; i < 6; ++i) check(springHess[i],  expected_s0_hess[i], "spring0.hess", i);

    if (fails == 0) {
        std::printf("test_avbd_solver: OK (vbd_init + spring_force on %u verts, %u springs, max_abs_diff=%g)\n",
                    N, N_S, max_abs_diff);
        return 0;
    }
    std::fprintf(stderr, "test_avbd_solver: %d FAIL\n", fails);
    return 1;
}
