// Smoke test for AvbdSolver — verifies the AVBD pipeline dispatches
// vbd_init + spring_force + vbd_gather_spring on real Metal hardware
// and produces the expected accumulated outputs.
//
// Test fixture (2 verts, 1 spring):
//   v0:  x=(0,0,0)  pred=(1,2,3)  m=2  → w=4 (inertial)
//   v1:  x=(3,0,0)  pred=(3,0,0)  m=1  → w=2 (inertial)
//   invH²=2
//   s0:  a=0, b=1, L=2, k=1
//        d = p_a − p_b = (−3,0,0), len=3, c=1
//        gradA = (k·c/len)·d = (1/3)·(−3,0,0) = (−1,0,0)
//        hess  = (k/len²)·d⊗d packed = (1/9)·[9,0,0,0,0,0] = [1,0,0,0,0,0]
//   CSR:  offsets=[0,1,2], idx=[0,0], role=[0,1]
//
// Pipeline:
//   vbd_init     → g[0]=(−4,−8,−12) h[0..5]=[4,0,0,4,0,4]
//                  g[1]=(0,0,0)     h[6..11]=[2,0,0,2,0,2]
//   spring_force → gradA[0]=(−1,0,0), hess[0..5]=[1,0,0,0,0,0]
//   gather_spring (role-aware sign flip on gradient; Hessian no flip):
//     v0 role 0 sign=+1: g[0] += (−1,0,0)  →  (−5,−8,−12)
//                        h += hess         →  [5,0,0,4,0,4]
//     v1 role 1 sign=−1: g[1] += (+1,0,0)  →  (1,0,0)
//                        h += hess         →  [3,0,0,2,0,2]

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

    constexpr uint32_t N = 2;
    const float positions[3 * N] = {
        0.0f, 0.0f, 0.0f,
        3.0f, 0.0f, 0.0f,
    };
    const float predicted[3 * N] = {
        1.0f, 2.0f, 3.0f,
        3.0f, 0.0f, 0.0f,
    };
    const float mass[N] = {2.0f, 1.0f};
    constexpr float invHSquared = 2.0f;
    solver.setupMesh(N, positions, predicted, mass, invHSquared);

    constexpr uint32_t N_S = 1;
    const uint32_t p1Idx[N_S]    = {0u};
    const uint32_t p2Idx[N_S]    = {1u};
    const float restLen[N_S]     = {2.0f};
    const float stiffness[N_S]   = {1.0f};
    solver.uploadSprings(N_S, p1Idx, p2Idx, restLen, stiffness);

    const int rc = solver.step();
    if (rc != 0) {
        std::fprintf(stderr, "test_avbd_solver: step() returned %d\n", rc);
        return 1;
    }

    std::vector<float> gOut, hOut;
    solver.readScratch(gOut, hOut);

    const float expected_g[3 * N] = {
        -5.0f, -8.0f, -12.0f,
         1.0f,  0.0f,   0.0f,
    };
    const float expected_h[6 * N] = {
        5.0f, 0.0f, 0.0f, 4.0f, 0.0f, 4.0f,
        3.0f, 0.0f, 0.0f, 2.0f, 0.0f, 2.0f,
    };

    int fails = 0;
    float max_abs_diff = 0.0f;
    auto check = [&](float got, float ref, const char* label, uint32_t i) {
        const float d = std::fabs(got - ref);
        if (d > max_abs_diff) max_abs_diff = d;
        if (d > 1e-5f) {
            if (fails < 5) {
                std::fprintf(stderr,
                    "%s mismatch at i=%u: got %g, expected %g (diff %g)\n",
                    label, i, got, ref, d);
            }
            ++fails;
        }
    };
    for (uint32_t i = 0; i < 3 * N; ++i) check(gOut[i], expected_g[i], "g", i);
    for (uint32_t i = 0; i < 6 * N; ++i) check(hOut[i], expected_h[i], "h", i);

    if (fails == 0) {
        std::printf("test_avbd_solver: OK (vbd_init+spring_force+gather_spring on %u verts/%u spring, max_abs_diff=%g)\n",
                    N, N_S, max_abs_diff);
        return 0;
    }
    std::fprintf(stderr, "test_avbd_solver: %d FAIL\n", fails);
    return 1;
}
