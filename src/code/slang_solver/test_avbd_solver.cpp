// Smoke test for AvbdSolver — verifies the AVBD vbd_init kernel
// dispatches correctly on real Metal hardware and produces the
// expected inertial term in (gScratch, hScratch).
//
// Test fixture (2 verts, no constraints):
//   v0:  x=(0,0,0), pred=(1,2,3), m=2, invH²=2 → w=4
//        expected g=(-4,-8,-12), h=[4,0,0,4,0,4]
//   v1:  x=(5,5,5), pred=(5,5,5), m=1, invH²=2 → w=2
//        expected g=(0,0,0), h=[2,0,0,2,0,2]
//
// Usage:
//   ./test_avbd_solver <metallib_dir>

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
        5.0f, 5.0f, 5.0f,
    };
    const float predicted[3 * N] = {
        1.0f, 2.0f, 3.0f,
        5.0f, 5.0f, 5.0f,
    };
    const float mass[N] = {2.0f, 1.0f};
    constexpr float invHSquared = 2.0f;

    solver.setupMesh(N, positions, predicted, mass, invHSquared);

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
    };
    const float expected_h[6 * N] = {
        4.0f, 0.0f, 0.0f, 4.0f, 0.0f, 4.0f,
        2.0f, 0.0f, 0.0f, 2.0f, 0.0f, 2.0f,
    };

    int fails = 0;
    float max_abs_diff = 0.0f;
    for (uint32_t i = 0; i < 3 * N; ++i) {
        const float d = std::fabs(gOut[i] - expected_g[i]);
        if (d > max_abs_diff) max_abs_diff = d;
        if (d > 1e-5f) {
            std::fprintf(stderr,
                "g mismatch at i=%u: got %g, expected %g\n",
                i, gOut[i], expected_g[i]);
            ++fails;
        }
    }
    for (uint32_t i = 0; i < 6 * N; ++i) {
        const float d = std::fabs(hOut[i] - expected_h[i]);
        if (d > max_abs_diff) max_abs_diff = d;
        if (d > 1e-5f) {
            std::fprintf(stderr,
                "h mismatch at i=%u: got %g, expected %g\n",
                i, hOut[i], expected_h[i]);
            ++fails;
        }
    }

    if (fails == 0) {
        std::printf("test_avbd_solver: OK (vbd_init dispatched on %u verts, max_abs_diff=%g)\n",
                    N, max_abs_diff);
        return 0;
    }
    std::fprintf(stderr, "test_avbd_solver: %d FAIL\n", fails);
    return 1;
}
