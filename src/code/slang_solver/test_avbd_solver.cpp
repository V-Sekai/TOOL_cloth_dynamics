// Smoke test for AvbdSolver — verifies the AVBD pipeline dispatches
// all wired kernels (vbd_init + spring_force + vbd_gather_spring +
// attachment_force + vbd_gather_attachment + vbd_solve_apply) on real
// Metal hardware and produces the expected updated positions.
//
// Test fixture (3 verts, 1 spring, 1 attachment):
//   v0:  x=(0,0,0)  pred=(1,2,3)  m=2
//   v1:  x=(3,0,0)  pred=(3,0,0)  m=1
//   v2:  x=(0,0,0)  pred=(0,0,0)  m=1
//   invH²=2
//
//   s0:  a=0, b=1, L=2, k=1  (same as PR #60 spring test)
//   a0:  pins v2 to fixedPos=(0,0,10), stiffness k=4
//
// vbd_init writes (w=2m):
//   v0: g=(-4,-8,-12) h_diag=4
//   v1: g=(0,0,0)     h_diag=2
//   v2: g=(0,0,0)     h_diag=2
//
// Spring path: (matches PR #60)
//   v0: g+=(-1,0,0)  h+=[1,0,0,0,0,0]  →  g=(-5,-8,-12) h=[5,0,0,4,0,4]
//   v1: g+=(1,0,0)   h+=[1,0,0,0,0,0]  →  g=(1,0,0)     h=[3,0,0,2,0,2]
//
// Attachment path:
//   v2 has one attachment c=0; gradV = k·(p_v - fixed) = 4·(0-0, 0-0, 0-10) = (0,0,-40)
//   hessScalar = k = 4
//   gather: g[2] += (0,0,-40)  →  (0,0,-40)
//           h[2] diag += 4     →  [6,0,0,6,0,6]
//
// vbd_solve_apply per vertex:
//   v0:  H=diag(5,4,4), g=(-5,-8,-12)
//        Δx = -(1/80)·(16·-5, 20·-8, 20·-12) = (1, 2, 3)
//        new pos = (1, 2, 3)
//   v1:  H=diag(3,2,2), g=(1,0,0)
//        Δx = -(1/12)·(4·1, 0, 0) = (-1/3, 0, 0)
//        new pos = (3-1/3, 0, 0) = (8/3, 0, 0)
//   v2:  H=diag(6,6,6), g=(0,0,-40)
//        det = 6·(36-0) + 0 + 0 = 216
//        Δx = -(1/216)·(36·0, 36·0, 36·-40) = (0, 0, 40/6) ≈ (0,0,6.6667)
//        new pos = (0, 0, 0) + (0, 0, 40/6) = (0, 0, 20/3)
//        — attachment pulls v2 partially toward the (0,0,10) anchor

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
        3.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 0.0f,
    };
    const float predicted[3 * N] = {
        1.0f, 2.0f, 3.0f,
        3.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 0.0f,
    };
    const float mass[N] = {2.0f, 1.0f, 1.0f};
    constexpr float invHSquared = 2.0f;
    solver.setupMesh(N, positions, predicted, mass, invHSquared);

    constexpr uint32_t N_S = 1;
    const uint32_t spP1[N_S]   = {0u};
    const uint32_t spP2[N_S]   = {1u};
    const float spLen[N_S]     = {2.0f};
    const float spK[N_S]       = {1.0f};
    solver.uploadSprings(N_S, spP1, spP2, spLen, spK);

    constexpr uint32_t N_A = 1;
    const uint32_t atVert[N_A] = {2u};
    const float atFixed[3 * N_A] = {0.0f, 0.0f, 10.0f};
    const float atK[N_A]       = {4.0f};
    solver.uploadAttachments(N_A, atVert, atFixed, atK);

    const int rc = solver.step();
    if (rc != 0) {
        std::fprintf(stderr, "test_avbd_solver: step() returned %d\n", rc);
        return 1;
    }

    std::vector<float> posOut;
    solver.readPositions(posOut);

    const float expected_pos[3 * N] = {
        1.0f, 2.0f, 3.0f,
        8.0f / 3.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 20.0f / 3.0f,
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
    for (uint32_t i = 0; i < 3 * N; ++i) check(posOut[i], expected_pos[i], "pos", i);

    if (fails == 0) {
        std::printf("test_avbd_solver: OK (full AVBD step on %u verts, %u spring, %u attachment, max_abs_diff=%g)\n",
                    N, N_S, N_A, max_abs_diff);
        std::printf("  v0 pos: (%g, %g, %g) — expected (1, 2, 3)\n",      posOut[0], posOut[1], posOut[2]);
        std::printf("  v1 pos: (%g, %g, %g) — expected (8/3, 0, 0)\n",    posOut[3], posOut[4], posOut[5]);
        std::printf("  v2 pos: (%g, %g, %g) — expected (0, 0, 20/3)\n",   posOut[6], posOut[7], posOut[8]);
        return 0;
    }
    std::fprintf(stderr, "test_avbd_solver: %d FAIL\n", fails);
    return 1;
}
