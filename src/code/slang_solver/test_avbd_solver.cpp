// Smoke test for AvbdSolver — full AVBD chain over springs + attachments
// + triangle membrane on real Metal hardware.
//
// Test fixture (3 verts, 1 spring, 1 attachment, 1 triangle):
//   v0:  x=(0,0,0)  pred=(1,2,3)  m=2
//   v1:  x=(3,0,0)  pred=(3,0,0)  m=1
//   v2:  x=(0,4,0)  pred=(0,4,0)  m=1
//   invH²=2
//
//   s0:  spring(0,1) L=2 k=1
//   a0:  attach v2 to (0,4,10) k=4
//   T0:  triangle (0,1,2) k=1  — current shape is a 3-4-5 right tri,
//        rest shape (after corotational projection) is unit edges in x,y;
//        deformation residual drives a per-vertex force/Hessian.
//
// vbd_init (w=2m):
//   v0: g=(-4,-8,-12), h_diag=4
//   v1: g=(0,0,0),     h_diag=2
//   v2: g=(0,0,0),     h_diag=2
//
// Spring (a=0, b=1, d=(-3,0,0), len=3, c=1, scale=1/3 → gradA=(-1,0,0), hess=[1,0,0,0,0,0]):
//   v0 role 0, sign=+1: g+=(-1,0,0)  h+=[1,0,0,0,0,0]
//   v1 role 1, sign=-1: g+=(1,0,0)   h+=[1,0,0,0,0,0]
//
// Attachment (pins v2 to (0,4,10), k=4):
//   gradV = k·(p-fixed) = 4·(0,0,-10) = (0,0,-40), hessScalar = 4
//   v2: g+=(0,0,-40)   h_diag+=4
//
// Triangle membrane:
//   F = [(3,0,0)|(0,4,0)]; a=3, b=0, d=4, R=I
//   newF.col(0)=(1,0,0), newF.col(1)=(0,1,0)
//   e0 = (2,0,0), e1_resid=(0,3,0)
//   With k=1:
//     grad[0] = -(2,3,0)  hessScalar[0] = 2
//     grad[1] = +(2,0,0)  hessScalar[1] = 1
//     grad[2] = +(0,3,0)  hessScalar[2] = 1
//   v0 (role 0): g+=(-2,-3,0)  h_diag+=2
//   v1 (role 1): g+=(2,0,0)    h_diag+=1
//   v2 (role 2): g+=(0,3,0)    h_diag+=1
//
// Accumulated per-vertex (g, H=diag(Hxx,Hyy,Hzz)):
//   v0:  g=(-7,-11,-12)  H=diag(7,6,6)
//   v1:  g=(3,0,0)       H=diag(4,3,3)
//   v2:  g=(0,3,-40)     H=diag(7,7,7)
//
// vbd_solve_apply Δx = -H⁻¹·g (diagonal H → componentwise):
//   v0:  Δx = (7/7, 11/6, 12/6) = (1, 11/6, 2)
//        new pos = (1, 11/6, 2)
//   v1:  Δx = (-3/4, 0, 0)
//        new pos = (9/4, 0, 0)
//   v2:  Δx = (0, -3/7, 40/7)
//        new pos = (0, 25/7, 40/7)

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
        0.0f, 4.0f, 0.0f,
    };
    const float predicted[3 * N] = {
        1.0f, 2.0f, 3.0f,
        3.0f, 0.0f, 0.0f,
        0.0f, 4.0f, 0.0f,
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
    const uint32_t atVert[N_A]     = {2u};
    const float atFixed[3 * N_A]   = {0.0f, 4.0f, 10.0f};
    const float atK[N_A]           = {4.0f};
    solver.uploadAttachments(N_A, atVert, atFixed, atK);

    constexpr uint32_t N_T = 1;
    const uint32_t triIdx[3 * N_T] = {0u, 1u, 2u};
    const float triK[N_T]          = {1.0f};
    solver.uploadTriangles(N_T, triIdx, triK);

    const int rc = solver.step();
    if (rc != 0) {
        std::fprintf(stderr, "test_avbd_solver: step() returned %d\n", rc);
        return 1;
    }

    std::vector<float> posOut;
    solver.readPositions(posOut);

    const float expected_pos[3 * N] = {
        1.0f, 11.0f / 6.0f, 2.0f,
        9.0f / 4.0f, 0.0f, 0.0f,
        0.0f, 25.0f / 7.0f, 40.0f / 7.0f,
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
        std::printf("test_avbd_solver: OK (full AVBD step on %u verts: %u springs, %u attach, %u tri, max_abs_diff=%g)\n",
                    N, N_S, N_A, N_T, max_abs_diff);
        std::printf("  v0 pos: (%g, %g, %g) — expected (1, 11/6, 2)\n",       posOut[0], posOut[1], posOut[2]);
        std::printf("  v1 pos: (%g, %g, %g) — expected (9/4, 0, 0)\n",        posOut[3], posOut[4], posOut[5]);
        std::printf("  v2 pos: (%g, %g, %g) — expected (0, 25/7, 40/7)\n",    posOut[6], posOut[7], posOut[8]);
        return 0;
    }
    std::fprintf(stderr, "test_avbd_solver: %d FAIL\n", fails);
    return 1;
}
