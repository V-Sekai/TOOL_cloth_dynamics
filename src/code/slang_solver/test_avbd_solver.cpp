// Smoke test for AvbdSolver — full AVBD pipeline (all 10 kernels)
// over springs + attachments + triangle membrane + bending on real
// Metal hardware.
//
// Test fixture (4 verts, 1 spring, 1 attachment, 1 triangle, 1 bending):
//   v0:  x=(0,0,0)  pred=(1,2,3)  m=2
//   v1:  x=(3,0,0)  pred=(3,0,0)  m=1
//   v2:  x=(0,4,0)  pred=(0,4,0)  m=1
//   v3:  x=(3,4,0)  pred=(3,4,0)  m=1
//   invH²=2
//
//   s0:  spring(0,1) L=2 k=1
//   a0:  attach v2 to (0,4,10) k=4
//   T0:  triangle (0,1,2) k=1 (3-4-5 right tri being projected toward
//        unit-edges rest)
//   B0:  bending stencil (0,1,2,3) weights=(1,1,-1,-1) nTarget=4 k=1
//        Σ w·p = (0,0,0)+(3,0,0)-(0,4,0)-(3,4,0) = (0,-8,0)
//        |s|=8, scale=4/8=0.5, om=0.5, resid=(0,-4,0)
//        Per-corner: grad = k·w·resid; hessScalar = k·w²
//          v0 (w=+1):  grad=(0,-4,0)  hs=1
//          v1 (w=+1):  grad=(0,-4,0)  hs=1
//          v2 (w=-1):  grad=(0,+4,0)  hs=1
//          v3 (w=-1):  grad=(0,+4,0)  hs=1
//
// Accumulated per-vertex (g, H=diag(Hxx,Hyy,Hzz)):
//   v0: inertial(-4,-8,-12)/4 + spring(-1,0,0)/+1 + tri(-2,-3,0)/+2 + bend(0,-4,0)/+1
//       → g=(-7,-15,-12)  H=diag(8,7,7)
//   v1: inertial 0/2 + spring(+1,0,0)/+1 + tri(+2,0,0)/+1 + bend(0,-4,0)/+1
//       → g=(3,-4,0)      H=diag(5,4,4)
//   v2: inertial 0/2 + attachment(0,0,-40)/+4 + tri(0,+3,0)/+1 + bend(0,+4,0)/+1
//       → g=(0,7,-40)     H=diag(8,8,8)
//   v3: inertial 0/2 + bend(0,+4,0)/+1
//       → g=(0,4,0)       H=diag(3,3,3)
//
// vbd_solve_apply Δx = −H⁻¹·g (diagonal):
//   v0: Δx=(7/8, 15/7, 12/7)     pos=(7/8, 15/7, 12/7)
//   v1: Δx=(-3/5, 1, 0)          pos=(3-3/5, 1, 0) = (12/5, 1, 0)
//   v2: Δx=(0, -7/8, 5)          pos=(0, 4-7/8, 5) = (0, 25/8, 5)
//   v3: Δx=(0, -4/3, 0)          pos=(3, 4-4/3, 0) = (3, 8/3, 0)

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

    constexpr uint32_t N = 4;
    const float positions[3 * N] = {
        0.0f, 0.0f, 0.0f,
        3.0f, 0.0f, 0.0f,
        0.0f, 4.0f, 0.0f,
        3.0f, 4.0f, 0.0f,
    };
    const float predicted[3 * N] = {
        1.0f, 2.0f, 3.0f,
        3.0f, 0.0f, 0.0f,
        0.0f, 4.0f, 0.0f,
        3.0f, 4.0f, 0.0f,
    };
    const float mass[N] = {2.0f, 1.0f, 1.0f, 1.0f};
    constexpr float invHSquared = 2.0f;
    solver.setupMesh(N, positions, predicted, mass, invHSquared);

    constexpr uint32_t N_S = 1;
    const uint32_t spP1[N_S] = {0u};
    const uint32_t spP2[N_S] = {1u};
    const float spLen[N_S]   = {2.0f};
    const float spK[N_S]     = {1.0f};
    solver.uploadSprings(N_S, spP1, spP2, spLen, spK);

    constexpr uint32_t N_A = 1;
    const uint32_t atVert[N_A]   = {2u};
    const float atFixed[3 * N_A] = {0.0f, 4.0f, 10.0f};
    const float atK[N_A]         = {4.0f};
    solver.uploadAttachments(N_A, atVert, atFixed, atK);

    constexpr uint32_t N_T = 1;
    const uint32_t triIdx[3 * N_T] = {0u, 1u, 2u};
    // Identity inv_deltaUV — canonical rest, same as pre-PR-G behavior.
    const float triInvUV[4 * N_T]  = {1.0f, 0.0f, 0.0f, 1.0f};
    const float triK[N_T]          = {1.0f};
    solver.uploadTriangles(N_T, triIdx, triInvUV, triK);

    constexpr uint32_t N_B = 1;
    const uint32_t bendIdx[4 * N_B]    = {0u, 1u, 2u, 3u};
    const float bendWeight[4 * N_B]    = {1.0f, 1.0f, -1.0f, -1.0f};
    const float bendNTarget[N_B]       = {4.0f};
    const float bendK[N_B]             = {1.0f};
    solver.uploadBendings(N_B, bendIdx, bendWeight, bendNTarget, bendK);

    const int rc = solver.step();
    if (rc != 0) {
        std::fprintf(stderr, "test_avbd_solver: step() returned %d\n", rc);
        return 1;
    }

    std::vector<float> posOut;
    solver.readPositions(posOut);

    const float expected_pos[3 * N] = {
        7.0f / 8.0f, 15.0f / 7.0f, 12.0f / 7.0f,
        12.0f / 5.0f, 1.0f, 0.0f,
        0.0f, 25.0f / 8.0f, 5.0f,
        3.0f, 8.0f / 3.0f, 0.0f,
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

    if (fails != 0) {
        std::fprintf(stderr, "test_avbd_solver: forward %d FAIL\n", fails);
        return 1;
    }
    std::printf("test_avbd_solver: forward OK (full AVBD step on %u verts: %u spring %u attach %u tri %u bend, max_abs_diff=%g)\n",
                N, N_S, N_A, N_T, N_B, max_abs_diff);
    std::printf("  v0 pos: (%g, %g, %g) — expected (7/8, 15/7, 12/7)\n",  posOut[0], posOut[1], posOut[2]);
    std::printf("  v1 pos: (%g, %g, %g) — expected (12/5, 1, 0)\n",       posOut[3], posOut[4], posOut[5]);
    std::printf("  v2 pos: (%g, %g, %g) — expected (0, 25/8, 5)\n",       posOut[6], posOut[7], posOut[8]);
    std::printf("  v3 pos: (%g, %g, %g) — expected (3, 8/3, 0)\n",        posOut[9], posOut[10], posOut[11]);

    // ----- PR-G / CHI-13: backward smoke ----------------------------------
    // Pass an arbitrary loss cotangent on the final positions and confirm
    // stepBackward() runs end-to-end and produces non-degenerate gradients
    // for every parameter category. This is a smoke test — full FD
    // validation lives in tests/slang_validate/*_backward_test.cpp.
    std::vector<float> vLoss(3 * N, 0.0f);
    // Loss = x_0_z (sensitive to all params via v0's update). Cotangent:
    // (0,0,1) on vertex 0, zero on others.
    vLoss[2] = 1.0f;
    const int rcBwd = solver.stepBackward(vLoss.data());
    if (rcBwd != 0) {
        std::fprintf(stderr, "test_avbd_solver: stepBackward returned %d\n", rcBwd);
        return 1;
    }

    std::vector<float> dPos, dSpringL, dSpringK;
    std::vector<float> dAttachFixed, dAttachK, dAttachLam;
    std::vector<float> dTriK, dTriLam0, dTriLam1;
    std::vector<float> dBendN, dBendK, dBendLam;
    solver.readPositionsGrad(dPos);
    solver.readSpringGrad(dSpringL, dSpringK);
    solver.readAttachGrad(dAttachFixed, dAttachK, dAttachLam);
    solver.readTriGrad(dTriK, dTriLam0, dTriLam1);
    solver.readBendGrad(dBendN, dBendK, dBendLam);

    auto allFinite = [](const std::vector<float>& v) {
        for (float x : v) if (!std::isfinite(x)) return false;
        return true;
    };
    auto reportNF = [](const std::vector<float>& v, const char* name) {
        for (size_t i = 0; i < v.size(); ++i) {
            if (!std::isfinite(v[i])) {
                std::fprintf(stderr, "  %s[%zu] = %g (non-finite)\n", name, i, v[i]);
            }
        }
    };
    bool bad = false;
    if (!allFinite(dPos))         { reportNF(dPos,         "dPos");         bad = true; }
    if (!allFinite(dSpringL))     { reportNF(dSpringL,     "dSpringL");     bad = true; }
    if (!allFinite(dSpringK))     { reportNF(dSpringK,     "dSpringK");     bad = true; }
    if (!allFinite(dAttachFixed)) { reportNF(dAttachFixed, "dAttachFixed"); bad = true; }
    if (!allFinite(dAttachK))     { reportNF(dAttachK,     "dAttachK");     bad = true; }
    if (!allFinite(dAttachLam))   { reportNF(dAttachLam,   "dAttachLam");   bad = true; }
    if (!allFinite(dTriK))        { reportNF(dTriK,        "dTriK");        bad = true; }
    if (!allFinite(dTriLam0))     { reportNF(dTriLam0,     "dTriLam0");     bad = true; }
    if (!allFinite(dTriLam1))     { reportNF(dTriLam1,     "dTriLam1");     bad = true; }
    if (!allFinite(dBendN))       { reportNF(dBendN,       "dBendN");       bad = true; }
    if (!allFinite(dBendK))       { reportNF(dBendK,       "dBendK");       bad = true; }
    if (!allFinite(dBendLam))     { reportNF(dBendLam,     "dBendLam");     bad = true; }
    if (bad) {
        std::fprintf(stderr, "test_avbd_solver: backward produced non-finite gradients\n");
        return 1;
    }

    std::printf("test_avbd_solver: backward OK (stepBackward + read accessors)\n");
    std::printf("  ∂L/∂x      : v0=(%.3g, %.3g, %.3g)  v1=(%.3g, %.3g, %.3g)\n",
                dPos[0], dPos[1], dPos[2], dPos[3], dPos[4], dPos[5]);
    std::printf("  ∂L/∂L (s0) : %.3g     ∂L/∂k (s0)  : %.3g\n", dSpringL[0], dSpringK[0]);
    std::printf("  ∂L/∂anchor : (%.3g, %.3g, %.3g)  ∂L/∂k_attach: %.3g\n",
                dAttachFixed[0], dAttachFixed[1], dAttachFixed[2], dAttachK[0]);
    std::printf("  ∂L/∂k_tri  : %.3g     ∂L/∂λ0_tri  : (%.3g, %.3g, %.3g)\n",
                dTriK[0], dTriLam0[0], dTriLam0[1], dTriLam0[2]);
    std::printf("  ∂L/∂n_bend : %.3g     ∂L/∂k_bend  : %.3g\n", dBendN[0], dBendK[0]);

    return 0;
}
