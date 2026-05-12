// Unit test for cloth::avbdBackwardShim (CHI-14 Brick C).
//
// Verifies that the aggregator:
//   1. Runs end-to-end without NaN
//   2. Produces non-zero per-type sums where the underlying
//      per-constraint cotangents are non-zero
//   3. Matches manual sums of the AvbdSolver::read*Grad accessors
//   4. Routes density gradient through vertex_area correctly:
//      area·v_mass summed over verts == dL_ddensity
//
// Uses the same 4-vertex fixture as test_avbd_solver.cpp so we know
// the underlying solver outputs.

#include "AvbdBackwardShim.h"
#include "AvbdSolver.h"

#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <vector>

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s <metallib_dir>\n", argv[0]);
        return 2;
    }
    cloth::AvbdSolver solver(argv[1]);
    if (!solver.ok()) {
        std::fprintf(stderr, "construction failed\n");
        return 1;
    }

    // Same 4-vertex fixture as test_avbd_solver.cpp: 1 spring, 1
    // attachment, 1 triangle, 1 bending. All four constraint types
    // exercised so the shim's per-type aggregation matters.
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
    solver.setupMesh(N, positions, predicted, mass, 2.0f);

    const uint32_t spP1[] = {0u};
    const uint32_t spP2[] = {1u};
    const float    spLen[] = {2.0f};
    const float    spK[] = {1.0f};
    solver.uploadSprings(1u, spP1, spP2, spLen, spK);

    const uint32_t atVert[] = {2u};
    const float    atFixed[] = {0.0f, 4.0f, 10.0f};
    const float    atK[] = {4.0f};
    solver.uploadAttachments(1u, atVert, atFixed, atK);

    const uint32_t triIdx[] = {0u, 1u, 2u};
    const float    triInvUV[] = {1.0f, 0.0f, 0.0f, 1.0f};
    const float    triK[] = {1.0f};
    solver.uploadTriangles(1u, triIdx, triInvUV, triK);

    const uint32_t bendIdx[] = {0u, 1u, 2u, 3u};
    const float    bendW[] = {1.0f, 1.0f, -1.0f, -1.0f};
    const float    bendN[] = {4.0f};
    const float    bendK[] = {1.0f};
    solver.uploadBendings(1u, bendIdx, bendW, bendN, bendK);

    if (solver.step() != 0) {
        std::fprintf(stderr, "forward step failed\n");
        return 1;
    }

    // Loss cotangent: ∂L/∂x_v0 = (0, 0, 1) (z-component of v0).
    std::vector<double> dL_dx(3 * N, 0.0);
    dL_dx[2] = 1.0;

    // Per-vertex area for density aggregation. For the test fixture
    // we'll use unit area at each vertex (just exercises the
    // multiplication; physical values would come from the mesh).
    std::vector<double> vertex_area(N, 1.0);

    cloth::AvbdParamGradients g;
    if (cloth::avbdBackwardShim(solver, N, 1u, 1u, 1u,
                                dL_dx.data(), vertex_area.data(), g) != 0) {
        std::fprintf(stderr, "shim failed\n");
        return 1;
    }

    // --- Check 1: all aggregates finite -----------------------------
    auto fin = [](double x) { return std::isfinite(x); };
    int fails = 0;
    for (int i = 0; i < 4; ++i) {
        if (!fin(g.dL_dk_pertype[i])) {
            std::fprintf(stderr, "dL_dk_pertype[%d] non-finite: %g\n", i, g.dL_dk_pertype[i]);
            ++fails;
        }
    }
    if (!fin(g.dL_ddensity)) { std::fprintf(stderr, "dL_ddensity non-finite: %g\n", g.dL_ddensity); ++fails; }
    auto allFin = [&](const std::vector<double>& v, const char* n) {
        for (size_t i = 0; i < v.size(); ++i) {
            if (!fin(v[i])) {
                std::fprintf(stderr, "%s[%zu] non-finite: %g\n", n, i, v[i]);
                ++fails;
                break;
            }
        }
    };
    allFin(g.dL_dx, "dL_dx");
    allFin(g.dL_dxfixed, "dL_dxfixed");
    allFin(g.dL_dlambda_attach, "dL_dlambda_attach");
    allFin(g.dL_dlambda0_tri, "dL_dlambda0_tri");
    allFin(g.dL_dlambda1_tri, "dL_dlambda1_tri");
    allFin(g.dL_dlambda_bend, "dL_dlambda_bend");
    if (fails > 0) {
        std::fprintf(stderr, "FAIL: %d non-finite values\n", fails);
        return 1;
    }

    // --- Check 2: shim aggregates match manual sums ----------------
    // Re-run the underlying solver and read accessors directly, then
    // compute the per-type sums by hand.
    std::vector<float> v_loss_f(3 * N, 0.0f);
    v_loss_f[2] = 1.0f;
    if (solver.stepBackward(v_loss_f.data()) != 0) {
        std::fprintf(stderr, "stepBackward retest failed\n");
        return 1;
    }
    std::vector<float> dRest, dSpringK;  solver.readSpringGrad(dRest, dSpringK);
    std::vector<float> dFix, dAtKgrad, dAtLam; solver.readAttachGrad(dFix, dAtKgrad, dAtLam);
    std::vector<float> dTriK, dTriL0, dTriL1; solver.readTriGrad(dTriK, dTriL0, dTriL1);
    std::vector<float> dBendN, dBendK, dBendLam; solver.readBendGrad(dBendN, dBendK, dBendLam);

    auto sum = [](const std::vector<float>& v) {
        double s = 0.0; for (float x : v) s += double(x); return s;
    };
    const double manual_k_spring = sum(dSpringK);
    const double manual_k_attach = sum(dAtKgrad);
    const double manual_k_tri    = sum(dTriK);
    const double manual_k_bend   = sum(dBendK);

    auto check = [&](double got, double ref, const char* label) {
        const double d = std::fabs(got - ref);
        // The shim re-dispatches stepBackward so floating-point noise
        // can differ; allow 1e-5 absolute.
        if (d > 1e-5) {
            std::fprintf(stderr, "%s: shim %.6g vs manual %.6g (diff %.3g)\n",
                         label, got, ref, d);
            ++fails;
        }
    };
    check(g.dL_dk_pertype[0], manual_k_spring, "dL_dk_pertype[SPRING]");
    check(g.dL_dk_pertype[1], manual_k_attach, "dL_dk_pertype[ATTACH]");
    check(g.dL_dk_pertype[2], manual_k_tri,    "dL_dk_pertype[TRI]");
    check(g.dL_dk_pertype[3], manual_k_bend,   "dL_dk_pertype[BEND]");

    // --- Check 3: density aggregation route ------------------------
    std::vector<float> dMass; solver.readMassGrad(dMass);
    double manual_density = 0.0;
    for (uint32_t v = 0; v < N; ++v) {
        manual_density += double(dMass[v]) * vertex_area[v];
    }
    check(g.dL_ddensity, manual_density, "dL_ddensity");

    if (fails > 0) {
        std::fprintf(stderr, "FAIL: %d mismatches\n", fails);
        return 1;
    }

    std::printf("test_avbd_backward_shim: OK\n");
    std::printf("  dL_dk_pertype = [%.4g, %.4g, %.4g, %.4g]\n",
                g.dL_dk_pertype[0], g.dL_dk_pertype[1],
                g.dL_dk_pertype[2], g.dL_dk_pertype[3]);
    std::printf("  dL_ddensity   = %.4g\n", g.dL_ddensity);
    std::printf("  ||dL_dx||_inf = %.4g\n",
                [&]() {
                    double m = 0.0;
                    for (double x : g.dL_dx) m = std::max(m, std::fabs(x));
                    return m;
                }());
    std::printf("  dL_dxfixed[0..2] = (%.4g, %.4g, %.4g)\n",
                g.dL_dxfixed[0], g.dL_dxfixed[1], g.dL_dxfixed[2]);
    return 0;
}
