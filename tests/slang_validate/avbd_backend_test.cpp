// Backend-agnostic AvbdSolver driver. Builds a synthetic cloth (grid with
// springs, membrane triangles, bending stencils, pinned top row), runs a few
// forward steps and one backward pass, and dumps everything to a text file.
// Build it once against AvbdSolver.mm (Metal) and once against
// AvbdSolverRhi.cpp (slang-rhi), then diff the dumps with
// compare_avbd_dumps.py.
//
// usage: avbd_backend_test <kernel dir> <out.txt> [gridN=24] [steps=3] [iters=4]
#include "AvbdSolver.h"
#include "AvbdIft.h"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

using namespace cloth;

static void dump(FILE* f, const char* name, const std::vector<float>& v) {
    std::fprintf(f, "%s %zu\n", name, v.size());
    for (float x : v) std::fprintf(f, "%.9g\n", x);
}

int main(int argc, char** argv) {
    if (argc < 3) { std::fprintf(stderr, "usage: %s <dir> <out.txt> [N] [steps] [iters]\n", argv[0]); return 2; }
    const uint32_t N = argc > 3 ? uint32_t(std::atoi(argv[3])) : 24;
    const int steps = argc > 4 ? std::atoi(argv[4]) : 3;
    const int iters = argc > 5 ? std::atoi(argv[5]) : 4;
    const uint32_t nV = N * N;
    const float h = 0.01f, invH2 = 1.0f / (h * h);
    const float spacing = 0.05f;

    // AVBD_TEST_SKIP=s|a|t|b (any combination) drops springs/attachments/triangles/bendings.
    const char* skipEnv = std::getenv("AVBD_TEST_SKIP");
    const std::string skip = skipEnv ? skipEnv : "";
    auto skipped = [&](char c) { return skip.find(c) != std::string::npos; };

    AvbdSolver solver(argv[1]);
    if (!solver.ok()) { std::fprintf(stderr, "solver not ok\n"); return 1; }

    // Vertices: a saddle y = 2xz, so the bending stencil has a well-conditioned rest magnitude.
    std::vector<float> pos(3 * nV), pred(3 * nV), mass(nV, 0.02f);
    for (uint32_t j = 0; j < N; ++j)
        for (uint32_t i = 0; i < N; ++i) {
            uint32_t v = j * N + i;
            pos[3*v+0] = i * spacing;
            pos[3*v+1] = 2.0f * (i * spacing) * (j * spacing);   // saddle: constant bending stencil
            pos[3*v+2] = j * spacing;
            pred[3*v+0] = pos[3*v+0];
            pred[3*v+1] = pos[3*v+1] - 0.5f * 9.8f * h * h;   // gravity prediction
            pred[3*v+2] = pos[3*v+2];
            mass[v] = 0.02f + 0.001f * float(v % 7);
        }
    solver.setupMesh(nV, pos.data(), pred.data(), mass.data(), invH2);

    // Springs along grid edges + shear.
    std::vector<uint32_t> sp1, sp2; std::vector<float> srest, sk;
    auto addSpring = [&](uint32_t a, uint32_t b) {
        float dx = pos[3*a]-pos[3*b], dy = pos[3*a+1]-pos[3*b+1], dz = pos[3*a+2]-pos[3*b+2];
        sp1.push_back(a); sp2.push_back(b);
        srest.push_back(std::sqrt(dx*dx+dy*dy+dz*dz)); sk.push_back(2000.0f);
    };
    for (uint32_t j = 0; j < N; ++j)
        for (uint32_t i = 0; i < N; ++i) {
            uint32_t v = j * N + i;
            if (i + 1 < N) addSpring(v, v + 1);
            if (j + 1 < N) addSpring(v, v + N);
            if (i + 1 < N && j + 1 < N) addSpring(v, v + N + 1);
        }
    if (skipped('s')) { sp1.clear(); sp2.clear(); srest.clear(); sk.clear(); }
    solver.uploadSprings(uint32_t(sp1.size()), sp1.data(), sp2.data(), srest.data(), sk.data());

    // Pinned top row.
    std::vector<uint32_t> av; std::vector<float> afix, ak;
    for (uint32_t i = 0; i < N; ++i) {
        av.push_back(i);
        for (int c = 0; c < 3; ++c) afix.push_back(pos[3*i+c]);
        ak.push_back(5000.0f);
    }
    if (skipped('a')) { av.clear(); afix.clear(); ak.clear(); }
    solver.uploadAttachments(uint32_t(av.size()), av.data(), afix.data(), ak.data());

    // Membrane triangles + bending stencils.
    std::vector<uint32_t> tri, bend; std::vector<float> invUV, tk, bw, bn, bk;
    auto sub = [&](uint32_t a, uint32_t b, float* e) { for (int c = 0; c < 3; ++c) e[c] = pos[3*a+c] - pos[3*b+c]; };
    auto addTri = [&](uint32_t a, uint32_t b, uint32_t c) {
        tri.push_back(a); tri.push_back(b); tri.push_back(c);
        // Kernel computes F = [e0 e1] * M, so M = inverse of the 2D rest edge matrix D = [u1 u2].
        float e0[3], e1[3]; sub(b, a, e0); sub(c, a, e1);
        float l0 = std::sqrt(e0[0]*e0[0] + e0[1]*e0[1] + e0[2]*e0[2]);
        float d01 = e0[0]*e1[0] + e0[1]*e1[1] + e0[2]*e1[2];
        float cx = e0[1]*e1[2] - e0[2]*e1[1], cy = e0[2]*e1[0] - e0[0]*e1[2], cz = e0[0]*e1[1] - e0[1]*e1[0];
        float u1x = l0, u2x = d01 / l0, u2y = std::sqrt(cx*cx + cy*cy + cz*cz) / l0;
        float det = u1x * u2y;
        invUV.push_back(u2y / det);  invUV.push_back(-u2x / det);   // m00 m01
        invUV.push_back(0.0f);       invUV.push_back(u1x / det);    // m10 m11
        tk.push_back(800.0f);
    };
    for (uint32_t j = 0; j + 1 < N; ++j)
        for (uint32_t i = 0; i + 1 < N; ++i) {
            uint32_t v = j * N + i;
            addTri(v, v + 1, v + N);
            addTri(v + 1, v + N + 1, v + N);
            // bending across the shared diagonal (v+1, v+N); opposite corners v and v+N+1
            bend.push_back(v + 1); bend.push_back(v + N); bend.push_back(v); bend.push_back(v + N + 1);
            bw.push_back(1.0f); bw.push_back(1.0f); bw.push_back(-1.0f); bw.push_back(-1.0f);
            {   // rest magnitude of the weighted stencil sum
                float sx = 0, sy = 0, sz = 0;
                const uint32_t q[4] = { v + 1, v + N, v, v + N + 1 };
                const float w[4] = { 1.f, 1.f, -1.f, -1.f };
                for (int k = 0; k < 4; ++k) { sx += w[k]*pos[3*q[k]]; sy += w[k]*pos[3*q[k]+1]; sz += w[k]*pos[3*q[k]+2]; }
                bn.push_back(std::sqrt(sx*sx + sy*sy + sz*sz));
                bk.push_back(30.0f);
            }
        }
    if (skipped('t')) { tri.clear(); invUV.clear(); tk.clear(); }
    if (skipped('b')) { bend.clear(); bw.clear(); bn.clear(); bk.clear(); }
    solver.uploadTriangles(uint32_t(tk.size()), tri.data(), invUV.data(), tk.data());
    solver.uploadBendings(uint32_t(bk.size()), bend.data(), bw.data(), bn.data(), bk.data());
    solver.buildColoring();
    solver.uploadSelfCollisionRadii(std::vector<float>(nV, 0.01f).data(), 16u);

    if (std::getenv("AVBD_TEST_IFT")) {
        // HVP / IFT sanity: operator symmetry, step-size consistency, block-diagonal agreement, CG.
        std::vector<float> x0 = pos, y0 = pred;
        solver.updateState(x0.data(), y0.data());
        std::vector<float> g0, hb;
        solver.computeGradient(x0.data(), g0, &hb);
        double g0n = 0; for (float v : g0) g0n += double(v) * v;
        std::printf("|g(x0)| = %.4g\n", std::sqrt(g0n));
        const size_t n = 3 * size_t(nV);
        auto hvp = [&](const std::vector<double>& v, double eps) {
            std::vector<float> xp(n), xm(n), gp, gm; std::vector<double> out(n);
            for (size_t i = 0; i < n; ++i) { xp[i] = float(x0[i] + eps * v[i]); xm[i] = float(x0[i] - eps * v[i]); }
            solver.computeGradient(xp.data(), gp, nullptr); solver.computeGradient(xm.data(), gm, nullptr);
            for (size_t i = 0; i < n; ++i) out[i] = (double(gp[i]) - double(gm[i])) / (2 * eps);
            return out;
        };
        std::vector<double> v(n), w(n);
        unsigned seed = 12345; auto rnd = [&]() { seed = seed * 1664525u + 1013904223u; return double(seed >> 8) / double(1 << 24) - 0.5; };
        for (size_t i = 0; i < n; ++i) { v[i] = rnd(); w[i] = rnd(); }
        double ext = N * spacing;
        for (double rel : {1e-4, 1e-3, 1e-2}) {
            double eps = rel * ext / 0.5;
            auto Hv = hvp(v, eps), Hw = hvp(w, eps);
            double wHv = 0, vHw = 0, vHv = 0, nv = 0;
            for (size_t i = 0; i < n; ++i) { wHv += w[i]*Hv[i]; vHw += v[i]*Hw[i]; vHv += v[i]*Hv[i]; nv += Hv[i]*Hv[i]; }
            std::printf("eps_rel=%g  w'Hv=%.6g v'Hw=%.6g  asym=%.3g  v'Hv=%.6g |Hv|=%.4g\n", rel, wHv, vHw,
                        std::fabs(wHv - vHw) / (std::fabs(wHv) + std::fabs(vHw) + 1e-30), vHv, std::sqrt(nv));
        }
        // block-diagonal agreement: perturb only vertex v0
        { uint32_t v0 = nV / 2; std::vector<double> e(n, 0.0); e[3*v0] = 1.0;
          auto He = hvp(e, 1e-3 * ext);
          std::printf("H e_x at v0 (fd): %.6g %.6g %.6g   block col0 (hScratch): %.6g %.6g %.6g\n", He[3*v0], He[3*v0+1], He[3*v0+2],
                      hb[6*v0+0], hb[6*v0+1], hb[6*v0+2]); }
        std::vector<float> c(n), z; for (size_t i = 0; i < n; ++i) c[i] = float(rnd());
        {   // IFT-mode backward: does vbd_init_backward produce a mass cotangent?
            std::vector<float> zr(n); for (size_t i = 0; i < n; ++i) zr[i] = 1e-3f * float(rnd());
            solver.updateState(x0.data(), y0.data());
            solver.setIftMode(true);
            int rcb = solver.stepBackward(zr.data());
            solver.setIftMode(false);
            std::vector<float> dm, dpred; solver.readMassGrad(dm); solver.readPredictedGrad(dpred);
            double mx = 0, mp = 0; for (float v : dm) mx = std::max(mx, double(std::fabs(v))); for (float v : dpred) mp = std::max(mp, double(std::fabs(v)));
            std::printf("IFT-mode stepBackward rc=%d  max|dMass|=%.4g  max|dPred|=%.4g\n", rcb, mx, mp);
        }
        AvbdIft ift(solver);
        for (int rep = 0; rep < 3; ++rep) {
            float rr = 0; int it = ift.solve(x0.data(), y0.data(), nullptr, c.data(), 60, 1e-3f, z, &rr);
            std::printf("IFT solve %d: cg_iters=%d relres=%.3g refactors=%d\n", rep, it, rr, ift.refactorizations());
        }
        return 0;
    }

    FILE* f = std::fopen(argv[2], "w");
    if (!f) { std::fprintf(stderr, "cannot write %s\n", argv[2]); return 1; }

    std::vector<float> outPos;
    auto t0 = std::chrono::steady_clock::now();
    for (int s = 0; s < steps; ++s) {
        solver.readPositions(outPos);
        for (uint32_t v = 0; v < nV; ++v) {
            pred[3*v+1] = outPos[3*v+1] - 0.5f * 9.8f * h * h;
            pred[3*v+0] = outPos[3*v+0]; pred[3*v+2] = outPos[3*v+2];
        }
        solver.updateState(outPos.data(), pred.data());
        for (int it = 0; it < iters; ++it) {
            solver.step();
            solver.stepDualAttachments();
            solver.stepDualMembrane();
            solver.stepDualBending();
        }
        solver.readPositions(outPos);
        char name[32]; std::snprintf(name, sizeof(name), "positions_step%d", s);
        dump(f, name, outPos);
    }
    auto t1 = std::chrono::steady_clock::now();
    std::printf("forward: %d steps x %d iters in %.2f ms (%.3f ms/iter)\n", steps, iters,
                std::chrono::duration<double, std::milli>(t1 - t0).count(),
                std::chrono::duration<double, std::milli>(t1 - t0).count() / (steps * iters));

    std::vector<float> g, h6;
    solver.readScratch(g, h6);
    dump(f, "gScratch", g); dump(f, "hScratch", h6);

    // Backward: cotangent = small multiple of the final positions.
    std::vector<float> vloss(3 * nV);
    for (uint32_t i = 0; i < 3 * nV; ++i) vloss[i] = 1e-3f * outPos[i];
    int rc = solver.stepBackward(vloss.data());
    std::printf("stepBackward rc=%d\n", rc);
    std::vector<float> a, b, c;
    solver.readPositionsGrad(a); dump(f, "dL_dx", a);
    solver.readMassGrad(a);      dump(f, "dL_dmass", a);
    solver.readSpringGrad(a, b); dump(f, "dL_dSpringRest", a); dump(f, "dL_dSpringStiff", b);
    solver.readAttachGrad(a, b, c); dump(f, "dL_dAttachFixed", a); dump(f, "dL_dAttachStiff", b);
    solver.readTriGrad(a, b, c); dump(f, "dL_dTriStiff", a);
    solver.readBendGrad(a, b, c); dump(f, "dL_dBendNTarget", a); dump(f, "dL_dBendStiff", b);

    std::vector<std::pair<uint32_t, uint32_t>> pairs;
    int src = solver.detectSelfCollisions(pairs);
    std::fprintf(f, "selfcoll_pairs %d %zu\n", src, pairs.size());
    std::fclose(f);
    return 0;
}
