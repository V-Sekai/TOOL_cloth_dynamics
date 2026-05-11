// First end-to-end PD step. Composes six slangc-emitted kernels in
// one harness:
//
//   spring_project       — local step for the stretch spring
//   attachment_project   — local step for the pin
//   assemble_b           — gather per-constraint projections into b
//   spmv                 — A·p in the CG inner loop
//   saxpby               — vector updates in the CG inner loop
//   dot_reduce_serial    — inner products in the CG inner loop
//
// System under test (smallest non-trivial PD case):
//
//   2 vertices.  v0 pinned at (0, 0, 0) with sqrtW_att = 1.
//                v1 connected to v0 by a spring, rest length l0 = 1,
//                sqrtW_spring = 1.
//   Initial positions:  x = [ (0,0,0),  (2,0,0) ].
//   Predictor (zero velocity, no gravity, dt = 1): s = x.
//   Mass:  m_0 = m_1 = 1   (so M/h² = identity).
//
// Hand-derived single-PD-step answer (working through the energy
// minimisation directly):
//
//   E(q_0, q_1) = ½‖q_0 − 0‖² + ½‖q_1 − 2‖²       (mass / inertia)
//               + ½·1·‖q_0 − 0‖²                   (attachment)
//               + ½·1·‖(q_0 − q_1) − (−1)‖²        (spring; p_s = (−1,0,0))
//
//   ∂E/∂q_0 :  3·q_0 − q_1 + 1 = 0
//   ∂E/∂q_1 : −q_0 + 2·q_1 − 3 = 0
//
//   → q_0 = 0.2,  q_1 = 1.6.
//
// Expected:  q_new = [(0.2, 0, 0),  (1.6, 0, 0)].
//
// The harness mirrors the cg_demo pattern's namespace + EXTERN_C
// neutralisation, extended to six co-included emit.cpp's.

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <vector>

#include "slang-cpp-prelude.h"

#define WRAP_KERNEL(NS, EMIT)                                       \
    namespace NS {                                                  \
        _Pragma("push_macro(\"SLANG_PRELUDE_EXTERN_C\")")           \
        _Pragma("push_macro(\"SLANG_PRELUDE_EXTERN_C_START\")")     \
        _Pragma("push_macro(\"SLANG_PRELUDE_EXTERN_C_END\")")       \
    }

namespace pd_spring {
    #pragma push_macro("SLANG_PRELUDE_EXTERN_C")
    #pragma push_macro("SLANG_PRELUDE_EXTERN_C_START")
    #pragma push_macro("SLANG_PRELUDE_EXTERN_C_END")
    #undef  SLANG_PRELUDE_EXTERN_C
    #undef  SLANG_PRELUDE_EXTERN_C_START
    #undef  SLANG_PRELUDE_EXTERN_C_END
    #define SLANG_PRELUDE_EXTERN_C
    #define SLANG_PRELUDE_EXTERN_C_START
    #define SLANG_PRELUDE_EXTERN_C_END
    #include "spring_project_emit.cpp"
    #pragma pop_macro("SLANG_PRELUDE_EXTERN_C")
    #pragma pop_macro("SLANG_PRELUDE_EXTERN_C_START")
    #pragma pop_macro("SLANG_PRELUDE_EXTERN_C_END")
}

namespace pd_attach {
    #pragma push_macro("SLANG_PRELUDE_EXTERN_C")
    #pragma push_macro("SLANG_PRELUDE_EXTERN_C_START")
    #pragma push_macro("SLANG_PRELUDE_EXTERN_C_END")
    #undef  SLANG_PRELUDE_EXTERN_C
    #undef  SLANG_PRELUDE_EXTERN_C_START
    #undef  SLANG_PRELUDE_EXTERN_C_END
    #define SLANG_PRELUDE_EXTERN_C
    #define SLANG_PRELUDE_EXTERN_C_START
    #define SLANG_PRELUDE_EXTERN_C_END
    #include "attachment_project_emit.cpp"
    #pragma pop_macro("SLANG_PRELUDE_EXTERN_C")
    #pragma pop_macro("SLANG_PRELUDE_EXTERN_C_START")
    #pragma pop_macro("SLANG_PRELUDE_EXTERN_C_END")
}

namespace pd_assemble {
    #pragma push_macro("SLANG_PRELUDE_EXTERN_C")
    #pragma push_macro("SLANG_PRELUDE_EXTERN_C_START")
    #pragma push_macro("SLANG_PRELUDE_EXTERN_C_END")
    #undef  SLANG_PRELUDE_EXTERN_C
    #undef  SLANG_PRELUDE_EXTERN_C_START
    #undef  SLANG_PRELUDE_EXTERN_C_END
    #define SLANG_PRELUDE_EXTERN_C
    #define SLANG_PRELUDE_EXTERN_C_START
    #define SLANG_PRELUDE_EXTERN_C_END
    #include "assemble_b_emit.cpp"
    #pragma pop_macro("SLANG_PRELUDE_EXTERN_C")
    #pragma pop_macro("SLANG_PRELUDE_EXTERN_C_START")
    #pragma pop_macro("SLANG_PRELUDE_EXTERN_C_END")
}

namespace pd_spmv {
    #pragma push_macro("SLANG_PRELUDE_EXTERN_C")
    #pragma push_macro("SLANG_PRELUDE_EXTERN_C_START")
    #pragma push_macro("SLANG_PRELUDE_EXTERN_C_END")
    #undef  SLANG_PRELUDE_EXTERN_C
    #undef  SLANG_PRELUDE_EXTERN_C_START
    #undef  SLANG_PRELUDE_EXTERN_C_END
    #define SLANG_PRELUDE_EXTERN_C
    #define SLANG_PRELUDE_EXTERN_C_START
    #define SLANG_PRELUDE_EXTERN_C_END
    #include "spmv_emit.cpp"
    #pragma pop_macro("SLANG_PRELUDE_EXTERN_C")
    #pragma pop_macro("SLANG_PRELUDE_EXTERN_C_START")
    #pragma pop_macro("SLANG_PRELUDE_EXTERN_C_END")
}

namespace pd_saxpby {
    #pragma push_macro("SLANG_PRELUDE_EXTERN_C")
    #pragma push_macro("SLANG_PRELUDE_EXTERN_C_START")
    #pragma push_macro("SLANG_PRELUDE_EXTERN_C_END")
    #undef  SLANG_PRELUDE_EXTERN_C
    #undef  SLANG_PRELUDE_EXTERN_C_START
    #undef  SLANG_PRELUDE_EXTERN_C_END
    #define SLANG_PRELUDE_EXTERN_C
    #define SLANG_PRELUDE_EXTERN_C_START
    #define SLANG_PRELUDE_EXTERN_C_END
    #include "saxpby_emit.cpp"
    #pragma pop_macro("SLANG_PRELUDE_EXTERN_C")
    #pragma pop_macro("SLANG_PRELUDE_EXTERN_C_START")
    #pragma pop_macro("SLANG_PRELUDE_EXTERN_C_END")
}

namespace pd_dot {
    #pragma push_macro("SLANG_PRELUDE_EXTERN_C")
    #pragma push_macro("SLANG_PRELUDE_EXTERN_C_START")
    #pragma push_macro("SLANG_PRELUDE_EXTERN_C_END")
    #undef  SLANG_PRELUDE_EXTERN_C
    #undef  SLANG_PRELUDE_EXTERN_C_START
    #undef  SLANG_PRELUDE_EXTERN_C_END
    #define SLANG_PRELUDE_EXTERN_C
    #define SLANG_PRELUDE_EXTERN_C_START
    #define SLANG_PRELUDE_EXTERN_C_END
    #include "dot_reduce_serial_emit.cpp"
    #pragma pop_macro("SLANG_PRELUDE_EXTERN_C")
    #pragma pop_macro("SLANG_PRELUDE_EXTERN_C_START")
    #pragma pop_macro("SLANG_PRELUDE_EXTERN_C_END")
}

static constexpr double kBudgetSeconds = 5.0;

int main() {
    const auto t0 = std::chrono::steady_clock::now();

    constexpr uint32_t V = 2;
    constexpr uint32_t N = 3 * V;          // CG vector dim (one float per (vertex, dim))
    constexpr uint32_t MAX_CG = 30;
    constexpr double   CG_TOL = 1e-6;

    // ---- Vertex positions and per-vertex state ---------------------------
    // Flat float[] for the CG side; Vector<float,3>[] for the constraint
    // kernels. The two layouts are bit-identical (Vector<float,3> is three
    // contiguous floats).
    std::vector<Vector<float, 3>> positions(64, Vector<float, 3>(0.0f));
    positions[0] = Vector<float, 3>(0.0f, 0.0f, 0.0f);
    positions[1] = Vector<float, 3>(2.0f, 0.0f, 0.0f);

    // ---- Spring kernel inputs --------------------------------------------
    // One real spring (v0—v1, l0 = 1, sqrtW = 1); padded slots are inert.
    constexpr uint32_t GROUP_SIZE = 64;
    std::vector<uint32_t> sp_p1(GROUP_SIZE, 1u);
    std::vector<uint32_t> sp_p2(GROUP_SIZE, 0u);
    std::vector<float>    sp_rest(GROUP_SIZE, 0.0f);
    std::vector<float>    sp_sqrtW(GROUP_SIZE, 0.0f);
    sp_p1[0] = 0u; sp_p2[0] = 1u; sp_rest[0] = 1.0f; sp_sqrtW[0] = 1.0f;

    std::vector<Vector<float, 3>> sp_out(GROUP_SIZE, Vector<float, 3>(0.0f));

    pd_spring::GlobalParams_0 gp_spring{};
    gp_spring.positions_0.data  = positions.data();  gp_spring.positions_0.count  = positions.size();
    gp_spring.projected_0.data  = sp_out.data();     gp_spring.projected_0.count  = sp_out.size();
    gp_spring.p1Idx_0.data      = sp_p1.data();      gp_spring.p1Idx_0.count      = sp_p1.size();
    gp_spring.p2Idx_0.data      = sp_p2.data();      gp_spring.p2Idx_0.count      = sp_p2.size();
    gp_spring.restLen_0.data    = sp_rest.data();    gp_spring.restLen_0.count    = sp_rest.size();
    gp_spring.sqrtWeight_0.data = sp_sqrtW.data();   gp_spring.sqrtWeight_0.count = sp_sqrtW.size();

    // ---- Attachment kernel inputs ----------------------------------------
    // One real pin (v0 → world (0,0,0)); padded slots inert.
    std::vector<Vector<float, 3>> at_fixed(GROUP_SIZE, Vector<float, 3>(0.0f));
    std::vector<float>            at_sqrtW(GROUP_SIZE, 0.0f);
    at_fixed[0] = Vector<float, 3>(0.0f, 0.0f, 0.0f);
    at_sqrtW[0] = 1.0f;

    std::vector<Vector<float, 3>> at_out(GROUP_SIZE, Vector<float, 3>(0.0f));

    pd_attach::GlobalParams_0 gp_attach{};
    gp_attach.projected_0.data  = at_out.data();    gp_attach.projected_0.count  = at_out.size();
    gp_attach.fixedPos_0.data   = at_fixed.data();  gp_attach.fixedPos_0.count   = at_fixed.size();
    gp_attach.sqrtWeight_0.data = at_sqrtW.data();  gp_attach.sqrtWeight_0.count = at_sqrtW.size();

    ComputeVaryingInput vi{};
    vi.startGroupID = uint3(0, 0, 0);
    vi.endGroupID   = uint3(1, 1, 1);

    // ---- Local step: dispatch both constraint kernels --------------------
    pd_spring::main_0(&vi, nullptr, &gp_spring);
    pd_attach::main_0(&vi, nullptr, &gp_attach);

    // ---- Stack the constraint outputs into one projections buffer --------
    //   slot 0: spring output (sp_out[0])
    //   slot 1: attachment output (at_out[0])
    std::vector<float> projections(3 * 2);
    projections[0] = sp_out[0][0]; projections[1] = sp_out[0][1]; projections[2] = sp_out[0][2];
    projections[3] = at_out[0][0]; projections[4] = at_out[0][1]; projections[5] = at_out[0][2];

    // ---- assemble_b inputs -----------------------------------------------
    //   v0 ← +1·slot 0 (spring's p1 endpoint) + 1·slot 1 (attachment)
    //   v1 ← −1·slot 0 (spring's p2 endpoint)
    std::vector<float> s(N), mass(V, 1.0f), b(N, 0.0f);
    s[0] = 0.0f; s[1] = 0.0f; s[2] = 0.0f;    // s[v=0] = x[0] (zero velocity, no gravity)
    s[3] = 2.0f; s[4] = 0.0f; s[5] = 0.0f;    // s[v=1] = x[1]

    std::vector<uint32_t> ctxStart  = {0u, 2u, 3u};
    std::vector<uint32_t> ctxSlot   = {0u, 1u, 0u};
    std::vector<float>    ctxWeight = {+1.0f, +1.0f, -1.0f};

    pd_assemble::AssembleBParams_0 abp{V};
    pd_assemble::GlobalParams_0    gp_ab{};
    gp_ab.params_0           = &abp;
    gp_ab.s_0.data           = s.data();           gp_ab.s_0.count           = s.size();
    gp_ab.mass_0.data        = mass.data();        gp_ab.mass_0.count        = mass.size();
    gp_ab.projections_0.data = projections.data(); gp_ab.projections_0.count = projections.size();
    gp_ab.ctxStart_0.data    = ctxStart.data();    gp_ab.ctxStart_0.count    = ctxStart.size();
    gp_ab.ctxSlot_0.data     = ctxSlot.data();     gp_ab.ctxSlot_0.count     = ctxSlot.size();
    gp_ab.ctxWeight_0.data   = ctxWeight.data();   gp_ab.ctxWeight_0.count   = ctxWeight.size();
    gp_ab.b_0.data           = b.data();           gp_ab.b_0.count           = b.size();

    pd_assemble::main_0(&vi, nullptr, &gp_ab);

    // Sanity: hand-derived b for the x-component should be [-1, 3].
    // (The y/z components are zero by symmetry.) We rely on the bit-exact
    // assemble_b validation; this is a print, not a hard check.
    // b[0] = m·s[0] + 1·(-1) + 1·0    = -1
    // b[3] = m·s[3] + (-1)·(-1)        = 2 + 1 = 3
    std::printf("  after local+assemble:  b = [%g, %g, %g,  %g, %g, %g]\n",
                b[0], b[1], b[2], b[3], b[4], b[5]);

    // ---- Build A = M/h² + S^T·S in CSR. Hand-computed for this system: --
    //
    //   v0:  spring (+1, p_v0=v0)·(+1, p_v0=v0) = +1   in (v0,v0)
    //   v0:  spring (+1, p_v0=v0)·(−1, p_v0=v1) = −1   in (v0,v1)
    //   v0:  attach (+1, p_v0=v0)·(+1, p_v0=v0) = +1   in (v0,v0)  (additional)
    //   v1:  spring (−1, p_v0=v1)·(+1, p_v0=v0) = −1   in (v1,v0)
    //   v1:  spring (−1, p_v0=v1)·(−1, p_v0=v1) = +1   in (v1,v1)
    //
    //   plus M/h² diagonal: +1 on every (vid, vid)
    //
    //   per-dim x/y/z block (each dim decouples; we encode all six rows):
    //     A[0,0] = 1+2 = 3       A[0,3] = -1
    //     A[3,0] = -1            A[3,3] = 1+1 = 2
    //   diagonal-only for y/z (no spring couples y or z components across)
    //     A[1,1] = 3   A[2,2] = 3
    //     A[4,4] = 2   A[5,5] = 2
    //   Wait: the spring's S has +1 in (c+d, v0_dim) and -1 in (c+d, v1_dim).
    //   Sum over c+d: contributes 1 to (v0d, v0d), -1 to (v0d, v1d), -1 to (v1d, v0d),
    //   1 to (v1d, v1d) — for every dim d, since spring is isotropic.
    //
    //   So A is block-diagonal across dims (each dim block is 2x2):
    //     [3 -1]
    //     [-1 2]
    //   (this is the x-block; y-block and z-block are identical).
    //   The 6x6 A in row-major has 12 nonzeros (4 per dim block).
    //
    //   CSR ordering rows 0..5 = v0x, v0y, v0z, v1x, v1y, v1z.
    //   Each row has 2 nonzeros.
    std::vector<int32_t> A_rowPtr = {0, 2, 4, 6, 8, 10, 12};
    std::vector<int32_t> A_colIdx = {0, 3,
                                     1, 4,
                                     2, 5,
                                     0, 3,
                                     1, 4,
                                     2, 5};
    std::vector<float>   A_vals   = { 3.0f, -1.0f,
                                      3.0f, -1.0f,
                                      3.0f, -1.0f,
                                     -1.0f,  2.0f,
                                     -1.0f,  2.0f,
                                     -1.0f,  2.0f};
    constexpr uint32_t A_NNZ = 12;

    pd_spmv::SpmvParams_0  spp{N};
    pd_spmv::GlobalParams_0 gp_sp{};
    gp_sp.params_0      = &spp;
    gp_sp.rowPtr_0.data = A_rowPtr.data(); gp_sp.rowPtr_0.count = A_rowPtr.size();
    gp_sp.colIdx_0.data = A_colIdx.data(); gp_sp.colIdx_0.count = A_colIdx.size();
    gp_sp.values_0.data = A_vals.data();   gp_sp.values_0.count = A_vals.size();

    // ---- CG outer loop (re-uses cg_demo pattern, now on the PD system) ---
    std::vector<float> q(N, 0.0f);              // initial guess
    std::vector<float> r(N), p_dir(N), qprod(N);
    for (uint32_t i = 0; i < N; ++i) {
        r[i]     = b[i];                        // r₀ = b − A·0 = b
        p_dir[i] = b[i];
    }

    pd_dot::DotReduceSerialParams_0 dp{N};
    float dot_dst[2] = {0.0f, 0.0f};
    pd_dot::GlobalParams_0 gp_dot{};
    gp_dot.params_0   = &dp;
    gp_dot.dst_0.data = dot_dst; gp_dot.dst_0.count = 2;

    auto kernelDot = [&](float* aa, float* bb) -> double {
        gp_dot.a_0.data = aa; gp_dot.a_0.count = N;
        gp_dot.b_0.data = bb; gp_dot.b_0.count = N;
        pd_dot::main_0(&vi, nullptr, &gp_dot);
        return double(dot_dst[0]) + double(dot_dst[1]);
    };

    double delta_new = kernelDot(r.data(), r.data());
    const double delta_0 = delta_new;

    pd_saxpby::SaxpbyParams_0 spy{N, 0.0f, 0.0f};
    pd_saxpby::GlobalParams_0 gp_sax{};
    gp_sax.params_0 = &spy;

    int cg_iters = 0;
    for (; cg_iters < int(MAX_CG); ++cg_iters) {
        // qprod = A · p_dir
        gp_sp.x_0.data = p_dir.data(); gp_sp.x_0.count = N;
        gp_sp.y_0.data = qprod.data(); gp_sp.y_0.count = N;
        pd_spmv::main_0(&vi, nullptr, &gp_sp);

        const double pq    = kernelDot(p_dir.data(), qprod.data());
        const double alpha = delta_new / pq;

        // q := 1·q + α·p_dir
        spy.alpha_0 = 1.0f; spy.beta_0 = float(alpha);
        gp_sax.x_0.data = q.data();     gp_sax.x_0.count   = N;
        gp_sax.y_0.data = p_dir.data(); gp_sax.y_0.count   = N;
        gp_sax.dst_0.data = q.data();   gp_sax.dst_0.count = N;
        pd_saxpby::main_0(&vi, nullptr, &gp_sax);

        // r := 1·r + (−α)·qprod
        spy.alpha_0 = 1.0f; spy.beta_0 = float(-alpha);
        gp_sax.x_0.data = r.data();     gp_sax.x_0.count   = N;
        gp_sax.y_0.data = qprod.data(); gp_sax.y_0.count   = N;
        gp_sax.dst_0.data = r.data();   gp_sax.dst_0.count = N;
        pd_saxpby::main_0(&vi, nullptr, &gp_sax);

        const double delta_old = delta_new;
        delta_new = kernelDot(r.data(), r.data());
        if (delta_new < CG_TOL * CG_TOL * delta_0) { ++cg_iters; break; }

        const double beta = delta_new / delta_old;

        // p_dir := 1·r + β·p_dir
        spy.alpha_0 = 1.0f; spy.beta_0 = float(beta);
        gp_sax.x_0.data = r.data();     gp_sax.x_0.count   = N;
        gp_sax.y_0.data = p_dir.data(); gp_sax.y_0.count   = N;
        gp_sax.dst_0.data = p_dir.data(); gp_sax.dst_0.count = N;
        pd_saxpby::main_0(&vi, nullptr, &gp_sax);
    }

    // ---- Compare against hand-derived answer ------------------------------
    const float expected[N] = {
        0.2f, 0.0f, 0.0f,
        1.6f, 0.0f, 0.0f,
    };
    int   fails        = 0;
    float max_abs_diff = 0.0f;
    for (uint32_t i = 0; i < N; ++i) {
        const float d = std::fabs(q[i] - expected[i]);
        if (d > max_abs_diff) max_abs_diff = d;
        if (d > 1e-5f) {
            if (fails < 5) {
                std::fprintf(stderr,
                    "pd_step_demo mismatch at i=%u: got %g, expected %g (diff %g)\n",
                    i, q[i], expected[i], d);
            }
            ++fails;
        }
    }

    const auto t1 = std::chrono::steady_clock::now();
    const double elapsed = std::chrono::duration<double>(t1 - t0).count();
    if (elapsed > kBudgetSeconds) {
        std::fprintf(stderr, "pd_step_demo: TIMEOUT — %.3fs > %.1fs\n",
                     elapsed, kBudgetSeconds);
        return 2;
    }
    if (fails == 0) {
        std::printf(
            "pd_step_demo: %u/%u dims OK (cg_iters=%d, max_abs_diff=%g, "
            "q=[%g,%g,%g, %g,%g,%g], %.1fms)\n",
            N, N, cg_iters, max_abs_diff,
            q[0], q[1], q[2], q[3], q[4], q[5],
            elapsed * 1000.0);
        return 0;
    }
    std::fprintf(stderr,
        "pd_step_demo: %d FAIL (cg_iters=%d, q=[%g,%g,%g, %g,%g,%g])\n",
        fails, cg_iters, q[0], q[1], q[2], q[3], q[4], q[5]);
    return 1;
}
