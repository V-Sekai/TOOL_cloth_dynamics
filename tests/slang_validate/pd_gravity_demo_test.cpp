// Hanging-mass demo: same 2-vert 1-spring 1-pin system as
// pd_gravity_demo, but with gravity applied to both vertices. The
// pin is soft (sqrtW_att = 1) so it slides under the load — the
// equilibrium is *not* the pin's fixed point.
//
// System:
//   v0  pinned at (0, 0, 0)     sqrtW_att    = 1
//   v1  spring to v0             sqrtW_spring = 1,  l0 = 1
//   x(0) = [(0,0,0), (0,-1.5,0)] (spring along -y, slightly stretched)
//   v(0) = 0,  mass[v] = 1
//   gravity g = (0, -1, 0) m/s²  (applied to both verts)
//
// Predictor (semi-implicit Euler with external force):
//   s_n = x_n + dt · v_n + dt² · g
//
// Hand-derived equilibrium (energy minimum of pin + spring + gravity):
//
//   v1 (no pin): spring force = gravity
//     k_spring · (L − l0) = m·g  →  L = l0 + m·g/k = 1 + 1·1/1 = 2
//
//   v0 (pin):    pin force + spring force + gravity = 0
//     k_att · (0 − y0) + (−k_spring · (L − l0)) + (−m·g) = 0
//     −y0 = m·g + k_spring·(L − 1) = 1 + 1 = 2  →  y0 = −2
//
//   ⇒  v0 settles at (0, −2, 0),  v1 at (0, y0 − L, 0) = (0, −4, 0).
//
// Final-state assertions (loose tolerance because soft-pin equilibrium
// is reached only in the limit):
//   |v0 − (0,−2,0)| < 0.05
//   |v1 − (0,−4,0)| < 0.05
//   kinetic_energy < 0.01

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <vector>

#include "slang-cpp-prelude.h"

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

static constexpr double kBudgetSeconds = 15.0;
static constexpr uint32_t V          = 2;
static constexpr uint32_t N          = 3 * V;
static constexpr uint32_t GROUP_SIZE = 64;
static constexpr uint32_t MAX_CG     = 30;
static constexpr double   CG_TOL     = 1e-6;

// One PD step: takes predictor s, produces new positions q.
// All six kernels are dispatched here.  CG inner loop matches the
// pd_step_demo pattern verbatim.
static void pdStep(
    const std::vector<Vector<float, 3>>& positions_for_projection,
    const std::vector<float>& s,
    std::vector<float>& q_out,
    /* spring kernel binding */     pd_spring::GlobalParams_0& gp_spring,
    /* attachment kernel binding */ pd_attach::GlobalParams_0&  gp_attach,
    std::vector<Vector<float, 3>>& sp_out,
    std::vector<Vector<float, 3>>& at_out,
    /* assemble_b binding */        pd_assemble::AssembleBParams_0& abp,
    pd_assemble::GlobalParams_0& gp_ab,
    std::vector<float>& b,
    std::vector<float>& projections,
    /* spmv binding */              pd_spmv::GlobalParams_0& gp_sp,
    /* saxpby binding */            pd_saxpby::SaxpbyParams_0& spy,
    pd_saxpby::GlobalParams_0& gp_sax,
    /* dot binding */               pd_dot::GlobalParams_0& gp_dot,
    float* dot_dst,
    ComputeVaryingInput& vi)
{
    (void)positions_for_projection;
    // ---- Local step: dispatch constraint kernels (they read positions
    // via the bindings already set up by the caller) ----------------------
    pd_spring::main_0(&vi, nullptr, &gp_spring);
    pd_attach::main_0(&vi, nullptr, &gp_attach);

    // Stack constraint outputs into projections buffer.
    projections[0] = sp_out[0][0]; projections[1] = sp_out[0][1]; projections[2] = sp_out[0][2];
    projections[3] = at_out[0][0]; projections[4] = at_out[0][1]; projections[5] = at_out[0][2];

    // ---- Wire s, b through assemble_b binding -----------------------------
    gp_ab.s_0.data = const_cast<float*>(s.data()); gp_ab.s_0.count = s.size();
    gp_ab.b_0.data = b.data();                     gp_ab.b_0.count = b.size();
    pd_assemble::main_0(&vi, nullptr, &gp_ab);

    // ---- CG inner loop on the fixed A from pd_step_demo (same system) ----
    std::vector<float> r(N), p_dir(N), qprod(N);
    for (uint32_t i = 0; i < N; ++i) {
        q_out[i] = 0.0f;          // initial guess
        r[i]     = b[i];          // r₀ = b
        p_dir[i] = b[i];
    }

    auto kernelDot = [&](float* aa, float* bb) -> double {
        gp_dot.a_0.data = aa; gp_dot.a_0.count = N;
        gp_dot.b_0.data = bb; gp_dot.b_0.count = N;
        pd_dot::main_0(&vi, nullptr, &gp_dot);
        return double(dot_dst[0]) + double(dot_dst[1]);
    };

    double delta_new = kernelDot(r.data(), r.data());
    const double delta_0 = delta_new;

    for (uint32_t k = 0; k < MAX_CG; ++k) {
        gp_sp.x_0.data = p_dir.data(); gp_sp.x_0.count = N;
        gp_sp.y_0.data = qprod.data(); gp_sp.y_0.count = N;
        pd_spmv::main_0(&vi, nullptr, &gp_sp);

        const double pq    = kernelDot(p_dir.data(), qprod.data());
        const double alpha = delta_new / pq;

        spy.alpha_0 = 1.0f; spy.beta_0 = float(alpha);
        gp_sax.x_0.data = q_out.data();   gp_sax.x_0.count   = N;
        gp_sax.y_0.data = p_dir.data();   gp_sax.y_0.count   = N;
        gp_sax.dst_0.data = q_out.data(); gp_sax.dst_0.count = N;
        pd_saxpby::main_0(&vi, nullptr, &gp_sax);

        spy.alpha_0 = 1.0f; spy.beta_0 = float(-alpha);
        gp_sax.x_0.data = r.data();     gp_sax.x_0.count   = N;
        gp_sax.y_0.data = qprod.data(); gp_sax.y_0.count   = N;
        gp_sax.dst_0.data = r.data();   gp_sax.dst_0.count = N;
        pd_saxpby::main_0(&vi, nullptr, &gp_sax);

        const double delta_old = delta_new;
        delta_new = kernelDot(r.data(), r.data());
        if (delta_new < CG_TOL * CG_TOL * delta_0) break;

        const double beta = delta_new / delta_old;
        spy.alpha_0 = 1.0f; spy.beta_0 = float(beta);
        gp_sax.x_0.data = r.data();       gp_sax.x_0.count   = N;
        gp_sax.y_0.data = p_dir.data();   gp_sax.y_0.count   = N;
        gp_sax.dst_0.data = p_dir.data(); gp_sax.dst_0.count = N;
        pd_saxpby::main_0(&vi, nullptr, &gp_sax);
    }
}

int main() {
    const auto t0 = std::chrono::steady_clock::now();

    // ---- Simulation parameters --------------------------------------------
    constexpr float dt          = 0.1f;
    constexpr float damping     = 0.95f;
    constexpr int   N_STEPS     = 500;
    constexpr float gravity_y   = -1.0f;       // m/s² in y

    // ---- Vertex state -----------------------------------------------------
    std::vector<Vector<float, 3>> positions(GROUP_SIZE, Vector<float, 3>(0.0f));
    positions[0] = Vector<float, 3>( 0.0f,  0.0f, 0.0f);
    positions[1] = Vector<float, 3>( 0.0f, -1.5f, 0.0f);   // spring along −y, slightly stretched

    std::vector<float> velocity(N, 0.0f);
    std::vector<float> s(N, 0.0f);
    std::vector<float> q(N, 0.0f);
    std::vector<float> mass(V, 1.0f);
    std::vector<float> b(N, 0.0f);
    std::vector<float> projections(3 * 2);

    // ---- Constraint kernel inputs (identical to pd_step_demo) ------------
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

    std::vector<Vector<float, 3>> at_fixed(GROUP_SIZE, Vector<float, 3>(0.0f));
    std::vector<float>            at_sqrtW(GROUP_SIZE, 0.0f);
    at_fixed[0] = Vector<float, 3>(0.0f, 0.0f, 0.0f);
    at_sqrtW[0] = 1.0f;
    std::vector<Vector<float, 3>> at_out(GROUP_SIZE, Vector<float, 3>(0.0f));

    pd_attach::GlobalParams_0 gp_attach{};
    gp_attach.projected_0.data  = at_out.data();    gp_attach.projected_0.count  = at_out.size();
    gp_attach.fixedPos_0.data   = at_fixed.data();  gp_attach.fixedPos_0.count   = at_fixed.size();
    gp_attach.sqrtWeight_0.data = at_sqrtW.data();  gp_attach.sqrtWeight_0.count = at_sqrtW.size();

    // ---- assemble_b incidence (identical to pd_step_demo) ----------------
    std::vector<uint32_t> ctxStart  = {0u, 2u, 3u};
    std::vector<uint32_t> ctxSlot   = {0u, 1u, 0u};
    std::vector<float>    ctxWeight = {+1.0f, +1.0f, -1.0f};

    // Rescale assemble_b's mass for the new dt: A and b carry an h²
    // factor when written in the (M + h²·S^T·S)·q = M·s + h²·S^T·p
    // form. Equivalently, divide M by h² in the un-scaled form. We
    // pre-scale mass by 1/dt² and leave everything else.
    const float dt2 = dt * dt;
    std::vector<float> mass_scaled(V);
    for (uint32_t i = 0; i < V; ++i) mass_scaled[i] = mass[i] / dt2;

    pd_assemble::AssembleBParams_0 abp{V};
    pd_assemble::GlobalParams_0    gp_ab{};
    gp_ab.params_0           = &abp;
    gp_ab.mass_0.data        = mass_scaled.data();  gp_ab.mass_0.count        = V;
    gp_ab.projections_0.data = projections.data();  gp_ab.projections_0.count = projections.size();
    gp_ab.ctxStart_0.data    = ctxStart.data();     gp_ab.ctxStart_0.count    = ctxStart.size();
    gp_ab.ctxSlot_0.data     = ctxSlot.data();      gp_ab.ctxSlot_0.count     = ctxSlot.size();
    gp_ab.ctxWeight_0.data   = ctxWeight.data();    gp_ab.ctxWeight_0.count   = ctxWeight.size();

    // ---- A matrix CSR (same shape as pd_step_demo but with M/h² diagonal) -
    // The diagonal entries become M/h² + S^T·S contributions.
    const float diag_M_over_h2 = 1.0f / dt2;
    std::vector<int32_t> A_rowPtr = {0, 2, 4, 6, 8, 10, 12};
    std::vector<int32_t> A_colIdx = {0, 3,
                                     1, 4,
                                     2, 5,
                                     0, 3,
                                     1, 4,
                                     2, 5};
    std::vector<float>   A_vals   = {
        diag_M_over_h2 + 2.0f, -1.0f,    // v0x: M/h² + (spring + pin) = M/h² + 2
        diag_M_over_h2 + 2.0f, -1.0f,
        diag_M_over_h2 + 2.0f, -1.0f,
        -1.0f, diag_M_over_h2 + 1.0f,    // v1x: M/h² + spring = M/h² + 1
        -1.0f, diag_M_over_h2 + 1.0f,
        -1.0f, diag_M_over_h2 + 1.0f,
    };

    pd_spmv::SpmvParams_0  spp{N};
    pd_spmv::GlobalParams_0 gp_sp{};
    gp_sp.params_0      = &spp;
    gp_sp.rowPtr_0.data = A_rowPtr.data(); gp_sp.rowPtr_0.count = A_rowPtr.size();
    gp_sp.colIdx_0.data = A_colIdx.data(); gp_sp.colIdx_0.count = A_colIdx.size();
    gp_sp.values_0.data = A_vals.data();   gp_sp.values_0.count = A_vals.size();

    pd_saxpby::SaxpbyParams_0 spy{N, 0.0f, 0.0f};
    pd_saxpby::GlobalParams_0 gp_sax{};
    gp_sax.params_0 = &spy;

    pd_dot::DotReduceSerialParams_0 dp{N};
    float dot_dst[2] = {0.0f, 0.0f};
    pd_dot::GlobalParams_0 gp_dot{};
    gp_dot.params_0   = &dp;
    gp_dot.dst_0.data = dot_dst; gp_dot.dst_0.count = 2;

    ComputeVaryingInput vi{};
    vi.startGroupID = uint3(0, 0, 0);
    vi.endGroupID   = uint3(1, 1, 1);

    // ---- Time integration loop --------------------------------------------
    float spring_len_initial = 0.0f, spring_len_final = 0.0f;
    const float dt2_gy = dt * dt * gravity_y;    // = dt² · (g_y/m) since m = 1
    for (int step = 0; step < N_STEPS; ++step) {
        // s = x + dt·v + dt²·g     (semi-implicit Euler predictor with external force)
        for (uint32_t i = 0; i < V; ++i) {
            s[3*i + 0] = positions[i][0] + dt * velocity[3*i + 0];
            s[3*i + 1] = positions[i][1] + dt * velocity[3*i + 1] + dt2_gy;
            s[3*i + 2] = positions[i][2] + dt * velocity[3*i + 2];
        }

        pdStep(positions, s, q,
               gp_spring, gp_attach, sp_out, at_out,
               abp, gp_ab, b, projections,
               gp_sp, spy, gp_sax,
               gp_dot, dot_dst, vi);

        // v = damp · (q − x) / dt; x := q
        for (uint32_t i = 0; i < V; ++i) {
            for (int d = 0; d < 3; ++d) {
                const float dx = q[3*i + d] - positions[i][d];
                velocity[3*i + d] = damping * (dx / dt);
                positions[i][d]   = q[3*i + d];
            }
        }

        if (step == 0) {
            const float dx = positions[1][0] - positions[0][0];
            const float dy = positions[1][1] - positions[0][1];
            const float dz = positions[1][2] - positions[0][2];
            spring_len_initial = std::sqrt(dx*dx + dy*dy + dz*dz);
        }
    }

    // Final spring length.
    {
        const float dx = positions[1][0] - positions[0][0];
        const float dy = positions[1][1] - positions[0][1];
        const float dz = positions[1][2] - positions[0][2];
        spring_len_final = std::sqrt(dx*dx + dy*dy + dz*dz);
    }

    // ---- Final-state assertions -------------------------------------------
    const float v0_dx    = positions[0][0];
    const float v0_dy    = positions[0][1] - (-2.0f);
    const float v0_dz    = positions[0][2];
    const float v0_dist  = std::sqrt(v0_dx*v0_dx + v0_dy*v0_dy + v0_dz*v0_dz);

    const float v1_dx    = positions[1][0];
    const float v1_dy    = positions[1][1] - (-4.0f);
    const float v1_dz    = positions[1][2];
    const float v1_dist  = std::sqrt(v1_dx*v1_dx + v1_dy*v1_dy + v1_dz*v1_dz);

    double kinetic = 0.0;
    for (uint32_t i = 0; i < N; ++i) kinetic += double(velocity[i]) * double(velocity[i]);
    kinetic *= 0.5;

    int fails = 0;
    if (v0_dist > 0.05f)  { std::fprintf(stderr, "v0 − (0,−2,0): |.|=%g > 0.05\n", v0_dist); ++fails; }
    if (v1_dist > 0.05f)  { std::fprintf(stderr, "v1 − (0,−4,0): |.|=%g > 0.05\n", v1_dist); ++fails; }
    if (kinetic > 0.01)   { std::fprintf(stderr, "residual kinetic: %g > 0.01\n", kinetic); ++fails; }

    const auto t1 = std::chrono::steady_clock::now();
    const double elapsed = std::chrono::duration<double>(t1 - t0).count();
    if (elapsed > kBudgetSeconds) {
        std::fprintf(stderr, "pd_gravity_demo: TIMEOUT — %.3fs > %.1fs\n",
                     elapsed, kBudgetSeconds);
        return 2;
    }
    if (fails == 0) {
        std::printf(
            "pd_gravity_demo: %d steps OK (spring %.4f → %.4f, "
            "v0=(%.4f,%.4f,%.4f), v1=(%.4f,%.4f,%.4f), kin=%g, %.1fms)\n",
            N_STEPS, spring_len_initial, spring_len_final,
            positions[0][0], positions[0][1], positions[0][2],
            positions[1][0], positions[1][1], positions[1][2],
            kinetic, elapsed * 1000.0);
        return 0;
    }
    std::fprintf(stderr,
        "pd_gravity_demo: FAIL after %d steps (spring %.4f → %.4f, "
        "v0=[%g,%g,%g], v1=[%g,%g,%g], kin=%g)\n",
        N_STEPS, spring_len_initial, spring_len_final,
        positions[0][0], positions[0][1], positions[0][2],
        positions[1][0], positions[1][1], positions[1][2],
        kinetic);
    return 1;
}
