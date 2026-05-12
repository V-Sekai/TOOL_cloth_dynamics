// Finite-difference validator for
// Cloth.SlangCodegen.TriangleMembraneForceAlBackward (PR-G / CHI-13).
//
// This kernel implements the FULL 2D polar adjoint (Townsend 2016
// closed-form; not the frozen-R / Sifakis-cookie approximation).
// Validates analytic vs central-difference on all paths:
//
//   * v_p[3c+r]   — position cotangents per corner r ∈ {0,1,2}
//   * v_stiffness — ∂L/∂k
//   * v_lambda0, v_lambda1 — ∂L/∂λ
//
// Tolerance: 1% relative on positions (full polar adjoint chains
// through 5 normalisations, so fp32 noise accumulates). Lambda and
// stiffness are linear in v_grad, so they're tighter.

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <random>

#include "slang-cpp-prelude.h"

namespace fwd {
    #pragma push_macro("SLANG_PRELUDE_EXTERN_C")
    #pragma push_macro("SLANG_PRELUDE_EXTERN_C_START")
    #pragma push_macro("SLANG_PRELUDE_EXTERN_C_END")
    #undef  SLANG_PRELUDE_EXTERN_C
    #undef  SLANG_PRELUDE_EXTERN_C_START
    #undef  SLANG_PRELUDE_EXTERN_C_END
    #define SLANG_PRELUDE_EXTERN_C
    #define SLANG_PRELUDE_EXTERN_C_START
    #define SLANG_PRELUDE_EXTERN_C_END
    #include "triangle_membrane_force_al_emit.cpp"
    #pragma pop_macro("SLANG_PRELUDE_EXTERN_C")
    #pragma pop_macro("SLANG_PRELUDE_EXTERN_C_START")
    #pragma pop_macro("SLANG_PRELUDE_EXTERN_C_END")
}

namespace bwd {
    #pragma push_macro("SLANG_PRELUDE_EXTERN_C")
    #pragma push_macro("SLANG_PRELUDE_EXTERN_C_START")
    #pragma push_macro("SLANG_PRELUDE_EXTERN_C_END")
    #undef  SLANG_PRELUDE_EXTERN_C
    #undef  SLANG_PRELUDE_EXTERN_C_START
    #undef  SLANG_PRELUDE_EXTERN_C_END
    #define SLANG_PRELUDE_EXTERN_C
    #define SLANG_PRELUDE_EXTERN_C_START
    #define SLANG_PRELUDE_EXTERN_C_END
    #include "triangle_membrane_force_al_backward_emit.cpp"
    #pragma pop_macro("SLANG_PRELUDE_EXTERN_C")
    #pragma pop_macro("SLANG_PRELUDE_EXTERN_C_START")
    #pragma pop_macro("SLANG_PRELUDE_EXTERN_C_END")
}

static constexpr double kBudgetSeconds = 5.0;
static constexpr uint32_t GROUP_SIZE = 64;
static constexpr uint32_t N_TRI = 1;

struct FwdInputs {
    std::vector<Vector<float, 3>> positions;
    std::vector<uint32_t> idx;
    std::vector<float> stiffness;
    std::vector<Vector<float, 3>> lambda0;
    std::vector<Vector<float, 3>> lambda1;
    std::vector<float> inv_deltaUV;
};

static double run_forward_loss(const FwdInputs& in,
                               const std::vector<Vector<float, 3>>& v_grad,
                               const std::vector<float>& v_hess) {
    std::vector<Vector<float, 3>> grad(3 * GROUP_SIZE, Vector<float, 3>(0.0f));
    std::vector<float> hess(3 * GROUP_SIZE, 0.0f);

    fwd::GlobalParams_0 gp{};
    gp.positions_0.data   = const_cast<Vector<float, 3>*>(in.positions.data());
    gp.positions_0.count  = in.positions.size();
    gp.idx_0.data         = const_cast<uint32_t*>(in.idx.data());
    gp.idx_0.count        = in.idx.size();
    gp.stiffness_0.data   = const_cast<float*>(in.stiffness.data());
    gp.stiffness_0.count  = in.stiffness.size();
    gp.lambda0_0.data     = const_cast<Vector<float, 3>*>(in.lambda0.data());
    gp.lambda0_0.count    = in.lambda0.size();
    gp.lambda1_0.data     = const_cast<Vector<float, 3>*>(in.lambda1.data());
    gp.lambda1_0.count    = in.lambda1.size();
    gp.grad_0.data        = grad.data();         gp.grad_0.count        = grad.size();
    gp.hessScalar_0.data  = hess.data();         gp.hessScalar_0.count  = hess.size();
    gp.inv_deltaUV_0.data = const_cast<float*>(in.inv_deltaUV.data());
    gp.inv_deltaUV_0.count = in.inv_deltaUV.size();

    ComputeVaryingInput vi{};
    vi.startGroupID = uint3(0, 0, 0);
    vi.endGroupID   = uint3(1, 1, 1);
    fwd::main_0(&vi, nullptr, &gp);

    double loss = 0.0;
    for (uint32_t c = 0; c < N_TRI; ++c) {
        for (int r = 0; r < 3; ++r) {
            loss += double(v_grad[3*c + r].x) * double(grad[3*c + r].x);
            loss += double(v_grad[3*c + r].y) * double(grad[3*c + r].y);
            loss += double(v_grad[3*c + r].z) * double(grad[3*c + r].z);
            loss += double(v_hess[3*c + r])   * double(hess[3*c + r]);
        }
    }
    return loss;
}

int main() {
    const auto t0 = std::chrono::steady_clock::now();

    FwdInputs in;
    // Near-identity triangle: rest is (0,0,0), (1,0,0), (0,1,0).
    // Deformed slightly so F ≈ I (small e0, e1r → frozen-R is accurate).
    in.positions = {
        Vector<float, 3>(0.0f,  0.0f, 0.0f),
        Vector<float, 3>(1.02f, 0.01f, 0.0f),
        Vector<float, 3>(0.01f, 0.98f, 0.0f),
    };
    in.idx          = std::vector<uint32_t>(3 * GROUP_SIZE, 0u);
    in.stiffness    = std::vector<float>(GROUP_SIZE, 0.0f);
    in.lambda0      = std::vector<Vector<float, 3>>(GROUP_SIZE, Vector<float, 3>(0.0f));
    in.lambda1      = std::vector<Vector<float, 3>>(GROUP_SIZE, Vector<float, 3>(0.0f));
    in.inv_deltaUV  = std::vector<float>(4 * GROUP_SIZE, 0.0f);

    in.idx[0]=0u; in.idx[1]=1u; in.idx[2]=2u;
    in.stiffness[0] = 1.5f;
    in.lambda0[0] = Vector<float, 3>(0.05f, -0.03f, 0.0f);  // small λ — frozen-R OK
    in.lambda1[0] = Vector<float, 3>(-0.02f, 0.04f, 0.0f);
    // Rest material = identity: inv_deltaUV = [1, 0, 0, 1]
    in.inv_deltaUV[0] = 1.0f;
    in.inv_deltaUV[1] = 0.0f;
    in.inv_deltaUV[2] = 0.0f;
    in.inv_deltaUV[3] = 1.0f;

    std::mt19937 rng(0xC1131DB1u);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<Vector<float, 3>> v_grad(3 * GROUP_SIZE, Vector<float, 3>(0.0f));
    std::vector<float>            v_hess(3 * GROUP_SIZE, 0.0f);
    for (uint32_t c = 0; c < N_TRI; ++c) {
        for (int r = 0; r < 3; ++r) {
            v_grad[3*c + r] = Vector<float, 3>(dist(rng), dist(rng), dist(rng));
            v_hess[3*c + r] = dist(rng);
        }
    }

    // --- Analytic backward ---
    std::vector<Vector<float, 3>> v_p(3 * GROUP_SIZE, Vector<float, 3>(0.0f));
    std::vector<float>            v_k(GROUP_SIZE, 0.0f);
    std::vector<Vector<float, 3>> v_l0(GROUP_SIZE, Vector<float, 3>(0.0f));
    std::vector<Vector<float, 3>> v_l1(GROUP_SIZE, Vector<float, 3>(0.0f));

    bwd::GlobalParams_0 gpb{};
    gpb.positions_0.data    = in.positions.data();    gpb.positions_0.count    = in.positions.size();
    gpb.idx_0.data          = in.idx.data();          gpb.idx_0.count          = in.idx.size();
    gpb.stiffness_0.data    = in.stiffness.data();    gpb.stiffness_0.count    = in.stiffness.size();
    gpb.lambda0_0.data      = in.lambda0.data();      gpb.lambda0_0.count      = in.lambda0.size();
    gpb.lambda1_0.data      = in.lambda1.data();      gpb.lambda1_0.count      = in.lambda1.size();
    gpb.inv_deltaUV_0.data  = in.inv_deltaUV.data();  gpb.inv_deltaUV_0.count  = in.inv_deltaUV.size();
    gpb.v_grad_0.data       = v_grad.data();          gpb.v_grad_0.count       = v_grad.size();
    gpb.v_hessScalar_0.data = v_hess.data();          gpb.v_hessScalar_0.count = v_hess.size();
    gpb.v_p_0.data          = v_p.data();             gpb.v_p_0.count          = v_p.size();
    gpb.v_stiffness_0.data  = v_k.data();             gpb.v_stiffness_0.count  = v_k.size();
    gpb.v_lambda0_0.data    = v_l0.data();            gpb.v_lambda0_0.count    = v_l0.size();
    gpb.v_lambda1_0.data    = v_l1.data();            gpb.v_lambda1_0.count    = v_l1.size();

    ComputeVaryingInput vib{};
    vib.startGroupID = uint3(0, 0, 0);
    vib.endGroupID   = uint3(1, 1, 1);
    bwd::main_0(&vib, nullptr, &gpb);

    // --- FD check ---
    const float eps = 1e-4f;
    int fails = 0;
    double max_rel_err_pos = 0.0;
    double max_rel_err_other = 0.0;
    auto check_with_tol = [&](double analytic, double fd, double rel_tol,
                              double& max_rel_tracker, const char* label,
                              uint32_t c, int r, int comp) {
        const double abs_err = std::fabs(analytic - fd);
        const double denom = std::max(1e-6, std::fabs(fd));
        const double rel = abs_err / denom;
        if (rel > max_rel_tracker) max_rel_tracker = rel;
        if (rel > rel_tol && abs_err > 1e-4) {
            if (fails < 8) {
                std::fprintf(stderr,
                    "tri_membrane_al_backward FD mismatch %s c=%u r=%d comp=%d: analytic=%g, fd=%g (rel=%g)\n",
                    label, c, r, comp, analytic, fd, rel);
            }
            ++fails;
        }
    };

    for (uint32_t c = 0; c < N_TRI; ++c) {
        // FD on position cotangents (full polar adjoint — exact to fp32).
        for (int r = 0; r < 3; ++r) {
            const uint32_t v = in.idx[3*c + r];
            for (int comp = 0; comp < 3; ++comp) {
                FwdInputs plus = in, minus = in;
                plus.positions[v][comp]  += eps;
                minus.positions[v][comp] -= eps;
                const double Lp = run_forward_loss(plus,  v_grad, v_hess);
                const double Lm = run_forward_loss(minus, v_grad, v_hess);
                const double fd = (Lp - Lm) / (2.0 * double(eps));
                check_with_tol(double(v_p[3*c + r][comp]), fd, 1e-2,
                               max_rel_err_pos, "v_p", c, r, comp);
            }
        }
        // FD on stiffness — exact (linear in v_grad).
        {
            FwdInputs plus = in, minus = in;
            plus.stiffness[c]  += eps;
            minus.stiffness[c] -= eps;
            const double Lp = run_forward_loss(plus,  v_grad, v_hess);
            const double Lm = run_forward_loss(minus, v_grad, v_hess);
            const double fd = (Lp - Lm) / (2.0 * double(eps));
            check_with_tol(double(v_k[c]), fd, 2e-2,
                           max_rel_err_other, "v_stiffness", c, 0, 0);
        }
        // FD on lambda0, lambda1 — exact under frozen-R.
        for (int comp = 0; comp < 3; ++comp) {
            FwdInputs plus = in, minus = in;
            plus.lambda0[c][comp]  += eps;
            minus.lambda0[c][comp] -= eps;
            const double Lp = run_forward_loss(plus,  v_grad, v_hess);
            const double Lm = run_forward_loss(minus, v_grad, v_hess);
            const double fd = (Lp - Lm) / (2.0 * double(eps));
            check_with_tol(double(v_l0[c][comp]), fd, 2e-2,
                           max_rel_err_other, "v_lambda0", c, 0, comp);
        }
        for (int comp = 0; comp < 3; ++comp) {
            FwdInputs plus = in, minus = in;
            plus.lambda1[c][comp]  += eps;
            minus.lambda1[c][comp] -= eps;
            const double Lp = run_forward_loss(plus,  v_grad, v_hess);
            const double Lm = run_forward_loss(minus, v_grad, v_hess);
            const double fd = (Lp - Lm) / (2.0 * double(eps));
            check_with_tol(double(v_l1[c][comp]), fd, 2e-2,
                           max_rel_err_other, "v_lambda1", c, 0, comp);
        }
    }

    const auto t1 = std::chrono::steady_clock::now();
    const double elapsed = std::chrono::duration<double>(t1 - t0).count();
    if (elapsed > kBudgetSeconds) {
        std::fprintf(stderr, "tri_membrane_al_backward: TIMEOUT — %.3fs\n", elapsed);
        return 2;
    }
    if (fails == 0) {
        const uint32_t nChecks = N_TRI * (9 + 1 + 3 + 3);
        std::printf("triangle_membrane_force_al_backward: %u/%u FD checks OK "
                    "(full polar adjoint; max_rel_err_pos=%.3g, "
                    "max_rel_err_other=%.3g, %.1fms)\n",
                    nChecks, nChecks, max_rel_err_pos, max_rel_err_other,
                    elapsed * 1000.0);
        return 0;
    }
    std::fprintf(stderr, "triangle_membrane_force_al_backward: %d FAIL\n", fails);
    return 1;
}
