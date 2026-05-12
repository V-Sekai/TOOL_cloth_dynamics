// Finite-difference validator for
// Cloth.SlangCodegen.TriangleBendingForceAlBackward (PR-G / CHI-13).
//
// Loss for adjoint check (sum across all 4 corners r and all output
// channels):
//
//   L = Σ_r v_grad[4c+r] · grad[4c+r] + Σ_r v_hess[4c+r] · hess[4c+r]
//
// Validates analytic vs FD on each cotangent output (v_p per corner,
// v_lambda, v_stiffness, v_nTarget). Skips the degenerate constraint
// (n_target=0) — the kernel correctly zeros all outputs there.

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
    #include "triangle_bending_force_al_emit.cpp"
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
    #include "triangle_bending_force_al_backward_emit.cpp"
    #pragma pop_macro("SLANG_PRELUDE_EXTERN_C")
    #pragma pop_macro("SLANG_PRELUDE_EXTERN_C_START")
    #pragma pop_macro("SLANG_PRELUDE_EXTERN_C_END")
}

static constexpr double kBudgetSeconds = 5.0;
static constexpr uint32_t GROUP_SIZE = 64;
static constexpr uint32_t N_BEND = 1;

struct FwdInputs {
    std::vector<Vector<float, 3>> positions;
    std::vector<uint32_t> idx;
    std::vector<float> weight;
    std::vector<float> nTarget;
    std::vector<float> stiffness;
    std::vector<Vector<float, 3>> lambda;
};

static double run_forward_loss(const FwdInputs& in,
                               const std::vector<Vector<float, 3>>& v_grad,
                               const std::vector<float>& v_hess) {
    std::vector<Vector<float, 3>> grad(4 * GROUP_SIZE, Vector<float, 3>(0.0f));
    std::vector<float> hess(4 * GROUP_SIZE, 0.0f);

    fwd::GlobalParams_0 gp{};
    gp.positions_0.data  = const_cast<Vector<float, 3>*>(in.positions.data());
    gp.positions_0.count = in.positions.size();
    gp.idx_0.data        = const_cast<uint32_t*>(in.idx.data());
    gp.idx_0.count       = in.idx.size();
    gp.weight_0.data     = const_cast<float*>(in.weight.data());
    gp.weight_0.count    = in.weight.size();
    gp.nTarget_0.data    = const_cast<float*>(in.nTarget.data());
    gp.nTarget_0.count   = in.nTarget.size();
    gp.stiffness_0.data  = const_cast<float*>(in.stiffness.data());
    gp.stiffness_0.count = in.stiffness.size();
    gp.lambda_0.data     = const_cast<Vector<float, 3>*>(in.lambda.data());
    gp.lambda_0.count    = in.lambda.size();
    gp.grad_0.data       = grad.data();  gp.grad_0.count       = grad.size();
    gp.hessScalar_0.data = hess.data();  gp.hessScalar_0.count = hess.size();

    ComputeVaryingInput vi{};
    vi.startGroupID = uint3(0, 0, 0);
    vi.endGroupID   = uint3(1, 1, 1);
    fwd::main_0(&vi, nullptr, &gp);

    double loss = 0.0;
    for (uint32_t c = 0; c < N_BEND; ++c) {
        for (int r = 0; r < 4; ++r) {
            loss += double(v_grad[4*c + r].x) * double(grad[4*c + r].x);
            loss += double(v_grad[4*c + r].y) * double(grad[4*c + r].y);
            loss += double(v_grad[4*c + r].z) * double(grad[4*c + r].z);
            loss += double(v_hess[4*c + r])   * double(hess[4*c + r]);
        }
    }
    return loss;
}

int main() {
    const auto t0 = std::chrono::steady_clock::now();

    FwdInputs in;
    // Non-degenerate stencil: 4 vertices forming a non-symmetric
    // configuration so all gradients are nonzero.
    in.positions = {
        Vector<float, 3>(0.1f, 0.2f, 0.0f),
        Vector<float, 3>(1.1f, 0.0f, 0.1f),
        Vector<float, 3>(0.0f, 1.0f, 0.3f),
        Vector<float, 3>(0.5f, 0.5f, 1.7f),
    };
    in.idx       = std::vector<uint32_t>(4 * GROUP_SIZE, 0u);
    in.weight    = std::vector<float>(4 * GROUP_SIZE, 0.0f);
    in.nTarget   = std::vector<float>(GROUP_SIZE, 0.0f);
    in.stiffness = std::vector<float>(GROUP_SIZE, 0.0f);
    in.lambda    = std::vector<Vector<float, 3>>(GROUP_SIZE, Vector<float, 3>(0.0f));

    in.idx[0]=0u; in.idx[1]=1u; in.idx[2]=2u; in.idx[3]=3u;
    in.weight[0]=0.8f; in.weight[1]=-0.3f; in.weight[2]=0.7f; in.weight[3]=-1.2f;
    in.nTarget[0] = 1.4f;
    in.stiffness[0] = 2.5f;
    in.lambda[0] = Vector<float, 3>(0.3f, -0.7f, 1.1f);

    // Random cotangents
    std::mt19937 rng(0xC1B0E13u);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<Vector<float, 3>> v_grad(4 * GROUP_SIZE, Vector<float, 3>(0.0f));
    std::vector<float>            v_hess(4 * GROUP_SIZE, 0.0f);
    for (uint32_t c = 0; c < N_BEND; ++c) {
        for (int r = 0; r < 4; ++r) {
            v_grad[4*c + r] = Vector<float, 3>(dist(rng), dist(rng), dist(rng));
            v_hess[4*c + r] = dist(rng);
        }
    }

    // --- Analytic backward ---
    std::vector<Vector<float, 3>> v_p(4 * GROUP_SIZE, Vector<float, 3>(0.0f));
    std::vector<float>            v_n(GROUP_SIZE, 0.0f);
    std::vector<float>            v_k(GROUP_SIZE, 0.0f);
    std::vector<Vector<float, 3>> v_l(GROUP_SIZE, Vector<float, 3>(0.0f));

    bwd::GlobalParams_0 gpb{};
    gpb.positions_0.data    = in.positions.data();    gpb.positions_0.count    = in.positions.size();
    gpb.idx_0.data          = in.idx.data();          gpb.idx_0.count          = in.idx.size();
    gpb.weight_0.data       = in.weight.data();       gpb.weight_0.count       = in.weight.size();
    gpb.nTarget_0.data      = in.nTarget.data();      gpb.nTarget_0.count      = in.nTarget.size();
    gpb.stiffness_0.data    = in.stiffness.data();    gpb.stiffness_0.count    = in.stiffness.size();
    gpb.lambda_0.data       = in.lambda.data();       gpb.lambda_0.count       = in.lambda.size();
    gpb.v_grad_0.data       = v_grad.data();          gpb.v_grad_0.count       = v_grad.size();
    gpb.v_hessScalar_0.data = v_hess.data();          gpb.v_hessScalar_0.count = v_hess.size();
    gpb.v_p_0.data          = v_p.data();             gpb.v_p_0.count          = v_p.size();
    gpb.v_nTarget_0.data    = v_n.data();             gpb.v_nTarget_0.count    = v_n.size();
    gpb.v_stiffness_0.data  = v_k.data();             gpb.v_stiffness_0.count  = v_k.size();
    gpb.v_lambda_0.data     = v_l.data();             gpb.v_lambda_0.count     = v_l.size();

    ComputeVaryingInput vib{};
    vib.startGroupID = uint3(0, 0, 0);
    vib.endGroupID   = uint3(1, 1, 1);
    bwd::main_0(&vib, nullptr, &gpb);

    // --- FD check ---
    const float eps = 1e-3f;
    int fails = 0;
    double max_rel_err = 0.0;
    auto check = [&](double analytic, double fd, const char* label, uint32_t c, int r, int comp) {
        const double abs_err = std::fabs(analytic - fd);
        const double denom = std::max(1e-6, std::fabs(fd));
        const double rel = abs_err / denom;
        if (rel > max_rel_err) max_rel_err = rel;
        if (rel > 2e-2 && abs_err > 1e-4) {
            if (fails < 8) {
                std::fprintf(stderr,
                    "tri_bending_al_backward FD mismatch %s c=%u r=%d comp=%d: analytic=%g, fd=%g (rel=%g)\n",
                    label, c, r, comp, analytic, fd, rel);
            }
            ++fails;
        }
    };

    for (uint32_t c = 0; c < N_BEND; ++c) {
        // FD on each corner's position
        for (int r = 0; r < 4; ++r) {
            const uint32_t v = in.idx[4*c + r];
            for (int comp = 0; comp < 3; ++comp) {
                FwdInputs plus = in, minus = in;
                plus.positions[v][comp]  += eps;
                minus.positions[v][comp] -= eps;
                const double Lp = run_forward_loss(plus,  v_grad, v_hess);
                const double Lm = run_forward_loss(minus, v_grad, v_hess);
                const double fd = (Lp - Lm) / (2.0 * double(eps));
                check(double(v_p[4*c + r][comp]), fd, "v_p", c, r, comp);
            }
        }
        // FD on stiffness[c]
        {
            FwdInputs plus = in, minus = in;
            plus.stiffness[c]  += eps;
            minus.stiffness[c] -= eps;
            const double Lp = run_forward_loss(plus,  v_grad, v_hess);
            const double Lm = run_forward_loss(minus, v_grad, v_hess);
            const double fd = (Lp - Lm) / (2.0 * double(eps));
            check(double(v_k[c]), fd, "v_stiffness", c, 0, 0);
        }
        // FD on nTarget[c]
        {
            FwdInputs plus = in, minus = in;
            plus.nTarget[c]  += eps;
            minus.nTarget[c] -= eps;
            const double Lp = run_forward_loss(plus,  v_grad, v_hess);
            const double Lm = run_forward_loss(minus, v_grad, v_hess);
            const double fd = (Lp - Lm) / (2.0 * double(eps));
            check(double(v_n[c]), fd, "v_nTarget", c, 0, 0);
        }
        // FD on lambda[c]
        for (int comp = 0; comp < 3; ++comp) {
            FwdInputs plus = in, minus = in;
            plus.lambda[c][comp]  += eps;
            minus.lambda[c][comp] -= eps;
            const double Lp = run_forward_loss(plus,  v_grad, v_hess);
            const double Lm = run_forward_loss(minus, v_grad, v_hess);
            const double fd = (Lp - Lm) / (2.0 * double(eps));
            check(double(v_l[c][comp]), fd, "v_lambda", c, 0, comp);
        }
    }

    const auto t1 = std::chrono::steady_clock::now();
    const double elapsed = std::chrono::duration<double>(t1 - t0).count();
    if (elapsed > kBudgetSeconds) {
        std::fprintf(stderr, "tri_bending_al_backward: TIMEOUT — %.3fs\n", elapsed);
        return 2;
    }
    if (fails == 0) {
        const uint32_t nChecks = N_BEND * (12 + 1 + 1 + 3);
        std::printf("triangle_bending_force_al_backward: %u/%u FD checks OK (max_rel_err=%.3g, %.1fms)\n",
                    nChecks, nChecks, max_rel_err, elapsed * 1000.0);
        return 0;
    }
    std::fprintf(stderr, "triangle_bending_force_al_backward: %d FAIL\n", fails);
    return 1;
}
