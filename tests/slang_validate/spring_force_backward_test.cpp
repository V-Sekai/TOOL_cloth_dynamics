// Finite-difference validator for Cloth.SlangCodegen.SpringForceBackward
// (PR-G / CHI-13). Numerically checks that the analytic adjoint emitted
// by the backward kernel agrees with central-difference gradients on
// the spring-force forward kernel.
//
// Setup. Per spring i with endpoints (a, b), rest length L, stiffness k:
//   d   = p_a − p_b
//   len = |d|
//   c   = len − L
//   gradA[i] = (k · c / len) · d
//   hess[6i..6i+5] packed sym 3x3 = k · (d ⊗ d) / len²
//
// Scalar loss for the adjoint check:
//   L_loss = Σ_i (v_gradA[i] · gradA[i]) + Σ_i (v_springHess[i] · trace(hess[i]))
//
// where trace(hess[i]) = hess[6i+0] + hess[6i+3] + hess[6i+5] = k
// (matching the gather backward's diagonal-sum convention).
//
// Validates:
//   v_p_d[i]       ≈ d L_loss / d (p_a − p_b)
//   v_restLen[i]   ≈ d L_loss / d L
//   v_stiffness[i] ≈ d L_loss / d k
//
// Two real springs over three vertices (same fixture as
// spring_force_test.cpp) plus randomly drawn cotangents. Central FD
// with ε=1e-3 on positions, restLen, stiffness. Tolerance 1e-2
// relative — generous to absorb fp32 fwd kernel + fwd loss summation
// noise.

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
    #include "spring_force_emit.cpp"
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
    #include "spring_force_backward_emit.cpp"
    #pragma pop_macro("SLANG_PRELUDE_EXTERN_C")
    #pragma pop_macro("SLANG_PRELUDE_EXTERN_C_START")
    #pragma pop_macro("SLANG_PRELUDE_EXTERN_C_END")
}

static constexpr double kBudgetSeconds = 5.0;

// Run the forward kernel and compute the scalar loss for given cotangents.
static double run_forward_loss(
    const std::vector<Vector<float, 3>>& positions,
    const std::vector<uint32_t>& p1Idx,
    const std::vector<uint32_t>& p2Idx,
    const std::vector<float>& restLen,
    const std::vector<float>& stiffness,
    const std::vector<Vector<float, 3>>& v_gradA,
    const std::vector<float>& v_springHess,
    uint32_t nSprings) {

    constexpr uint32_t GROUP_SIZE = 64;
    std::vector<Vector<float, 3>> gradA(GROUP_SIZE, Vector<float, 3>(0.0f));
    std::vector<float>            hess(6 * GROUP_SIZE, 0.0f);

    fwd::GlobalParams_0 gp{};
    gp.positions_0.data  = const_cast<Vector<float, 3>*>(positions.data());
    gp.positions_0.count = positions.size();
    gp.p1Idx_0.data      = const_cast<uint32_t*>(p1Idx.data());
    gp.p1Idx_0.count     = p1Idx.size();
    gp.p2Idx_0.data      = const_cast<uint32_t*>(p2Idx.data());
    gp.p2Idx_0.count     = p2Idx.size();
    gp.restLen_0.data    = const_cast<float*>(restLen.data());
    gp.restLen_0.count   = restLen.size();
    gp.stiffness_0.data  = const_cast<float*>(stiffness.data());
    gp.stiffness_0.count = stiffness.size();
    gp.gradA_0.data      = gradA.data();
    gp.gradA_0.count     = gradA.size();
    gp.hess_0.data       = hess.data();
    gp.hess_0.count      = hess.size();

    ComputeVaryingInput vi{};
    vi.startGroupID = uint3(0, 0, 0);
    vi.endGroupID   = uint3(1, 1, 1);
    fwd::main_0(&vi, nullptr, &gp);

    double loss = 0.0;
    for (uint32_t i = 0; i < nSprings; ++i) {
        loss += double(v_gradA[i].x) * double(gradA[i].x);
        loss += double(v_gradA[i].y) * double(gradA[i].y);
        loss += double(v_gradA[i].z) * double(gradA[i].z);
        // trace(hess) = hess[xx] + hess[yy] + hess[zz] = entries 0, 3, 5.
        const double tr = double(hess[6*i + 0]) + double(hess[6*i + 3]) + double(hess[6*i + 5]);
        loss += double(v_springHess[i]) * tr;
    }
    return loss;
}

int main() {
    const auto t0 = std::chrono::steady_clock::now();

    constexpr uint32_t N_SPRINGS = 2;
    constexpr uint32_t GROUP_SIZE = 64;

    std::vector<Vector<float, 3>> positions = {
        Vector<float, 3>(0.0f,  0.0f, 0.0f),
        Vector<float, 3>(3.0f,  0.0f, 0.0f),
        Vector<float, 3>(0.0f,  4.0f, 0.0f),
    };

    // Padding (slots 2..63): a=1, b=0 with restLen=0, stiffness=0 → kernel
    // outputs zero so they don't perturb the loss.
    std::vector<uint32_t> p1Idx(GROUP_SIZE, 1u);
    std::vector<uint32_t> p2Idx(GROUP_SIZE, 0u);
    std::vector<float>    restLen(GROUP_SIZE, 0.0f);
    std::vector<float>    stiffness(GROUP_SIZE, 0.0f);

    p1Idx[0] = 1u; p2Idx[0] = 0u; restLen[0] = 2.0f; stiffness[0] = 1.5f;
    p1Idx[1] = 2u; p2Idx[1] = 0u; restLen[1] = 3.5f; stiffness[1] = 2.5f;

    // Random cotangents (fixed seed for reproducibility).
    std::mt19937 rng(0xC411D13u);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<Vector<float, 3>> v_gradA(GROUP_SIZE, Vector<float, 3>(0.0f));
    std::vector<float>            v_springHess(GROUP_SIZE, 0.0f);
    for (uint32_t i = 0; i < N_SPRINGS; ++i) {
        v_gradA[i] = Vector<float, 3>(dist(rng), dist(rng), dist(rng));
        v_springHess[i] = dist(rng);
    }

    // --- Run backward kernel for analytical gradients ---
    std::vector<Vector<float, 3>> v_p_d_buf(GROUP_SIZE, Vector<float, 3>(0.0f));
    std::vector<float>            v_restLen_buf(GROUP_SIZE, 0.0f);
    std::vector<float>            v_stiffness_buf(GROUP_SIZE, 0.0f);

    bwd::GlobalParams_0 gpb{};
    gpb.positions_0.data    = positions.data();    gpb.positions_0.count    = positions.size();
    gpb.p1Idx_0.data        = p1Idx.data();        gpb.p1Idx_0.count        = p1Idx.size();
    gpb.p2Idx_0.data        = p2Idx.data();        gpb.p2Idx_0.count        = p2Idx.size();
    gpb.restLen_0.data      = restLen.data();      gpb.restLen_0.count      = restLen.size();
    gpb.stiffness_0.data    = stiffness.data();    gpb.stiffness_0.count    = stiffness.size();
    gpb.v_gradA_0.data      = v_gradA.data();      gpb.v_gradA_0.count      = v_gradA.size();
    gpb.v_springHess_0.data = v_springHess.data(); gpb.v_springHess_0.count = v_springHess.size();
    gpb.v_p_d_0.data        = v_p_d_buf.data();    gpb.v_p_d_0.count        = v_p_d_buf.size();
    gpb.v_restLen_0.data    = v_restLen_buf.data();  gpb.v_restLen_0.count    = v_restLen_buf.size();
    gpb.v_stiffness_0.data  = v_stiffness_buf.data(); gpb.v_stiffness_0.count = v_stiffness_buf.size();

    ComputeVaryingInput vib{};
    vib.startGroupID = uint3(0, 0, 0);
    vib.endGroupID   = uint3(1, 1, 1);
    bwd::main_0(&vib, nullptr, &gpb);

    // --- Finite-difference reference ---
    const float eps = 1e-3f;
    int fails = 0;
    double max_rel_err = 0.0;

    auto check = [&](double analytic, double fd, const char* label, uint32_t i, int comp) {
        const double abs_err = std::fabs(analytic - fd);
        const double denom = std::max(1e-6, std::fabs(fd));
        const double rel = abs_err / denom;
        if (rel > max_rel_err) max_rel_err = rel;
        // 2% relative tolerance — fp32 fwd-loss + fwd central-diff noise.
        if (rel > 2e-2 && abs_err > 1e-4) {
            if (fails < 8) {
                std::fprintf(stderr,
                    "spring_force_backward FD mismatch %s[%u,%d]: analytic=%g, fd=%g (rel=%g)\n",
                    label, i, comp, analytic, fd, rel);
            }
            ++fails;
        }
    };

    for (uint32_t i = 0; i < N_SPRINGS; ++i) {
        const uint32_t a = p1Idx[i];

        // FD on d = p_a − p_b: perturb p_a positively, p_b negatively
        // (equivalent to perturbing d directly).
        for (int comp = 0; comp < 3; ++comp) {
            auto pos_plus  = positions;
            auto pos_minus = positions;
            pos_plus[a][comp]  += eps;
            pos_minus[a][comp] -= eps;
            // Also subtract from b to perturb d (since d = p_a − p_b).
            // No — perturbing only p_a perturbs d by +eps which is what we
            // want for ∂L/∂d. The kernel writes ∂L/∂d, and via Newton's 3rd
            // the per-vertex contributions on a and b are ±∂L/∂d.
            // So FD on (p_a) gives exactly ∂L/∂d_comp.
            const double L_plus  = run_forward_loss(pos_plus,  p1Idx, p2Idx, restLen, stiffness, v_gradA, v_springHess, N_SPRINGS);
            const double L_minus = run_forward_loss(pos_minus, p1Idx, p2Idx, restLen, stiffness, v_gradA, v_springHess, N_SPRINGS);
            const double fd = (L_plus - L_minus) / (2.0 * double(eps));
            const double analytic = double(v_p_d_buf[i][comp]);
            check(analytic, fd, "v_p_d", i, comp);
        }

        // FD on restLen[i]
        {
            auto rl_plus  = restLen;
            auto rl_minus = restLen;
            rl_plus[i]  += eps;
            rl_minus[i] -= eps;
            const double L_plus  = run_forward_loss(positions, p1Idx, p2Idx, rl_plus,  stiffness, v_gradA, v_springHess, N_SPRINGS);
            const double L_minus = run_forward_loss(positions, p1Idx, p2Idx, rl_minus, stiffness, v_gradA, v_springHess, N_SPRINGS);
            const double fd = (L_plus - L_minus) / (2.0 * double(eps));
            const double analytic = double(v_restLen_buf[i]);
            check(analytic, fd, "v_restLen", i, 0);
        }

        // FD on stiffness[i]
        {
            auto st_plus  = stiffness;
            auto st_minus = stiffness;
            st_plus[i]  += eps;
            st_minus[i] -= eps;
            const double L_plus  = run_forward_loss(positions, p1Idx, p2Idx, restLen, st_plus,  v_gradA, v_springHess, N_SPRINGS);
            const double L_minus = run_forward_loss(positions, p1Idx, p2Idx, restLen, st_minus, v_gradA, v_springHess, N_SPRINGS);
            const double fd = (L_plus - L_minus) / (2.0 * double(eps));
            const double analytic = double(v_stiffness_buf[i]);
            check(analytic, fd, "v_stiffness", i, 0);
        }
    }

    const auto t1 = std::chrono::steady_clock::now();
    const double elapsed = std::chrono::duration<double>(t1 - t0).count();
    if (elapsed > kBudgetSeconds) {
        std::fprintf(stderr, "spring_force_backward: TIMEOUT — %.3fs > %.1fs\n",
                     elapsed, kBudgetSeconds);
        return 2;
    }
    if (fails == 0) {
        const uint32_t nChecks = N_SPRINGS * 5;  // 3 pos + 1 rest + 1 stiff
        std::printf("spring_force_backward: %u/%u FD checks OK (max_rel_err=%.3g, %.1fms)\n",
                    nChecks, nChecks, max_rel_err, elapsed * 1000.0);
        return 0;
    }
    std::fprintf(stderr, "spring_force_backward: %d FAIL\n", fails);
    return 1;
}
