// Inverse-design MVP for AvbdSolver::stepBackward (CHI-14 Brick A).
//
// Smallest end-to-end proof that AVBD's reverse-mode adjoint composes
// with L-BFGS-B to recover a ground-truth parameter from a target
// position. No DiffCloth-side plumbing — drives the AvbdSolver C++ API
// directly.
//
// Setup. 2 verts, 1 spring, 1 attachment:
//   v0 attached to (0,0,0) with k_attach = 1000      (rigid pin)
//   v1 free, mass = 1
//   spring (a=1, b=0), rest length L = 1, stiffness k_spring (to fit)
//   predicted: positions + h²·g  (gravity bias toward rest)
//   invHSquared = 100
//
// Forward (1 AVBD outer iter):
//   v0 stays put (pinned).
//   v1: g = inertial(1·100·Δs) + spring(k·c·n/r)
//       H = diag(100+k, 100, 100)   (inertial + spring GN)
//       Δx = −H⁻¹·g
//   x_v1_after = x_v1_before + Δx
//
// We run the forward at k = k_gt (4.0) to obtain x_target, then ask
// LBFGSpp to find k that minimises ½‖x_v1 − x_target‖² starting from
// k = 1.0. With AVBD's analytic adjoint feeding the gradient,
// convergence to k_gt should be well within 20 LBFGS iterations.

#include "AvbdSolver.h"

#include <Eigen/Core>
#include <LBFGSB.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

using VecXd = Eigen::VectorXd;

namespace {

struct Mesh {
    static constexpr uint32_t N = 2;
    static constexpr float L  = 1.0f;   // spring rest length
    static constexpr float K_ATTACH = 1000.0f;
    static constexpr float INV_HSQ  = 100.0f;
    static constexpr float MASS     = 1.0f;
    // Predicted position perturbs v1 slightly toward shorter rest;
    // this is what makes the 1-iter Δx non-zero and dependent on k.
    static constexpr float V1_PREDICTED_X = 1.9f;

    std::vector<float> positions = {0.0f, 0.0f, 0.0f,  2.0f, 0.0f, 0.0f};
    std::vector<float> predicted = {0.0f, 0.0f, 0.0f,  V1_PREDICTED_X, 0.0f, 0.0f};
    std::vector<float> mass      = {MASS, MASS};

    std::vector<uint32_t> spP1 = {1u};     // endpoint a
    std::vector<uint32_t> spP2 = {0u};     // endpoint b
    std::vector<float>    spLen= {L};

    std::vector<uint32_t> atVert  = {0u};
    std::vector<float>    atFixed = {0.0f, 0.0f, 0.0f};
    std::vector<float>    atK     = {K_ATTACH};
};

// Run one AVBD outer iter at the given spring stiffness and return the
// post-step v1 position (3 floats).
void run_forward(cloth::AvbdSolver& solver, const Mesh& m, float k_spring,
                 std::vector<float>& positions_out) {
    solver.setupMesh(Mesh::N, m.positions.data(), m.predicted.data(),
                     m.mass.data(), Mesh::INV_HSQ);
    const float k_spring_arr[1] = {k_spring};
    solver.uploadSprings(1u, m.spP1.data(), m.spP2.data(), m.spLen.data(),
                         k_spring_arr);
    solver.uploadAttachments(1u, m.atVert.data(), m.atFixed.data(), m.atK.data());
    if (solver.step() != 0) {
        std::fprintf(stderr, "test_avbd_inverse_design: forward step failed\n");
        std::exit(1);
    }
    solver.readPositions(positions_out);
}

class InverseDesignProblem {
public:
    InverseDesignProblem(cloth::AvbdSolver& solver, const Mesh& mesh,
                         const std::vector<float>& target)
        : solver_(solver), mesh_(mesh), target_(target) {}

    int iters = 0;

    // LBFGSpp calls this: returns loss, fills grad. Convention: x has 1
    // entry (k_spring); grad has 1 entry (∂loss/∂k_spring).
    double operator()(const VecXd& x, VecXd& grad) {
        ++iters;
        const float k = static_cast<float>(x[0]);

        std::vector<float> pos;
        run_forward(solver_, mesh_, k, pos);

        // Loss = ½ Σ (x_v - target_v)²  over both verts. (Only v1 moves;
        // v0 stays pinned ≈ at target. Including v0 keeps the cotangent
        // dimension consistent with stepBackward's input layout.)
        double loss = 0.0;
        std::vector<float> v_loss(3 * Mesh::N, 0.0f);
        for (uint32_t v = 0; v < Mesh::N; ++v) {
            for (int c = 0; c < 3; ++c) {
                const double diff = double(pos[3*v+c]) - double(target_[3*v+c]);
                loss += 0.5 * diff * diff;
                v_loss[3*v+c] = static_cast<float>(diff);
            }
        }

        // Backward: dL/dx → dL/dk via AVBD adjoint.
        if (solver_.stepBackward(v_loss.data()) != 0) {
            std::fprintf(stderr, "test_avbd_inverse_design: stepBackward failed\n");
            std::exit(1);
        }
        std::vector<float> dRestLen, dStiff;
        solver_.readSpringGrad(dRestLen, dStiff);
        grad[0] = double(dStiff[0]);

        std::printf("  iter %2d:  k=%.6f  loss=%.6g  ∂L/∂k=%.6g\n",
                    iters, x[0], loss, grad[0]);
        return loss;
    }

private:
    cloth::AvbdSolver& solver_;
    const Mesh& mesh_;
    const std::vector<float>& target_;
};

}  // anonymous namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s <metallib_dir>\n", argv[0]);
        return 2;
    }
    cloth::AvbdSolver solver(argv[1]);
    if (!solver.ok()) {
        std::fprintf(stderr, "test_avbd_inverse_design: construction failed\n");
        return 1;
    }

    const Mesh mesh;

    // 1. Build the target: run forward at k_gt = 4.0 and capture x.
    constexpr double K_GT     = 4.0;
    constexpr double K_INITIAL = 1.0;
    constexpr double K_LO = 0.1, K_HI = 100.0;

    std::vector<float> target;
    run_forward(solver, mesh, static_cast<float>(K_GT), target);
    std::printf("Ground truth: k = %.4f → x_v1 = (%.6f, %.6f, %.6f)\n",
                K_GT, target[3], target[4], target[5]);

    // 2. LBFGSpp setup.
    LBFGSpp::LBFGSBParam<double> param;
    param.epsilon = 1e-8;
    param.epsilon_rel = 1e-8;
    param.delta = 1e-12;            // tight objective-change stop
    param.past = 1;
    param.max_iterations = 50;
    param.max_linesearch = 40;
    LBFGSpp::LBFGSBSolver<double> lbfgs(param);

    InverseDesignProblem fn(solver, mesh, target);
    VecXd x(1);  x[0] = K_INITIAL;
    VecXd lb(1); lb[0] = K_LO;
    VecXd ub(1); ub[0] = K_HI;

    double fx = 0.0;
    int niter = 0;
    try {
        niter = lbfgs.minimize(fn, x, fx, lb, ub);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "L-BFGS-B threw: %s\n", e.what());
        return 1;
    }

    std::printf("L-BFGS-B converged in %d iters (%d fn evals)\n", niter, fn.iters);
    std::printf("  k_recovered = %.6f   (ground truth %.6f)\n", x[0], K_GT);
    std::printf("  final loss  = %.6g\n", fx);

    const double k_err = std::fabs(x[0] - K_GT) / K_GT;
    constexpr double K_TOL = 1e-3;  // 0.1% relative
    if (k_err > K_TOL) {
        std::fprintf(stderr,
            "test_avbd_inverse_design: FAIL — k recovered %.6f, gt %.6f, "
            "rel err %.4g > tol %.4g\n",
            x[0], K_GT, k_err, K_TOL);
        return 1;
    }
    std::printf("test_avbd_inverse_design: OK (rel err %.3g < tol %.3g)\n",
                k_err, K_TOL);
    return 0;
}
