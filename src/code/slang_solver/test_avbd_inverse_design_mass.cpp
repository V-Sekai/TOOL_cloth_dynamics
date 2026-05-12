// Density / mass inverse-design MVP (CHI-14 Brick B).
//
// Sibling of test_avbd_inverse_design.cpp (which fits k_spring). This
// one fits a per-vertex MASS to match a target post-step position,
// using the v_mass cotangent emitted by `vbd_init_backward` and routed
// through `AvbdSolver::stepBackward` + `readMassGrad`.
//
// Setup. Same 2-vert / 1-spring / 1-attach fixture as the stiffness
// MVP. Vertex v1 is the free DoF; varying its mass m_v1 changes the
// inertial weight w = m·invHSq → changes the per-vertex Hessian
// → changes Δx → changes final position. Spring stiffness held fixed
// at k = 4.0 (no longer the optimized variable).
//
// LBFGSpp drives a SCALAR x = m_v1, starting at 0.5 with bounds
// [0.05, 10]. Ground truth m_v1 = 2.0. Expected: convergence within
// 1e-3 relative error in <10 fn evals.

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
    static constexpr float L  = 1.0f;
    static constexpr float K_SPRING = 4.0f;
    static constexpr float K_ATTACH = 1000.0f;
    static constexpr float INV_HSQ  = 100.0f;
    static constexpr float M_V0     = 1.0f;   // v0 is pinned; mass is irrelevant
    static constexpr float V1_PRED_X = 1.9f;

    std::vector<float> positions = {0.0f, 0.0f, 0.0f,  2.0f, 0.0f, 0.0f};
    std::vector<float> predicted = {0.0f, 0.0f, 0.0f,  V1_PRED_X, 0.0f, 0.0f};

    std::vector<uint32_t> spP1 = {1u};
    std::vector<uint32_t> spP2 = {0u};
    std::vector<float>    spLen= {L};
    std::vector<float>    spK  = {K_SPRING};

    std::vector<uint32_t> atVert  = {0u};
    std::vector<float>    atFixed = {0.0f, 0.0f, 0.0f};
    std::vector<float>    atK     = {K_ATTACH};
};

void run_forward(cloth::AvbdSolver& solver, const Mesh& m, float m_v1,
                 std::vector<float>& positions_out) {
    const float mass[Mesh::N] = {Mesh::M_V0, m_v1};
    solver.setupMesh(Mesh::N, m.positions.data(), m.predicted.data(), mass,
                     Mesh::INV_HSQ);
    solver.uploadSprings(1u, m.spP1.data(), m.spP2.data(), m.spLen.data(),
                         m.spK.data());
    solver.uploadAttachments(1u, m.atVert.data(), m.atFixed.data(), m.atK.data());
    if (solver.step() != 0) {
        std::fprintf(stderr, "test_avbd_inverse_design_mass: forward failed\n");
        std::exit(1);
    }
    solver.readPositions(positions_out);
}

class MassProblem {
public:
    MassProblem(cloth::AvbdSolver& solver, const Mesh& mesh,
                const std::vector<float>& target)
        : solver_(solver), mesh_(mesh), target_(target) {}

    int iters = 0;

    double operator()(const VecXd& x, VecXd& grad) {
        ++iters;
        const float m_v1 = static_cast<float>(x[0]);

        std::vector<float> pos;
        run_forward(solver_, mesh_, m_v1, pos);

        double loss = 0.0;
        std::vector<float> v_loss(3 * Mesh::N, 0.0f);
        for (uint32_t v = 0; v < Mesh::N; ++v) {
            for (int c = 0; c < 3; ++c) {
                const double diff = double(pos[3*v+c]) - double(target_[3*v+c]);
                loss += 0.5 * diff * diff;
                v_loss[3*v+c] = static_cast<float>(diff);
            }
        }

        if (solver_.stepBackward(v_loss.data()) != 0) {
            std::fprintf(stderr, "stepBackward failed\n");
            std::exit(1);
        }
        std::vector<float> dMass;
        solver_.readMassGrad(dMass);
        // ∂L/∂m_v1 — gradient of loss w.r.t. v1's mass.
        grad[0] = double(dMass[1]);

        std::printf("  iter %2d:  m=%.6f  loss=%.6g  ∂L/∂m=%.6g\n",
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
        std::fprintf(stderr, "construction failed\n");
        return 1;
    }

    const Mesh mesh;
    constexpr double M_GT      = 2.0;
    constexpr double M_INITIAL = 0.5;
    constexpr double M_LO = 0.05, M_HI = 10.0;

    std::vector<float> target;
    run_forward(solver, mesh, static_cast<float>(M_GT), target);
    std::printf("Ground truth: m_v1 = %.4f → x_v1 = (%.6f, %.6f, %.6f)\n",
                M_GT, target[3], target[4], target[5]);

    LBFGSpp::LBFGSBParam<double> param;
    param.epsilon = 1e-8;
    param.epsilon_rel = 1e-8;
    param.delta = 1e-12;
    param.past = 1;
    param.max_iterations = 50;
    param.max_linesearch = 40;
    LBFGSpp::LBFGSBSolver<double> lbfgs(param);

    MassProblem fn(solver, mesh, target);
    VecXd x(1);  x[0] = M_INITIAL;
    VecXd lb(1); lb[0] = M_LO;
    VecXd ub(1); ub[0] = M_HI;

    double fx = 0.0;
    int niter = 0;
    try {
        niter = lbfgs.minimize(fn, x, fx, lb, ub);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "L-BFGS-B threw: %s\n", e.what());
        return 1;
    }

    std::printf("L-BFGS-B converged in %d iters (%d fn evals)\n", niter, fn.iters);
    std::printf("  m_recovered = %.6f   (ground truth %.6f)\n", x[0], M_GT);
    std::printf("  final loss  = %.6g\n", fx);

    const double m_err = std::fabs(x[0] - M_GT) / M_GT;
    constexpr double M_TOL = 1e-3;
    if (m_err > M_TOL) {
        std::fprintf(stderr,
            "test_avbd_inverse_design_mass: FAIL — m %.6f vs gt %.6f, "
            "rel err %.4g > tol %.4g\n", x[0], M_GT, m_err, M_TOL);
        return 1;
    }
    std::printf("test_avbd_inverse_design_mass: OK (rel err %.3g < tol %.3g)\n",
                m_err, M_TOL);
    return 0;
}
