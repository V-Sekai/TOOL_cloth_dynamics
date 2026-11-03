// AvbdIft — implicit-function (IFT) adjoint solve for the AVBD implicit-Euler step.
//
// The converged step satisfies g(x*; y, p) = 0 with g the total energy gradient
// the AVBD kernels already compute. The adjoint needs z = H^-1 c, H = dg/dx.
// H is never formed analytically: it is assembled from Hessian-vector products
// (central differences of AvbdSolver::computeGradient) with distance-2 colored
// probing, factored with sparse LDLT, and the factor is reused as a PCG
// preconditioner for neighbouring steps (H changes slowly) until CG stalls.
#ifndef CLOTH_AVBD_IFT_H
#define CLOTH_AVBD_IFT_H

#include <cstdint>
#include <memory>
#include <vector>

namespace cloth {

class AvbdSolver;

class AvbdIft {
public:
    explicit AvbdIft(AvbdSolver& solver);
    ~AvbdIft();

    // z = H^-1 c at (x, y, fixedPos); x, y, c are 3*nV floats (tight).
    // Returns CG iterations used (>= 0) or a negative error. Leaves the solver
    // at (x, y, fixedPos) so the per-constraint backward kernels see x*.
    int solve(const float* x, const float* y, const float* fixedPos, const float* c,
              int maxIter, float relTol, std::vector<float>& z, float* relRes);

    int refactorizations() const;
    int solves() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace cloth

#endif
