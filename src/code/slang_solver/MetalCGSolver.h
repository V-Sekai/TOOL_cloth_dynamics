// MetalCGSolver — pure-C++ interface to the Slang/Metal CG solver
// pipeline (spmv + saxpby + dot_reduce). Wraps the Metal dispatcher
// so DiffCloth's simulation code can call into our Tier-3 perf path
// without pulling Foundation/Metal headers into Eigen-using TUs.
//
// Lifecycle:
//   - Construct once per system matrix (does CSR upload + PSO compile).
//   - `solve(b)` per PD outer iteration; returns x such that A·x = b.
//   - Destruct to free Metal resources.
//
// Precision caveat: kernels are fp32. The CG inner loop accumulates
// in df32 via dot_reduce_serial / dot_reduce, but spmv and saxpby
// are pure fp32. DiffCloth uses fp64 (Eigen VecXd = double). This
// wrapper accepts and returns double for API compatibility, but the
// solve happens in fp32 internally with df32 inner products. Drift
// vs Eigen's LLT solve is expected at ~1e-5..1e-7 relative.

#ifndef CLOTH_METAL_CG_SOLVER_H
#define CLOTH_METAL_CG_SOLVER_H

#include <cstddef>
#include <cstdint>
#include <vector>

namespace cloth {

// Sparse CSR matrix in 32-bit indices + 32-bit values. The wrapper
// takes whatever (rowPtr, colIdx, values) you give it; the caller
// is responsible for ensuring A is SPD (which DiffCloth's system
// matrix is).
struct CSRSpMatF {
    std::vector<int32_t> rowPtr;   // length = rows + 1
    std::vector<int32_t> colIdx;   // length = nnz
    std::vector<float>   values;   // length = nnz
    uint32_t rows = 0;
};

class MetalCGSolver {
public:
    // Construct from a CSR matrix. Uploads to Metal buffers; builds
    // compute pipelines for spmv / saxpby / dot_reduce. The
    // metallibPath must point at the directory containing
    // spmv.metallib, saxpby.metallib, dot_reduce.metallib (so the
    // wrapper can `MTLDevice newLibraryWithURL:` them at runtime).
    MetalCGSolver(const CSRSpMatF& A, const char* metallibPath);
    ~MetalCGSolver();

    // Solve A·x = b. b.size() == A.rows. Output x.size() == A.rows.
    // CG inner loop runs until ||r|| <= tol·||r₀|| or max_iter reached.
    // Returns the number of iterations taken (≤ max_iter). If the
    // solver failed to construct, returns -1 and leaves x untouched.
    int solve(const std::vector<double>& b, std::vector<double>& x,
              double tol = 1e-6, int max_iter = 200);

    bool ok() const;

private:
    struct Impl;
    Impl* impl_;
};

}  // namespace cloth

#endif  // CLOTH_METAL_CG_SOLVER_H
