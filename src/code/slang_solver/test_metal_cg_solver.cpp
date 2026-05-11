// Standalone correctness test for MetalCGSolver. Builds a tiny SPD
// CSR matrix, solves A·x = b through the wrapper, and compares against
// the analytic answer.
//
// System (same as cg_demo_test from #15):
//
//   A = [ 4 1 0 ]      b = [5, 7, 6]
//       [ 1 3 0 ]      A⁻¹·b = [8/11, 23/11, 3]
//       [ 0 0 2 ]               ≈ [0.7272727, 2.0909091, 3.0]
//
//   CSR:
//     rowPtr = [0, 2, 4, 5]
//     colIdx = [0, 1, 0, 1, 2]
//     values = [4, 1, 1, 3, 2]
//
// CG on a 3×3 SPD matrix converges in ≤ 3 iters in exact arithmetic;
// fp32 may need 4-5. Tolerance: 1e-4 absolute (loose because fp32).

#include "MetalCGSolver.h"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

int main(int argc, char** argv) {
    // Path to the directory containing spmv.metallib, saxpby.metallib,
    // dot_reduce.metallib. Default points at our existing slang_validate
    // build dir; can be overridden with argv[1].
    std::string libDir =
        "/Users/ernest.lee/Desktop/TOOL_cloth_dynamics/tests/slang_validate/build";
    if (argc > 1) libDir = argv[1];

    cloth::CSRSpMatF A;
    A.rows    = 3;
    A.rowPtr  = {0, 2, 4, 5};
    A.colIdx  = {0, 1, 0, 1, 2};
    A.values  = {4.0f, 1.0f, 1.0f, 3.0f, 2.0f};

    cloth::MetalCGSolver solver(A, libDir.c_str());
    if (!solver.ok()) {
        std::fprintf(stderr, "test_metal_cg_solver: solver init failed.  "
                              "Make sure metallibs exist in:\n  %s\n"
                              "Run `make -C tests/slang_validate test` first.\n",
                     libDir.c_str());
        return 2;
    }

    std::vector<double> b = {5.0, 7.0, 6.0};
    std::vector<double> x;

    const auto t0 = std::chrono::steady_clock::now();
    int iters = solver.solve(b, x, /*tol=*/1e-6, /*max_iter=*/50);
    const double ms =
        std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - t0).count();

    if (iters < 0) {
        std::fprintf(stderr, "test_metal_cg_solver: solve failed\n");
        return 1;
    }

    const double expected[] = {8.0 / 11.0, 23.0 / 11.0, 3.0};
    double max_diff = 0.0;
    for (size_t i = 0; i < x.size(); ++i) {
        const double d = std::fabs(x[i] - expected[i]);
        if (d > max_diff) max_diff = d;
    }

    std::printf("test_metal_cg_solver: iters=%d  x=[%.6f, %.6f, %.6f]\n"
                "                       expected=[%.6f, %.6f, %.6f]\n"
                "                       max_abs_diff=%g  wall=%.2f ms\n",
                iters, x[0], x[1], x[2],
                expected[0], expected[1], expected[2],
                max_diff, ms);

    if (max_diff > 1e-4) {
        std::fprintf(stderr, "test_metal_cg_solver: FAIL (diff %g > 1e-4)\n",
                     max_diff);
        return 1;
    }
    std::printf("test_metal_cg_solver: OK\n");
    return 0;
}
