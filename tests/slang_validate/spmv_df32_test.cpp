// Real-input validator for Cloth.SlangCodegen.SpmvDf32 — the df32-
// accumulating sparse matrix-vector kernel.
//
// Should produce the same result as the original fp32 Spmv (#13) on
// the same fixture — at this small N (3x3, 6 nnz) the fp32 round-off
// is far below 1e-6 so both kernels land at the same answer.
//
// The win for SpmvDf32 only shows up at scale: large N or
// ill-conditioned matrices where the per-row sum builds up enough
// fp32 error to matter. That win is observable in CG iter count
// rather than per-spmv accuracy.
//
// Test fixture: 3x3 sparse matrix
//
//   A = [ 2 0 1 ]      x = [1, 2, 3]
//       [ 0 3 0 ]      y = A*x = [5, 6, 32]
//       [ 4 5 6 ]
//
//   CSR:
//     rowPtr = [0, 2, 3, 6]
//     colIdx = [0, 2, 1, 0, 1, 2]
//     values = [2, 1, 3, 4, 5, 6]

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>

#include "spmv_df32_emit.cpp"

static constexpr double kBudgetSeconds = 5.0;

int main() {
    const auto t0 = std::chrono::steady_clock::now();

    constexpr uint32_t ROWS = 3;
    constexpr uint32_t NNZ  = 6;

    SpmvDf32Params_0 params{ROWS};

    int32_t rowPtr[ROWS + 1] = {0, 2, 3, 6};
    int32_t colIdx[NNZ]      = {0, 2, 1, 0, 1, 2};
    float   values[NNZ]      = {2.0f, 1.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    float   x[ROWS]          = {1.0f, 2.0f, 3.0f};
    float   y[ROWS]          = {0.0f, 0.0f, 0.0f};

    const float expected[ROWS] = {
        2.0f * 1.0f + 1.0f * 3.0f,                    // = 5
        3.0f * 2.0f,                                  // = 6
        4.0f * 1.0f + 5.0f * 2.0f + 6.0f * 3.0f,      // = 32
    };

    GlobalParams_0 gp{};
    gp.params_0      = &params;
    gp.rowPtr_0.data = rowPtr;   gp.rowPtr_0.count = ROWS + 1;
    gp.colIdx_0.data = colIdx;   gp.colIdx_0.count = NNZ;
    gp.values_0.data = values;   gp.values_0.count = NNZ;
    gp.x_0.data      = x;        gp.x_0.count      = ROWS;
    gp.y_0.data      = y;        gp.y_0.count      = ROWS;

    ComputeVaryingInput vi{};
    vi.startGroupID = uint3(0, 0, 0);
    vi.endGroupID   = uint3(1, 1, 1);
    main_0(&vi, nullptr, &gp);

    int   fails        = 0;
    float max_abs_diff = 0.0f;
    for (uint32_t i = 0; i < ROWS; ++i) {
        const float d = std::fabs(y[i] - expected[i]);
        if (d > max_abs_diff) max_abs_diff = d;
        if (d > 1e-6f) {
            if (fails < 5) {
                std::fprintf(stderr,
                    "spmv_df32 mismatch at i=%u: got %g, expected %g (diff %g)\n",
                    i, y[i], expected[i], d);
            }
            ++fails;
        }
    }

    const auto t1 = std::chrono::steady_clock::now();
    const double elapsed = std::chrono::duration<double>(t1 - t0).count();
    if (elapsed > kBudgetSeconds) {
        std::fprintf(stderr, "spmv_df32: TIMEOUT — %.3fs > %.1fs\n",
                     elapsed, kBudgetSeconds);
        return 2;
    }
    if (fails == 0) {
        std::printf("spmv_df32: %u/%u rows OK (nnz=%u, max_abs_diff=%g, %.1fms)\n",
                    ROWS, ROWS, NNZ, max_abs_diff, elapsed * 1000.0);
        return 0;
    }
    std::fprintf(stderr, "spmv_df32: %d FAIL out of %u\n", fails, ROWS);
    return 1;
}
