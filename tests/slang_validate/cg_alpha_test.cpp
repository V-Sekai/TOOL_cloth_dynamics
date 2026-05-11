// Real-input validator for Cloth.SlangCodegen.CGAlpha — the
// single-thread GPU-side α / −α kernel for the CG inner loop.
//
//   α      = dot(r, r) / dot(p, q)
//   out[0] = +α
//   out[1] = −α
//
// dot inputs are df32 (hi, lo) pairs from dot_reduce_serial.
//
// Test fixture matches cg_demo's iter-1 values from #15:
//   dot(p, q) = 389         → dotPQ    = [389, 0]
//   dot(r, r) = 110         → dotDelta = [110, 0]
//   expected α = 110/389 ≈ 0.282776
//   expected out = [+α, −α] = [+0.282776, −0.282776]

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>

#include "cg_alpha_emit.cpp"

static constexpr double kBudgetSeconds = 5.0;

int main() {
    const auto t0 = std::chrono::steady_clock::now();

    float dotPQ[2]    = {389.0f, 0.0f};
    float dotDelta[2] = {110.0f, 0.0f};
    float alphaOut[2] = {0.0f, 0.0f};

    GlobalParams_0 gp{};
    gp.dotPQ_0.data    = dotPQ;     gp.dotPQ_0.count    = 2;
    gp.dotDelta_0.data = dotDelta;  gp.dotDelta_0.count = 2;
    gp.alphaOut_0.data = alphaOut;  gp.alphaOut_0.count = 2;

    ComputeVaryingInput vi{};
    vi.startGroupID = uint3(0, 0, 0);
    vi.endGroupID   = uint3(1, 1, 1);
    main_0(&vi, nullptr, &gp);

    const float expected_a = 110.0f / 389.0f;
    const float ref[2] = {+expected_a, -expected_a};

    int   fails        = 0;
    float max_abs_diff = 0.0f;
    for (int i = 0; i < 2; ++i) {
        const float d = std::fabs(alphaOut[i] - ref[i]);
        if (d > max_abs_diff) max_abs_diff = d;
        if (d > 1e-6f) {
            std::fprintf(stderr,
                "cg_alpha mismatch at i=%d: got %g, expected %g (diff %g)\n",
                i, alphaOut[i], ref[i], d);
            ++fails;
        }
    }

    const auto t1 = std::chrono::steady_clock::now();
    const double elapsed = std::chrono::duration<double>(t1 - t0).count();
    if (elapsed > kBudgetSeconds) {
        std::fprintf(stderr, "cg_alpha: TIMEOUT — %.3fs > %.1fs\n",
                     elapsed, kBudgetSeconds);
        return 2;
    }
    if (fails == 0) {
        std::printf("cg_alpha: 2/2 OK (α=%.6f, out=[%.6f, %.6f], max_abs_diff=%g, %.1fms)\n",
                    expected_a, alphaOut[0], alphaOut[1], max_abs_diff,
                    elapsed * 1000.0);
        return 0;
    }
    std::fprintf(stderr, "cg_alpha: %d FAIL\n", fails);
    return 1;
}
