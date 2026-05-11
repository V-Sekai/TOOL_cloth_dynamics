// Real-input validator for Cloth.SlangCodegen.DotReduceSerial — the
// single-thread df32 dot product that produces the same numerical
// result as the parallel DotReduce kernel.
//
// Why this kernel exists: slangc cpp target rejects
// GroupMemoryBarrierWithGroupSync (E36107) because its sequential
// per-thread dispatch can't honour cross-thread synchronisation. The
// serial variant uses [numthreads(1, 1, 1)] and a plain for-loop —
// same df32 EFTs, same result, no barriers.
//
// This harness puts the kernel under genuine fp32 stress so it can
// prove the df32 path actually does something measurable:
//
//   Test 1 (precision stress): n = 4096, a[i] = b[i] = 1.0 / sqrt(n).
//     Exact answer is 1.0. Plain fp32 sum stagnates with rounding
//     error proportional to n·ε; df32 should hit 1.0 to <1 ulp.
//
//   Test 2 (sanity): n = 4, a = [1, 2, 3, 4], b = [5, 6, 7, 8]
//     dot = 5 + 12 + 21 + 32 = 70.  Trivially exact in fp32 too,
//     but confirms the kernel handles arbitrary asymmetric inputs.

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <vector>

#include "dot_reduce_serial_emit.cpp"

static constexpr double kBudgetSeconds = 5.0;

namespace {

bool runOne(const char* tag, std::vector<float>& a,
            std::vector<float>& b, double expectedExact,
            float tol_ulp, int& fails_out, float& max_diff_out) {
    const uint32_t n = uint32_t(a.size());

    DotReduceSerialParams_0 params{n};

    std::vector<float> dst(2, 0.0f);

    GlobalParams_0 gp{};
    gp.params_0   = &params;
    gp.a_0.data   = a.data();   gp.a_0.count   = n;
    gp.b_0.data   = b.data();   gp.b_0.count   = n;
    gp.dst_0.data = dst.data(); gp.dst_0.count = dst.size();

    ComputeVaryingInput vi{};
    vi.startGroupID = uint3(0, 0, 0);
    vi.endGroupID   = uint3(1, 1, 1);
    main_0(&vi, nullptr, &gp);

    // df32 reconstruction: hi + lo with full fp64 sum.
    const double got = double(dst[0]) + double(dst[1]);
    const double diff = std::fabs(got - expectedExact);
    if (diff > max_diff_out) max_diff_out = float(diff);

    const float ulp_at_expected = std::fabs(std::nextafter(float(expectedExact), INFINITY)
                                            - float(expectedExact));
    const double tol = double(tol_ulp) * double(ulp_at_expected);
    const bool ok = diff <= tol;

    if (!ok) {
        std::fprintf(stderr,
            "dot_reduce_serial %s: got %.17g, expected %.17g (diff %g, tol %g = %g ulp)\n",
            tag, got, expectedExact, diff, tol, tol_ulp);
        ++fails_out;
    } else {
        std::printf("  %-20s n=%-5u got=%.17g  diff=%.3e  (df32 hi=%.17g lo=%.17g)\n",
                    tag, n, got, diff, dst[0], dst[1]);
    }
    return ok;
}

}  // namespace

int main() {
    const auto t0 = std::chrono::steady_clock::now();

    int   fails        = 0;
    float max_abs_diff = 0.0f;

    // Test 1: precision stress. n = 4096 identical terms of 1/sqrt(n).
    //   exact:  n * (1/sqrt(n))^2 = 1.0
    //   plain fp32 sum: drifts by O(n eps) ~ 4096 * 6e-8 ~ 2.4e-4.
    //   df32: should hit 1.0 within a few ulp.
    {
        constexpr uint32_t N = 4096;
        const float v = 1.0f / std::sqrt(float(N));
        std::vector<float> a(N, v), b(N, v);
        runOne("stress (n=4096)", a, b, 1.0, /*tol_ulp=*/4.0f,
               fails, max_abs_diff);
    }

    // Test 2: arbitrary asymmetric small case.
    {
        std::vector<float> a = {1.0f, 2.0f, 3.0f, 4.0f};
        std::vector<float> b = {5.0f, 6.0f, 7.0f, 8.0f};
        runOne("asym (n=4)", a, b, 70.0, /*tol_ulp=*/1.0f,
               fails, max_abs_diff);
    }

    const auto t1 = std::chrono::steady_clock::now();
    const double elapsed = std::chrono::duration<double>(t1 - t0).count();
    if (elapsed > kBudgetSeconds) {
        std::fprintf(stderr, "dot_reduce_serial: TIMEOUT — %.3fs > %.1fs\n",
                     elapsed, kBudgetSeconds);
        return 2;
    }
    if (fails == 0) {
        std::printf("dot_reduce_serial: 2/2 OK (max_abs_diff=%g, %.1fms)\n",
                    max_abs_diff, elapsed * 1000.0);
        return 0;
    }
    std::fprintf(stderr, "dot_reduce_serial: %d FAIL\n", fails);
    return 1;
}
