// Real-input validator for Cloth.SlangCodegen.AttachmentProject — the
// DiffCloth PD per-pin local step. The current particle position is
// ignored; the kernel just emits sqrtWeight * fixedPos per attachment.
//
// Reference: src/code/simulation/AttachmentSpring.cpp:project.
//
// Test fixture (3 real attachments + 61 padded slots):
//   pin 0:  fixedPos = (5, 0, 0),  sqrtWeight = 2.0
//             → out = (10, 0, 0)
//   pin 1:  fixedPos = (0, 3, 0),  sqrtWeight = 1.0
//             → out = (0, 3, 0)
//   pin 2:  fixedPos = (1, 2, 3),  sqrtWeight = 4.0
//             → out = (4, 8, 12)
//   pins 3..63: sqrtWeight = 0 → out = (0, 0, 0) (not checked).

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <vector>

#include "attachment_project_emit.cpp"

static constexpr double kBudgetSeconds = 5.0;

int main() {
    const auto t0 = std::chrono::steady_clock::now();

    constexpr uint32_t N_REAL = 3;
    constexpr uint32_t GROUP_SIZE = 64;

    std::vector<Vector<float, 3>> projected(GROUP_SIZE, Vector<float, 3>(0.0f));

    std::vector<Vector<float, 3>> fixedPos(GROUP_SIZE, Vector<float, 3>(0.0f));
    fixedPos[0] = Vector<float, 3>(5.0f, 0.0f, 0.0f);
    fixedPos[1] = Vector<float, 3>(0.0f, 3.0f, 0.0f);
    fixedPos[2] = Vector<float, 3>(1.0f, 2.0f, 3.0f);

    std::vector<float> sqrtWeight(GROUP_SIZE, 0.0f);
    sqrtWeight[0] = 2.0f;
    sqrtWeight[1] = 1.0f;
    sqrtWeight[2] = 4.0f;

    GlobalParams_0 gp{};
    gp.projected_0.data   = projected.data();   gp.projected_0.count   = projected.size();
    gp.fixedPos_0.data    = fixedPos.data();    gp.fixedPos_0.count    = fixedPos.size();
    gp.sqrtWeight_0.data  = sqrtWeight.data();  gp.sqrtWeight_0.count  = sqrtWeight.size();

    ComputeVaryingInput vi{};
    vi.startGroupID = uint3(0, 0, 0);
    vi.endGroupID   = uint3(1, 1, 1);
    main_0(&vi, nullptr, &gp);

    const float expected[N_REAL][3] = {
        {10.0f,  0.0f,  0.0f},
        { 0.0f,  3.0f,  0.0f},
        { 4.0f,  8.0f, 12.0f},
    };

    int   fails        = 0;
    float max_abs_diff = 0.0f;
    for (uint32_t c = 0; c < N_REAL; ++c) {
        for (int k = 0; k < 3; ++k) {
            const float got = projected[c][k];
            const float ref = expected[c][k];
            const float d   = std::fabs(got - ref);
            if (d > max_abs_diff) max_abs_diff = d;
            if (d > 1e-6f) {
                if (fails < 5) {
                    std::fprintf(stderr,
                        "attachment_project mismatch at c=%u, k=%d: got %g, expected %g (diff %g)\n",
                        c, k, got, ref, d);
                }
                ++fails;
            }
        }
    }

    const auto t1 = std::chrono::steady_clock::now();
    const double elapsed = std::chrono::duration<double>(t1 - t0).count();
    if (elapsed > kBudgetSeconds) {
        std::fprintf(stderr, "attachment_project: TIMEOUT — %.3fs > %.1fs\n",
                     elapsed, kBudgetSeconds);
        return 2;
    }
    if (fails == 0) {
        std::printf("attachment_project: %u/%u pins OK (max_abs_diff=%g, %.1fms)\n",
                    N_REAL, N_REAL, max_abs_diff, elapsed * 1000.0);
        return 0;
    }
    std::fprintf(stderr, "attachment_project: %d FAIL out of %u\n",
                 fails, N_REAL * 3u);
    return 1;
}
