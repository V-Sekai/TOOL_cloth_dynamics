// Real-input validator for Cloth.SlangCodegen.VbdInit — first kernel
// of the AVBD vertex-block-update pipeline (PR-D part 1).
//
// Per vertex v: writes the inertial term to scratch buffers:
//   w = m · invHSquared
//   gScratch[v]            = w · (x − x̃)
//   hScratch[6v..6v+5]     = [w, 0, 0, w, 0, w]    (packed sym 3x3 = w·I)
//
// Test fixture (2 verts):
//   v0:  x=(0,0,0)  pred=(1,2,3)  m=2, invH²=2  → w=4
//        g = 4 · (-1,-2,-3) = (-4,-8,-12)
//        H = [4, 0, 0, 4, 0, 4]
//   v1:  x=(5,5,5)  pred=(5,5,5)  m=1, invH²=2  → w=2
//        g = 2 · (0,0,0) = (0,0,0)
//        H = [2, 0, 0, 2, 0, 2]
//
// Padded slots 2..63: x=pred=(0,0,0), m=0 → w=0, all outputs zero.

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <vector>

#include "vbd_init_emit.cpp"

static constexpr double kBudgetSeconds = 5.0;

int main() {
    const auto t0 = std::chrono::steady_clock::now();

    constexpr uint32_t N_VERTS   = 2;
    constexpr uint32_t GROUP_SIZE = 64;

    VbdInitParams_0 params{};
    params.invHSquared_0 = 2.0f;

    std::vector<Vector<float, 3>> positions(GROUP_SIZE, Vector<float, 3>(0.0f));
    std::vector<Vector<float, 3>> predicted(GROUP_SIZE, Vector<float, 3>(0.0f));
    std::vector<float>            mass(GROUP_SIZE, 0.0f);
    std::vector<Vector<float, 3>> gScratch(GROUP_SIZE, Vector<float, 3>(0.0f));
    std::vector<float>            hScratch(6 * GROUP_SIZE, 0.0f);

    positions[0] = Vector<float, 3>(0.0f, 0.0f, 0.0f);
    predicted[0] = Vector<float, 3>(1.0f, 2.0f, 3.0f);
    mass[0]      = 2.0f;

    positions[1] = Vector<float, 3>(5.0f, 5.0f, 5.0f);
    predicted[1] = Vector<float, 3>(5.0f, 5.0f, 5.0f);
    mass[1]      = 1.0f;

    GlobalParams_0 gp{};
    gp.params_0       = &params;
    gp.positions_0.data = positions.data(); gp.positions_0.count = positions.size();
    gp.predicted_0.data = predicted.data(); gp.predicted_0.count = predicted.size();
    gp.mass_0.data      = mass.data();      gp.mass_0.count      = mass.size();
    gp.gScratch_0.data  = gScratch.data();  gp.gScratch_0.count  = gScratch.size();
    gp.hScratch_0.data  = hScratch.data();  gp.hScratch_0.count  = hScratch.size();

    ComputeVaryingInput vi{};
    vi.startGroupID = uint3(0, 0, 0);
    vi.endGroupID   = uint3(1, 1, 1);
    main_0(&vi, nullptr, &gp);

    const float expected_g[N_VERTS][3] = {
        {-4.0f, -8.0f, -12.0f},
        { 0.0f,  0.0f,   0.0f},
    };
    const float expected_h[N_VERTS][6] = {
        {4.0f, 0.0f, 0.0f, 4.0f, 0.0f, 4.0f},
        {2.0f, 0.0f, 0.0f, 2.0f, 0.0f, 2.0f},
    };

    int   fails        = 0;
    float max_abs_diff = 0.0f;
    auto check = [&](float got, float ref, const char* label, uint32_t v, int axis) {
        const float d = std::fabs(got - ref);
        if (d > max_abs_diff) max_abs_diff = d;
        if (d > 1e-6f) {
            if (fails < 5) {
                std::fprintf(stderr,
                    "vbd_init %s mismatch at v=%u, axis=%d: got %g, expected %g (diff %g)\n",
                    label, v, axis, got, ref, d);
            }
            ++fails;
        }
    };
    for (uint32_t v = 0; v < N_VERTS; ++v) {
        for (int axis = 0; axis < 3; ++axis) check(gScratch[v][axis], expected_g[v][axis], "g", v, axis);
        for (int j = 0; j < 6; ++j)          check(hScratch[6*v + j], expected_h[v][j],    "h", v, j);
    }

    const auto t1 = std::chrono::steady_clock::now();
    const double elapsed = std::chrono::duration<double>(t1 - t0).count();
    if (elapsed > kBudgetSeconds) {
        std::fprintf(stderr, "vbd_init: TIMEOUT — %.3fs > %.1fs\n",
                     elapsed, kBudgetSeconds);
        return 2;
    }
    if (fails == 0) {
        std::printf("vbd_init: %u/%u verts OK (max_abs_diff=%g, %.1fms)\n",
                    N_VERTS, N_VERTS, max_abs_diff, elapsed * 1000.0);
        return 0;
    }
    std::fprintf(stderr, "vbd_init: %d FAIL out of %u checks\n", fails, N_VERTS * 9u);
    return 1;
}
