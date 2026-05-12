// Real-input validator for Cloth.SlangCodegen.VbdSolveApply —
// final kernel of the AVBD vertex-block-update pipeline (PR-D part 2).
//
// Per vertex v, read g and packed sym 3x3 H from scratch, compute
// Δx = −H⁻¹ g via closed-form adjugate/det, write positions[v] += Δx.
//
// Test fixture (3 verts, three different H/g combinations):
//
// v0:  H = 2·I (diagonal), g = (−4, −8, −12)
//      x_start = (0, 0, 0)
//      H⁻¹ = (1/2)·I
//      Δx = −H⁻¹·g = (2, 4, 6)
//      x_end = (2, 4, 6)
//
// v1:  H = I (diagonal), g = (0, 0, 0)   -- no displacement
//      x_start = (5, 5, 5)
//      Δx = 0
//      x_end = (5, 5, 5)
//
// v2:  H = [[4,1,0],[1,3,0],[0,0,5]] (sym, off-diag), g = (8, 3, 10)
//      adj00 = 3·5 − 0² = 15
//      adj11 = 4·5 − 0² = 20
//      adj22 = 4·3 − 1² = 11
//      adj01 = 0·0 − 1·5 = −5
//      adj02 = 1·0 − 0·3 = 0
//      adj12 = 1·0 − 4·0 = 0
//      det   = 4·15 + 1·(−5) + 0·0 = 55
//      Δx_x  = −(1/55)·(15·8 + (−5)·3 + 0·10) = −(1/55)·(120 − 15) = −(1/55)·105 = −21/11
//      Δx_y  = −(1/55)·((−5)·8 + 20·3 + 0·10) = −(1/55)·(−40 + 60) = −20/55 = −4/11
//      Δx_z  = −(1/55)·(0·8 + 0·3 + 11·10) = −(1/55)·110 = −2
//      x_start = (10, 10, 10)
//      x_end ≈ (10 − 21/11, 10 − 4/11, 10 − 2) = (8.0909..., 9.6363..., 8)

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <vector>

#include "vbd_solve_apply_emit.cpp"

static constexpr double kBudgetSeconds = 5.0;

int main() {
    const auto t0 = std::chrono::steady_clock::now();

    constexpr uint32_t N_VERTS   = 3;
    constexpr uint32_t GROUP_SIZE = 64;

    std::vector<Vector<float, 3>> positions(GROUP_SIZE, Vector<float, 3>(0.0f));
    std::vector<Vector<float, 3>> gScratch(GROUP_SIZE, Vector<float, 3>(0.0f));
    // hScratch defaults to identity matrix per slot so padded threads
    // don't hit a singular Hessian (det != 0).
    std::vector<float>            hScratch(6 * GROUP_SIZE, 0.0f);
    for (uint32_t i = 0; i < GROUP_SIZE; ++i) {
        hScratch[6*i + 0] = 1.0f;  // Hxx
        hScratch[6*i + 3] = 1.0f;  // Hyy
        hScratch[6*i + 5] = 1.0f;  // Hzz
    }

    // v0
    positions[0] = Vector<float, 3>(0.0f, 0.0f, 0.0f);
    gScratch[0]  = Vector<float, 3>(-4.0f, -8.0f, -12.0f);
    hScratch[0]  = 2.0f;  hScratch[3] = 2.0f;  hScratch[5] = 2.0f;
    hScratch[1]  = 0.0f;  hScratch[2] = 0.0f;  hScratch[4] = 0.0f;

    // v1
    positions[1] = Vector<float, 3>(5.0f, 5.0f, 5.0f);
    gScratch[1]  = Vector<float, 3>(0.0f, 0.0f, 0.0f);
    // hScratch[6..11] stays at identity from above

    // v2 with off-diagonal Hessian.
    positions[2] = Vector<float, 3>(10.0f, 10.0f, 10.0f);
    gScratch[2]  = Vector<float, 3>(8.0f, 3.0f, 10.0f);
    hScratch[12] = 4.0f;  // Hxx
    hScratch[13] = 1.0f;  // Hxy
    hScratch[14] = 0.0f;  // Hxz
    hScratch[15] = 3.0f;  // Hyy
    hScratch[16] = 0.0f;  // Hyz
    hScratch[17] = 5.0f;  // Hzz

    GlobalParams_0 gp{};
    gp.gScratch_0.data  = gScratch.data();  gp.gScratch_0.count  = gScratch.size();
    gp.hScratch_0.data  = hScratch.data();  gp.hScratch_0.count  = hScratch.size();
    gp.positions_0.data = positions.data(); gp.positions_0.count = positions.size();

    ComputeVaryingInput vi{};
    vi.startGroupID = uint3(0, 0, 0);
    vi.endGroupID   = uint3(1, 1, 1);
    main_0(&vi, nullptr, &gp);

    const float expected_pos[N_VERTS][3] = {
        {2.0f, 4.0f, 6.0f},
        {5.0f, 5.0f, 5.0f},
        {10.0f - 21.0f/11.0f, 10.0f - 4.0f/11.0f, 8.0f},
    };

    int   fails        = 0;
    float max_abs_diff = 0.0f;
    for (uint32_t v = 0; v < N_VERTS; ++v) {
        for (int axis = 0; axis < 3; ++axis) {
            const float got = positions[v][axis];
            const float ref = expected_pos[v][axis];
            const float d   = std::fabs(got - ref);
            if (d > max_abs_diff) max_abs_diff = d;
            if (d > 1e-5f) {
                if (fails < 5) {
                    std::fprintf(stderr,
                        "vbd_solve_apply mismatch at v=%u, axis=%d: got %g, expected %g (diff %g)\n",
                        v, axis, got, ref, d);
                }
                ++fails;
            }
        }
    }

    const auto t1 = std::chrono::steady_clock::now();
    const double elapsed = std::chrono::duration<double>(t1 - t0).count();
    if (elapsed > kBudgetSeconds) {
        std::fprintf(stderr, "vbd_solve_apply: TIMEOUT — %.3fs > %.1fs\n",
                     elapsed, kBudgetSeconds);
        return 2;
    }
    if (fails == 0) {
        std::printf("vbd_solve_apply: %u/%u verts OK (max_abs_diff=%g, %.1fms)\n",
                    N_VERTS, N_VERTS, max_abs_diff, elapsed * 1000.0);
        return 0;
    }
    std::fprintf(stderr, "vbd_solve_apply: %d FAIL out of %u checks\n", fails, N_VERTS * 3u);
    return 1;
}
