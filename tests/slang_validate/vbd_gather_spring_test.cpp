// Real-input validator for Cloth.SlangCodegen.VbdGatherSpring —
// gather kernel between vbd_init and vbd_solve_apply (PR-D part 3).
//
// Per vertex v, loops over the CSR adjacency slice and accumulates
// every incident spring's force/Hessian into the per-vertex scratch:
//   sign = 1 − 2·float(role[k])         -- +1 for endpoint a, −1 for b
//   gScratch[v] += sign · springGradA[c]
//   hScratch[6v..6v+5] += springHess[6c..6c+5]    (no sign flip; GN
//                                                   diagonal block is
//                                                   the same for both
//                                                   endpoints)
//
// Test fixture (2 verts, 1 spring between them):
//   spring 0: a=0, b=1
//   springGradA[0] = (1, 0, 0),  springHess[0..5] = [1, 0, 0, 0, 0, 0]
//   CSR (from AdjacencySpring.build 2 [(0,1)]):
//     vertSpringOffset = [0, 1, 2, 2, ...]   (size GROUP_SIZE + 1)
//     vertSpringIdx    = [0, 0]
//     vertSpringRole   = [0, 1]
//   Initial scratch:
//     gScratch[0] = (10, 0, 0)   hScratch[0..5]  = [100, 0, 0, 100, 0, 100]
//     gScratch[1] = (20, 0, 0)   hScratch[6..11] = [200, 0, 0, 200, 0, 200]
//
// Expected (after one spring gather):
//   v0 role 0, sign=+1:  g=(10+1, 0, 0)=(11, 0, 0),
//                        h=[100+1, 0, 0, 100, 0, 100]=[101, 0, 0, 100, 0, 100]
//   v1 role 1, sign=−1:  g=(20−1, 0, 0)=(19, 0, 0),
//                        h=[200+1, 0, 0, 200, 0, 200]=[201, 0, 0, 200, 0, 200]

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <vector>

#include "vbd_gather_spring_emit.cpp"

static constexpr double kBudgetSeconds = 5.0;

int main() {
    const auto t0 = std::chrono::steady_clock::now();

    constexpr uint32_t N_VERTS    = 2;
    constexpr uint32_t GROUP_SIZE = 64;

    // Spring outputs (from a prior spring_force dispatch). Padded to
    // GROUP_SIZE but only spring 0 is real.
    std::vector<Vector<float, 3>> springGradA(GROUP_SIZE, Vector<float, 3>(0.0f));
    std::vector<float>            springHess(6 * GROUP_SIZE, 0.0f);
    springGradA[0] = Vector<float, 3>(1.0f, 0.0f, 0.0f);
    springHess[0]  = 1.0f;  // Hxx of spring 0; others stay 0

    // CSR adjacency for verts 0 and 1. Padded verts 2..GROUP_SIZE point
    // at offset 2 so their loop range is empty (no gather happens).
    std::vector<uint32_t> vertSpringOffset(GROUP_SIZE + 1, 2u);
    vertSpringOffset[0] = 0u;
    vertSpringOffset[1] = 1u;
    vertSpringOffset[2] = 2u;
    std::vector<uint32_t> vertSpringIdx  = { 0u, 0u };
    std::vector<uint32_t> vertSpringRole = { 0u, 1u };

    // Initial scratch (would normally come from vbd_init).
    std::vector<Vector<float, 3>> gScratch(GROUP_SIZE, Vector<float, 3>(0.0f));
    std::vector<float>            hScratch(6 * GROUP_SIZE, 0.0f);
    gScratch[0] = Vector<float, 3>(10.0f, 0.0f, 0.0f);
    gScratch[1] = Vector<float, 3>(20.0f, 0.0f, 0.0f);
    hScratch[0]  = 100.0f; hScratch[3]  = 100.0f; hScratch[5]  = 100.0f;
    hScratch[6]  = 200.0f; hScratch[9]  = 200.0f; hScratch[11] = 200.0f;

    // Identity vertex permutation + zero color offset — same lane-to-vert
    // mapping as the pre-coloring version, so reference math is bit-exact.
    std::vector<uint32_t> vertPerm(GROUP_SIZE);
    for (uint32_t i = 0; i < GROUP_SIZE; ++i) vertPerm[i] = i;
    VbdGatherSpringParams_0 paramsBuf{};
    paramsBuf.colorOffset_0 = 0u;

    GlobalParams_0 gp{};
    gp.springGradA_0.data      = springGradA.data();      gp.springGradA_0.count      = springGradA.size();
    gp.springHess_0.data       = springHess.data();       gp.springHess_0.count       = springHess.size();
    gp.vertSpringOffset_0.data = vertSpringOffset.data(); gp.vertSpringOffset_0.count = vertSpringOffset.size();
    gp.vertSpringIdx_0.data    = vertSpringIdx.data();    gp.vertSpringIdx_0.count    = vertSpringIdx.size();
    gp.vertSpringRole_0.data   = vertSpringRole.data();   gp.vertSpringRole_0.count   = vertSpringRole.size();
    gp.gScratch_0.data         = gScratch.data();         gp.gScratch_0.count         = gScratch.size();
    gp.hScratch_0.data         = hScratch.data();         gp.hScratch_0.count         = hScratch.size();
    gp.vertPerm_0.data         = vertPerm.data();         gp.vertPerm_0.count         = vertPerm.size();
    gp.params_0                = &paramsBuf;

    ComputeVaryingInput vi{};
    vi.startGroupID = uint3(0, 0, 0);
    vi.endGroupID   = uint3(1, 1, 1);
    main_0(&vi, nullptr, &gp);

    const float expected_g[N_VERTS][3] = {
        {11.0f, 0.0f, 0.0f},
        {19.0f, 0.0f, 0.0f},
    };
    const float expected_h[N_VERTS][6] = {
        {101.0f, 0.0f, 0.0f, 100.0f, 0.0f, 100.0f},
        {201.0f, 0.0f, 0.0f, 200.0f, 0.0f, 200.0f},
    };

    int   fails        = 0;
    float max_abs_diff = 0.0f;
    auto check = [&](float got, float ref, const char* label, uint32_t v, int axis) {
        const float d = std::fabs(got - ref);
        if (d > max_abs_diff) max_abs_diff = d;
        if (d > 1e-6f) {
            if (fails < 5) {
                std::fprintf(stderr,
                    "vbd_gather_spring %s mismatch at v=%u, axis=%d: got %g, expected %g (diff %g)\n",
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
        std::fprintf(stderr, "vbd_gather_spring: TIMEOUT — %.3fs > %.1fs\n",
                     elapsed, kBudgetSeconds);
        return 2;
    }
    if (fails == 0) {
        std::printf("vbd_gather_spring: %u/%u verts OK (max_abs_diff=%g, %.1fms)\n",
                    N_VERTS, N_VERTS, max_abs_diff, elapsed * 1000.0);
        return 0;
    }
    std::fprintf(stderr, "vbd_gather_spring: %d FAIL out of %u checks\n", fails, N_VERTS * 9u);
    return 1;
}
