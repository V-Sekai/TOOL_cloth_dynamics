// Real-input validator for Cloth.SlangCodegen.VbdGatherTriangle —
// triangle-membrane gather (K=3) in the AVBD vertex-block-update pipeline.
//
// Per vertex v, loops over the CSR adjacency slice and accumulates each
// incident triangle's per-corner force + scalar Hessian (at slot
// 3·c + r) into per-vertex scratch.
//
// Test fixture (3 verts, 1 triangle T0=(0,1,2)):
//   triGrad[3*0+0..2] = [(1,0,0), (0,1,0), (0,0,1)]
//   triHessScalar[3*0+0..2] = [3, 4, 5]
//
//   CSR (from AdjacencyKwise.build 3 [[0,1,2]]):
//     vertTriOffset = [0, 1, 2, 3]
//     vertTriIdx    = [0, 0, 0]
//     vertTriRole   = [0, 1, 2]
//
//   Initial scratch all zero.
//
// Expected:
//   v0:  g=(1,0,0), h=[3, 0, 0, 3, 0, 3]   (role 0 picks slot 0)
//   v1:  g=(0,1,0), h=[4, 0, 0, 4, 0, 4]   (role 1 picks slot 1)
//   v2:  g=(0,0,1), h=[5, 0, 0, 5, 0, 5]   (role 2 picks slot 2)

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <vector>

#include "vbd_gather_triangle_emit.cpp"

static constexpr double kBudgetSeconds = 5.0;

int main() {
    const auto t0 = std::chrono::steady_clock::now();

    constexpr uint32_t N_VERTS    = 3;
    constexpr uint32_t GROUP_SIZE = 64;

    std::vector<Vector<float, 3>> triGrad(3 * GROUP_SIZE, Vector<float, 3>(0.0f));
    std::vector<float>            triHessScalar(3 * GROUP_SIZE, 0.0f);
    triGrad[0] = Vector<float, 3>(1.0f, 0.0f, 0.0f);
    triGrad[1] = Vector<float, 3>(0.0f, 1.0f, 0.0f);
    triGrad[2] = Vector<float, 3>(0.0f, 0.0f, 1.0f);
    triHessScalar[0] = 3.0f;
    triHessScalar[1] = 4.0f;
    triHessScalar[2] = 5.0f;

    std::vector<uint32_t> vertTriOffset(GROUP_SIZE + 1, 3u);
    vertTriOffset[0] = 0u;
    vertTriOffset[1] = 1u;
    vertTriOffset[2] = 2u;
    vertTriOffset[3] = 3u;
    std::vector<uint32_t> vertTriIdx  = { 0u, 0u, 0u };
    std::vector<uint32_t> vertTriRole = { 0u, 1u, 2u };

    std::vector<Vector<float, 3>> gScratch(GROUP_SIZE, Vector<float, 3>(0.0f));
    std::vector<float>            hScratch(6 * GROUP_SIZE, 0.0f);

    GlobalParams_0 gp{};
    gp.triGrad_0.data       = triGrad.data();       gp.triGrad_0.count       = triGrad.size();
    gp.triHessScalar_0.data = triHessScalar.data(); gp.triHessScalar_0.count = triHessScalar.size();
    gp.vertTriOffset_0.data = vertTriOffset.data(); gp.vertTriOffset_0.count = vertTriOffset.size();
    gp.vertTriIdx_0.data    = vertTriIdx.data();    gp.vertTriIdx_0.count    = vertTriIdx.size();
    gp.vertTriRole_0.data   = vertTriRole.data();   gp.vertTriRole_0.count   = vertTriRole.size();
    gp.gScratch_0.data      = gScratch.data();      gp.gScratch_0.count      = gScratch.size();
    gp.hScratch_0.data      = hScratch.data();      gp.hScratch_0.count      = hScratch.size();

    ComputeVaryingInput vi{};
    vi.startGroupID = uint3(0, 0, 0);
    vi.endGroupID   = uint3(1, 1, 1);
    main_0(&vi, nullptr, &gp);

    const float expected_g[N_VERTS][3] = {
        {1.0f, 0.0f, 0.0f},
        {0.0f, 1.0f, 0.0f},
        {0.0f, 0.0f, 1.0f},
    };
    const float expected_h[N_VERTS][6] = {
        {3.0f, 0.0f, 0.0f, 3.0f, 0.0f, 3.0f},
        {4.0f, 0.0f, 0.0f, 4.0f, 0.0f, 4.0f},
        {5.0f, 0.0f, 0.0f, 5.0f, 0.0f, 5.0f},
    };

    int   fails        = 0;
    float max_abs_diff = 0.0f;
    auto check = [&](float got, float ref, const char* label, uint32_t v, int axis) {
        const float d = std::fabs(got - ref);
        if (d > max_abs_diff) max_abs_diff = d;
        if (d > 1e-6f) {
            if (fails < 5) {
                std::fprintf(stderr,
                    "vbd_gather_triangle %s mismatch at v=%u, axis=%d: got %g, expected %g (diff %g)\n",
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
        std::fprintf(stderr, "vbd_gather_triangle: TIMEOUT — %.3fs > %.1fs\n",
                     elapsed, kBudgetSeconds);
        return 2;
    }
    if (fails == 0) {
        std::printf("vbd_gather_triangle: %u/%u verts OK (max_abs_diff=%g, %.1fms)\n",
                    N_VERTS, N_VERTS, max_abs_diff, elapsed * 1000.0);
        return 0;
    }
    std::fprintf(stderr, "vbd_gather_triangle: %d FAIL out of %u checks\n", fails, N_VERTS * 9u);
    return 1;
}
