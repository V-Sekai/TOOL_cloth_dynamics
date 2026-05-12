// Real-input validator for Cloth.SlangCodegen.VbdGatherBending —
// dihedral-bending gather (K=4) in the AVBD vertex-block-update pipeline.
//
// Same shape as vbd_gather_triangle (#54) but K=4: the bending stencil
// touches 4 vertices and triangle_bending_force writes per-vertex outputs
// at slot 4·c + r.
//
// Test fixture (4 verts, 1 bending constraint B0=(0,1,2,3)):
//   bendGrad[4*0+0..3] = [(1,0,0), (0,1,0), (0,0,1), (1,1,1)]
//   bendHessScalar     = [2, 3, 5, 7]
//
//   CSR (from AdjacencyKwise.build 4 [[0,1,2,3]]):
//     vertBendOffset = [0, 1, 2, 3, 4]
//     vertBendIdx    = [0, 0, 0, 0]
//     vertBendRole   = [0, 1, 2, 3]
//
// Expected (initial scratch zero):
//   v0: g=(1,0,0), h_diag=2    (role 0 → slot 0)
//   v1: g=(0,1,0), h_diag=3    (role 1 → slot 1)
//   v2: g=(0,0,1), h_diag=5    (role 2 → slot 2)
//   v3: g=(1,1,1), h_diag=7    (role 3 → slot 3)

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <vector>

#include "vbd_gather_bending_emit.cpp"

static constexpr double kBudgetSeconds = 5.0;

int main() {
    const auto t0 = std::chrono::steady_clock::now();

    constexpr uint32_t N_VERTS    = 4;
    constexpr uint32_t GROUP_SIZE = 64;

    std::vector<Vector<float, 3>> bendGrad(4 * GROUP_SIZE, Vector<float, 3>(0.0f));
    std::vector<float>            bendHessScalar(4 * GROUP_SIZE, 0.0f);
    bendGrad[0] = Vector<float, 3>(1.0f, 0.0f, 0.0f);
    bendGrad[1] = Vector<float, 3>(0.0f, 1.0f, 0.0f);
    bendGrad[2] = Vector<float, 3>(0.0f, 0.0f, 1.0f);
    bendGrad[3] = Vector<float, 3>(1.0f, 1.0f, 1.0f);
    bendHessScalar[0] = 2.0f;
    bendHessScalar[1] = 3.0f;
    bendHessScalar[2] = 5.0f;
    bendHessScalar[3] = 7.0f;

    std::vector<uint32_t> vertBendOffset(GROUP_SIZE + 1, 4u);
    vertBendOffset[0] = 0u;
    vertBendOffset[1] = 1u;
    vertBendOffset[2] = 2u;
    vertBendOffset[3] = 3u;
    vertBendOffset[4] = 4u;
    std::vector<uint32_t> vertBendIdx  = { 0u, 0u, 0u, 0u };
    std::vector<uint32_t> vertBendRole = { 0u, 1u, 2u, 3u };

    std::vector<Vector<float, 3>> gScratch(GROUP_SIZE, Vector<float, 3>(0.0f));
    std::vector<float>            hScratch(6 * GROUP_SIZE, 0.0f);

    GlobalParams_0 gp{};
    gp.bendGrad_0.data       = bendGrad.data();       gp.bendGrad_0.count       = bendGrad.size();
    gp.bendHessScalar_0.data = bendHessScalar.data(); gp.bendHessScalar_0.count = bendHessScalar.size();
    gp.vertBendOffset_0.data = vertBendOffset.data(); gp.vertBendOffset_0.count = vertBendOffset.size();
    gp.vertBendIdx_0.data    = vertBendIdx.data();    gp.vertBendIdx_0.count    = vertBendIdx.size();
    gp.vertBendRole_0.data   = vertBendRole.data();   gp.vertBendRole_0.count   = vertBendRole.size();
    gp.gScratch_0.data       = gScratch.data();       gp.gScratch_0.count       = gScratch.size();
    gp.hScratch_0.data       = hScratch.data();       gp.hScratch_0.count       = hScratch.size();

    ComputeVaryingInput vi{};
    vi.startGroupID = uint3(0, 0, 0);
    vi.endGroupID   = uint3(1, 1, 1);
    main_0(&vi, nullptr, &gp);

    const float expected_g[N_VERTS][3] = {
        {1.0f, 0.0f, 0.0f},
        {0.0f, 1.0f, 0.0f},
        {0.0f, 0.0f, 1.0f},
        {1.0f, 1.0f, 1.0f},
    };
    const float expected_h[N_VERTS][6] = {
        {2.0f, 0.0f, 0.0f, 2.0f, 0.0f, 2.0f},
        {3.0f, 0.0f, 0.0f, 3.0f, 0.0f, 3.0f},
        {5.0f, 0.0f, 0.0f, 5.0f, 0.0f, 5.0f},
        {7.0f, 0.0f, 0.0f, 7.0f, 0.0f, 7.0f},
    };

    int   fails        = 0;
    float max_abs_diff = 0.0f;
    auto check = [&](float got, float ref, const char* label, uint32_t v, int axis) {
        const float d = std::fabs(got - ref);
        if (d > max_abs_diff) max_abs_diff = d;
        if (d > 1e-6f) {
            if (fails < 5) {
                std::fprintf(stderr,
                    "vbd_gather_bending %s mismatch at v=%u, axis=%d: got %g, expected %g (diff %g)\n",
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
        std::fprintf(stderr, "vbd_gather_bending: TIMEOUT — %.3fs > %.1fs\n",
                     elapsed, kBudgetSeconds);
        return 2;
    }
    if (fails == 0) {
        std::printf("vbd_gather_bending: %u/%u verts OK (max_abs_diff=%g, %.1fms)\n",
                    N_VERTS, N_VERTS, max_abs_diff, elapsed * 1000.0);
        return 0;
    }
    std::fprintf(stderr, "vbd_gather_bending: %d FAIL out of %u checks\n", fails, N_VERTS * 9u);
    return 1;
}
