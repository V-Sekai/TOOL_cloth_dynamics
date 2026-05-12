// Real-input validator for Cloth.SlangCodegen.VbdGatherAttachment —
// attachment gather in the AVBD vertex-block-update pipeline.
//
// Per vertex v, loops over the CSR adjacency slice and accumulates
// every incident attachment's force into gScratch[v] and scalar
// Hessian (k·I) into the three diagonal entries of hScratch.
//
// Test fixture (3 verts, 2 attachments):
//   attach 0 pins v0:  gradV=(1, 2, 3),   hessScalar=5
//   attach 1 pins v2:  gradV=(-1, 0, 0),  hessScalar=2
//   v1 has no attachment.
//
//   CSR (from AdjacencyKwise.build 3 [[0], [2]]):
//     vertAttachOffset = [0, 1, 1, 2]
//     vertAttachIdx    = [0, 1]
//
//   Initial scratch (would normally come from vbd_init / earlier gathers):
//     g[0] = (0, 0, 0)   h[0..5]  = [10, 0, 0, 10, 0, 10]
//     g[1] = (0, 0, 0)   h[6..11] = [10, 0, 0, 10, 0, 10]
//     g[2] = (0, 0, 0)   h[12..17]= [10, 0, 0, 10, 0, 10]
//
// Expected:
//   v0:  g=(1, 2, 3),    h=[15, 0, 0, 15, 0, 15]   (+5 on Hxx/Hyy/Hzz)
//   v1:  unchanged       h unchanged
//   v2:  g=(-1, 0, 0),   h=[12, 0, 0, 12, 0, 12]   (+2 on Hxx/Hyy/Hzz)

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <vector>

#include "vbd_gather_attachment_emit.cpp"

static constexpr double kBudgetSeconds = 5.0;

int main() {
    const auto t0 = std::chrono::steady_clock::now();

    constexpr uint32_t N_VERTS    = 3;
    constexpr uint32_t GROUP_SIZE = 64;

    std::vector<Vector<float, 3>> attachGradV(GROUP_SIZE, Vector<float, 3>(0.0f));
    std::vector<float>            attachHessScalar(GROUP_SIZE, 0.0f);
    attachGradV[0]      = Vector<float, 3>(1.0f, 2.0f, 3.0f);
    attachHessScalar[0] = 5.0f;
    attachGradV[1]      = Vector<float, 3>(-1.0f, 0.0f, 0.0f);
    attachHessScalar[1] = 2.0f;

    std::vector<uint32_t> vertAttachOffset(GROUP_SIZE + 1, 2u);
    vertAttachOffset[0] = 0u;
    vertAttachOffset[1] = 1u;
    vertAttachOffset[2] = 1u;
    vertAttachOffset[3] = 2u;
    std::vector<uint32_t> vertAttachIdx = { 0u, 1u };

    std::vector<Vector<float, 3>> gScratch(GROUP_SIZE, Vector<float, 3>(0.0f));
    std::vector<float>            hScratch(6 * GROUP_SIZE, 0.0f);
    for (uint32_t v = 0; v < N_VERTS; ++v) {
        hScratch[6*v + 0] = 10.0f;  // Hxx
        hScratch[6*v + 3] = 10.0f;  // Hyy
        hScratch[6*v + 5] = 10.0f;  // Hzz
    }

    GlobalParams_0 gp{};
    gp.attachGradV_0.data      = attachGradV.data();      gp.attachGradV_0.count      = attachGradV.size();
    gp.attachHessScalar_0.data = attachHessScalar.data(); gp.attachHessScalar_0.count = attachHessScalar.size();
    gp.vertAttachOffset_0.data = vertAttachOffset.data(); gp.vertAttachOffset_0.count = vertAttachOffset.size();
    gp.vertAttachIdx_0.data    = vertAttachIdx.data();    gp.vertAttachIdx_0.count    = vertAttachIdx.size();
    gp.gScratch_0.data         = gScratch.data();         gp.gScratch_0.count         = gScratch.size();
    gp.hScratch_0.data         = hScratch.data();         gp.hScratch_0.count         = hScratch.size();

    ComputeVaryingInput vi{};
    vi.startGroupID = uint3(0, 0, 0);
    vi.endGroupID   = uint3(1, 1, 1);
    main_0(&vi, nullptr, &gp);

    const float expected_g[N_VERTS][3] = {
        { 1.0f,  2.0f, 3.0f},
        { 0.0f,  0.0f, 0.0f},
        {-1.0f,  0.0f, 0.0f},
    };
    const float expected_h[N_VERTS][6] = {
        {15.0f, 0.0f, 0.0f, 15.0f, 0.0f, 15.0f},
        {10.0f, 0.0f, 0.0f, 10.0f, 0.0f, 10.0f},
        {12.0f, 0.0f, 0.0f, 12.0f, 0.0f, 12.0f},
    };

    int   fails        = 0;
    float max_abs_diff = 0.0f;
    auto check = [&](float got, float ref, const char* label, uint32_t v, int axis) {
        const float d = std::fabs(got - ref);
        if (d > max_abs_diff) max_abs_diff = d;
        if (d > 1e-6f) {
            if (fails < 5) {
                std::fprintf(stderr,
                    "vbd_gather_attachment %s mismatch at v=%u, axis=%d: got %g, expected %g (diff %g)\n",
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
        std::fprintf(stderr, "vbd_gather_attachment: TIMEOUT — %.3fs > %.1fs\n",
                     elapsed, kBudgetSeconds);
        return 2;
    }
    if (fails == 0) {
        std::printf("vbd_gather_attachment: %u/%u verts OK (max_abs_diff=%g, %.1fms)\n",
                    N_VERTS, N_VERTS, max_abs_diff, elapsed * 1000.0);
        return 0;
    }
    std::fprintf(stderr, "vbd_gather_attachment: %d FAIL out of %u checks\n", fails, N_VERTS * 9u);
    return 1;
}
