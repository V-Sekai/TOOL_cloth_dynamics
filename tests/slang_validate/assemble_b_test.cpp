// Real-input validator for Cloth.SlangCodegen.AssembleB — the PD
// right-hand-side gather kernel.
//
//   b[3v+d] = mass[v] * s[3v+d]
//           + Σ_{(slot, w) in incident[v]}  w * projections[3*slot + d]
//
// Test fixture: 2-vertex system with 1 spring + 1 attachment.
//
//   verts:   v0, v1
//   slot 0:  spring projection p_s = (1, 0, 0)
//             touches v0 with w = +1 (the +sqrtW endpoint)
//             touches v1 with w = −1 (the −sqrtW endpoint)
//   slot 1:  attachment projection p_a = (10, 0, 0)
//             touches v0 with w = +2 (the pinned vertex, sqrtW = 2)
//
//   predicted positions s = [0.5, 0, 0,  1.0, 0, 0]
//   diagonal mass       m = [1.0, 1.0]
//
// Expected:
//   b[v=0] = 1·(0.5, 0, 0) + (+1)·(1,0,0) + (+2)·(10,0,0) = (21.5, 0, 0)
//   b[v=1] = 1·(1.0, 0, 0) + (−1)·(1,0,0)                  = ( 0.0, 0, 0)
//
// CSR encoding:
//   ctxStart  = [0, 2, 3]
//   ctxSlot   = [0, 1, 0]
//   ctxWeight = [+1, +2, −1]

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>

#include "assemble_b_emit.cpp"

static constexpr double kBudgetSeconds = 5.0;

int main() {
    const auto t0 = std::chrono::steady_clock::now();

    constexpr uint32_t V          = 2;
    constexpr uint32_t SLOT_COUNT = 2;
    constexpr uint32_t INCIDENCES = 3;

    AssembleBParams_0 params{V};

    float s[3 * V] = {
        0.5f, 0.0f, 0.0f,    // v0
        1.0f, 0.0f, 0.0f,    // v1
    };
    float mass[V] = {1.0f, 1.0f};
    float projections[3 * SLOT_COUNT] = {
         1.0f, 0.0f, 0.0f,   // slot 0 — spring projection
        10.0f, 0.0f, 0.0f,   // slot 1 — attachment projection
    };
    uint32_t ctxStart[V + 1]   = {0u, 2u, 3u};
    uint32_t ctxSlot[INCIDENCES]   = {0u, 1u, 0u};
    float    ctxWeight[INCIDENCES] = {+1.0f, +2.0f, -1.0f};

    float b[3 * V] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

    GlobalParams_0 gp{};
    gp.params_0          = &params;
    gp.s_0.data          = s;            gp.s_0.count          = 3 * V;
    gp.mass_0.data       = mass;         gp.mass_0.count       = V;
    gp.projections_0.data = projections; gp.projections_0.count = 3 * SLOT_COUNT;
    gp.ctxStart_0.data   = ctxStart;     gp.ctxStart_0.count   = V + 1;
    gp.ctxSlot_0.data    = ctxSlot;      gp.ctxSlot_0.count    = INCIDENCES;
    gp.ctxWeight_0.data  = ctxWeight;    gp.ctxWeight_0.count  = INCIDENCES;
    gp.b_0.data          = b;            gp.b_0.count          = 3 * V;

    ComputeVaryingInput vi{};
    vi.startGroupID = uint3(0, 0, 0);
    vi.endGroupID   = uint3(1, 1, 1);
    main_0(&vi, nullptr, &gp);

    const float expected[3 * V] = {
        21.5f, 0.0f, 0.0f,    // v0: 0.5 + 1 + 20
         0.0f, 0.0f, 0.0f,    // v1: 1 - 1
    };

    int   fails        = 0;
    float max_abs_diff = 0.0f;
    for (uint32_t i = 0; i < 3 * V; ++i) {
        const float d = std::fabs(b[i] - expected[i]);
        if (d > max_abs_diff) max_abs_diff = d;
        if (d > 1e-6f) {
            if (fails < 5) {
                std::fprintf(stderr,
                    "assemble_b mismatch at i=%u: got %g, expected %g (diff %g)\n",
                    i, b[i], expected[i], d);
            }
            ++fails;
        }
    }

    const auto t1 = std::chrono::steady_clock::now();
    const double elapsed = std::chrono::duration<double>(t1 - t0).count();
    if (elapsed > kBudgetSeconds) {
        std::fprintf(stderr, "assemble_b: TIMEOUT — %.3fs > %.1fs\n",
                     elapsed, kBudgetSeconds);
        return 2;
    }
    if (fails == 0) {
        std::printf(
            "assemble_b: %u/%u OK (V=%u, slots=%u, incidences=%u, max_abs_diff=%g, %.1fms)\n",
            V, V, V, SLOT_COUNT, INCIDENCES, max_abs_diff, elapsed * 1000.0);
        return 0;
    }
    std::fprintf(stderr, "assemble_b: %d FAIL out of %u\n", fails, 3 * V);
    return 1;
}
