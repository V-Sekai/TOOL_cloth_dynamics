// Perf baseline: time each of the five hottest kernels on a
// realistic-N input through slangc-cpp's sequential simulator.
//
// **What this measures (Tier 2 baseline).** slangc-cpp emits each
// kernel as `main_0(varyingInput, params, globals)` that loops over
// thread groups in plain C++ — single-threaded, no SIMD pragmas, no
// vectorisation hints. So the numbers below are *sequential CPU
// runtime of the kernel's algorithmic work*. They establish:
//
//   1. A regression floor — any new PR that doubles a kernel's
//      runtime here will probably also do so on GPU.
//   2. A relative cost ranking — spmv vs saxpby vs dot_reduce vs
//      assemble_b at one fixed N. Useful for budget planning.
//   3. A scaling sanity check — at N = 30 000 the bind-time
//      structures (rowPtr, ctxStart, …) actually fit and the
//      kernels actually complete without time-outing.
//
// **What this does NOT measure.** GPU performance. Memory
// bandwidth. Wavefront-level cost. Coalesced vs scattered access
// penalties. Cache pressure beyond what L1/L2 already absorb at
// these tiny working sets. None of those are reachable through
// the slangc-cpp target — Tier 3 (Metal/Vulkan dispatch) is the
// only path to numbers that predict shipping perf.
//
// Five kernels, sized for ~10k-vertex cloth (= 30 000 scalar dims):
//
//   spring_project       N_springs = 30 000
//   saxpby               N = 30 000
//   spmv                 N_rows = 30 000, nnz ≈ 210 000
//   dot_reduce_serial    N = 30 000
//   assemble_b           V = 10 000, incidences = 30 000

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <vector>

#include "slang-cpp-prelude.h"

namespace pf_spring {
    #pragma push_macro("SLANG_PRELUDE_EXTERN_C")
    #pragma push_macro("SLANG_PRELUDE_EXTERN_C_START")
    #pragma push_macro("SLANG_PRELUDE_EXTERN_C_END")
    #undef  SLANG_PRELUDE_EXTERN_C
    #undef  SLANG_PRELUDE_EXTERN_C_START
    #undef  SLANG_PRELUDE_EXTERN_C_END
    #define SLANG_PRELUDE_EXTERN_C
    #define SLANG_PRELUDE_EXTERN_C_START
    #define SLANG_PRELUDE_EXTERN_C_END
    #include "spring_project_emit.cpp"
    #pragma pop_macro("SLANG_PRELUDE_EXTERN_C")
    #pragma pop_macro("SLANG_PRELUDE_EXTERN_C_START")
    #pragma pop_macro("SLANG_PRELUDE_EXTERN_C_END")
}

namespace pf_saxpby {
    #pragma push_macro("SLANG_PRELUDE_EXTERN_C")
    #pragma push_macro("SLANG_PRELUDE_EXTERN_C_START")
    #pragma push_macro("SLANG_PRELUDE_EXTERN_C_END")
    #undef  SLANG_PRELUDE_EXTERN_C
    #undef  SLANG_PRELUDE_EXTERN_C_START
    #undef  SLANG_PRELUDE_EXTERN_C_END
    #define SLANG_PRELUDE_EXTERN_C
    #define SLANG_PRELUDE_EXTERN_C_START
    #define SLANG_PRELUDE_EXTERN_C_END
    #include "saxpby_emit.cpp"
    #pragma pop_macro("SLANG_PRELUDE_EXTERN_C")
    #pragma pop_macro("SLANG_PRELUDE_EXTERN_C_START")
    #pragma pop_macro("SLANG_PRELUDE_EXTERN_C_END")
}

namespace pf_spmv {
    #pragma push_macro("SLANG_PRELUDE_EXTERN_C")
    #pragma push_macro("SLANG_PRELUDE_EXTERN_C_START")
    #pragma push_macro("SLANG_PRELUDE_EXTERN_C_END")
    #undef  SLANG_PRELUDE_EXTERN_C
    #undef  SLANG_PRELUDE_EXTERN_C_START
    #undef  SLANG_PRELUDE_EXTERN_C_END
    #define SLANG_PRELUDE_EXTERN_C
    #define SLANG_PRELUDE_EXTERN_C_START
    #define SLANG_PRELUDE_EXTERN_C_END
    #include "spmv_emit.cpp"
    #pragma pop_macro("SLANG_PRELUDE_EXTERN_C")
    #pragma pop_macro("SLANG_PRELUDE_EXTERN_C_START")
    #pragma pop_macro("SLANG_PRELUDE_EXTERN_C_END")
}

namespace pf_dot {
    #pragma push_macro("SLANG_PRELUDE_EXTERN_C")
    #pragma push_macro("SLANG_PRELUDE_EXTERN_C_START")
    #pragma push_macro("SLANG_PRELUDE_EXTERN_C_END")
    #undef  SLANG_PRELUDE_EXTERN_C
    #undef  SLANG_PRELUDE_EXTERN_C_START
    #undef  SLANG_PRELUDE_EXTERN_C_END
    #define SLANG_PRELUDE_EXTERN_C
    #define SLANG_PRELUDE_EXTERN_C_START
    #define SLANG_PRELUDE_EXTERN_C_END
    #include "dot_reduce_serial_emit.cpp"
    #pragma pop_macro("SLANG_PRELUDE_EXTERN_C")
    #pragma pop_macro("SLANG_PRELUDE_EXTERN_C_START")
    #pragma pop_macro("SLANG_PRELUDE_EXTERN_C_END")
}

namespace pf_ab {
    #pragma push_macro("SLANG_PRELUDE_EXTERN_C")
    #pragma push_macro("SLANG_PRELUDE_EXTERN_C_START")
    #pragma push_macro("SLANG_PRELUDE_EXTERN_C_END")
    #undef  SLANG_PRELUDE_EXTERN_C
    #undef  SLANG_PRELUDE_EXTERN_C_START
    #undef  SLANG_PRELUDE_EXTERN_C_END
    #define SLANG_PRELUDE_EXTERN_C
    #define SLANG_PRELUDE_EXTERN_C_START
    #define SLANG_PRELUDE_EXTERN_C_END
    #include "assemble_b_emit.cpp"
    #pragma pop_macro("SLANG_PRELUDE_EXTERN_C")
    #pragma pop_macro("SLANG_PRELUDE_EXTERN_C_START")
    #pragma pop_macro("SLANG_PRELUDE_EXTERN_C_END")
}

static constexpr double kBudgetSeconds = 30.0;

// ---- Dispatch-shape helpers --------------------------------------------
static ComputeVaryingInput viOver(uint32_t numElements, uint32_t threadsPerGroup) {
    ComputeVaryingInput vi{};
    const uint32_t nGroups = (numElements + threadsPerGroup - 1) / threadsPerGroup;
    vi.startGroupID = uint3(0, 0, 0);
    vi.endGroupID   = uint3(nGroups, 1, 1);
    return vi;
}

// Volatile sink: prevents clang -O2 from dead-code-eliminating the
// kernel dispatch on the grounds that its outputs are never read.
// Each bench function dumps a checksum here after timing the loop;
// main() then prints all checksums so the compiler can't argue the
// stores are dead.
static volatile float g_dce_sink = 0.0f;

template <typename F>
static double timeMs(F&& fn, int reps) {
    const auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < reps; ++i) fn();
    const auto t1 = std::chrono::steady_clock::now();
    return std::chrono::duration<double, std::milli>(t1 - t0).count();
}

// ---- Per-kernel benches ------------------------------------------------

static double benchSpringProject(uint32_t N, int reps) {
    std::vector<Vector<float, 3>> positions(N + 1, Vector<float, 3>(0.0f));
    for (uint32_t i = 0; i < N + 1; ++i)
        positions[i] = Vector<float, 3>(float(i), float((i * 7) % 11), float((i * 13) % 19));
    std::vector<Vector<float, 3>> out(N, Vector<float, 3>(0.0f));
    std::vector<uint32_t> p1(N), p2(N);
    std::vector<float>    rest(N, 1.0f), sqrtW(N, 1.0f);
    for (uint32_t i = 0; i < N; ++i) { p1[i] = i; p2[i] = i + 1; }

    pf_spring::GlobalParams_0 gp{};
    gp.positions_0.data  = positions.data();  gp.positions_0.count  = positions.size();
    gp.projected_0.data  = out.data();        gp.projected_0.count  = out.size();
    gp.p1Idx_0.data      = p1.data();         gp.p1Idx_0.count      = p1.size();
    gp.p2Idx_0.data      = p2.data();         gp.p2Idx_0.count      = p2.size();
    gp.restLen_0.data    = rest.data();       gp.restLen_0.count    = rest.size();
    gp.sqrtWeight_0.data = sqrtW.data();      gp.sqrtWeight_0.count = sqrtW.size();

    ComputeVaryingInput vi = viOver(N, 64);
    const double ms = timeMs([&]{ pf_spring::main_0(&vi, nullptr, &gp); }, reps);
    g_dce_sink += out[N / 2][0] + out[N / 2][1] + out[N / 2][2];
    return ms;
}

static double benchSaxpby(uint32_t N, int reps) {
    pf_saxpby::SaxpbyParams_0 params{N, 1.5f, -0.25f};
    std::vector<float> x(N, 1.0f), y(N, 2.0f), dst(N, 0.0f);

    pf_saxpby::GlobalParams_0 gp{};
    gp.params_0   = &params;
    gp.x_0.data   = x.data();   gp.x_0.count   = N;
    gp.y_0.data   = y.data();   gp.y_0.count   = N;
    gp.dst_0.data = dst.data(); gp.dst_0.count = N;

    ComputeVaryingInput vi = viOver(N, 256);
    const double ms = timeMs([&]{ pf_saxpby::main_0(&vi, nullptr, &gp); }, reps);
    g_dce_sink += dst[N / 2];
    return ms;
}

static double benchSpmv(uint32_t N, int reps, uint32_t& nnz_out) {
    // Synthetic tridiagonal-ish SPD-ish CSR. Each row has ~3 nonzeros
    // except rows near the boundary (2). nnz ≈ 3N − 2.
    std::vector<int32_t> rowPtr(N + 1, 0);
    std::vector<int32_t> colIdx;
    std::vector<float>   values;
    colIdx.reserve(3 * N);
    values.reserve(3 * N);
    for (uint32_t i = 0; i < N; ++i) {
        rowPtr[i] = int32_t(colIdx.size());
        if (i > 0)       { colIdx.push_back(int32_t(i - 1)); values.push_back(-1.0f); }
                          colIdx.push_back(int32_t(i));     values.push_back( 4.0f);
        if (i + 1 < N)   { colIdx.push_back(int32_t(i + 1)); values.push_back(-1.0f); }
    }
    rowPtr[N] = int32_t(colIdx.size());
    nnz_out = uint32_t(colIdx.size());

    std::vector<float> x(N), y(N, 0.0f);
    for (uint32_t i = 0; i < N; ++i) x[i] = float((i % 13) - 6);

    pf_spmv::SpmvParams_0 params{N};
    pf_spmv::GlobalParams_0 gp{};
    gp.params_0      = &params;
    gp.rowPtr_0.data = rowPtr.data(); gp.rowPtr_0.count = rowPtr.size();
    gp.colIdx_0.data = colIdx.data(); gp.colIdx_0.count = colIdx.size();
    gp.values_0.data = values.data(); gp.values_0.count = values.size();
    gp.x_0.data      = x.data();      gp.x_0.count      = x.size();
    gp.y_0.data      = y.data();      gp.y_0.count      = y.size();

    ComputeVaryingInput vi = viOver(N, 256);
    const double ms = timeMs([&]{ pf_spmv::main_0(&vi, nullptr, &gp); }, reps);
    g_dce_sink += y[N / 2];
    return ms;
}

static double benchDotReduceSerial(uint32_t N, int reps) {
    pf_dot::DotReduceSerialParams_0 params{N};
    std::vector<float> a(N), b(N);
    for (uint32_t i = 0; i < N; ++i) {
        a[i] = float((i % 17) - 8);
        b[i] = float((i % 23) - 11);
    }
    float dst[2] = {0.0f, 0.0f};

    pf_dot::GlobalParams_0 gp{};
    gp.params_0   = &params;
    gp.a_0.data   = a.data();   gp.a_0.count   = N;
    gp.b_0.data   = b.data();   gp.b_0.count   = N;
    gp.dst_0.data = dst;        gp.dst_0.count = 2;

    // dot_reduce_serial is [numthreads(1, 1, 1)] — exactly one thread.
    ComputeVaryingInput vi = viOver(1, 1);
    const double ms = timeMs([&]{ pf_dot::main_0(&vi, nullptr, &gp); }, reps);
    g_dce_sink += dst[0] + dst[1];
    return ms;
}

static double benchAssembleB(uint32_t V, uint32_t incidencesPerVert,
                             int reps, uint32_t& slots_out) {
    const uint32_t totalInc = V * incidencesPerVert;
    const uint32_t slots    = V;
    slots_out = slots;

    std::vector<float> s(3 * V), mass(V, 1.0f), b(3 * V, 0.0f);
    std::vector<float> projections(3 * slots, 1.0f);
    std::vector<uint32_t> ctxStart(V + 1, 0u);
    std::vector<uint32_t> ctxSlot(totalInc);
    std::vector<float>    ctxWeight(totalInc);

    for (uint32_t v = 0; v < V; ++v) {
        ctxStart[v] = v * incidencesPerVert;
        s[3*v + 0] = float(v); s[3*v + 1] = 0.0f; s[3*v + 2] = 0.0f;
        for (uint32_t k = 0; k < incidencesPerVert; ++k) {
            const uint32_t idx = v * incidencesPerVert + k;
            ctxSlot[idx]   = (v + k) % slots;
            ctxWeight[idx] = ((k & 1) ? -1.0f : +1.0f);
        }
    }
    ctxStart[V] = totalInc;

    pf_ab::AssembleBParams_0 params{V};
    pf_ab::GlobalParams_0 gp{};
    gp.params_0           = &params;
    gp.s_0.data           = s.data();           gp.s_0.count           = s.size();
    gp.mass_0.data        = mass.data();        gp.mass_0.count        = mass.size();
    gp.projections_0.data = projections.data(); gp.projections_0.count = projections.size();
    gp.ctxStart_0.data    = ctxStart.data();    gp.ctxStart_0.count    = ctxStart.size();
    gp.ctxSlot_0.data     = ctxSlot.data();     gp.ctxSlot_0.count     = ctxSlot.size();
    gp.ctxWeight_0.data   = ctxWeight.data();   gp.ctxWeight_0.count   = ctxWeight.size();
    gp.b_0.data           = b.data();           gp.b_0.count           = b.size();

    ComputeVaryingInput vi = viOver(V, 64);
    const double ms = timeMs([&]{ pf_ab::main_0(&vi, nullptr, &gp); }, reps);
    g_dce_sink += b[3 * (V / 2)] + b[3 * (V / 2) + 1] + b[3 * (V / 2) + 2];
    return ms;
}

int main() {
    const auto wallStart = std::chrono::steady_clock::now();

    constexpr uint32_t N         = 30000;     // = 3 · 10 000 verts
    constexpr uint32_t V_AB      = 10000;
    constexpr uint32_t INC_PER_V = 3;          // ≈ 3 incidences per vert (avg)
    constexpr int      REPS_FAST = 200;
    constexpr int      REPS_DOT  = 50;         // dot_reduce_serial is single-thread; 50 reps still sub-second

    std::printf("\nperf_baseline (slangc-cpp; sequential CPU; NOT GPU-predictive)\n");
    std::printf("=================================================================\n");
    std::printf("%-22s %-12s %-12s %-14s %s\n",
                "kernel", "N", "reps", "ms/invoke", "throughput");
    std::printf("-----------------------------------------------------------------\n");

    {
        const double ms = benchSpringProject(N, REPS_FAST);
        const double per = ms / REPS_FAST;
        std::printf("%-22s %-12u %-12d %-14.4f %.1f M springs/s\n",
                    "spring_project", N, REPS_FAST, per, (double(N) / per) / 1e3);
    }
    {
        const double ms = benchSaxpby(N, REPS_FAST);
        const double per = ms / REPS_FAST;
        std::printf("%-22s %-12u %-12d %-14.4f %.1f M elems/s\n",
                    "saxpby", N, REPS_FAST, per, (double(N) / per) / 1e3);
    }
    {
        uint32_t nnz = 0;
        const double ms = benchSpmv(N, REPS_FAST, nnz);
        const double per = ms / REPS_FAST;
        std::printf("%-22s %-12u %-12d %-14.4f %.1f M nnz/s  (nnz=%u)\n",
                    "spmv", N, REPS_FAST, per, (double(nnz) / per) / 1e3, nnz);
    }
    {
        const double ms = benchDotReduceSerial(N, REPS_DOT);
        const double per = ms / REPS_DOT;
        std::printf("%-22s %-12u %-12d %-14.4f %.1f M elems/s  [single-thread]\n",
                    "dot_reduce_serial", N, REPS_DOT, per, (double(N) / per) / 1e3);
    }
    {
        uint32_t slots = 0;
        const double ms = benchAssembleB(V_AB, INC_PER_V, REPS_FAST, slots);
        const double per = ms / REPS_FAST;
        const uint32_t inc = V_AB * INC_PER_V;
        std::printf("%-22s %-12u %-12d %-14.4f %.1f M inc/s  (V=%u, inc=%u, slots=%u)\n",
                    "assemble_b", V_AB, REPS_FAST, per,
                    (double(inc) / per) / 1e3, V_AB, inc, slots);
    }

    std::printf("-----------------------------------------------------------------\n");
    const double wall_ms =
        std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - wallStart).count();
    std::printf("total wall: %.1f ms   (dce_sink=%g — read to defeat optimiser)\n\n",
                wall_ms, double(g_dce_sink));

    if (wall_ms / 1000.0 > kBudgetSeconds) {
        std::fprintf(stderr, "perf_baseline: TIMEOUT — %.3fs > %.1fs\n",
                     wall_ms / 1000.0, kBudgetSeconds);
        return 2;
    }
    return 0;
}
