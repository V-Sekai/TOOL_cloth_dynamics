// Tier-3 perf baseline on real GPU via Metal. Dispatches the same
// five hottest kernels as perf_baseline_test.cpp through Metal at the
// same N (30 000 for vector-shaped kernels, 10 000 verts for
// assemble_b), times each via CPU-side wall-clock around
// commandBuffer.waitUntilCompleted, and compares against the
// slangc-cpp numbers from #22.
//
// **Batched dispatch.** This harness batches all `reps` invocations
// into ONE command buffer with ONE commit + ONE waitUntilCompleted,
// so the ~100-200µs Metal submission round-trip is amortised across
// `reps` invocations. The per-invoke numbers below approach
// steady-state GPU compute throughput rather than dispatch latency
// (#23 measured the one-commit-per-invoke regime).
//
// **Warm-up.** A single un-timed dispatch is paid up-front so
// pipeline-state compile, page-in, and command-queue init don't
// pollute the steady-state measurement.
//
// Pipeline (per kernel):
//
//   .slang  -- slangc -target metal -->  build/<k>.metal
//   .metal  -- xcrun metal -c       -->  build/<k>.air
//   .air    -- xcrun metallib       -->  build/<k>.metallib
//   .metallib -- MTLDevice newLibraryWithURL -->  MTLLibrary
//   MTLLibrary -- newFunctionWithName "main_0" -->  MTLFunction
//   MTLFunction -- newComputePipelineState -->  MTLComputePipelineState
//
// Bindings follow Slang's [[vk::binding(N, 0)]] numbering:
//
//   saxpby:             0=params, 1=x, 2=y, 3=dst
//   spmv:               0=params, 1=rowPtr, 2=colIdx, 3=values, 4=x, 5=y
//   dot_reduce_serial:  0=params, 1=a, 2=b, 3=dst
//   assemble_b:         0=params, 1=s, 2=mass, 3=projections, 4=ctxStart,
//                       5=ctxSlot, 6=ctxWeight, 7=b
//   spring_project:     0=positions, 1=projected, 2=p1Idx, 3=p2Idx,
//                       4=restLen, 5=sqrtWeight  (no params buffer)
//
// Threadgroup sizes per kernel (from the Slang [numthreads(...)] attrs):
//
//   saxpby:             256
//   spmv:               256
//   dot_reduce_serial:  1     (single-threaded by design)
//   assemble_b:         64
//   spring_project:     64
//
// Notable: dot_reduce_serial dispatches a single thread group of one
// thread. The parallel dot_reduce isn't ported here yet — it would
// need its own metallib + a different parameter buffer.

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

// ---- Common helpers ---------------------------------------------------

static double seconds() {
    return std::chrono::duration<double>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}

static id<MTLLibrary> loadLib(id<MTLDevice> dev, const char* path) {
    NSError* err = nil;
    NSURL* url = [NSURL fileURLWithPath:[NSString stringWithUTF8String:path]];
    id<MTLLibrary> lib = [dev newLibraryWithURL:url error:&err];
    if (!lib) {
        std::fprintf(stderr, "newLibraryWithURL %s failed: %s\n",
                     path, err ? err.localizedDescription.UTF8String : "(null)");
    }
    return lib;
}

static id<MTLComputePipelineState> makePipe(id<MTLDevice> dev,
                                             id<MTLLibrary> lib,
                                             const char* fnName) {
    NSError* err = nil;
    id<MTLFunction> fn = [lib newFunctionWithName:[NSString stringWithUTF8String:fnName]];
    if (!fn) {
        std::fprintf(stderr, "newFunctionWithName %s failed (no such function)\n",
                     fnName);
        return nil;
    }
    id<MTLComputePipelineState> pipe = [dev newComputePipelineStateWithFunction:fn
                                                                          error:&err];
    if (!pipe) {
        std::fprintf(stderr, "newComputePipelineState failed: %s\n",
                     err ? err.localizedDescription.UTF8String : "(null)");
    }
    return pipe;
}

template <typename T>
static id<MTLBuffer> makeBuf(id<MTLDevice> dev, const std::vector<T>& v) {
    return [dev newBufferWithBytes:v.data()
                            length:v.size() * sizeof(T)
                           options:MTLResourceStorageModeShared];
}

template <typename T>
static id<MTLBuffer> makeBuf(id<MTLDevice> dev, const T& v) {
    return [dev newBufferWithBytes:&v
                            length:sizeof(T)
                           options:MTLResourceStorageModeShared];
}

static id<MTLBuffer> makeEmptyBuf(id<MTLDevice> dev, size_t bytes) {
    return [dev newBufferWithLength:bytes options:MTLResourceStorageModeShared];
}

// Batched dispatch: all `reps` invocations encoded into ONE command
// buffer with ONE commit + ONE waitUntilCompleted. This amortises the
// ~100–200 µs Metal command-submission round-trip across `reps`,
// approaching steady-state GPU throughput.
//
// Same encoder + same buffers across all reps: between dispatches the
// encoder serialises them on the resources, so each invocation
// actually runs (the GPU doesn't elide them as redundant). The first
// dispatch in the batch also pays a one-time pipeline-state-cache
// warm-up that we don't separate out.
static double timeReps(id<MTLCommandQueue> queue,
                       id<MTLComputePipelineState> pipe,
                       NSArray<id<MTLBuffer>>* buffers,
                       NSArray<NSNumber*>* offsets,
                       MTLSize gridSize,
                       MTLSize tgSize,
                       int reps) {
    // Warm-up dispatch: pays one-time costs (PSO compile, page-in,
    // command-queue init) outside the timing window.
    {
        id<MTLCommandBuffer> warm = [queue commandBuffer];
        id<MTLComputeCommandEncoder> wenc = [warm computeCommandEncoder];
        [wenc setComputePipelineState:pipe];
        for (NSUInteger i = 0; i < buffers.count; ++i) {
            [wenc setBuffer:buffers[i] offset:offsets[i].unsignedIntegerValue atIndex:i];
        }
        [wenc dispatchThreads:gridSize threadsPerThreadgroup:tgSize];
        [wenc endEncoding];
        [warm commit];
        [warm waitUntilCompleted];
    }

    id<MTLCommandBuffer> cb = [queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    [enc setComputePipelineState:pipe];
    for (NSUInteger i = 0; i < buffers.count; ++i) {
        [enc setBuffer:buffers[i] offset:offsets[i].unsignedIntegerValue atIndex:i];
    }
    const double t0 = seconds();
    for (int i = 0; i < reps; ++i) {
        [enc dispatchThreads:gridSize threadsPerThreadgroup:tgSize];
    }
    [enc endEncoding];
    [cb commit];
    [cb waitUntilCompleted];
    const double total = seconds() - t0;
    (void)reps;
    return total * 1000.0;
}

// Per-invocation timing kept for reference but no longer used in the
// main benches. (One commit per invoke is what #23 measured.)
static double timeRepsUnbatched(id<MTLCommandQueue> queue,
                                id<MTLComputePipelineState> pipe,
                                NSArray<id<MTLBuffer>>* buffers,
                                NSArray<NSNumber*>* offsets,
                                MTLSize gridSize,
                                MTLSize tgSize,
                                int reps) {
    double total = 0.0;
    for (int i = 0; i < reps; ++i) {
        id<MTLCommandBuffer> cb = [queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:pipe];
        for (NSUInteger j = 0; j < buffers.count; ++j) {
            [enc setBuffer:buffers[j] offset:offsets[j].unsignedIntegerValue atIndex:j];
        }
        [enc dispatchThreads:gridSize threadsPerThreadgroup:tgSize];
        [enc endEncoding];
        const double t0 = seconds();
        [cb commit];
        [cb waitUntilCompleted];
        total += seconds() - t0;
    }
    return total * 1000.0;
}

// ---- Per-kernel bench helpers -----------------------------------------

struct SaxpbyParams { uint32_t n; float alpha; float beta; };
struct SpmvParams   { uint32_t rows; };
struct DotParams    { uint32_t n; };
struct AssembleBParams { uint32_t numVerts; };

static double benchSaxpby(id<MTLDevice> dev, id<MTLCommandQueue> queue,
                          id<MTLLibrary> lib, uint32_t N, int reps) {
    id<MTLComputePipelineState> pipe = makePipe(dev, lib, "main_0");
    if (!pipe) return -1;

    SaxpbyParams p{N, 1.5f, -0.25f};
    std::vector<float> x(N, 1.0f), y(N, 2.0f), dst(N, 0.0f);

    id<MTLBuffer> bp   = makeBuf(dev, p);
    id<MTLBuffer> bx   = makeBuf(dev, x);
    id<MTLBuffer> by   = makeBuf(dev, y);
    id<MTLBuffer> bdst = makeBuf(dev, dst);

    return timeReps(queue, pipe,
                    @[bp, bx, by, bdst],
                    @[@0, @0, @0, @0],
                    MTLSizeMake(N, 1, 1),
                    MTLSizeMake(256, 1, 1),
                    reps);
}

static double benchSpmv(id<MTLDevice> dev, id<MTLCommandQueue> queue,
                        id<MTLLibrary> lib, uint32_t N, int reps,
                        uint32_t& nnz_out) {
    id<MTLComputePipelineState> pipe = makePipe(dev, lib, "main_0");
    if (!pipe) return -1;

    std::vector<int32_t> rowPtr(N + 1, 0);
    std::vector<int32_t> colIdx;
    std::vector<float>   values;
    colIdx.reserve(3 * N);
    values.reserve(3 * N);
    for (uint32_t i = 0; i < N; ++i) {
        rowPtr[i] = int32_t(colIdx.size());
        if (i > 0)     { colIdx.push_back(int32_t(i - 1)); values.push_back(-1.0f); }
                        colIdx.push_back(int32_t(i));     values.push_back( 4.0f);
        if (i + 1 < N) { colIdx.push_back(int32_t(i + 1)); values.push_back(-1.0f); }
    }
    rowPtr[N] = int32_t(colIdx.size());
    nnz_out = uint32_t(colIdx.size());

    std::vector<float> xv(N), yv(N, 0.0f);
    for (uint32_t i = 0; i < N; ++i) xv[i] = float((i % 13) - 6);

    SpmvParams p{N};
    id<MTLBuffer> bp     = makeBuf(dev, p);
    id<MTLBuffer> bRowP  = makeBuf(dev, rowPtr);
    id<MTLBuffer> bColI  = makeBuf(dev, colIdx);
    id<MTLBuffer> bVals  = makeBuf(dev, values);
    id<MTLBuffer> bx     = makeBuf(dev, xv);
    id<MTLBuffer> by     = makeBuf(dev, yv);

    return timeReps(queue, pipe,
                    @[bp, bRowP, bColI, bVals, bx, by],
                    @[@0, @0, @0, @0, @0, @0],
                    MTLSizeMake(N, 1, 1),
                    MTLSizeMake(256, 1, 1),
                    reps);
}

static double benchDotReduceSerial(id<MTLDevice> dev, id<MTLCommandQueue> queue,
                                   id<MTLLibrary> lib, uint32_t N, int reps) {
    id<MTLComputePipelineState> pipe = makePipe(dev, lib, "main_0");
    if (!pipe) return -1;

    DotParams p{N};
    std::vector<float> a(N), b(N);
    for (uint32_t i = 0; i < N; ++i) {
        a[i] = float((i % 17) - 8);
        b[i] = float((i % 23) - 11);
    }
    std::vector<float> dst(2, 0.0f);

    id<MTLBuffer> bp  = makeBuf(dev, p);
    id<MTLBuffer> ba  = makeBuf(dev, a);
    id<MTLBuffer> bb  = makeBuf(dev, b);
    id<MTLBuffer> bd  = makeBuf(dev, dst);

    return timeReps(queue, pipe,
                    @[bp, ba, bb, bd],
                    @[@0, @0, @0, @0],
                    MTLSizeMake(1, 1, 1),       // [numthreads(1,1,1)]
                    MTLSizeMake(1, 1, 1),
                    reps);
}

// Parallel dot_reduce: one threadgroup of 256 threads doing a
// grid-strided df32 fold + groupshared tree reduce. Compiles fine
// through slangc-metal (Metal natively supports threadgroup memory +
// threadgroup_barrier); only slangc-cpp can't run it, which is why
// the serial variant exists.
static double benchDotReduceParallel(id<MTLDevice> dev, id<MTLCommandQueue> queue,
                                     id<MTLLibrary> lib, uint32_t N, int reps) {
    id<MTLComputePipelineState> pipe = makePipe(dev, lib, "main_0");
    if (!pipe) return -1;

    DotParams p{N};
    std::vector<float> a(N), b(N);
    for (uint32_t i = 0; i < N; ++i) {
        a[i] = float((i % 17) - 8);
        b[i] = float((i % 23) - 11);
    }
    std::vector<float> dst(2, 0.0f);

    id<MTLBuffer> bp  = makeBuf(dev, p);
    id<MTLBuffer> ba  = makeBuf(dev, a);
    id<MTLBuffer> bb  = makeBuf(dev, b);
    id<MTLBuffer> bd  = makeBuf(dev, dst);

    return timeReps(queue, pipe,
                    @[bp, ba, bb, bd],
                    @[@0, @0, @0, @0],
                    MTLSizeMake(256, 1, 1),    // one threadgroup of 256
                    MTLSizeMake(256, 1, 1),
                    reps);
}

static double benchAssembleB(id<MTLDevice> dev, id<MTLCommandQueue> queue,
                             id<MTLLibrary> lib, uint32_t V,
                             uint32_t incPerV, int reps, uint32_t& slots_out) {
    id<MTLComputePipelineState> pipe = makePipe(dev, lib, "main_0");
    if (!pipe) return -1;

    const uint32_t totalInc = V * incPerV;
    const uint32_t slots    = V;
    slots_out = slots;

    std::vector<float> s(3 * V), mass(V, 1.0f), bvec(3 * V, 0.0f);
    std::vector<float> projections(3 * slots, 1.0f);
    std::vector<uint32_t> ctxStart(V + 1, 0u);
    std::vector<uint32_t> ctxSlot(totalInc);
    std::vector<float>    ctxWeight(totalInc);

    for (uint32_t v = 0; v < V; ++v) {
        ctxStart[v] = v * incPerV;
        s[3*v] = float(v); s[3*v + 1] = 0.0f; s[3*v + 2] = 0.0f;
        for (uint32_t k = 0; k < incPerV; ++k) {
            const uint32_t idx = v * incPerV + k;
            ctxSlot[idx]   = (v + k) % slots;
            ctxWeight[idx] = ((k & 1) ? -1.0f : +1.0f);
        }
    }
    ctxStart[V] = totalInc;

    AssembleBParams p{V};
    id<MTLBuffer> bp     = makeBuf(dev, p);
    id<MTLBuffer> bs     = makeBuf(dev, s);
    id<MTLBuffer> bmass  = makeBuf(dev, mass);
    id<MTLBuffer> bproj  = makeBuf(dev, projections);
    id<MTLBuffer> bcs    = makeBuf(dev, ctxStart);
    id<MTLBuffer> bcsl   = makeBuf(dev, ctxSlot);
    id<MTLBuffer> bcw    = makeBuf(dev, ctxWeight);
    id<MTLBuffer> bb     = makeBuf(dev, bvec);

    return timeReps(queue, pipe,
                    @[bp, bs, bmass, bproj, bcs, bcsl, bcw, bb],
                    @[@0, @0, @0, @0, @0, @0, @0, @0],
                    MTLSizeMake(V, 1, 1),
                    MTLSizeMake(64, 1, 1),
                    reps);
}

static double benchSpringProject(id<MTLDevice> dev, id<MTLCommandQueue> queue,
                                 id<MTLLibrary> lib, uint32_t N, int reps) {
    id<MTLComputePipelineState> pipe = makePipe(dev, lib, "main_0");
    if (!pipe) return -1;

    // Pad positions so all (p1, p2) pairs are valid indices.
    std::vector<float> positions(3 * (N + 1));
    for (uint32_t i = 0; i < N + 1; ++i) {
        positions[3*i + 0] = float(i);
        positions[3*i + 1] = float((i * 7) % 11);
        positions[3*i + 2] = float((i * 13) % 19);
    }
    std::vector<float> projected(3 * N, 0.0f);
    std::vector<uint32_t> p1(N), p2(N);
    std::vector<float>    rest(N, 1.0f), sqrtW(N, 1.0f);
    for (uint32_t i = 0; i < N; ++i) { p1[i] = i; p2[i] = i + 1; }

    // No ConstantBuffer in spring_project — bindings start at 0 with positions.
    id<MTLBuffer> bpos   = makeBuf(dev, positions);
    id<MTLBuffer> bproj  = makeBuf(dev, projected);
    id<MTLBuffer> bp1    = makeBuf(dev, p1);
    id<MTLBuffer> bp2    = makeBuf(dev, p2);
    id<MTLBuffer> brest  = makeBuf(dev, rest);
    id<MTLBuffer> bsqW   = makeBuf(dev, sqrtW);

    return timeReps(queue, pipe,
                    @[bpos, bproj, bp1, bp2, brest, bsqW],
                    @[@0, @0, @0, @0, @0, @0],
                    MTLSizeMake(N, 1, 1),
                    MTLSizeMake(64, 1, 1),
                    reps);
}

// ---- main -------------------------------------------------------------

int main(int argc, const char** argv) {
    (void)argc; (void)argv;
    @autoreleasepool {
        id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
        if (!dev) {
            std::fprintf(stderr, "MTLCreateSystemDefaultDevice returned nil — "
                                  "no GPU available, skipping.\n");
            return 0;     // Don't fail the build on headless CI runners.
        }
        std::printf("\nmetal_perf (real GPU dispatch — batched, all reps per command buffer)\n");
        std::printf("=================================================================\n");
        std::printf("device: %s\n", dev.name.UTF8String);
        std::printf("threadgroup memory: %lu B\n", (unsigned long)dev.maxThreadgroupMemoryLength);
        std::printf("\n");
        std::printf("All reps are encoded into ONE command buffer with ONE commit/wait,\n");
        std::printf("so the ~100–200µs Metal submission overhead is paid once and the\n");
        std::printf("per-invoke numbers approach steady-state GPU throughput (Metal's\n");
        std::printf("compute encoder serialises same-resource dispatches between\n");
        std::printf("invocations, so each one actually runs and is not elided).\n");
        std::printf("A one-shot warm-up dispatch (not timed) is paid first to keep PSO\n");
        std::printf("compile + page-in out of the measurement.\n");
        std::printf("-----------------------------------------------------------------\n");
        std::printf("%-22s %-12s %-12s %-14s %s\n",
                    "kernel", "N", "reps", "ms/invoke", "throughput");
        std::printf("-----------------------------------------------------------------\n");

        id<MTLCommandQueue> queue = [dev newCommandQueue];

        const std::string libRoot = "build/";

        constexpr uint32_t N         = 30000;
        constexpr uint32_t V_AB      = 10000;
        constexpr uint32_t INC_PER_V = 3;
        constexpr int      REPS_FAST = 200;
        constexpr int      REPS_DOT  = 50;

        auto runOne = [&](const char* tag, const char* libName,
                          auto&& bench, uint32_t n, int reps,
                          const char* unit, double countMul) {
            id<MTLLibrary> lib = loadLib(dev, (libRoot + libName).c_str());
            if (!lib) return;
            const double ms  = bench(lib);
            const double per = ms / reps;
            std::printf("%-22s %-12u %-12d %-14.4f %.1f M %s/s\n",
                        tag, n, reps, per,
                        (double(n) * countMul / per) / 1e3, unit);
        };

        runOne("spring_project", "spring_project.metallib",
               [&](id<MTLLibrary> lib){ return benchSpringProject(dev, queue, lib, N, REPS_FAST); },
               N, REPS_FAST, "springs", 1.0);

        runOne("saxpby", "saxpby.metallib",
               [&](id<MTLLibrary> lib){ return benchSaxpby(dev, queue, lib, N, REPS_FAST); },
               N, REPS_FAST, "elems", 1.0);

        {
            id<MTLLibrary> lib = loadLib(dev, (libRoot + "spmv.metallib").c_str());
            if (lib) {
                uint32_t nnz = 0;
                const double ms  = benchSpmv(dev, queue, lib, N, REPS_FAST, nnz);
                const double per = ms / REPS_FAST;
                std::printf("%-22s %-12u %-12d %-14.4f %.1f M nnz/s  (nnz=%u)\n",
                            "spmv", N, REPS_FAST, per,
                            (double(nnz) / per) / 1e3, nnz);
            }
        }

        {
            id<MTLLibrary> lib = loadLib(dev, (libRoot + "dot_reduce_serial.metallib").c_str());
            if (lib) {
                const double ms  = benchDotReduceSerial(dev, queue, lib, N, REPS_DOT);
                const double per = ms / REPS_DOT;
                std::printf("%-22s %-12u %-12d %-14.4f %.1f M elems/s  [serial, 1 SIMT lane]\n",
                            "dot_reduce_serial", N, REPS_DOT, per,
                            (double(N) / per) / 1e3);
            }
        }

        {
            id<MTLLibrary> lib = loadLib(dev, (libRoot + "dot_reduce.metallib").c_str());
            if (lib) {
                const double ms  = benchDotReduceParallel(dev, queue, lib, N, REPS_FAST);
                const double per = ms / REPS_FAST;
                std::printf("%-22s %-12u %-12d %-14.4f %.1f M elems/s  [parallel, 256 threads, df32]\n",
                            "dot_reduce", N, REPS_FAST, per,
                            (double(N) / per) / 1e3);
            }
        }

        {
            id<MTLLibrary> lib = loadLib(dev, (libRoot + "assemble_b.metallib").c_str());
            if (lib) {
                uint32_t slots = 0;
                const double ms  = benchAssembleB(dev, queue, lib, V_AB, INC_PER_V,
                                                  REPS_FAST, slots);
                const double per = ms / REPS_FAST;
                const uint32_t inc = V_AB * INC_PER_V;
                std::printf("%-22s %-12u %-12d %-14.4f %.1f M inc/s  (V=%u, inc=%u, slots=%u)\n",
                            "assemble_b", V_AB, REPS_FAST, per,
                            (double(inc) / per) / 1e3, V_AB, inc, slots);
            }
        }

        std::printf("-----------------------------------------------------------------\n");
    }
    return 0;
}
