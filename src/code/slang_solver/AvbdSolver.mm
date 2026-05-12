// AvbdSolver implementation (PR-E stub). Loads the six AVBD kernel
// metallibs and builds compute pipeline state objects. Per-step
// dispatch is a stub — actual GPU loop wiring lands in follow-up PRs.

#include "AvbdSolver.h"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <cstdio>
#include <string>

namespace cloth {

struct AvbdSolver::Impl {
    id<MTLDevice>               device   = nil;
    id<MTLCommandQueue>         queue    = nil;

    // Six AVBD pipeline kernels. Each loaded from a separate
    // metallib at construction time.
    id<MTLComputePipelineState> psoInit             = nil;
    id<MTLComputePipelineState> psoGatherSpring     = nil;
    id<MTLComputePipelineState> psoGatherAttachment = nil;
    id<MTLComputePipelineState> psoGatherTriangle   = nil;
    id<MTLComputePipelineState> psoGatherBending    = nil;
    id<MTLComputePipelineState> psoSolveApply       = nil;

    bool ok = false;

    static id<MTLLibrary> loadLib(id<MTLDevice> dev, const std::string& path,
                                   const char* name) {
        NSError* err = nil;
        NSURL* url = [NSURL fileURLWithPath:[NSString stringWithUTF8String:path.c_str()]];
        id<MTLLibrary> lib = [dev newLibraryWithURL:url error:&err];
        if (!lib) {
            std::fprintf(stderr, "AvbdSolver: failed to load %s from %s: %s\n",
                         name, path.c_str(),
                         err ? err.localizedDescription.UTF8String : "(null)");
        }
        return lib;
    }

    static id<MTLComputePipelineState> makePSO(id<MTLDevice> dev,
                                                 id<MTLLibrary> lib,
                                                 const char* fn,
                                                 const char* tag) {
        NSError* err = nil;
        id<MTLFunction> f = [lib newFunctionWithName:[NSString stringWithUTF8String:fn]];
        if (!f) {
            std::fprintf(stderr, "AvbdSolver: no fn %s in %s\n", fn, tag);
            return nil;
        }
        id<MTLComputePipelineState> pso =
            [dev newComputePipelineStateWithFunction:f error:&err];
        if (!pso) {
            std::fprintf(stderr, "AvbdSolver: PSO %s failed: %s\n",
                         tag, err ? err.localizedDescription.UTF8String : "(null)");
        }
        return pso;
    }
};

AvbdSolver::AvbdSolver(const char* metallibPath)
    : impl_(new Impl()) {
    impl_->device = MTLCreateSystemDefaultDevice();
    if (!impl_->device) {
        std::fprintf(stderr, "AvbdSolver: no Metal device\n");
        return;
    }
    impl_->queue = [impl_->device newCommandQueue];

    std::string root = metallibPath;
    if (!root.empty() && root.back() != '/') root.push_back('/');

    id<MTLLibrary> libInit       = Impl::loadLib(impl_->device, root + "vbd_init.metallib",             "vbd_init");
    id<MTLLibrary> libSpring     = Impl::loadLib(impl_->device, root + "vbd_gather_spring.metallib",    "vbd_gather_spring");
    id<MTLLibrary> libAttachment = Impl::loadLib(impl_->device, root + "vbd_gather_attachment.metallib","vbd_gather_attachment");
    id<MTLLibrary> libTriangle   = Impl::loadLib(impl_->device, root + "vbd_gather_triangle.metallib",  "vbd_gather_triangle");
    id<MTLLibrary> libBending    = Impl::loadLib(impl_->device, root + "vbd_gather_bending.metallib",   "vbd_gather_bending");
    id<MTLLibrary> libSolve      = Impl::loadLib(impl_->device, root + "vbd_solve_apply.metallib",      "vbd_solve_apply");
    if (!libInit || !libSpring || !libAttachment || !libTriangle || !libBending || !libSolve) return;

    impl_->psoInit             = Impl::makePSO(impl_->device, libInit,       "main_0", "vbd_init");
    impl_->psoGatherSpring     = Impl::makePSO(impl_->device, libSpring,     "main_0", "vbd_gather_spring");
    impl_->psoGatherAttachment = Impl::makePSO(impl_->device, libAttachment, "main_0", "vbd_gather_attachment");
    impl_->psoGatherTriangle   = Impl::makePSO(impl_->device, libTriangle,   "main_0", "vbd_gather_triangle");
    impl_->psoGatherBending    = Impl::makePSO(impl_->device, libBending,    "main_0", "vbd_gather_bending");
    impl_->psoSolveApply       = Impl::makePSO(impl_->device, libSolve,      "main_0", "vbd_solve_apply");
    if (!impl_->psoInit || !impl_->psoGatherSpring || !impl_->psoGatherAttachment ||
        !impl_->psoGatherTriangle || !impl_->psoGatherBending || !impl_->psoSolveApply) return;

    impl_->ok = true;
    std::fprintf(stderr,
        "AvbdSolver: all 6 AVBD kernels loaded (init, gather_{spring,attachment,triangle,bending}, solve_apply)\n");
}

AvbdSolver::~AvbdSolver() { delete impl_; }

bool AvbdSolver::ok() const { return impl_ && impl_->ok; }

int AvbdSolver::step() {
    // PR-E stub. Real dispatch comes in follow-up PRs.
    if (!ok()) return -1;
    return 0;
}

}  // namespace cloth
