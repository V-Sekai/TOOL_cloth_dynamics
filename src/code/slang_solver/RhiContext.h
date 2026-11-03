// RhiContext — thin slang-rhi wrapper that mirrors the Metal host's encoder
// idiom (setPipeline / setBuffer / setBytes / dispatchThreads) so that
// AvbdSolver can be a near line-for-line port of AvbdSolver.mm. The backend
// (Vulkan, D3D12, CUDA, Metal) is chosen by slang-rhi.
//
// Kernels are the Lean-emitted `.slang` files, compiled at runtime. Every AVBD
// kernel is `[numthreads(64,1,1)]` with no bounds check on `tid.x`, because
// Metal's dispatchThreads() masks the tail. Other backends round up to whole
// groups, so loadKernel() injects a guard against a `count` constant to keep
// the same semantics.
//
// Buf mirrors a Metal shared-storage buffer: it has a host shadow, so the
// existing `.contents` style host access keeps working. The host copy is
// uploaded before the first dispatch that binds it, and refreshed from the GPU
// the next time the host asks for it after a dispatch touched it.
//
// float3 layout: with these shaders, StructuredBuffer<float3> is tightly
// packed (12 B) on Vulkan, whereas Metal pads to 16 B. See kVec3Floats.
#ifndef CLOTH_RHI_CONTEXT_H
#define CLOTH_RHI_CONTEXT_H

#include <cstddef>
#include <cstdint>
#include <set>
#include <string>
#include <map>
#include <vector>

#include <slang-rhi.h>

namespace cloth {

class RhiContext;

// Floats per float3 element in a GPU buffer (12 B stride).
constexpr uint32_t kVec3Floats = 3;

class Buf {
public:
    Buf() = default;
    explicit operator bool() const { return gpu_ != nullptr; }
    size_t length = 0;

    // Host pointer for writing (or read-modify-write). Marks the host copy dirty.
    void* contents();
    // Host pointer for reading only.
    const void* cdata() const;

private:
    friend class RhiContext;
    RhiContext* ctx_ = nullptr;
    rhi::ComPtr<rhi::IBuffer> gpu_;
    mutable std::vector<uint8_t> host_;
    mutable bool hostDirty_ = false;
    mutable bool gpuDirty_ = false;
};

class RhiContext {
public:
    static constexpr uint32_t kGuardBinding = 31;
    static constexpr uint32_t kGroupSize = 64;

    explicit RhiContext(rhi::DeviceType type = rhi::DeviceType::Vulkan);
    ~RhiContext();

    bool ok() const { return device_ != nullptr; }
    const std::string& error() const { return error_; }
    rhi::IDevice* device() const { return device_.get(); }
    const char* adapterName() const { return adapterName_.c_str(); }

    // Loads `<dir>/<name>.slang`, injects the tail guard, builds a pipeline.
    bool loadKernel(const std::string& dir, const std::string& name);
    bool hasKernel(const std::string& name) const { return kernels_.count(name) != 0; }

    // Creates a buffer of `bytes` (min 4), initialised from `init` or zeroed.
    Buf makeBuffer(const void* init, size_t bytes);

    // --- Metal-style encoder facade ---
    void setPipeline(const char* kernel);
    void setBuffer(uint32_t index, Buf& buf);
    void setBytes(uint32_t index, const void* data, size_t size);
    void dispatchThreads(uint32_t count);
    void copyBuffer(Buf& dst, Buf& src);
    // Submits recorded work and waits for completion.
    void flush();

private:
    friend class Buf;
    struct Kernel {
        rhi::ComPtr<rhi::IComputePipeline> pipeline;
        std::map<uint32_t, std::string> nameByBinding;
    };
    void ensureEncoder();
    void syncUp(Buf& b);
    void syncDown(const Buf& b);
    void fail(const std::string& msg);

    rhi::ComPtr<rhi::IDevice> device_;
    rhi::ComPtr<rhi::ICommandEncoder> encoder_;
    std::map<std::string, Kernel> kernels_;
    std::string error_;
    std::string adapterName_;

    // Pending encoder state.
    std::string cur_;
    std::vector<std::pair<uint32_t, Buf*>> pendingBufs_;
    std::vector<std::pair<uint32_t, std::vector<uint8_t>>> pendingBytes_;
    // Buffers bound in the current (unflushed) batch.
    std::set<Buf*> touched_;
};

} // namespace cloth

#endif
