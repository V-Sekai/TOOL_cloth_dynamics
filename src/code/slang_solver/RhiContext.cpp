#include "RhiContext.h"

#include <slang-rhi/shader-cursor.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>

namespace cloth {
using namespace rhi;

// ---------------------------------------------------------------- Buf

void* Buf::contents() {
    if (!ctx_) return nullptr;
    if (ctx_->touched_.count(this)) ctx_->flush();   // keep host/GPU order
    if (gpuDirty_) ctx_->syncDown(*this);
    hostDirty_ = true;
    return host_.data();
}

const void* Buf::cdata() const {
    if (!ctx_) return nullptr;
    if (ctx_->touched_.count(const_cast<Buf*>(this))) ctx_->flush();
    if (gpuDirty_) ctx_->syncDown(*this);
    return host_.data();
}

// -------------------------------------------------------------- RhiContext

void RhiContext::fail(const std::string& msg) {
    error_ = msg;
    std::fprintf(stderr, "RhiContext: %s\n", msg.c_str());
}

RhiContext::RhiContext(DeviceType type) {
    DeviceDesc desc = {};
    desc.deviceType = type;
    Result r = getRHI()->createDevice(desc, device_.writeRef());
    if (SLANG_FAILED(r)) {
        error_ = "createDevice failed";
        device_ = nullptr;
        return;
    }
    const DeviceInfo& info = device_->getInfo();
    adapterName_ = info.adapterName ? info.adapterName : "";
}

RhiContext::~RhiContext() {
    flush();
}

bool RhiContext::loadKernel(const std::string& dir, const std::string& name) {
    if (!device_) return false;
    std::string path = dir;
    if (!path.empty() && path.back() != '/' && path.back() != '\\') path.push_back('/');
    std::ifstream f(path + name + ".slang", std::ios::binary);
    if (!f) { error_ = "missing " + path + name + ".slang"; return false; }
    std::stringstream ss;
    ss << f.rdbuf();
    std::string src = ss.str();

    // Tail guard (see header).
    const std::string sig = "void main(uint3 tid : SV_DispatchThreadID) {";
    size_t at = src.find(sig);
    if (at == std::string::npos) { error_ = name + ": unexpected entry signature"; return false; }
    src.insert(at + sig.size(), "\n  if (tid.x >= __guard.count) return;");
    std::string decl = "struct __Guard { uint count; };\n"
                       "[[vk::binding(" + std::to_string(kGuardBinding) + ", 0)]]\n"
                       "ConstantBuffer<__Guard> __guard;\n";
    src = decl + src;

    slang::ISession* session = device_->getSlangSession();
    ComPtr<slang::IBlob> diag;
    slang::IModule* mod = session->loadModuleFromSourceString(
        name.c_str(), (name + ".slang").c_str(), src.c_str(), diag.writeRef());
    if (!mod) {
        error_ = name + ": compile failed: " +
                 (diag ? (const char*)diag->getBufferPointer() : "");
        return false;
    }
    ComPtr<slang::IEntryPoint> ep;
    if (SLANG_FAILED(mod->findEntryPointByName("main", ep.writeRef()))) {
        error_ = name + ": no entry point main";
        return false;
    }
    slang::IComponentType* parts[] = { mod, ep.get() };
    ComPtr<slang::IComponentType> composed, linked;
    if (SLANG_FAILED(session->createCompositeComponentType(parts, 2, composed.writeRef(), diag.writeRef())) ||
        SLANG_FAILED(composed->link(linked.writeRef(), diag.writeRef()))) {
        error_ = name + ": link failed: " +
                 (diag ? (const char*)diag->getBufferPointer() : "");
        return false;
    }

    Kernel k;
    slang::ProgramLayout* layout = linked->getLayout();
    for (unsigned i = 0; i < layout->getParameterCount(); ++i) {
        slang::VariableLayoutReflection* p = layout->getParameterByIndex(i);
        k.nameByBinding[p->getBindingIndex()] = p->getName();
    }

    ShaderProgramDesc pd = {};
    pd.slangGlobalScope = linked.get();
    ComPtr<IShaderProgram> prog;
    if (SLANG_FAILED(device_->createShaderProgram(pd, prog.writeRef(), diag.writeRef()))) {
        error_ = name + ": createShaderProgram failed: " +
                 (diag ? (const char*)diag->getBufferPointer() : "");
        return false;
    }
    ComputePipelineDesc cpd = {};
    cpd.program = prog.get();
    if (SLANG_FAILED(device_->createComputePipeline(cpd, k.pipeline.writeRef()))) {
        error_ = name + ": createComputePipeline failed";
        return false;
    }
    kernels_[name] = std::move(k);
    return true;
}

Buf RhiContext::makeBuffer(const void* init, size_t bytes) {
    Buf b;
    if (!device_) return b;
    const size_t size = bytes ? bytes : 4;
    b.host_.assign(size, 0);
    if (init && bytes) std::memcpy(b.host_.data(), init, bytes);

    BufferDesc bd = {};
    bd.size = size;
    bd.usage = BufferUsage::ShaderResource | BufferUsage::UnorderedAccess |
               BufferUsage::CopyDestination | BufferUsage::CopySource;
    bd.defaultState = ResourceState::UnorderedAccess;
    bd.memoryType = MemoryType::DeviceLocal;
    if (SLANG_FAILED(device_->createBuffer(bd, b.host_.data(), b.gpu_.writeRef()))) {
        fail("createBuffer failed");
        return Buf();
    }
    b.ctx_ = this;
    b.length = bytes;
    return b;
}

void RhiContext::ensureEncoder() {
    if (!encoder_) encoder_ = device_->getQueue(QueueType::Graphics)->createCommandEncoder();
}

void RhiContext::syncUp(Buf& b) {
    if (!b.hostDirty_) return;
    ensureEncoder();
    encoder_->uploadBufferData(b.gpu_, 0, b.host_.size(), b.host_.data());
    b.hostDirty_ = false;
}

void RhiContext::syncDown(const Buf& b) {
    flush();
    if (SLANG_FAILED(device_->readBuffer(b.gpu_, 0, b.host_.size(), b.host_.data())))
        fail("readBuffer failed");
    b.gpuDirty_ = false;
}

void RhiContext::setPipeline(const char* kernel) {
    cur_ = kernel;
    pendingBufs_.clear();
    pendingBytes_.clear();
}

void RhiContext::setBuffer(uint32_t index, Buf& buf) {
    pendingBufs_.emplace_back(index, &buf);
}

void RhiContext::setBytes(uint32_t index, const void* data, size_t size) {
    const uint8_t* p = static_cast<const uint8_t*>(data);
    pendingBytes_.emplace_back(index, std::vector<uint8_t>(p, p + size));
}

void RhiContext::dispatchThreads(uint32_t count) {
    auto it = kernels_.find(cur_);
    if (it == kernels_.end()) { fail("unknown kernel " + cur_); return; }
    if (count == 0) return;
    Kernel& k = it->second;

    for (auto& pb : pendingBufs_) {
        if (!pb.second || !*pb.second) { fail(cur_ + ": unbound buffer at " + std::to_string(pb.first)); return; }
        syncUp(*pb.second);
    }
    ensureEncoder();
    auto pass = encoder_->beginComputePass();
    auto root = pass->bindPipeline(k.pipeline);
    ShaderCursor cur(root);

    auto nameOf = [&](uint32_t idx) -> const char* {
        auto n = k.nameByBinding.find(idx);
        return n == k.nameByBinding.end() ? nullptr : n->second.c_str();
    };
    bool good = true;
    for (auto& pb : pendingBufs_) {
        const char* n = nameOf(pb.first);
        if (!n) { fail(cur_ + ": no binding " + std::to_string(pb.first)); good = false; break; }
        if (SLANG_FAILED(cur[n].setBinding(pb.second->gpu_.get()))) {
            fail(cur_ + ": setBinding failed for " + n); good = false; break;
        }
        touched_.insert(pb.second);
    }
    if (good) {
        for (auto& pc : pendingBytes_) {
            const char* n = nameOf(pc.first);
            if (!n) { fail(cur_ + ": no binding " + std::to_string(pc.first)); good = false; break; }
            if (SLANG_FAILED(cur[n].getDereferenced().setData(pc.second.data(), pc.second.size()))) {
                fail(cur_ + ": setData failed for " + n); good = false; break;
            }
        }
    }
    if (good) {
        uint32_t guard = count;
        if (SLANG_FAILED(cur["__guard"].getDereferenced().setData(&guard, sizeof(guard)))) {
            fail(cur_ + ": guard setData failed"); good = false;
        }
    }
    if (good) pass->dispatchCompute((count + kGroupSize - 1) / kGroupSize, 1, 1);
    pass->end();
    // Metal serialises dispatches within an encoder; do the same.
    encoder_->globalBarrier();
}

void RhiContext::copyBuffer(Buf& dst, Buf& src) {
    syncUp(src);
    ensureEncoder();
    const size_t n = src.host_.size() < dst.host_.size() ? src.host_.size() : dst.host_.size();
    encoder_->copyBuffer(dst.gpu_, 0, src.gpu_, 0, n);
    encoder_->globalBarrier();
    touched_.insert(&dst);
    touched_.insert(&src);
    dst.hostDirty_ = false;
}

void RhiContext::flush() {
    if (!encoder_ || !device_) return;
    auto q = device_->getQueue(QueueType::Graphics);
    q->submit(encoder_->finish());
    q->waitOnHost();
    encoder_ = nullptr;
    for (Buf* b : touched_) b->gpuDirty_ = true;
    touched_.clear();
}

} // namespace cloth
