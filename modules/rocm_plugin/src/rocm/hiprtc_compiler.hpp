// HIP kernel wrapper: load pre-compiled HSACO or JIT-compile via hipRTC.
// Used by:
//   - Triton Flash Attention (AOT-compiled HSACO from Python subprocess)
//   - WMMA Attention (JIT-compiled HIP C++ via hipRTC)
#pragma once
#include <hip/hip_runtime.h>
#include <hip/hiprtc.h>
#include <string>
#include <vector>
#include <memory>
#include <cstdio>

namespace ov {
namespace rocm_gpu {
namespace hiprtc {

struct CompiledKernel {
    hipModule_t   module = nullptr;
    hipFunction_t func   = nullptr;
    std::vector<char> binary;
    std::string func_name;

    ~CompiledKernel() {
        if (module) hipModuleUnload(module);
    }

    CompiledKernel() = default;
    CompiledKernel(CompiledKernel&& o) noexcept
        : module(o.module), func(o.func),
          binary(std::move(o.binary)), func_name(std::move(o.func_name)) {
        o.module = nullptr; o.func = nullptr;
    }
    CompiledKernel& operator=(CompiledKernel&& o) noexcept {
        if (this != &o) {
            if (module) hipModuleUnload(module);
            module = o.module; func = o.func;
            binary = std::move(o.binary); func_name = std::move(o.func_name);
            o.module = nullptr; o.func = nullptr;
        }
        return *this;
    }
    CompiledKernel(const CompiledKernel&) = delete;
    CompiledKernel& operator=(const CompiledKernel&) = delete;
};

// JIT compile HIP C++ source to a loaded kernel via hipRTC.
// func_name: the __global__ function to extract.
// arch: e.g. "gfx1100" (auto-prefixed with --gpu-architecture=).
// tag: human-readable label for error messages.
// Returns shared_ptr to loaded kernel, or nullptr on failure.
inline std::shared_ptr<CompiledKernel> compile(
        const std::string& source, const std::string& func_name,
        const std::string& arch, const std::string& tag = "") {
    hiprtcProgram prog;
    hiprtcResult r = hiprtcCreateProgram(&prog, source.c_str(), tag.c_str(), 0, nullptr, nullptr);
    if (r != HIPRTC_SUCCESS) {
        fprintf(stderr, "[hiprtc] createProgram failed: %s\n", hiprtcGetErrorString(r));
        return nullptr;
    }

    std::string arch_flag = "--gpu-architecture=" + arch;
    const char* opts[] = { arch_flag.c_str(), "-O3" };
    r = hiprtcCompileProgram(prog, 2, opts);
    if (r != HIPRTC_SUCCESS) {
        size_t log_sz = 0;
        hiprtcGetProgramLogSize(prog, &log_sz);
        std::string log(log_sz, '\0');
        hiprtcGetProgramLog(prog, &log[0]);
        fprintf(stderr, "[hiprtc] compile failed (%s): %s\n", tag.c_str(), log.c_str());
        hiprtcDestroyProgram(&prog);
        return nullptr;
    }

    size_t code_sz = 0;
    hiprtcGetCodeSize(prog, &code_sz);
    auto k = std::make_shared<CompiledKernel>();
    k->binary.resize(code_sz);
    hiprtcGetCode(prog, k->binary.data());
    hiprtcDestroyProgram(&prog);

    k->func_name = func_name;
    if (hipModuleLoadData(&k->module, k->binary.data()) != hipSuccess) {
        fprintf(stderr, "[hiprtc] moduleLoad failed (%s)\n", tag.c_str());
        return nullptr;
    }
    if (hipModuleGetFunction(&k->func, k->module, func_name.c_str()) != hipSuccess) {
        fprintf(stderr, "[hiprtc] getFunction '%s' failed (%s)\n", func_name.c_str(), tag.c_str());
        return nullptr;
    }
    return k;
}

// Launch a compiled kernel.
inline void launch(const CompiledKernel& k, hipStream_t stream,
                   unsigned grid_x, unsigned block_x,
                   void* args, size_t args_size,
                   unsigned shared_mem = 0) {
    void* config[] = {
        HIP_LAUNCH_PARAM_BUFFER_POINTER, args,
        HIP_LAUNCH_PARAM_BUFFER_SIZE, &args_size,
        HIP_LAUNCH_PARAM_END
    };
    hipModuleLaunchKernel(k.func, grid_x, 1, 1, block_x, 1, 1,
                          shared_mem, stream, nullptr, config);
}

}  // namespace hiprtc
}  // namespace rocm_gpu
}  // namespace ov
