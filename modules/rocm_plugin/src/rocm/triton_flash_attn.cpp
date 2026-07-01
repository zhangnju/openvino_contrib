#include "triton_flash_attn.hpp"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <cmath>

namespace ov { namespace rocm_gpu { namespace triton_fa {

namespace {
struct __attribute__((packed)) FAArgs {
    const void* Q; const void* K; const void* V; const void* Bias;
    void* Out; void* Lse; void* TMP;
    float softmax_scale;
    int32_t stride_qb, stride_qh, stride_qm;
    int32_t stride_kb, stride_kh, stride_kn;
    int32_t stride_vb, stride_vh, stride_vn;
    int32_t stride_bb, stride_bh, stride_bm;
    int32_t stride_ob, stride_oh, stride_om;
    int32_t nheads, seqlen_q, seqlen_k, seqlen_q_rounded, headdim;
    int32_t cache_key_sq, cache_key_sk;
};
// Args are packed contiguously (no trailing padding)
} // anon

std::shared_ptr<TritonFAKernel> compile(
    int seqlen_q, int seqlen_k, int headdim,
    const std::string& arch, const std::string& bias_type, bool causal)
{
    static const bool trace = std::getenv("ROCM_TRACE_TRITON_FA") != nullptr;

    // Find the AOT script (shipped alongside the plugin .so)
    // Search paths: next to this .so, /tmp, ROCM_TRITON_FA_SCRIPT env
    std::string script;
    if (auto* env = std::getenv("ROCM_TRITON_FA_SCRIPT")) {
        script = env;
    } else {
        for (auto& p : {"/tmp/triton_flash_attn_aot.py",
                        "/home/openvino_rocm/openvino_contrib/modules/rocm_plugin/src/rocm/triton_flash_attn_aot.py"}) {
            std::ifstream f(p);
            if (f.good()) { script = p; break; }
        }
    }
    if (script.empty()) {
        fprintf(stderr, "[triton-fa] AOT script not found\n");
        return nullptr;
    }

    char cmd[1024];
    snprintf(cmd, sizeof(cmd),
        "python3 %s --seqlen_q=%d --seqlen_k=%d --headdim=%d "
        "--bias_type=%s --causal=%d --arch=%s 2>/dev/null",
        script.c_str(), seqlen_q, seqlen_k, headdim,
        bias_type.c_str(), causal ? 1 : 0, arch.c_str());

    if (trace) fprintf(stderr, "[triton-fa] %s\n", cmd);

    FILE* pipe = popen(cmd, "r");
    if (!pipe) {
        fprintf(stderr, "[triton-fa] popen failed\n");
        return nullptr;
    }

    char hsaco_path[512] = {}, json_path[512] = {};
    if (!fgets(hsaco_path, sizeof(hsaco_path), pipe) ||
        !fgets(json_path, sizeof(json_path), pipe)) {
        pclose(pipe);
        fprintf(stderr, "[triton-fa] AOT compile failed\n");
        return nullptr;
    }
    pclose(pipe);

    // Trim newlines
    auto trim = [](char* s) { char* p = s + strlen(s) - 1; while (p >= s && (*p == '\n' || *p == '\r')) *p-- = 0; };
    trim(hsaco_path); trim(json_path);

    if (trace) fprintf(stderr, "[triton-fa] hsaco=%s json=%s\n", hsaco_path, json_path);

    // Load HSACO
    std::ifstream hf(hsaco_path, std::ios::binary);
    if (!hf.good()) {
        fprintf(stderr, "[triton-fa] cannot open %s\n", hsaco_path);
        return nullptr;
    }
    std::vector<char> hsaco((std::istreambuf_iterator<char>(hf)), std::istreambuf_iterator<char>());

    auto k = std::make_shared<hiprtc::CompiledKernel>();
    k->binary = std::move(hsaco);
    k->func_name = "_fwd_kernel";

    if (hipModuleLoadData(&k->module, k->binary.data()) != hipSuccess) {
        fprintf(stderr, "[triton-fa] hipModuleLoadData failed\n");
        return nullptr;
    }
    if (hipModuleGetFunction(&k->func, k->module, k->func_name.c_str()) != hipSuccess) {
        fprintf(stderr, "[triton-fa] hipModuleGetFunction failed for %s\n", k->func_name.c_str());
        return nullptr;
    }

    // Parse JSON for launch config
    KernelMeta meta;
    std::ifstream jf(json_path);
    if (jf.good()) {
        std::string line, content;
        while (std::getline(jf, line)) content += line;
        auto extract_int = [&](const char* key) -> int {
            auto pos = content.find(std::string("\"") + key + "\"");
            if (pos == std::string::npos) return 0;
            pos = content.find(':', pos);
            return std::atoi(content.c_str() + pos + 1);
        };
        meta.shared_mem = extract_int("shared_mem");
        meta.num_warps = extract_int("num_warps");
        meta.block_m = extract_int("block_m");
        meta.block_n = extract_int("block_n");
        auto name_pos = content.find("\"kernel_name\"");
        if (name_pos != std::string::npos) {
            auto q1 = content.find('"', content.find(':', name_pos) + 1);
            auto q2 = content.find('"', q1 + 1);
            if (q1 != std::string::npos && q2 != std::string::npos)
                meta.kernel_name = content.substr(q1 + 1, q2 - q1 - 1);
        }
    }

    if (trace)
        fprintf(stderr, "[triton-fa] loaded: shared=%d warps=%d block_m=%d kernel=%s\n",
                meta.shared_mem, meta.num_warps, meta.block_m, meta.kernel_name.c_str());

    auto result = std::make_shared<TritonFAKernel>();
    result->kernel = std::move(k);
    result->meta = meta;
    return result;
}

void launch(const TritonFAKernel& tk, hipStream_t stream,
            void* args_buf,
            const void* Q, const void* K, const void* V, const void* Bias, void* O,
            void* Lse, void* TMP,
            int batch, int heads, int seqlen_q, int seqlen_k, int headdim,
            float softmax_scale,
            int stride_qb, int stride_qh, int stride_qm,
            int stride_kb, int stride_kh, int stride_kn,
            int stride_vb, int stride_vh, int stride_vn,
            int stride_ob, int stride_oh, int stride_om)
{
    int sq_rounded = ((seqlen_q + 127) / 128) * 128;

    // Write args into the caller-provided persistent buffer.
    // This buffer must outlive the async kernel launch (use a mutable workbuffer).
    FAArgs* args = static_cast<FAArgs*>(args_buf);
    args->Q = Q; args->K = K; args->V = V; args->Bias = Bias;
    args->Out = O; args->Lse = Lse; args->TMP = TMP;
    args->softmax_scale = softmax_scale;
    args->stride_qb = stride_qb; args->stride_qh = stride_qh; args->stride_qm = stride_qm;
    args->stride_kb = stride_kb; args->stride_kh = stride_kh; args->stride_kn = stride_kn;
    args->stride_vb = stride_vb; args->stride_vh = stride_vh; args->stride_vn = stride_vn;
    args->stride_bb = 0; args->stride_bh = 0; args->stride_bm = 0;
    args->stride_ob = stride_ob; args->stride_oh = stride_oh; args->stride_om = stride_om;
    args->nheads = heads; args->seqlen_q = seqlen_q; args->seqlen_k = seqlen_k;
    args->seqlen_q_rounded = sq_rounded; args->headdim = headdim;
    args->cache_key_sq = seqlen_q / 32; args->cache_key_sk = seqlen_k / 32;

    unsigned grid_x = (seqlen_q + tk.meta.block_m - 1) / tk.meta.block_m;
    unsigned grid_y = batch * heads;
    unsigned block_x = tk.meta.num_warps * 32;

    size_t args_size = sizeof(FAArgs);
    void* config[] = {
        HIP_LAUNCH_PARAM_BUFFER_POINTER, args,
        HIP_LAUNCH_PARAM_BUFFER_SIZE, &args_size,
        HIP_LAUNCH_PARAM_END
    };

    hipModuleLaunchKernel(tk.kernel->func,
                          grid_x, grid_y, 1,
                          block_x, 1, 1,
                          tk.meta.shared_mem, stream,
                          nullptr, config);

}

}}} // namespace
