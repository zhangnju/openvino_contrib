// FusedRTD op: compiles and launches a rocMLIR reshape_transpose_reshape_dot kernel.
// Supports both input-side and output-side Reshape→Transpose→Reshape transforms.

#include "fused_rtd_op.hpp"
#include <rocm_operation_registry.hpp>
#include <openvino/core/except.hpp>
#include "transformer/nodes/fused_rtd_node.hpp"
#include "rocm/rocmlir_compiler.hpp"

#include <fmt/format.h>
#include <sstream>
#include <fstream>
#include <cstdio>
#include <sys/stat.h>

namespace ov {
namespace rocm_gpu {

namespace {

static std::string find_driver() {
    for (const char* p : {"/home/rocmlir_install/bin/rocmlir-driver",
                           "/opt/rocm/bin/rocmlir-driver"}) {
        struct stat st;
        if (stat(p, &st) == 0) return p;
    }
    return "rocmlir-driver";
}

// Compute row-major strides for given dims
static std::vector<int64_t> row_strides(const std::vector<int64_t>& dims) {
    std::vector<int64_t> st(dims.size());
    st.back() = 1;
    for (int i = (int)dims.size() - 2; i >= 0; i--)
        st[i] = st[i + 1] * dims[i + 1];
    return st;
}

// Compute strides after a non-contiguous reshape (dimension split/merge).
// This mirrors MIGraphX's reshape stride computation:
// When reshaping from pre_dims with pre_strides to new_dims, the new strides
// are computed by splitting/merging the stride hierarchy.
static std::vector<int64_t> reshape_strides(const std::vector<int64_t>& pre_dims,
                                             const std::vector<int64_t>& pre_strides,
                                             const std::vector<int64_t>& new_dims) {
    std::vector<int64_t> result(new_dims.size());

    // Walk through both dimension lists, matching element counts
    size_t pi = 0, ni = 0;
    while (pi < pre_dims.size() && ni < new_dims.size()) {
        int64_t pre_count = pre_dims[pi];
        int64_t new_count = new_dims[ni];

        if (pre_count == new_count) {
            result[ni] = pre_strides[pi];
            pi++; ni++;
        } else if (pre_count > new_count) {
            // Splitting: pre_dim is being split into multiple new_dims
            // First new dim gets the original stride
            result[ni] = pre_strides[pi];
            int64_t accum = new_dims[ni];
            ni++;
            while (ni < new_dims.size() && accum < pre_count) {
                // Subsequent splits: stride = parent_stride / accumulated_factor
                result[ni] = pre_strides[pi] / accum;
                accum *= new_dims[ni];
                ni++;
            }
            pi++;
        } else {
            // Merging: multiple pre_dims merge into one new_dim
            result[ni] = pre_strides[pi];
            int64_t accum = pre_dims[pi];
            pi++;
            while (pi < pre_dims.size() && accum < new_count) {
                accum *= pre_dims[pi];
                pi++;
            }
            ni++;
        }
    }
    // Fill any remaining
    for (; ni < new_dims.size(); ni++) result[ni] = 1;

    return result;
}

// Apply transpose permutation to dims and strides
static void apply_transpose(const std::vector<int64_t>& dims,
                            const std::vector<int64_t>& strides,
                            const std::vector<int64_t>& perm,
                            std::vector<int64_t>& out_dims,
                            std::vector<int64_t>& out_strides) {
    out_dims.resize(perm.size());
    out_strides.resize(perm.size());
    for (size_t i = 0; i < perm.size(); i++) {
        out_dims[i] = dims[perm[i]];
        out_strides[i] = strides[perm[i]];
    }
}

// Format a migraphx.shaped type string
static std::string shaped(const std::vector<int64_t>& dims,
                          const std::vector<int64_t>& strides) {
    std::ostringstream o;
    o << "!migraphx.shaped<";
    for (size_t i = 0; i < dims.size(); i++) { if (i) o << "x"; o << dims[i]; }
    o << "xf16, ";
    for (size_t i = 0; i < strides.size(); i++) { if (i) o << "x"; o << strides[i]; }
    o << ">";
    return o.str();
}

// Short form (inside ops, no !migraphx.shaped prefix)
static std::string shaped_short(const std::vector<int64_t>& dims,
                                const std::vector<int64_t>& strides) {
    std::ostringstream o;
    o << "<";
    for (size_t i = 0; i < dims.size(); i++) { if (i) o << "x"; o << dims[i]; }
    o << "xf16, ";
    for (size_t i = 0; i < strides.size(); i++) { if (i) o << "x"; o << strides[i]; }
    o << ">";
    return o.str();
}

static std::string dims_str(const std::vector<int64_t>& v) {
    std::ostringstream o;
    for (size_t i = 0; i < v.size(); i++) { if (i) o << ", "; o << v[i]; }
    return o.str();
}

// Generate MLIR IR for optional input transform + dot + optional output transform.
// Input-side:  reshape(in_pre→in_r1) → transpose(in_perm) → [implicit reshape to dot input]
// Output-side: dot → transpose(out_perm) → reshape(out_r2)
static std::string make_rtd_mlir(
    // Actual A input shape (before any input transform)
    const std::vector<int64_t>& A_shape,
    const std::vector<int64_t>& B_shape,
    bool transB,
    // Input-side transform (empty in_perm = no transform)
    const std::vector<int64_t>& in_pre,
    const std::vector<int64_t>& in_r1,
    const std::vector<int64_t>& in_perm,
    const std::vector<int64_t>& in_r2,
    // Output-side transform
    const std::vector<int64_t>& dot_out_shape,
    const std::vector<int64_t>& out_perm,
    const std::vector<int64_t>& out_r2,
    const std::string& arch, int num_cu)
{
    std::ostringstream s;
    const std::string archTriple = "amdgcn-amd-amdhsa:" + arch;
    bool has_in = !in_perm.empty();

    // ── Compute A type ──
    auto a_st = row_strides(A_shape);
    std::string A_type = shaped(A_shape, a_st);

    // ── Compute B type ──
    auto b_st = row_strides(B_shape);
    std::string B_type = shaped(B_shape, b_st);

    // ── Compute final output type ──
    auto r2_st = row_strides(out_r2);
    std::string final_type = shaped(out_r2, r2_st);

    // ── For identity output perm, check if we need output transform ──
    bool has_out = false;
    for (size_t i = 0; i < out_perm.size(); i++) {
        if (out_perm[i] != (int64_t)i) { has_out = true; break; }
    }

    // If input transform present, the dot operates in the high-dimensional space
    // after transpose. We need multibroadcast on B to match.

    s << "module {\n";
    s << "  func.func @mlir_reshape_transpose_reshape_dot(\n";
    s << "      %arg0: " << A_type << ",\n";
    s << "      %arg1: " << B_type << ")\n";
    s << "      -> " << final_type << "\n";
    s << fmt::format("      attributes {{arch = \"{}\", kernel = \"mixr\", num_cu = {} : i64}} {{\n",
                     archTriple, num_cu);

    int v = 0;
    std::string cur_a;

    if (has_in) {
        // ── Input-side transform ──
        // Step 1: reshape A from A_shape (in_pre) to in_r1
        auto r1_st = reshape_strides(A_shape, a_st, in_r1);
        s << fmt::format("    %{} = migraphx.reshape %arg0 {{dims = [{}]}} : {} -> {}\n",
                         v, dims_str(in_r1), shaped_short(A_shape, a_st), shaped_short(in_r1, r1_st));
        cur_a = fmt::format("%{}", v); v++;

        // Step 2: transpose
        std::vector<int64_t> tp_dims, tp_st;
        apply_transpose(in_r1, r1_st, in_perm, tp_dims, tp_st);
        s << fmt::format("    %{} = migraphx.transpose {} {{permutation = [{}]}} : {} -> {}\n",
                         v, cur_a, dims_str(in_perm), shaped_short(in_r1, r1_st), shaped_short(tp_dims, tp_st));
        cur_a = fmt::format("%{}", v); v++;

        // The dot input A is now tp_dims with tp_st (non-contiguous).
        // B needs to be multibroadcast to match batch dims.
        // Dot contracts last dim of A with (transB ? last : second-to-last) dim of B.
        // Build the multibroadcast dims for B:
        // A is [batch..., M, K], B is [K, N] or [N, K] → broadcast to [batch..., K, N] or [batch..., N, K]
        size_t a_ndim = tp_dims.size();
        size_t b_ndim = B_shape.size();

        // Build broadcast shape: leading batch dims from A + B original dims
        std::vector<int64_t> bcast_dims;
        std::vector<int64_t> bcast_st;
        for (size_t i = 0; i < a_ndim - 2; i++) {
            bcast_dims.push_back(tp_dims[i]);
            bcast_st.push_back(0);  // broadcast
        }
        // Append B's own dims with their strides
        for (size_t i = 0; i < b_ndim; i++) {
            bcast_dims.push_back(B_shape[i]);
            bcast_st.push_back(b_st[i]);
        }

        std::string B_bcast_type = shaped(bcast_dims, bcast_st);
        s << fmt::format("    %{} = migraphx.multibroadcast %arg1 {{out_dyn_dims = [], out_lens = [{}]}} : {} -> {}\n",
                         v, dims_str(bcast_dims), shaped_short(B_shape, b_st), shaped_short(bcast_dims, bcast_st));
        std::string cur_b = fmt::format("%{}", v); v++;

        // Dot: A[batch.., M, K] x B[batch.., K, N] → C[batch.., M, N]
        std::vector<int64_t> dot_dims;
        for (size_t i = 0; i < a_ndim - 1; i++) dot_dims.push_back(tp_dims[i]);
        // N = transB ? B_shape[0] : B_shape.back()
        int64_t N = transB ? B_shape[0] : B_shape.back();
        dot_dims.push_back(N);
        auto dot_st_out = row_strides(dot_dims);

        s << fmt::format("    %{} = migraphx.dot {}, {} : {}, {} -> {}\n",
                         v, cur_a, cur_b,
                         shaped_short(tp_dims, tp_st),
                         shaped_short(bcast_dims, bcast_st),
                         shaped_short(dot_dims, dot_st_out));
        std::string cur = fmt::format("%{}", v); v++;

        if (has_out) {
            // Output-side transpose
            std::vector<int64_t> otp_dims(out_perm.size());
            for (size_t i = 0; i < out_perm.size(); i++) otp_dims[i] = dot_dims[out_perm[i]];
            auto otp_st_transposed = row_strides(dot_dims);
            std::vector<int64_t> otp_st(out_perm.size());
            for (size_t i = 0; i < out_perm.size(); i++) otp_st[i] = otp_st_transposed[out_perm[i]];

            s << fmt::format("    %{} = migraphx.transpose {} {{permutation = [{}]}} : {} -> {}\n",
                             v, cur, dims_str(out_perm),
                             shaped_short(dot_dims, dot_st_out),
                             shaped_short(otp_dims, otp_st));
            cur = fmt::format("%{}", v); v++;

            // Final reshape
            s << fmt::format("    %{} = migraphx.reshape {} {{dims = [{}]}} : {} -> {}\n",
                             v, cur, dims_str(out_r2), shaped_short(otp_dims, otp_st), shaped_short(out_r2, r2_st));
            cur = fmt::format("%{}", v); v++;
        } else {
            // Reshape dot output to final shape
            if (dot_dims != out_r2) {
                s << fmt::format("    %{} = migraphx.reshape {} {{dims = [{}]}} : {} -> {}\n",
                                 v, cur, dims_str(out_r2),
                                 shaped_short(dot_dims, dot_st_out),
                                 shaped_short(out_r2, r2_st));
                cur = fmt::format("%{}", v); v++;
            }
        }

        s << "    return " << cur << " : " << final_type << "\n";
    } else {
        // No input transform — original output-only logic
        auto dot_st_out = row_strides(dot_out_shape);
        std::string dot_type = shaped(dot_out_shape, dot_st_out);

        // Transposed dims after perm applied to dot_out_shape
        std::vector<int64_t> tp_dims(out_perm.size());
        for (size_t i = 0; i < out_perm.size(); i++) tp_dims[i] = dot_out_shape[out_perm[i]];
        std::vector<int64_t> tp_st(out_perm.size());
        for (size_t i = 0; i < out_perm.size(); i++) tp_st[i] = dot_st_out[out_perm[i]];
        std::string tp_type = shaped(tp_dims, tp_st);

        // dot
        s << fmt::format("    %{} = migraphx.dot %arg0, %arg1 : {}, {} -> {}\n",
                         v, A_type, B_type, dot_type);
        std::string cur = fmt::format("%{}", v); v++;

        // transpose
        s << fmt::format("    %{} = migraphx.transpose {} {{permutation = [{}]}} : {} -> {}\n",
                         v, cur, dims_str(out_perm), dot_type, tp_type);
        cur = fmt::format("%{}", v); v++;

        // reshape to final output
        s << fmt::format("    %{} = migraphx.reshape {} {{dims = [{}]}} : {} -> {}\n",
                         v, cur, dims_str(out_r2), tp_type, final_type);
        cur = fmt::format("%{}", v); v++;

        s << "    return " << cur << " : " << final_type << "\n";
    }

    s << "  }\n}\n";
    return s.str();
}

}  // namespace

FusedRTDOp::FusedRTDOp(
        const CreationContext& ctx,
        const std::shared_ptr<ov::Node>& node,
        IndexCollection&& inputs, IndexCollection&& outputs)
    : OperationBase(ctx, node, std::move(inputs), std::move(outputs))
{
    auto rtd = std::dynamic_pointer_cast<nodes::FusedReshapeTransposeDot>(node);
    OPENVINO_ASSERT(rtd, "Expected FusedReshapeTransposeDot node");

    auto a_shape = node->get_input_shape(0);
    auto b_shape = node->get_input_shape(1);
    auto dot_out_shape = rtd->get_out_r1();

    std::vector<int64_t> a_dims(a_shape.begin(), a_shape.end());
    std::vector<int64_t> b_dims(b_shape.begin(), b_shape.end());

    auto props = ctx.device().props();
    std::string arch(props.gcnArchName);
    auto colon = arch.find(':');
    if (colon != std::string::npos) arch = arch.substr(0, colon);

    std::string mlir_ir = make_rtd_mlir(
        a_dims, b_dims, rtd->get_transpose_b(),
        rtd->get_in_pre(), rtd->get_in_r1(), rtd->get_in_perm(), rtd->get_in_r2(),
        dot_out_shape, rtd->get_out_perm(), rtd->get_out_r2(),
        arch, props.multiProcessorCount);

    if (getenv("ROCM_TRACE_RTD_IR"))
        fprintf(stderr, "[FusedRTD] IR:\n%s\n", mlir_ir.c_str());

    std::string driver = find_driver();
    try {
        auto compiled = rocmlir::compile_mlir_ir_migraphx(mlir_ir, arch, driver);
        grid_x_ = compiled.grid_x;
        block_x_ = compiled.block_x;

        hipError_t err = hipModuleLoadData(&module_, compiled.hsaco.data());
        if (err != hipSuccess) OPENVINO_THROW("FusedRTD: hipModuleLoadData failed");
        err = hipModuleGetFunction(&func_, module_, compiled.kernel_name.c_str());
        if (err != hipSuccess) OPENVINO_THROW("FusedRTD: hipModuleGetFunction failed");

        fprintf(stderr, "[FusedRTD] compiled %s grid=%d blk=%d%s\n",
                compiled.kernel_name.c_str(), grid_x_, block_x_,
                rtd->has_input_transform() ? " (input+output)" : "");
    } catch (const std::exception& e) {
        fprintf(stderr, "[FusedRTD] compilation failed: %s — will throw\n", e.what());
        throw;
    }
}

FusedRTDOp::~FusedRTDOp() {
    if (module_) hipModuleUnload(module_);
}

void FusedRTDOp::Execute(
        const InferenceRequestContext& ctx,
        Inputs inputs, Outputs outputs,
        const Workbuffers&) const {
    void* A = const_cast<void*>(inputs[0].get());
    void* B = const_cast<void*>(inputs[1].get());
    void* C = outputs[0].get();
    void* args[] = {&A, &B, &C};

    hipModuleLaunchKernel(func_, grid_x_, 1, 1, block_x_, 1, 1,
                          0, ctx.getThreadContext().stream().get(),
                          args, nullptr);
}

OPERATION_REGISTER(FusedRTDOp, FusedReshapeTransposeDot);

}  // namespace rocm_gpu
}  // namespace ov
