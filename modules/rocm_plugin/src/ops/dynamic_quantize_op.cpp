// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// Executor for nodes::DynamicQuantizeStats: one kernel makes a single pass over
// the activation tensor computing global min and max simultaneously, then emits
//   span = max(0, max(x)) - min(0, min(x))
// as an f32 scalar. Replaces two separate full-tensor reductions (ReduceMax +
// ReduceMin) plus Maximum/Minimum/Subtract with a single read of x.

#include "dynamic_quantize_op.hpp"
#include <rocm_operation_registry.hpp>
#include <openvino/core/except.hpp>
#include <hip/hip_runtime.h>

namespace ov {
namespace rocm_gpu {

namespace {

__device__ inline float warp_min(float v) {
    for (int o = 16; o > 0; o >>= 1) v = fminf(v, __shfl_xor(v, o));
    return v;
}
__device__ inline float warp_max(float v) {
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor(v, o));
    return v;
}

// atomicMin/Max for float via int bit-reinterpretation (monotonic ordering trick).
__device__ inline void atomicMinF(float* addr, float val) {
    int* iaddr = (int*)addr;
    int old = *iaddr, assumed;
    do {
        assumed = old;
        float cur = __int_as_float(assumed);
        if (cur <= val) break;
        old = atomicCAS(iaddr, assumed, __float_as_int(val));
    } while (old != assumed);
}
__device__ inline void atomicMaxF(float* addr, float val) {
    int* iaddr = (int*)addr;
    int old = *iaddr, assumed;
    do {
        assumed = old;
        float cur = __int_as_float(assumed);
        if (cur >= val) break;
        old = atomicCAS(iaddr, assumed, __float_as_int(val));
    } while (old != assumed);
}

// Initialize the {min,max} accumulator to {0,0} (range includes 0 per ONNX spec).
__global__ void dq_init(float* __restrict__ acc) {
    acc[0] = 0.f;  // global min
    acc[1] = 0.f;  // global max
}

// Multi-block grid-stride reduction: each block reduces its slice, then one
// atomic min/max per block updates the global accumulator. High parallelism.
__global__ void dq_reduce(const float* __restrict__ x, size_t n, float* __restrict__ acc) {
    __shared__ float s_min[32];
    __shared__ float s_max[32];
    int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;

    float vmin = 0.f, vmax = 0.f;
    for (size_t i = (size_t)blockIdx.x * blockDim.x + tid;
         i < n; i += (size_t)gridDim.x * blockDim.x) {
        float v = x[i];
        vmin = fminf(vmin, v);
        vmax = fmaxf(vmax, v);
    }
    vmin = warp_min(vmin); vmax = warp_max(vmax);
    if (lane == 0) { s_min[warp] = vmin; s_max[warp] = vmax; }
    __syncthreads();

    if (warp == 0) {
        int nwarps = (blockDim.x + 31) >> 5;
        float mn = (lane < nwarps) ? s_min[lane] : 0.f;
        float mx = (lane < nwarps) ? s_max[lane] : 0.f;
        mn = warp_min(mn); mx = warp_max(mx);
        if (lane == 0) { atomicMinF(&acc[0], mn); atomicMaxF(&acc[1], mx); }
    }
}

// span_out = max - min ; xmin_out = min
__global__ void dq_finalize(const float* __restrict__ acc,
                            float* __restrict__ span_out,
                            float* __restrict__ xmin_out) {
    span_out[0] = acc[1] - acc[0];
    xmin_out[0] = acc[0];
}

}  // namespace

DynamicQuantizeOp::DynamicQuantizeOp(const CreationContext& context,
                                     const std::shared_ptr<ov::Node>& node,
                                     IndexCollection&& inputIds,
                                     IndexCollection&& outputIds)
    : OperationBase(context, node, std::move(inputIds), std::move(outputIds)) {
    auto dq = std::dynamic_pointer_cast<nodes::DynamicQuantizeStats>(node);
    OPENVINO_ASSERT(dq, "DynamicQuantizeOp: expected DynamicQuantizeStats node, got: ",
                    node->get_type_name());
    x_type_ = node->get_input_element_type(0);
    OPENVINO_ASSERT(x_type_ == ov::element::f32,
                    "DynamicQuantizeOp: only f32 input supported, got ", x_type_.get_type_name());
    n_elems_ = ov::shape_size(node->get_input_shape(0));
}

WorkbufferRequest DynamicQuantizeOp::GetWorkBufferRequest() const {
    // mutable buffer [0]: 2 floats {global_min, global_max} accumulator.
    return {{}, {2 * sizeof(float)}};
}

void DynamicQuantizeOp::Execute(const InferenceRequestContext& context,
                                Inputs inputs,
                                Outputs outputs,
                                const Workbuffers& wbs) const {
    OPENVINO_ASSERT(inputs.size() == 1, GetName(), ": expected 1 input");
    OPENVINO_ASSERT(outputs.size() == 2, GetName(), ": expected 2 outputs");
    OPENVINO_ASSERT(!wbs.mutable_buffers.empty(), GetName(), ": need acc workbuffer");

    auto stream = context.getThreadContext().stream().get();
    const float* x  = static_cast<const float*>(inputs[0].get());
    float* span_out = static_cast<float*>(outputs[0].get());
    float* xmin_out = static_cast<float*>(outputs[1].get());
    float* acc      = static_cast<float*>(wbs.mutable_buffers[0].get());

    // Cap grid so each block does meaningful work; enough blocks for full occupancy.
    constexpr unsigned kBlock = 256;
    unsigned grid = (unsigned)((n_elems_ + kBlock - 1) / kBlock);
    if (grid > 512) grid = 512;
    if (grid == 0) grid = 1;

    dq_init<<<1, 1, 0, stream>>>(acc);
    dq_reduce<<<grid, kBlock, 0, stream>>>(x, n_elems_, acc);
    dq_finalize<<<1, 1, 0, stream>>>(acc, span_out, xmin_out);
}

OPERATION_REGISTER(DynamicQuantizeOp, DynamicQuantizeStats);

}  // namespace rocm_gpu
}  // namespace ov
