// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "softmax.hpp"
#include "rocm_attention_softmax.hpp"

#include <fmt/format.h>

#include <rocm/dnn.hpp>
#include <rocm_operation_registry.hpp>
#include <openvino/core/except.hpp>

#include "converters.hpp"
#include "kernels/fused_reduce.hpp"
#include "rocm/constant_factory.hpp"

namespace ov {
namespace rocm_gpu {

static int take(const ov::Shape& shape, size_t i) noexcept { return i < shape.size() ? shape[i] : 1; }

static constexpr long long prod(int a, int b) noexcept { return static_cast<long long>(a) * static_cast<long long>(b); }

/** @brief Dimension Mapping Rationales
 *
 * 1. Problem statement
 *
 * 1.1. miopenSoftmaxForward operates in terms of N,C,H,W, dimensions and mode of operation - channel or instance.
 * 1.2. OpenVINO defines Softmax operation in terms of shape and axis.
 * 1.3. miopenSoftmaxForward supports 4D and 5D tensors while OpenVINO supports any tensor rank
 *
 * 2. Excerpts from miopen Documentation
 *
 * @ref https://docs.rocm.com/deeplearning/miopen/api/index.html#miopenSoftmaxMode_t
 * 3.1.2.26. miopenSoftmaxMode_t
 * miopenSoftmaxMode_t is used to select over which data the miopenSoftmaxForward() and miopenSoftmaxBackward() are
 *computing their results. miopen_SOFTMAX_MODE_INSTANCE The softmax operation is computed per image (N) across the
 *dimensions C,H,W. miopen_SOFTMAX_MODE_CHANNEL  The softmax operation is computed per spatial location (H,W) per image
 *(N) across the dimension C.
 *
 * @ref https://docs.rocm.com/deeplearning/miopen/api/index.html#miopenSoftmaxForward
 * 3.2.98. miopenSoftmaxForward()
 * All tensor formats are supported for all modes and algorithms with 4 and 5D tensors.
 * Performance is expected to be highest with NCHW fully-packed tensors.
 * For more than 5 dimensions tensors must be packed in their spatial dimensions
 *
 * 3. Design Decisions
 *
 * 3.1. In the first release support only tensors of ranks 1-5
 * 3.2. Insert extra dimensions into the shape to get 4D tensor for ranks 1-3
 * 3.3. Merge some dimensions of the shape to get 4D tensor for ranks 4 or 5
 * 3.4. Use channel mode of miopenSoftmaxForward
 * 3.5. Assume default tensor format to be miopenTensorNCHW
 * 3.6. Map axis to dimension C
 *
 * 4. Rank/Axis Mapping of Shapes and Modes
 *
 *rank.axis,    miopenTensorNCHW
 *             N            C     H     W
 *   1.0  - {  1,          d0,    1,    1   },
 *   2.0  - {  1,          d0,   d1,    1   },
 *   2.1  - { d0,          d1,    1,    1   },
 *   3.0  - {  1,          d0,   d1,   d2   },
 *   3.1  - { d0,          d1,   d2,    1   },
 *   3.2  - {d0*d1,        d2,    1,    1   },
 *   4.0  - {  1,          d0,  d1*d2, d3   },
 *   4.1  - { d0,          d1,   d2,   d3   },
 *   4.2  - { d0*d1,       d2,   d3,    1   },
 *   4.3  - { d0*d1*d2,    d3,    1     1   },
 *   5.0  - {  1,          d0,  d1*d2,d3*d4 },
 *   5.1  - { d0,          d1,   d2,  d3*d4 },
 *   5.2  - { d0*d1,       d2,   d3,   d4   },
 *   5.3  - { d0*d1*d2,    d3,   d4,    1   },
 *   5.4  - { d0*d1*d2*d3, d4,    1,    1   },
 */
void SoftmaxOp::mapRankAxis(const ov::Shape& shape, int axis) {
    constexpr long long maxint = std::numeric_limits<int>::max();
    const auto rank = shape.size();
    OPENVINO_ASSERT(rank <= 5 && rank > 0);
    OPENVINO_ASSERT(axis < rank);

    const int d0 = shape[0];
    const int d1 = take(shape, 1);
    const int d2 = take(shape, 2);
    const int d3 = take(shape, 3);
    const int d4 = take(shape, 4);

    switch ((rank << 4) | axis) {
            // N            C     H      W
        case 0x10:
            shape_ = {1, d0, 1, 1};
            break;
        case 0x20:
            shape_ = {1, d0, d1, 1};
            break;
        case 0x21:
            shape_ = {d0, d1, 1, 1};
            break;
        case 0x30:
            shape_ = {1, d0, d1, d2};
            break;
        case 0x31:
            shape_ = {d0, d1, d2, 1};
            break;
        case 0x32:
            shape_ = {d0 * d1, d2, 1, 1};
            OPENVINO_ASSERT(prod(d0, d1) < maxint);
            break;
        case 0x40:
            shape_ = {1, d0, d1 * d2, d3};
            OPENVINO_ASSERT(prod(d1, d2) < maxint);
            break;
        case 0x41:
            shape_ = {d0, d1, d2, d3};
            break;
        case 0x42:
            shape_ = {d0 * d1, d2, d3, 1};
            OPENVINO_ASSERT(prod(d0, d1) < maxint);
            break;
        case 0x43:
            shape_ = {d0 * d1 * d2, d3, 1, 1};
            OPENVINO_ASSERT(prod(d0, d1) * d2 < maxint);
            break;
        case 0x50:
            shape_ = {1, d0, d1 * d2, d3 * d4};
            OPENVINO_ASSERT((prod(d1, d2) < maxint) && (prod(d3, d4) < maxint));
            break;
        case 0x51:
            shape_ = {d0, d1, d2, d3 * d4};
            OPENVINO_ASSERT(prod(d3, d4) < maxint);
            break;
        case 0x52:
            shape_ = {d0 * d1, d2, d3, d4};
            OPENVINO_ASSERT(prod(d0, d1) < maxint);
            break;
        case 0x53:
            shape_ = {d0 * d1 * d2, d3, d4, 1};
            OPENVINO_ASSERT(prod(d0, d1) * d2 < maxint);
            break;
        case 0x54:
            shape_ = {d0 * d1 * d2 * d3, d4, 1, 1};
            OPENVINO_ASSERT(prod(d0, d1) * d2 * d3 < maxint);
            break;
        default:
            throw_ov_exception(
                fmt::format("Unsupported combination of tensor rank ({}) and axis attribute ({})", rank, axis));
    }
}

inline bool isTypeSupported(miopenDataType_t type) {
    switch (type) {
        case miopenFloat:
        case miopenDouble:
        case miopenHalf:
        case miopenBFloat16:
        case miopenInt8:
            return true;
        default:
            return false;
    }
}

SoftmaxOp::SoftmaxOp(const CreationContext& context,
                     const NodeOp& node,
                     IndexCollection&& inputIds,
                     IndexCollection&& outputIds)
    : OperationMIOPEN{context, node, std::move(inputIds), std::move(outputIds)},
      type_{convertDataType<miopenDataType_t>(node.input(0).get_element_type())} {
    if (!isTypeSupported(type_)) {
        throw_ov_exception(fmt::format("SoftmaxOp: unsupported argument type: {}", toString(type_)));
    }
    const int axis = node.get_axis();
    mapRankAxis(node.input(0).get_shape(), axis);
    // Check if MIOpen will crash on this shape (N too large)
    // MIOpen softmax has internal issues when N > ~8M on gfx1100.
    // Fallback to custom warp kernel for fp16 with reasonable cols.
    const size_t N_collapsed = static_cast<size_t>(shape_[0]);
    const int softmax_dim = shape_[1];  // C dimension in NCHW mapping
    use_custom_kernel_ = false;
    if (type_ == miopenHalf && N_collapsed > 4000000 && softmax_dim <= 256) {
        use_custom_kernel_ = true;
        // For axis=last (most common), rows = product of all dims except last
        const auto& orig_shape = node.input(0).get_shape();
        custom_cols_ = static_cast<int>(orig_shape.back());
        custom_rows_ = 1;
        for (size_t i = 0; i < orig_shape.size() - 1; ++i)
            custom_rows_ *= orig_shape[i];
        fprintf(stderr, "[Softmax] Using custom kernel: rows=%zu cols=%d (MIOpen N=%zu too large)\n",
                custom_rows_, custom_cols_, N_collapsed);
    }

    if (!use_custom_kernel_)
        tensor_descriptor_.set(miopenTensorLayout_t::miopenTensorNCHW, type_, 4, shape_.data());
}

void SoftmaxOp::Execute(const InferenceRequestContext& context,
                        Inputs inputs,
                        Outputs outputs,
                        const Workbuffers&) const {
    OPENVINO_ASSERT(inputs.size() == 1);
    OPENVINO_ASSERT(outputs.size() == 1);

    // MIOpen miopenSoftmaxForward crashes when N (collapsed batch dim) is very large
    // (e.g. N=12582912 for swin_base_v2_bs8). Use custom warp kernel as fallback.
    // The custom kernel handles fp16 softmax over the last axis with cols <= 256.
    if (use_custom_kernel_) {
        const auto& stream = context.getThreadContext().stream();
        kernel::launch_plain_softmax(stream.get(), inputs[0].get(), outputs[0].get(),
                             custom_rows_, custom_cols_);
        return;
    }

    throwIfError(miopenSoftmaxForward(context.getThreadContext().dnnHandle().get(),
                                     &rocm::constants::one<float>::value,
                                     tensor_descriptor_.get(),
                                     inputs[0].get(),
                                     &rocm::constants::zero<float>::value,
                                     tensor_descriptor_.get(),
                                     outputs[0].get()));
}

rocmGraphCompatibility SoftmaxOp::GetrocmGraphCompatibility() const { return rocmGraphCompatibility::FULL; }

}  // namespace rocm_gpu
}  // namespace ov

// Factory for Softmax: use MIGraphX fused kernel for attention-tagged nodes.
namespace {
ov::rocm_gpu::OperationBase::Ptr softmaxFactory(
        const ov::rocm_gpu::CreationContext& context,
        const std::shared_ptr<ov::Node>& node,
        ov::rocm_gpu::OperationBase::IndexCollection&& inputIds,
        ov::rocm_gpu::OperationBase::IndexCollection&& outputIds) {
    using namespace ov::rocm_gpu;
    if (RocmAttentionSoftmaxOp::isEnabled() &&
        node->get_rt_info().count("rocm_attn_softmax")) {
        try {
            return std::make_shared<RocmAttentionSoftmaxOp>(
                context, *node,
                OperationBase::IndexCollection{inputIds},
                OperationBase::IndexCollection{outputIds});
        } catch (const std::exception& e) {
            std::cerr << "[AttnSoftmax] Fallback to MIOpen: " << e.what() << "\n";
        }
    }
    // SoftmaxOp constructor takes const NodeOp& (v1::Softmax)
    auto softmax_node = std::dynamic_pointer_cast<ov::op::v1::Softmax>(node);
    OPENVINO_ASSERT(softmax_node, "softmaxFactory: node is not v1::Softmax");
    return std::make_shared<SoftmaxOp>(context, *softmax_node, std::move(inputIds), std::move(outputIds));
}
}  // namespace

namespace ov { namespace rocm_gpu {
OPERATION_REGISTER_FACTORY(softmaxFactory, Softmax);
}}  // namespace ov::rocm_gpu
