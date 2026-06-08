// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// FusedConvolutionSlice: fused Conv+Bias+SiLU where the input is a channel-slice
// of a larger tensor.  The kernel receives the FULL (pre-split) input buffer and
// applies a zero-copy Slice transform (encoded in the rocMLIR MLIR) to extract
// channels [c_start, c_start + C) before computing the convolution.
//
// This bypasses OV's standard ConvolutionShapeInference (which would reject a
// full-C input feeding a C/2-filter) by implementing its own validate_and_infer_types().
//
// Node inputs:
//   0: full_input  [N, C_full, H, W]   — the pre-split tensor
//   1: filter      [K, C_slice/G, R, S]
//   2: bias        [K]
//
// Node attributes:
//   c_start  — first channel to use from full_input
//   strides, pads, dilations, auto_pad — same as Convolution

#pragma once

#include <openvino/op/op.hpp>
#include <openvino/op/convolution.hpp>
#include "activation_type.hpp"

namespace ov::rocm_gpu::nodes {

class FusedConvolutionSlice : public ov::op::Op {
public:
    OPENVINO_OP("FusedConvolutionSlice", "rocm_gpu");

    FusedConvolutionSlice() = default;

    FusedConvolutionSlice(const ov::Output<ov::Node>& full_input,
                          const ov::Output<ov::Node>& filter,
                          const ov::Output<ov::Node>& bias,
                          int c_start,
                          const ov::Strides& strides,
                          const ov::CoordinateDiff& pads_begin,
                          const ov::CoordinateDiff& pads_end,
                          const ov::Strides& dilations,
                          const ov::op::PadType& auto_pad)
        : Op({full_input, filter, bias})
        , c_start_(c_start)
        , strides_(strides)
        , pads_begin_(pads_begin)
        , pads_end_(pads_end)
        , dilations_(dilations)
        , auto_pad_(auto_pad)
    {
        constructor_validate_and_infer_types();
    }

    std::shared_ptr<ov::Node> clone_with_new_inputs(const ov::OutputVector& new_args) const override {
        ov::check_new_args_count(this, new_args);
        return std::make_shared<FusedConvolutionSlice>(
            new_args[0], new_args[1], new_args[2],
            c_start_, strides_, pads_begin_, pads_end_, dilations_, auto_pad_);
    }

    bool visit_attributes(ov::AttributeVisitor& visitor) override {
        visitor.on_attribute("c_start", c_start_);
        visitor.on_attribute("strides", strides_);
        visitor.on_attribute("pads_begin", pads_begin_);
        visitor.on_attribute("pads_end", pads_end_);
        visitor.on_attribute("dilations", dilations_);
        visitor.on_attribute("auto_pad", auto_pad_);
        return true;
    }

    void validate_and_infer_types() override {
        // full_input: [N, C_full, H, W]
        // filter:     [K, C_slice/G, R, S]
        // bias:       [K]
        OPENVINO_ASSERT(get_input_size() == 3);

        const auto& in_pshape  = get_input_partial_shape(0);
        const auto& flt_pshape = get_input_partial_shape(1);
        const auto& bias_pshape = get_input_partial_shape(2);
        const auto& in_type = get_input_element_type(0);

        // Must be NCHW (4D)
        OPENVINO_ASSERT(in_pshape.rank().is_static() && in_pshape.rank().get_length() == 4,
                        "FusedConvolutionSlice: input must be 4D NCHW");
        OPENVINO_ASSERT(flt_pshape.rank().is_static() && flt_pshape.rank().get_length() == 4,
                        "FusedConvolutionSlice: filter must be 4D KCRS");

        const auto N    = in_pshape[0];
        const auto H    = in_pshape[2];
        const auto W    = in_pshape[3];
        const auto K    = flt_pshape[0];  // output channels
        const auto R    = flt_pshape[2];
        const auto S    = flt_pshape[3];

        // Validate c_start in bounds
        if (in_pshape[1].is_static() && flt_pshape[1].is_static()) {
            const int C_full  = static_cast<int>(in_pshape[1].get_length());
            const int C_slice = static_cast<int>(flt_pshape[1].get_length());
            OPENVINO_ASSERT(c_start_ >= 0 && c_start_ + C_slice <= C_full,
                "FusedConvolutionSlice: c_start=", c_start_,
                " C_slice=", C_slice, " exceeds C_full=", C_full);
        }

        // Compute output spatial dims using the standard formula
        auto compute_out = [&](ov::Dimension in_dim, int flt, int pad_b, int pad_e, int str, int dil) {
            if (in_dim.is_dynamic()) return ov::Dimension::dynamic();
            int in  = static_cast<int>(in_dim.get_length());
            int eff = dil * (flt - 1) + 1;
            int out = (in + pad_b + pad_e - eff) / str + 1;
            return ov::Dimension(out);
        };

        int pad_h_b = pads_begin_.empty() ? 0 : static_cast<int>(pads_begin_[0]);
        int pad_h_e = pads_end_.empty()   ? 0 : static_cast<int>(pads_end_[0]);
        int pad_w_b = pads_begin_.size() < 2 ? 0 : static_cast<int>(pads_begin_[1]);
        int pad_w_e = pads_end_.size()   < 2 ? 0 : static_cast<int>(pads_end_[1]);
        int str_h   = strides_.empty()   ? 1 : static_cast<int>(strides_[0]);
        int str_w   = strides_.size() < 2 ? 1 : static_cast<int>(strides_[1]);
        int dil_h   = dilations_.empty() ? 1 : static_cast<int>(dilations_[0]);
        int dil_w   = dilations_.size() < 2 ? 1 : static_cast<int>(dilations_[1]);
        int r_val   = flt_pshape[2].is_static() ? static_cast<int>(flt_pshape[2].get_length()) : 1;
        int s_val   = flt_pshape[3].is_static() ? static_cast<int>(flt_pshape[3].get_length()) : 1;

        ov::Dimension OH = compute_out(H, r_val, pad_h_b, pad_h_e, str_h, dil_h);
        ov::Dimension OW = compute_out(W, s_val, pad_w_b, pad_w_e, str_w, dil_w);

        set_output_type(0, in_type, ov::PartialShape{N, K, OH, OW});
    }

    // Accessors used by FusedConvolutionRocMLIR
    int get_c_start() const { return c_start_; }
    const ov::Strides&         get_strides()    const { return strides_;    }
    const ov::CoordinateDiff&  get_pads_begin() const { return pads_begin_; }
    const ov::CoordinateDiff&  get_pads_end()   const { return pads_end_;   }
    const ov::Strides&         get_dilations()  const { return dilations_;  }
    const ov::op::PadType&     get_auto_pad()   const { return auto_pad_;   }

private:
    int c_start_ = 0;
    ov::Strides strides_;
    ov::CoordinateDiff pads_begin_, pads_end_;
    ov::Strides dilations_;
    ov::op::PadType auto_pad_ = ov::op::PadType::EXPLICIT;
};

}  // namespace ov::rocm_gpu::nodes
