// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// FusedConvolutionSliceOut: Conv+Bias+SliceOutput+SiLU+Add(skip) fused into one kernel.
//
// MIGraphX pattern: mlir_convolution_broadcast_slice_add_sigmoid_mul_add
// The convolution produces K output channels; we only keep channels [c_out_start, c_out_end).
// After slicing, SiLU is applied and the result is added to a skip-connection tensor.
//
// This is the output-side dual of FusedConvolutionSlice (which slices the INPUT).
// Here the FULL K channels are computed, then only a subset is written to output.
// MIGraphX encodes this via a Slice rock.transform on the conv output.
//
// Node inputs:
//   0: data       [N, C, H, W]
//   1: filter     [K, C/G, R, S]   (K = full output channels)
//   2: bias       [K]
//   3: skip_input [N, K_slice, OH, OW]  (same shape as output)
//
// Node attributes:
//   c_out_start, c_out_end  — channel range to keep from the K-channel conv output
//
// Output: [N, K_slice, OH, OW]  where K_slice = c_out_end - c_out_start

#pragma once

#include <openvino/op/op.hpp>
#include <openvino/op/convolution.hpp>
#include "activation_type.hpp"

namespace ov::rocm_gpu::nodes {

class FusedConvolutionSliceOut : public ov::op::Op {
public:
    OPENVINO_OP("FusedConvolutionSliceOut", "rocm_gpu");

    FusedConvolutionSliceOut() = default;

    // 4-input ctor: data, filter, bias, skip_input
    FusedConvolutionSliceOut(const ov::Output<ov::Node>& data,
                              const ov::Output<ov::Node>& filter,
                              const ov::Output<ov::Node>& bias,
                              const ov::Output<ov::Node>& skip_input,
                              int c_out_start,
                              int c_out_end,
                              const ov::Strides& strides,
                              const ov::CoordinateDiff& pads_begin,
                              const ov::CoordinateDiff& pads_end,
                              const ov::Strides& dilations,
                              const ov::op::PadType& auto_pad)
        : Op({data, filter, bias, skip_input})
        , c_out_start_(c_out_start)
        , c_out_end_(c_out_end)
        , strides_(strides)
        , pads_begin_(pads_begin)
        , pads_end_(pads_end)
        , dilations_(dilations)
        , auto_pad_(auto_pad)
    {
        constructor_validate_and_infer_types();
    }

    std::shared_ptr<ov::Node> clone_with_new_inputs(const ov::OutputVector& args) const override {
        ov::check_new_args_count(this, args);
        return std::make_shared<FusedConvolutionSliceOut>(
            args[0], args[1], args[2], args[3],
            c_out_start_, c_out_end_,
            strides_, pads_begin_, pads_end_, dilations_, auto_pad_);
    }

    bool visit_attributes(ov::AttributeVisitor& v) override {
        v.on_attribute("c_out_start", c_out_start_);
        v.on_attribute("c_out_end",   c_out_end_);
        v.on_attribute("strides",     strides_);
        v.on_attribute("pads_begin",  pads_begin_);
        v.on_attribute("pads_end",    pads_end_);
        v.on_attribute("dilations",   dilations_);
        v.on_attribute("auto_pad",    auto_pad_);
        return true;
    }

    void validate_and_infer_types() override {
        OPENVINO_ASSERT(get_input_size() == 4);
        const auto& in_pshape  = get_input_partial_shape(0);
        const auto& flt_pshape = get_input_partial_shape(1);
        const auto& in_type    = get_input_element_type(0);

        OPENVINO_ASSERT(in_pshape.rank().is_static() && in_pshape.rank().get_length() == 4);
        OPENVINO_ASSERT(flt_pshape.rank().is_static() && flt_pshape.rank().get_length() == 4);
        OPENVINO_ASSERT(c_out_end_ > c_out_start_ && c_out_start_ >= 0);

        const int K_slice = c_out_end_ - c_out_start_;
        const auto N = in_pshape[0];
        const auto H = in_pshape[2];
        const auto W = in_pshape[3];

        const int R = flt_pshape[2].is_static() ? static_cast<int>(flt_pshape[2].get_length()) : 1;
        const int S = flt_pshape[3].is_static() ? static_cast<int>(flt_pshape[3].get_length()) : 1;
        const int ph_b = pads_begin_.empty() ? 0 : static_cast<int>(pads_begin_[0]);
        const int ph_e = pads_end_.empty()   ? 0 : static_cast<int>(pads_end_[0]);
        const int pw_b = pads_begin_.size() < 2 ? 0 : static_cast<int>(pads_begin_[1]);
        const int pw_e = pads_end_.size()   < 2 ? 0 : static_cast<int>(pads_end_[1]);
        const int sh   = strides_.empty()   ? 1 : static_cast<int>(strides_[0]);
        const int sw   = strides_.size() < 2 ? 1 : static_cast<int>(strides_[1]);
        const int dh   = dilations_.empty() ? 1 : static_cast<int>(dilations_[0]);
        const int dw   = dilations_.size() < 2 ? 1 : static_cast<int>(dilations_[1]);

        auto compute_out = [](ov::Dimension d, int f, int pb, int pe, int s, int dil) {
            if (d.is_dynamic()) return ov::Dimension::dynamic();
            return ov::Dimension((static_cast<int>(d.get_length()) + pb + pe - dil*(f-1) - 1) / s + 1);
        };

        ov::Dimension OH = compute_out(H, R, ph_b, ph_e, sh, dh);
        ov::Dimension OW = compute_out(W, S, pw_b, pw_e, sw, dw);

        set_output_type(0, in_type, ov::PartialShape{N, K_slice, OH, OW});
    }

    int get_c_out_start() const { return c_out_start_; }
    int get_c_out_end()   const { return c_out_end_;   }
    const ov::Strides&        get_strides()    const { return strides_;    }
    const ov::CoordinateDiff& get_pads_begin() const { return pads_begin_; }
    const ov::CoordinateDiff& get_pads_end()   const { return pads_end_;   }
    const ov::Strides&        get_dilations()  const { return dilations_;  }
    const ov::op::PadType&    get_auto_pad()   const { return auto_pad_;   }

private:
    int c_out_start_ = 0, c_out_end_ = 0;
    ov::Strides strides_;
    ov::CoordinateDiff pads_begin_, pads_end_;
    ov::Strides dilations_;
    ov::op::PadType auto_pad_ = ov::op::PadType::EXPLICIT;
};

}  // namespace ov::rocm_gpu::nodes
