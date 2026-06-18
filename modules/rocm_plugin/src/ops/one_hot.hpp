// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <rocm_operation_base.hpp>
#include <kernels/one_hot.hpp>

namespace ov {
namespace rocm_gpu {

class OneHotOp : public OperationBase {
public:
    OneHotOp(const CreationContext& context,
             const std::shared_ptr<ov::Node>& node,
             IndexCollection&& inputIds,
             IndexCollection&& outputIds);

    void Execute(const InferenceRequestContext& context,
                 Inputs inputs,
                 Outputs outputs,
                 const Workbuffers&) const override;

    rocmGraphCompatibility GetrocmGraphCompatibility() const override {
        return rocmGraphCompatibility::FULL;
    }

private:
    int64_t  depth_{0};
    float    on_val_{1.f};
    float    off_val_{0.f};
    size_t   num_indices_{0};
    bool     indices_i32_{false};
    bool     depth_from_input_{false};
    bool     values_from_input_{false};
    kernel::Type_t out_type_{kernel::Type_t::f32};
};

}  // namespace rocm_gpu
}  // namespace ov
