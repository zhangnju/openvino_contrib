// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <rocm_operation_base.hpp>
#include <optional>
#include <vector>

namespace ov {
namespace rocm_gpu {

class TransposeOp : public OperationBase {
public:
    TransposeOp(const CreationContext& context,
                const std::shared_ptr<ov::Node>& node,
                IndexCollection&& inputIds,
                IndexCollection&& outputIds);

    void Execute(const InferenceRequestContext& context,
                 Inputs inputTensors,
                 Outputs outputTensors,
                 const Workbuffers& workbuffers) const override;

    rocmGraphCompatibility GetrocmGraphCompatibility() const override;

private:
    std::vector<int64_t> src_shape_;
    std::optional<std::vector<int32_t>> perm_;
    size_t element_size_;
    int32_t ndim_;
    ov::element::Type_t input_type_;
    ov::element::Type_t perm_type_;

    static bool isPermutationTensorSpecified(const ov::Node& node);
    static std::optional<std::vector<int32_t>> tryToExtractPermutation(const ov::Node& node, int32_t ndim);

    template <typename T>
    std::vector<int32_t> downloadPermutationVector(const InferenceRequestContext& context,
                                                    rocm::DevicePointer<const void*> ptr) const;
};

}  // namespace rocm_gpu
}  // namespace ov
