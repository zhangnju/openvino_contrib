// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <rocm/runtime.hpp>
#include <rocm_operation_base.hpp>

#include "kernels/convert.hpp"

namespace ov {
namespace rocm_gpu {

class ConvertOp : public OperationBase {
public:
    ConvertOp(const CreationContext& context,
              const std::shared_ptr<ov::Node>& node,
              IndexCollection&& inputIds,
              IndexCollection&& outputIds);

    void Execute(const InferenceRequestContext& context,
                 Inputs inputTensors,
                 Outputs outputTensors,
                 const Workbuffers& workbuffers) const override;

    rocmGraphCompatibility GetrocmGraphCompatibility() const override;

    using Type_t = ov::element::Type_t;
    using convert_t = void (*)(
        const rocm::Stream&, size_t, rocm::DevicePointer<void*>, rocm::DevicePointer<const void*>, unsigned, unsigned);

private:
    std::optional<kernel::Convert> convert_kernel_;
};

}  // namespace rocm_gpu
}  // namespace ov
