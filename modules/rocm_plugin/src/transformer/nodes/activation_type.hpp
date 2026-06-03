// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/core/attribute_adapter.hpp"
#include "openvino/core/enum_names.hpp"

namespace ov::rocm_gpu::nodes {

/**
 * @brief Activation modes for fused convolutions.
 *
 * Mirrors the MIOPEN MIOPENActivationMode_t enum
 */
enum class ActivationMode { SIGMOID, RELU, TANH, CLIPPED_RELU, ELU, SWISH, NO_ACTIVATION };

}  // namespace ov::rocm_gpu::nodes
namespace ov {
template <>
class AttributeAdapter<rocm_gpu::nodes::ActivationMode>
    : public EnumAttributeAdapterBase<rocm_gpu::nodes::ActivationMode> {
public:
    AttributeAdapter(rocm_gpu::nodes::ActivationMode& value)
        : EnumAttributeAdapterBase<rocm_gpu::nodes::ActivationMode>(value) {}

    OPENVINO_RTTI("AttributeAdapter<ActivationMode>");
};
}  // namespace ov
