// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "activation_type.hpp"

namespace ov {
std::ostream& operator<<(std::ostream& s, const rocm_gpu::nodes::ActivationMode& type) {
    return s << as_string(type);
}
template <>
EnumNames<rocm_gpu::nodes::ActivationMode>&
EnumNames<rocm_gpu::nodes::ActivationMode>::get() {
    static auto enum_names = EnumNames<rocm_gpu::nodes::ActivationMode>(
        "rocm_gpu::nodes::ActivationMode",
        {{"sigmoid", rocm_gpu::nodes::ActivationMode::SIGMOID},
         {"relu", rocm_gpu::nodes::ActivationMode::RELU},
         {"tanh", rocm_gpu::nodes::ActivationMode::TANH},
         {"clipped_relu", rocm_gpu::nodes::ActivationMode::CLIPPED_RELU},
         {"elu", rocm_gpu::nodes::ActivationMode::ELU},
         {"swish", rocm_gpu::nodes::ActivationMode::SWISH},
         {"no_activation", rocm_gpu::nodes::ActivationMode::NO_ACTIVATION}});
    return enum_names;
}
} // namespace ov