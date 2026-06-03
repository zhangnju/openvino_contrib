// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief A header for advanced hardware related properties for rocm plugin
 *        To use in set_property, compile_model, import_model, get_property methods
 *
 * @file rocm/properties.hpp
 */
#pragma once

#include "openvino/runtime/properties.hpp"

namespace ov {

/**
 * @brief Namespace with rocm GPU specific properties
 */
namespace rocm_gpu {

/**
 * @brief Defines if benchmarks should be run to determine fastest algorithms for some operations (e.g. Convolution)
 */
static constexpr Property<bool, PropertyMutability::RW> operation_benchmark{"ROCM_OPERATION_BENCHMARK"};

/**
 * @brief Specifies if rocm plugin attempts to use rocm Graph feature to speed up sequential network inferences
 */
static constexpr Property<bool, PropertyMutability::RW> use_hip_graph{"ROCM_USE_HIP_GRAPH"};

/**
 * @brief Read-only property showing number of used rocm Graphs
 */
static constexpr Property<size_t, PropertyMutability::RO> number_of_hip_graphs{"ROCM_NUMBER_OF_HIP_GRAPHS"};

}  // namespace rocm_gpu
}  // namespace ov
