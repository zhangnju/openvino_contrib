// Copyright (C) 2021-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "details/elementwise_binary.hpp"
#include <hip/hip_runtime.h>
#include <fmt/format.h>
#include "rocm/float16.hpp"
#include "details/rocm_type_traits.hpp"
#include "details/tensor_helpers.hpp"
#include "details/error.hpp"
#include "details/numpy_broadcast_mapper.hpp"

namespace ov {
namespace rocm_gpu {
namespace kernel {

template <typename T>
struct MultiplyOpImpl;

/**
 * Performs element-wise multiplication operation with two given tensors applying
 * broadcasting if needed.
 */
class Multiply {
public:
    Multiply(Type_t element_type, size_t out_num_elements, size_t max_threads_per_block);

    void operator()(hipStream_t stream,
                    const void* in0,
                    const NumpyBroadcastMapper& in0_mapper,
                    const void* in1,
                    const NumpyBroadcastMapper& in1_mapper,
                    void* out) const;

    // Fast periodic broadcast: out[i] = in0[i] * in1[i % period]
    // Defined in multiply.hip where MultiplyOpImpl<__half>::op is visible.
    void launch_bcast_periodic_f16(hipStream_t stream,
                                   const void* in0, const void* in1, void* out,
                                   size_t total, size_t period, size_t repeat) const;

    static constexpr bool has_bcast_periodic = true;

private:
    ElementwiseBinary<AllElementTypesSwitch, MultiplyOpImpl> impl_;
};

}  // namespace kernel
}  // namespace rocm_gpu
}  // namespace ov
