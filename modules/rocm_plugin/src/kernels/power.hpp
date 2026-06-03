// Copyright (C) 2021-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <hip/hip_runtime.h>
#include <fmt/format.h>
#include "rocm/float16.hpp"
#include "details/rocm_type_traits.hpp"
#include "details/tensor_helpers.hpp"
#include "details/error.hpp"
#include "details/numpy_broadcast_mapper.hpp" 
#include "details/rocm_type_traits.hpp"
//#include "details/elementwise_binary.hpp"

namespace ov {
namespace rocm_gpu {
namespace kernel {

template <typename T>
struct PowerOpImpl;

/**
 * Elementwise power operation.
 */
class Power {
public:
    Power(Type_t element_type, size_t out_num_elements, size_t max_threads_per_block);

    void operator()(hipStream_t stream,
                    const void* in0,
                    const NumpyBroadcastMapper& in0_mapper,
                    const void* in1,
                    const NumpyBroadcastMapper& in1_mapper,
                    void* out) const;

private:
    /*
    using SupportedElementTypes = ElementTypesSwitch<
#ifdef rocm_HAS_BF16_TYPE
        Type_t::bf16,
#endif
        Type_t::f16,
        Type_t::f32,
        Type_t::f64,
        Type_t::i8,
        Type_t::i16,
        Type_t::i32,
        Type_t::i64,
        Type_t::u8,
        Type_t::u16,
        Type_t::u32,
        Type_t::u64>;

    ElementwiseBinary<SupportedElementTypes, PowerOpImpl> impl_;
    */
    Type_t element_type_;
    size_t num_elements_;
    size_t max_threads_per_block_;

    size_t num_blocks_;
    size_t threads_per_block_;
};

}  // namespace kernel
}  // namespace rocm_gpu
}  // namespace ov
