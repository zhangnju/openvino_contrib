// Copyright (C) 2021-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <rocm/float16.hpp>

#include "details/rocm_type_traits.hpp"

namespace ov {
namespace rocm_gpu {
namespace kernel {

class Convert {
public:
    Convert(
        Type_t output_element_type, Type_t input_element_type, size_t size, size_t numBlocks, size_t threadsPerBlock);
    Convert(Convert&&) = default;
    Convert& operator=(Convert&&) = default;

    void operator()(hipStream_t, void*, const void*) const;
    using convert_t = void (*)(hipStream_t, size_t, void*, const void*, unsigned, unsigned);

private:
    convert_t convert_kernel_;
    size_t size_;
    size_t num_blocks_;
    size_t threads_per_block_;
};

#ifdef rocm_HAS_BF16_TYPE
template <typename TOutput, typename TInput>
__device__
    typename std::enable_if<std::is_same<TInput, __half>::value || std::is_same<TInput, __nv_bfloat16>::value ||
                                std::is_same<TOutput, __half>::value || std::is_same<TOutput, __nv_bfloat16>::value,
                            TOutput>::type
#else
template <typename TOutput, typename TInput>
__device__
    typename std::enable_if<std::is_same<TInput, __half>::value || std::is_same<TOutput, __half>::value, TOutput>::type
#endif
    cast(TInput in) {
    return static_cast<TOutput>(static_cast<float>(in));
}

#ifdef rocm_HAS_BF16_TYPE
template <typename TOutput, typename TInput>
__device__
    typename std::enable_if<!(std::is_same<TInput, __half>::value || std::is_same<TInput, __nv_bfloat16>::value ||
                              std::is_same<TOutput, __half>::value || std::is_same<TOutput, __nv_bfloat16>::value),
                            TOutput>::type
#else
template <typename TOutput, typename TInput>
__device__ typename std::enable_if<!(std::is_same<TInput, __half>::value || std::is_same<TOutput, __half>::value),
                                   TOutput>::type
#endif
    cast(TInput in) {
    return static_cast<TOutput>(in);
}

}  // namespace kernel
}  // namespace rocm_gpu
}  // namespace ov
