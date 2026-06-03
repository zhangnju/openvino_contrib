// Copyright (C) 2021-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once


#include <hip/hip_runtime.h>
#include <half/half.hpp>
using half    = half_float::half;
using float16 = half_float::half;
using __half    = half_float::half;
#include <set>
#include <vector>
#include <hip/hip_runtime.h>
#include <fmt/format.h>
#include "rocm/float16.hpp"
#include "details/rocm_type_traits.hpp"
#include "details/tensor_helpers.hpp"
#include "details/error.hpp"
#include "details/numpy_broadcast_mapper.hpp" 
#include "details/rocm_type_traits.hpp"
#include "details/error.hpp"

namespace ov {
namespace rocm_gpu {
namespace kernel {
template <typename T_INT>
class StridedSliceKernelOp {
public:
    StridedSliceKernelOp(const std::vector<T_INT> src_matrix_sizes,
                         const std::vector<T_INT> dst_matrix_sizes,
                         const std::set<size_t> reverse_axes,
                         const unsigned max_threads_per_block,
                         const unsigned blocks_number,
                         const unsigned threads_per_block,
                         const Type_t element_type,
                         const Type_t element_type_integer);

    void operator()(const hipStream_t stream,
                    const T_INT* src_matrix_sizes,
                    const void* src,
                    const T_INT* begin,
                    const T_INT* end,
                    const T_INT* stride,
                    const T_INT* dst_matrix_sizes,
                    void* dst) const;

private:
    template <typename T>
    void callKernels(const hipStream_t stream,
                     const T_INT* src_matrix_sizes,
                     const void* src,
                     const T_INT* begin,
                     const T_INT* end,
                     const T_INT* stride,
                     const T_INT* dst_matrix_sizes,
                     void* dst) const;
    template <typename T>
    void callStridedSliceKernel(const hipStream_t stream,
                                const T_INT* src_matrix_sizes,
                                const void* src,
                                const T_INT* begin,
                                const T_INT* end,
                                const T_INT* stride,
                                const T_INT* dst_matrix_sizes,
                                void* dst) const;
    template <typename T>
    void callReverseAxesKernel(const hipStream_t stream, void* dst) const;

private:
    std::vector<T_INT> src_matrix_sizes_;
    std::vector<T_INT> dst_matrix_sizes_;
    std::set<size_t> reverse_axes_;
    unsigned max_threads_per_block_;
    unsigned blocks_number_;
    unsigned threads_per_block_;
    Type_t element_type_;
    Type_t element_type_integer_;
};

template class StridedSliceKernelOp<int32_t>;
template class StridedSliceKernelOp<int64_t>;

}  // namespace kernel
}  // namespace rocm_gpu
}  // namespace ov
