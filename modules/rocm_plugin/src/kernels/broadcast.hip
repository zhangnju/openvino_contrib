#include "hip/hip_runtime.h"
// Copyright (C) 2021-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <fmt/format.h>

#include "broadcast.hpp"
#include "details/error.hpp"
#include "details/tensor_helpers.hpp"
#include "details/type_validator.hpp"

namespace ov {
namespace rocm_gpu {
namespace kernel {
//fix me, do we need to have a type restriction? 
template <typename T>
static __global__ void broadcast(const T* src, NumpyBroadcastMapper src_mapper, T* dst, size_t dst_num_elements) {
    const unsigned dst_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (dst_idx >= dst_num_elements) {
        return;
    }
    const unsigned src_idx = src_mapper.srcIndex(dst_idx);
    dst[dst_idx] = src[src_idx];
}

Broadcast::Broadcast(ov::rocm_gpu::kernel::Type_t element_type, size_t dst_num_elements, size_t max_threads_per_block)
    : element_type_{element_type}, dst_num_elements_{dst_num_elements} {
    TypeValidator<AllElementTypesSwitch>::check(element_type);
    std::tie(num_blocks_, threads_per_block_) = calculateElementwiseGrid(dst_num_elements_, max_threads_per_block);
}

void Broadcast::operator()(const hipStream_t stream,
                           const void* src,
                           const NumpyBroadcastMapper& src_mapper,
                           void* dst) const {
    AllElementTypesSwitch::switch_(element_type_, *this, stream, src, src_mapper, dst);
}

template <typename T>
constexpr void Broadcast::case_(hipStream_t stream,
                                const void* src,
                                const NumpyBroadcastMapper& src_mapper,
                                void* dst) const noexcept {
    broadcast<T><<<num_blocks_, threads_per_block_, 0, stream>>>(
        static_cast<const T*>(src), src_mapper, static_cast<T*>(dst), dst_num_elements_);
}

template <typename T>
void Broadcast::default_(T t, hipStream_t, const void*, const NumpyBroadcastMapper&, void*) const noexcept {
    throw_ov_exception(fmt::format("Element type = {} is not supported by Broadcast operation.", t));
}
}  // namespace kernel
}  // namespace rocm_gpu
}  // namespace ov
