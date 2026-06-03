// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <cstddef>

#include "details/rocm_type_traits.hpp"
#include "details/eltwise.hpp"

namespace ov {
namespace rocm_gpu {
namespace kernel {

class ConstModePad {
public:
    explicit ConstModePad(eltwise::KernelExecAttrs&& kernelExecAttrs,
                          kernel::Type_t dtype,
                          std::size_t outputRank,
                          int elementsPerThread,
                          size_t elementsNumber,
                          bool nchw_conv_padding);
    void operator()(hipStream_t stream,
                    const void* src,
                    void* dst,
                    const void* begin,
                    const std::size_t* srcShape,
                    const std::size_t* dstShape,
                    const void* padValue) const;

    template <typename T>
    void callKernel(hipStream_t stream,
                    const void* src,
                    void* dst,
                    const void* begin,
                    const std::size_t* srcShape,
                    const std::size_t* dstShape,
                    const void* padValue) const;

    template <typename T, int PayloadRank>
    void callKernel(hipStream_t stream,
                    const void* src,
                    void* dst,
                    const void* begin,
                    const std::size_t* srcShape,
                    const std::size_t* dstShape,
                    const void* padValue) const;

    template <typename T>
    void callNCHWFormatConvKernel(hipStream_t stream,
                                  const void* src,
                                  void* dst,
                                  const void* begin,
                                  const std::size_t* srcShape,
                                  const std::size_t* dstShape,
                                  const void* padValue) const;

    constexpr static unsigned kWarpsPerBlock = 8;
    constexpr static unsigned kElementsPerThread = 1;

private:
    eltwise::KernelExecAttrs kernel_exec_attrs_;
    kernel::Type_t dtype_;
    std::size_t output_rank_;

    int max_elements_per_thread_;
    size_t elements_number_;
    int blocks_number_;
    int threads_per_block_;
    bool nchw_conv_padding_;
};

}  // namespace kernel
}  // namespace rocm_gpu
}  // namespace ov
