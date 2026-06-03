// Copyright (C) 2022-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <vector>

#include "details/rocm_type_traits.hpp"
#include "details/error.hpp"
#include "interpolate_base.hpp"

namespace ov {
namespace rocm_gpu {
namespace kernel {

class InterpolateNearest : public InterpolateBase {
public:
    /// \brief Round modes for the nearest interpolation.
    enum class NearestMode { round_prefer_floor, round_prefer_ceil, floor, ceil, simple };

    InterpolateNearest(size_t num_blocks,
                       size_t threads_per_block,
                       ov::rocm_gpu::kernel::Type_t element_type,
                       bool upscale,
                       NearestMode nearest_mode,
                       CoordinateTransformMode transform_mode);

    void operator()(const hipStream_t stream,
                    const void* src,
                    const size_t* input_strides,
                    const size_t* output_strides,
                    const float* scales,
                    const size_t* input_shape,
                    const size_t* output_shape,
                    void* dst) const;

private:
    void callKernel(const hipStream_t stream,
                    const float* src,
                    const size_t* input_strides,
                    const size_t* output_strides,
                    const float* scales,
                    const size_t* input_shape,
                    const size_t* output_shape,
                    float* dst) const;

    void callKernel(const hipStream_t stream,
                    const __half* src,
                    const size_t* input_strides,
                    const size_t* output_strides,
                    const float* scales,
                    const size_t* input_shape,
                    const size_t* output_shape,
                    __half* dst) const;

private:
    size_t num_blocks_;
    size_t threads_per_block_;
    Type_t element_type_;
    bool use_optimized_kernel_;
    NearestMode nearest_mode_;
    CoordinateTransformMode transform_mode_;
};

}  // namespace kernel
}  // namespace rocm_gpu
}  // namespace ov
