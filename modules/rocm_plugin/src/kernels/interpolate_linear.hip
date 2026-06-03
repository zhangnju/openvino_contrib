#include "hip/hip_runtime.h"
// Copyright (C) 2022-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <fmt/format.h>

#include <algorithm>
#include <cmath>
#include <math.h>
#include "details/error.hpp"

//#include "convert.hpp"
#include "interpolate_details.hpp"
#include "interpolate_linear.hpp"

namespace ov {
namespace rocm_gpu {
namespace kernel {

//template <typename float>
static inline __device__ float triangle_coeffs(float dz) {
    return fmaxf(float{}, float{1.0f} - abs(dz));
}

//template <typename T, typename ComputeType>
static __global__ void interpolate_linear(const float* input,
                                          float* output,
                                          const InterpolateLinear::Props* props,
                                          const InterpolateLinear::Index* indices,
                                          const unsigned num_of_indices) {
    const unsigned output_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (output_idx >= shape_size(props->output_shape)) return;

    //using float = ComputeType;
    using details = InterpolateLinear::details;
    using IntShape = InterpolateLinear::IntShape;
    using UIntShape = InterpolateLinear::UIntShape;
    using CTShape = Shape<float, InterpolateLinear::MAX_SHAPE_RANK>;

    UIntShape output_coords{};
    shape_indices(props->output_shape, output_idx, output_coords);

    CTShape input_coords;
    details::shape_copy(output_coords, input_coords);
    IntShape input_coords_r;
    details::shape_copy(output_coords, input_coords_r);
    for (unsigned i = 0; i < props->num_of_axes; ++i) {
        const unsigned axis = props->axes[i];
        const float in_coord = details::get_original_coordinate(props->transform_mode,
                                                                 output_coords[axis],
                                                                 static_cast<float>(props->scales[i]),
                                                                 props->output_shape[axis],
                                                                 props->input_shape[axis]);
        input_coords[axis] = in_coord;
        input_coords_r[axis] = static_cast<int>(round(in_coord));
    }

    CTShape a;
    details::shape_copy(props->a, a);
    const float prod_of_a{props->prod_of_a};

    float summa = 0.0;
    float wsum = 0.0;
    for (unsigned index_i = 0; index_i < num_of_indices; ++index_i) {
        const IntShape& index = indices[index_i].v;
        UIntShape inner_coords;
        details::shape_copy(output_coords, inner_coords);
        bool inner_coords_are_valid = true;
        for (unsigned i = 0; i < props->num_of_axes; ++i) {
            const unsigned axis = props->axes[i];
            int coord = index[i] + input_coords_r[axis];
            if ((coord < 0) || (coord >= props->input_shape[axis])) {
                inner_coords_are_valid = false;
                break;
            }
            inner_coords[axis] = static_cast<unsigned>(coord);
        }
        if (!inner_coords_are_valid) continue;

        const unsigned inner_index = flat_address_by_shape(props->input_shape, inner_coords);
        float inner_value = static_cast<float>(input[inner_index]);

        CTShape dz{};
        for (unsigned i = 0; i < props->num_of_axes; ++i) {
            const unsigned axis = props->axes[i];
            dz[i] = input_coords[axis] - static_cast<float>(inner_coords[axis]);
        }

        float prod_of_triangle_coeffs = 1.0;
        for (unsigned i = 0; i < props->num_of_axes; ++i) {
            prod_of_triangle_coeffs *= triangle_coeffs(a[i] * dz[i]);
        }

        float w = prod_of_a * prod_of_triangle_coeffs;
        wsum += w;
        summa += w * inner_value;
    }

    if (wsum == float{})
        output[output_idx] = float{};
    else
        output[output_idx] = static_cast<float>(summa / wsum);
}

InterpolateLinear::InterpolateLinear(std::vector<size_t> in_shape,
                                     std::vector<size_t> axes,
                                     std::vector<float> scales,
                                     std::vector<size_t> out_shape,
                                     CoordinateTransformMode transform_mode,
                                     bool antialias,
                                     Type_t element_type,
                                     size_t max_threads_per_block)
    : InterpolateBase{element_type},
      props_device_ptr_{nullptr},
      indices_device_ptr_{nullptr},
      num_blocks_{},
      threads_per_block_{},
      element_type_{element_type} {
    assertThrow(in_shape.size() == out_shape.size(), "in_shape.size() != out_shape.size()");
    assertThrow(in_shape.size() <= MAX_SHAPE_RANK, "in_shape.size() > MAX_SHAPE_RANK");
    assertThrow(axes.size() == scales.size(), "axes.size() != scales.size()");

    std::copy(in_shape.begin(), in_shape.end(), props_.input_shape);
    std::copy(axes.begin(), axes.end(), props_.axes);
    std::copy(scales.begin(), scales.end(), props_.scales);
    std::copy(out_shape.begin(), out_shape.end(), props_.output_shape);

    bool is_downsample = false;
    for (size_t i = 0; i < scales.size(); ++i) {
        is_downsample = is_downsample || (scales[i] < 1.0f);
    }

    props_.num_of_axes = axes.size();

    props_.prod_of_a = 1.0f;
    for (size_t i = 0; i < props_.num_of_axes; ++i) {
        props_.a[i] = (is_downsample && antialias) ? props_.scales[i] : 1.0f;
        props_.prod_of_a *= props_.a[i];
    }

    for (size_t i = 0; i < props_.num_of_axes; ++i) {
        if (props_.scales[i] > 1.0) {
            props_.r[i] = 2;
        } else {
            props_.r[i] = static_cast<unsigned>(std::ceil(2.0f / props_.a[i]));
        }
    }

    std::vector<int> indices_shape;
    for (size_t i = 0; i < props_.num_of_axes; ++i) {
        indices_shape.push_back(props_.r[i] * 2 + 1);
    }
    details::ShapeIterator iter{indices_shape};
    while (!iter.end()) {
        Index index;
        for (size_t i = 0; i < props_.num_of_axes; ++i) {
            index.v[i] = iter.value()[i] - props_.r[i];
        }
        indices_.push_back(index);
        iter.advance();
    }

    props_.transform_mode = transform_mode;

    std::tie(num_blocks_, threads_per_block_) =
        calculateElementwiseGrid(shape_size(props_.output_shape), max_threads_per_block);
}

void InterpolateLinear::operator()(const hipStream_t stream, const void* input, void* output) const {
    const float* input_fp32 = nullptr;
    float *output_fp32 = nullptr;
    switch (element_type_) {
        /*
        case Type_t::f16:
            return callKernel<__half, __half>(stream, input, output);
        */
        case Type_t::f32:
            return callKernel(stream, input_fp32, output_fp32);
        /*
#ifdef rocm_HAS_BF16_TYPE
        case Type_t::bf16:
            return callKernel<hip_bfloat16, hip_bfloat16>(stream, input, output);
#endif
        case Type_t::i8:
            return callKernel<int8_t, float>(stream, input, output);
        case Type_t::u8:
            return callKernel<uint8_t, float>(stream, input, output);
        */
        default:
            throw_ov_exception(
                fmt::format("Element type = {} is not supported by InterpolateLinear operation.", element_type_));
    }
}

//template <typename T, typename float>
void InterpolateLinear::callKernel(const hipStream_t stream, const float* input, float* output) const {
    kernel::interpolate_linear<<<num_blocks_, threads_per_block_, 0, stream>>>(static_cast<const float*>(input),
                                                         static_cast<float*>(output),
                                                         static_cast<const Props*>(props_device_ptr_),
                                                         static_cast<const Index*>(indices_device_ptr_),
                                                         indices_.size());
}

std::vector<size_t> InterpolateLinear::immutableWorkbufferSizes() const {
    return {sizeof(Props), sizeof(Index) * indices_.size()};
}

void InterpolateLinear::initImmutableWorkbuffers(const std::vector<void*>& buffers) {
    kernel::throwIfError(
        hipMemcpyAsync(buffers[0], static_cast<const void*>(&props_), sizeof(props_), hipMemcpyHostToDevice));
    props_device_ptr_ = buffers[0];

    kernel::throwIfError(hipMemcpyAsync(
        buffers[1], static_cast<const void*>(&indices_[0]), sizeof(Index) * indices_.size(), hipMemcpyHostToDevice));
    indices_device_ptr_ = buffers[1];
}

}  // namespace kernel

}  // namespace rocm_gpu
}  // namespace ov
