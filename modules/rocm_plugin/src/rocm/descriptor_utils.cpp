// Copyright (C) 2021-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "descriptor_utils.hpp"

#include <array>
#include <ops/converters.hpp>

namespace rocm {

DnnTensorDescriptor makeDnnTensorDescr(const ov::element::Type& type, const ov::Shape& shape) {
    OPENVINO_ASSERT(!shape.empty());
    //OPENVINO_ASSERT(shape.size() <= MIOPEN_DIM_MAX);
    std::vector<int> dims;
    std::transform(shape.begin(), shape.end(), std::back_inserter(dims), [](auto v) { return static_cast<int>(v); });
    const int MIOPEN_DIM_MIN =
        4;  // see note here: https://docs.rocm.com/deeplearning/MIOPEN/api/index.html#MIOPENSetTensorNdDescriptor
    while (dims.size() < MIOPEN_DIM_MIN) {
        dims.push_back(1);
    }
    decltype(dims) strides(dims.size(), 0);
    strides.back() = 1;
    for (int i = dims.size() - 1; i > 0; i--) strides[i - 1] = strides[i] * dims[i];
    return DnnTensorDescriptor{}.set(
        ov::rocm_gpu::convertDataType<miopenDataType_t>(type), dims.size(), dims.data(), strides.data());
}

rocm::DnnTensorDescriptor makeInputDnnTensorDescr(const ov::Node& node, int n) {
    return makeDnnTensorDescr(node.get_input_element_type(n), node.get_input_shape(n));
}

rocm::DnnTensorDescriptor makeOutputDnnTensorDescr(const ov::Node& node, int n) {
    return makeDnnTensorDescr(node.get_output_element_type(n), node.get_output_shape(n));
}

}  // namespace rocm
