// Copyright (C) 2021-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <rocm/dnn.hpp>

#include "openvino/core/node.hpp"
#include "openvino/core/type/element_type.hpp"

namespace rocm {

DnnTensorDescriptor makeDnnTensorDescr(const ov::element::Type& type, const ov::Shape& shape);

rocm::DnnTensorDescriptor makeInputDnnTensorDescr(const ov::Node& node, int n);

rocm::DnnTensorDescriptor makeOutputDnnTensorDescr(const ov::Node& node, int n);

}  // namespace rocm
