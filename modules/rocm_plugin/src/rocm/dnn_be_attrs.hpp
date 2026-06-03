// Copyright (C) 2021-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#define MIOPEN_BETA_API 1
#include <miopen/miopen.h>

#include <cstdint>

namespace rocm {

/**
 * @brief Attribute type-id traits of MIOPEN backend descriptor.
 *
 * Binds together attribute type identifier and attribute value type.
 */
template <miopenBackendAttributeType_t TypeId>
struct DnnBEAttrType;

template <>
struct DnnBEAttrType<miopenBackendAttributeType_t::MIOPEN_TYPE_HANDLE> {
    using ValueType = miopenHandle_t;
};
template <>
struct DnnBEAttrType<miopenBackendAttributeType_t::MIOPEN_TYPE_NAN_PROPOGATION> {
    using ValueType = miopenNanPropagation_t;
};
template <>
struct DnnBEAttrType<miopenBackendAttributeType_t::MIOPEN_TYPE_DATA_TYPE> {
    using ValueType = miopenDataType_t;
};
template <>
struct DnnBEAttrType<miopenBackendAttributeType_t::MIOPEN_TYPE_CONVOLUTION_MODE> {
    using ValueType = miopenConvolutionMode_t;
};
template <>
struct DnnBEAttrType<miopenBackendAttributeType_t::MIOPEN_TYPE_BACKEND_DESCRIPTOR> {
    using ValueType = miopenBackendDescriptor_t;
};
template <>
struct DnnBEAttrType<miopenBackendAttributeType_t::MIOPEN_TYPE_POINTWISE_MODE> {
    using ValueType = miopenPointwiseMode_t;
};
template <>
struct DnnBEAttrType<miopenBackendAttributeType_t::MIOPEN_TYPE_HEUR_MODE> {
    using ValueType = miopenBackendHeurMode_t;
};
template <>
struct DnnBEAttrType<miopenBackendAttributeType_t::MIOPEN_TYPE_VOID_PTR> {
    using ValueType = const void*;
};
template <>
struct DnnBEAttrType<miopenBackendAttributeType_t::MIOPEN_TYPE_DOUBLE> {
    using ValueType = double;
};
template <>
struct DnnBEAttrType<miopenBackendAttributeType_t::MIOPEN_TYPE_FLOAT> {
    using ValueType = float;
};
template <>
struct DnnBEAttrType<miopenBackendAttributeType_t::MIOPEN_TYPE_INT64> {
    using ValueType = int64_t;
};
template <>
struct DnnBEAttrType<miopenBackendAttributeType_t::MIOPEN_TYPE_BOOLEAN> {
    using ValueType = bool;
};

/**
 * @brief Traits of MIOPEN backend descriptor attributes.
 *
 * Binds together attribute name and default type-id of it's value.
 * Most of attributes have a single value type. Some attributes can
 * accept several numeric types.
 */
template <miopenBackendAttributeName_t Name>
constexpr miopenBackendAttributeType_t GetDnnBEAttrTypeId() {
    switch (Name) {
        case miopenBackendAttributeName_t::MIOPEN_ATTR_EXECUTION_PLAN_ENGINE_CONFIG:
        case miopenBackendAttributeName_t::MIOPEN_ATTR_OPERATION_CONVOLUTION_FORWARD_X:
        case miopenBackendAttributeName_t::MIOPEN_ATTR_OPERATION_CONVOLUTION_FORWARD_W:
        case miopenBackendAttributeName_t::MIOPEN_ATTR_OPERATION_CONVOLUTION_FORWARD_Y:
        case miopenBackendAttributeName_t::MIOPEN_ATTR_OPERATION_CONVOLUTION_FORWARD_CONV_DESC:
        case miopenBackendAttributeName_t::MIOPEN_ATTR_OPERATION_POINTWISE_PW_DESCRIPTOR:
        case miopenBackendAttributeName_t::MIOPEN_ATTR_OPERATION_POINTWISE_XDESC:
        case miopenBackendAttributeName_t::MIOPEN_ATTR_OPERATION_POINTWISE_BDESC:
        case miopenBackendAttributeName_t::MIOPEN_ATTR_OPERATION_POINTWISE_YDESC:
        case miopenBackendAttributeName_t::MIOPEN_ATTR_OPERATION_POINTWISE_DXDESC:
        case miopenBackendAttributeName_t::MIOPEN_ATTR_OPERATION_POINTWISE_DYDESC:
        case miopenBackendAttributeName_t::MIOPEN_ATTR_OPERATION_POINTWISE_ALPHA1:
        case miopenBackendAttributeName_t::MIOPEN_ATTR_OPERATION_POINTWISE_ALPHA2:
        case miopenBackendAttributeName_t::MIOPEN_ATTR_OPERATIONGRAPH_OPS:
        case miopenBackendAttributeName_t::MIOPEN_ATTR_ENGINEHEUR_OPERATION_GRAPH:
        case miopenBackendAttributeName_t::MIOPEN_ATTR_ENGINE_OPERATION_GRAPH:
        case miopenBackendAttributeName_t::MIOPEN_ATTR_ENGINEHEUR_RESULTS:
        case miopenBackendAttributeName_t::MIOPEN_ATTR_ENGINECFG_ENGINE: {
            return miopenBackendAttributeType_t::MIOPEN_TYPE_BACKEND_DESCRIPTOR;
        }
        case miopenBackendAttributeName_t::MIOPEN_ATTR_EXECUTION_PLAN_HANDLE:
        case miopenBackendAttributeName_t::MIOPEN_ATTR_OPERATIONGRAPH_HANDLE: {
            return miopenBackendAttributeType_t::MIOPEN_TYPE_HANDLE;
        }
        case miopenBackendAttributeName_t::MIOPEN_ATTR_POINTWISE_RELU_LOWER_CLIP:
        case miopenBackendAttributeName_t::MIOPEN_ATTR_POINTWISE_RELU_UPPER_CLIP:
        case miopenBackendAttributeName_t::MIOPEN_ATTR_POINTWISE_RELU_LOWER_CLIP_SLOPE:
        case miopenBackendAttributeName_t::MIOPEN_ATTR_POINTWISE_ELU_ALPHA:
        case miopenBackendAttributeName_t::MIOPEN_ATTR_POINTWISE_SOFTPLUS_BETA:
        case miopenBackendAttributeName_t::MIOPEN_ATTR_POINTWISE_SWISH_BETA: {
            return miopenBackendAttributeType_t::MIOPEN_TYPE_DOUBLE;
        }
        case miopenBackendAttributeName_t::MIOPEN_ATTR_OPERATIONGRAPH_ENGINE_GLOBAL_COUNT:
        case miopenBackendAttributeName_t::MIOPEN_ATTR_EXECUTION_PLAN_WORKSPACE_SIZE:
        case miopenBackendAttributeName_t::MIOPEN_ATTR_VARIANT_PACK_UNIQUE_IDS:
        case miopenBackendAttributeName_t::MIOPEN_ATTR_TENSOR_DIMENSIONS:
        case miopenBackendAttributeName_t::MIOPEN_ATTR_TENSOR_STRIDES:
        case miopenBackendAttributeName_t::MIOPEN_ATTR_TENSOR_UNIQUE_ID:
        case miopenBackendAttributeName_t::MIOPEN_ATTR_TENSOR_BYTE_ALIGNMENT:
        case miopenBackendAttributeName_t::MIOPEN_ATTR_CONVOLUTION_SPATIAL_DIMS:
        case miopenBackendAttributeName_t::MIOPEN_ATTR_CONVOLUTION_PRE_PADDINGS:
        case miopenBackendAttributeName_t::MIOPEN_ATTR_CONVOLUTION_POST_PADDINGS:
        case miopenBackendAttributeName_t::MIOPEN_ATTR_CONVOLUTION_DILATIONS:
        case miopenBackendAttributeName_t::MIOPEN_ATTR_CONVOLUTION_FILTER_STRIDES:
        case miopenBackendAttributeName_t::MIOPEN_ATTR_ENGINE_GLOBAL_INDEX: {
            return miopenBackendAttributeType_t::MIOPEN_TYPE_INT64;
        }
        case miopenBackendAttributeName_t::MIOPEN_ATTR_POINTWISE_MATH_PREC:
        case miopenBackendAttributeName_t::MIOPEN_ATTR_TENSOR_DATA_TYPE:
        case miopenBackendAttributeName_t::MIOPEN_ATTR_CONVOLUTION_COMP_TYPE: {
            return miopenBackendAttributeType_t::MIOPEN_TYPE_DATA_TYPE;
        }
        case miopenBackendAttributeName_t::MIOPEN_ATTR_VARIANT_PACK_DATA_POINTERS:
        case miopenBackendAttributeName_t::MIOPEN_ATTR_VARIANT_PACK_WORKSPACE: {
            return miopenBackendAttributeType_t::MIOPEN_TYPE_VOID_PTR;
        }
        case miopenBackendAttributeName_t::MIOPEN_ATTR_TENSOR_IS_VIRTUAL: {
            return miopenBackendAttributeType_t::MIOPEN_TYPE_BOOLEAN;
        }
        case miopenBackendAttributeName_t::MIOPEN_ATTR_CONVOLUTION_CONV_MODE: {
            return miopenBackendAttributeType_t::MIOPEN_TYPE_CONVOLUTION_MODE;
        }
        case miopenBackendAttributeName_t::MIOPEN_ATTR_POINTWISE_MODE: {
            return miopenBackendAttributeType_t::MIOPEN_TYPE_POINTWISE_MODE;
        }
        case miopenBackendAttributeName_t::MIOPEN_ATTR_POINTWISE_NAN_PROPAGATION: {
            return miopenBackendAttributeType_t::MIOPEN_TYPE_NAN_PROPOGATION;
        }
        case miopenBackendAttributeName_t::MIOPEN_ATTR_ENGINEHEUR_MODE: {
            return miopenBackendAttributeType_t::MIOPEN_TYPE_HEUR_MODE;
        }
    }
}

}  // namespace rocm
