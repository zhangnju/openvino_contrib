// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <miopen/miopen.h>
#include <fmt/format.h>

#include <cstdint>
#include <rocm/float16.hpp>
#include <error.hpp>
#include <kernels/details/rocm_type_traits.hpp>
#include <openvino/core/except.hpp>
#include <string_view>
#include <type_traits>

#include "transformer/nodes/activation_type.hpp"

namespace ov {
namespace rocm_gpu {

/**
 * Converts OpenVINO data type to T
 * @tparam T Data type to convert
 * @param type OpenVINO data type
 * @return Converted T data type
 */
template <typename T>
T convertDataType(ov::element::Type type);

/**
 * @brief Converts OpenVINO data type to rocm data type
 */
template <>
inline constexpr hipDataType convertDataType<hipDataType>(const ov::element::Type type) {
    using ov::element::Type_t;
    switch (static_cast<Type_t>(type)) {

        case Type_t::bf16:
            return HIP_R_16BF;
        case Type_t::i16:
            return HIP_R_16I;
        case Type_t::u16:
            return HIP_R_16U;
        case Type_t::i64:
            return HIP_R_64I;
        case Type_t::u64:
            return HIP_R_64U;
        case Type_t::f16:
            return HIP_R_16F;
        case Type_t::f32:
            return HIP_R_32F;
        case Type_t::f64:
            return HIP_R_64F;
        case Type_t::i8:
            return HIP_R_8I;
        case Type_t::u8:
            return HIP_R_8U;
        case Type_t::i32:
            return HIP_R_32I;
        case Type_t::u32:
            return HIP_R_32U;
        default:
            throw_ov_exception(
                fmt::format("The ngraph element type {} is not supported by "
                            "the rocm library",
                            type.c_type_string()));
    }
}

/**
 * @brief Converts OpenVINO data type to miopen data type
 */
template <>
inline constexpr miopenDataType_t convertDataType<miopenDataType_t>(const ov::element::Type type) {
    using ov::element::Type_t;
    switch (static_cast<Type_t>(type)) {
        case Type_t::boolean:
            return miopenHalf;
        case Type_t::bf16:
            return miopenBFloat16;
        case Type_t::f16:
            return miopenHalf;
        case Type_t::f32:
            return miopenFloat;
        case Type_t::f64:
            return miopenDouble;
        case Type_t::i8:
            return miopenInt8;
        case Type_t::i32:
            return miopenInt32;
        case Type_t::i64:
            return miopenInt64;
        default:
            throw_ov_exception(
                fmt::format("The ngraph element type {} is not supported by "
                            "the miopen library",
                            type.c_type_string()));
    }
}

template <>
inline constexpr kernel::Type_t convertDataType<kernel::Type_t>(const ov::element::Type type) {
    using nType_t = ov::element::Type_t;
    using kType_t = kernel::Type_t;
    switch (static_cast<nType_t>(type)) {
        case nType_t::boolean:
            return kType_t::boolean;
#ifdef rocm_HAS_BF16_TYPE
        case nType_t::bf16:
            return kType_t::bf16;
#endif
        case nType_t::i16:
            return kType_t::i16;
        case nType_t::u16:
            return kType_t::u16;
        case nType_t::i64:
            return kType_t::i64;
        case nType_t::u64:
            return kType_t::u64;
        case nType_t::f16:
            return kType_t::f16;
        case nType_t::f32:
            return kType_t::f32;
        case nType_t::f64:
            return kType_t::f64;
        case nType_t::u1:
            return kType_t::u1;
        case nType_t::i4:
            return kType_t::i4;
        case nType_t::u4:
            return kType_t::u4;
        case nType_t::i8:
            return kType_t::i8;
        case nType_t::u8:
            return kType_t::u8;
        case nType_t::i32:
            return kType_t::i32;
        case nType_t::u32:
            return kType_t::u32;
        default:
            throw_ov_exception(
                fmt::format("The ngraph element type {} is not supported by "
                            "the rocm library",
                            type.c_type_string()));
    }
}

/**
 * @brief Retruns std::string representation of T type
 */
template <typename T>
std::string_view toString(const T& type);

/**
 * @brief Retruns std::string representation of hipDataType type
 */
template <>
inline constexpr std::string_view toString<hipDataType>(const hipDataType& type) {
    switch (type) {
        case HIP_R_16BF:
            return "HIP_R_16BF";
        case HIP_R_16I:
            return "HIP_R_16I";
        case HIP_R_16U:
            return "HIP_R_16U";
        case HIP_R_64I:
            return "HIP_R_64I";
        case HIP_R_64U:
            return "HIP_R_64U";
        case HIP_R_16F:
            return "HIP_R_16F";
        case HIP_R_32F:
            return "HIP_R_32F";
        case HIP_R_64F:
            return "HIP_R_64F";
        case HIP_R_8I:
            return "HIP_R_8I";
        case HIP_R_8U:
            return "HIP_R_8U";
        case HIP_R_32I:
            return "HIP_R_32I";
        case HIP_R_32U:
            return "HIP_R_32U";
        default:
            throw_ov_exception(
                fmt::format("ov::rocm_gpu::toString<hipDataType>(): Unsupported data type: type = {}", type));
    }
}

/**
 * @brief Retruns std::string representation of miopenDataType_t type
 */
template <>
inline constexpr std::string_view toString<miopenDataType_t>(const miopenDataType_t& type) {
    switch (type) {
        case miopenFloat:
            return "miopenFloat";
        case miopenDouble:
            return "miopenDouble";
        case miopenHalf:
            return "miopenHalf";
        case miopenInt8:
            return "miopenInt8";
        case miopenInt32:
            return "miopenInt32";
        case miopenBFloat16:
            return "miopenBFloat16";
        case miopenInt64:
            return "miopenInt64";
        default:
            throw_ov_exception(
                fmt::format("ov::rocm_gpu::toString<hipDataType>(): Unsupported data type: type = {}", type));
    }
}

/**
 * @brief Retruns the size of miopenDataType_t type value in bytes
 */
inline constexpr std::size_t elementSize(miopenDataType_t type) {
    switch (type) {
        case miopenFloat:
            return sizeof(float);
        case miopenDouble:
            return sizeof(double);
        case miopenHalf:
            return sizeof(float) / 2;
        case miopenInt8:
            return sizeof(std::int8_t);
        case miopenInt32:
            return sizeof(std::int32_t);
        case miopenBFloat16:
            return sizeof(std::uint16_t);
        case miopenInt64:
            return sizeof(std::int64_t);
        default:
            throw_ov_exception(
                fmt::format("The miopenDataType_t {} is not supported by the miopen library", toString(type)));
    }
}

/**
 * @brief Converts rocm plugin activation mode to miopen Backend API activation mode
 */
inline constexpr miopenPointwiseMode_t convertActivationModeToBE(const nodes::ActivationMode& mode) {
    switch (mode) {
        case nodes::ActivationMode::SIGMOID:
            return MIOPEN_POINTWISE_SIGMOID_FWD;
        case nodes::ActivationMode::RELU:
            return MIOPEN_POINTWISE_RELU_FWD;
        case nodes::ActivationMode::TANH:
            return MIOPEN_POINTWISE_TANH_FWD;
        case nodes::ActivationMode::ELU:
            return MIOPEN_POINTWISE_GELU_FWD;
        case nodes::ActivationMode::SWISH:
            return MIOPEN_POINTWISE_SWISH_FWD;
        default:
            throw_ov_exception(fmt::format("Unsupported activation: {}", mode));
    }
}

inline constexpr miopenActivationMode_t convertActivationMode(const nodes::ActivationMode& mode) {
    switch (mode) {
        case nodes::ActivationMode::SIGMOID:
            return miopenActivationLOGISTIC;
        case nodes::ActivationMode::RELU:
            return miopenActivationRELU;
        case nodes::ActivationMode::TANH:
            return miopenActivationTANH;
        case nodes::ActivationMode::CLIPPED_RELU:
            return miopenActivationCLIPPEDRELU;
        case nodes::ActivationMode::ELU:
            return miopenActivationELU;
        case nodes::ActivationMode::NO_ACTIVATION:
            return miopenActivationPASTHRU;
        default:
            throw_ov_exception(fmt::format("Unsupported activation: {}", mode));
    }
}

/**
 * @brief Auxilary structure for E type used in SwitchCase()
 * should contain type alias and static constexpr shift member of unsigned type, see example below
 */
template <typename E>
struct SwitchCaseTrait;

/**
 * @brief Auxilary structure for miopenDataType_t used in SwitchCase()
 */
template <>
struct SwitchCaseTrait<miopenDataType_t> {
    using type = uint32_t;
    static constexpr std::size_t shift = 16;
};

/**
 * @brief Packs two enum or integer values into one integer value allowing it to be used in switch() and case statements
 * e.g. switch (switchCase(in0, out)) {
 *          case switchCase(miopenFloat, miopenFloat):
 * Before using this function SwitchCaseTrait<E> structure specification has to be defined for E type, see example adove
 */
template <typename E>
inline constexpr typename SwitchCaseTrait<E>::type switchCase(E first, E second) {
    using I = typename SwitchCaseTrait<E>::type;
    static_assert(std::is_enum<E>() || std::is_integral<E>());
    static_assert(std::is_integral<I>());
    static_assert(std::is_unsigned<decltype(SwitchCaseTrait<E>::shift)>());
    constexpr auto shift = SwitchCaseTrait<E>::shift;
    const I firstI = static_cast<I>(first);
    const I secondI = static_cast<I>(second);
    const I result = (firstI << shift) + secondI;
    OPENVINO_ASSERT(static_cast<E>(secondI) == second);
    OPENVINO_ASSERT(static_cast<E>((result - secondI) >> shift) == first);
    return result;
}

// TODO: use in miopenTensorOpBase
/**
 * @brief Gets opTensorCompType for opTensorDesc used in miopenOpTensor()
 * See https://docs.rocm.com/deeplearning/miopen/api/index.html#miopenOpTensor for more details.
 */
inline constexpr miopenDataType_t getmiopenOpTensorCompType(miopenDataType_t in0,
                                                          miopenDataType_t in1,
                                                          miopenDataType_t out) {
    auto throwException = [=] {
        throw_ov_exception(
            fmt::format("ov::rocm_gpu::getmiopenOpTensorType(): Unsupported data types: in0 = {}, in1 = {} out = {}",
                        toString(in0),
                        toString(in1),
                        toString(out)));
    };
    if (in0 != in1) {
        throwException();
    }
    switch (switchCase(in0, out)) {
        case switchCase(miopenFloat, miopenFloat):
        case switchCase(miopenInt8, miopenFloat):
        case switchCase(miopenHalf, miopenFloat):
        case switchCase(miopenBFloat16, miopenFloat):
        case switchCase(miopenFloat, miopenHalf):
        case switchCase(miopenHalf, miopenHalf):
        case switchCase(miopenInt8, miopenInt8):
        case switchCase(miopenFloat, miopenInt8):
        case switchCase(miopenFloat, miopenBFloat16):
        case switchCase(miopenBFloat16, miopenBFloat16):
            return miopenFloat;
        case switchCase(miopenDouble, miopenDouble):
            return miopenDouble;
        default:
            throwException();
    }
    return miopenFloat;  // never reached
}

}  // namespace rocm_gpu
}  // namespace ov
