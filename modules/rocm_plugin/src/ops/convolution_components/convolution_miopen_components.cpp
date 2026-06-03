// Copyright (C) 2021-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "convolution_miopen_components.hpp"

#include <miopen/miopen.h>
#include <fmt/ostream.h>

#include <rocm_config.hpp>
#include <openvino/core/except.hpp>
#include <ops/converters.hpp>

namespace ov::rocm_gpu::Convolution::Details {

ConvolutionParamsmiopen::ConvolutionParamsmiopen(const Convolution::Details::ConvolutionParams& params)
    : number_of_dims_{static_cast<int>(params.NumberOfDims())},
      groups_{static_cast<int>(params.groups_)},
      data_type_{convertDataType<miopenDataType_t>(params.element_type_)} {
    if (params.padding_before_ != params.padding_after_) {
        throw_ov_exception(
            fmt::format("Asymmetric padding is not supported: padding_before: "
                        "{}, padding_after: {}",
                        params.padding_before_,
                        params.padding_after_));
    }

    OPENVINO_ASSERT(number_of_dims_ > NON_SPATIAL_DIMS_NUMBER);
    OPENVINO_ASSERT(params.input_shape_.size() == number_of_dims_);
    OPENVINO_ASSERT(params.filter_shape_.size() == number_of_dims_);
    OPENVINO_ASSERT(params.output_shape_.size() == number_of_dims_);

    const size_t number_of_spatial_dims = NumberOfSpatialDims();
    OPENVINO_ASSERT(params.strides_.size() == number_of_spatial_dims);
    OPENVINO_ASSERT(params.dilations_.size() == number_of_spatial_dims);
    OPENVINO_ASSERT(params.padding_before_.size() == number_of_spatial_dims);

    std::copy(params.input_shape_.begin(), params.input_shape_.end(), input_shape_.begin());
    std::copy(params.filter_shape_.begin(), params.filter_shape_.end(), filter_shape_.begin());
    std::copy(params.output_shape_.begin(), params.output_shape_.end(), output_shape_.begin());

    std::copy(params.strides_.begin(), params.strides_.end(), strides_.begin());
    std::copy(params.dilations_.begin(), params.dilations_.end(), dilations_.begin());
    std::copy(params.padding_before_.begin(), params.padding_before_.end(), paddings_.begin());
}

rocm::DnnTensorDescriptor ConvolutionParamsmiopen::MakeInputDescriptor() const {
    return rocm::DnnTensorDescriptor{}.set(
        miopenTensorLayout_t::miopenTensorNCHW, data_type_, number_of_dims_, input_shape_.data());
}

rocm::DnnFilterDescriptor ConvolutionParamsmiopen::MakeFilterDescriptor() const {
    return rocm::DnnFilterDescriptor{}.set(
        data_type_, miopenTensorLayout_t::miopenTensorNCHW, number_of_dims_, filter_shape_.data());
}

rocm::DnnTensorDescriptor ConvolutionParamsmiopen::MakeOutputDescriptor() const {
    return rocm::DnnTensorDescriptor{}.set(
        miopenTensorLayout_t::miopenTensorNCHW, data_type_, number_of_dims_, output_shape_.data());
}

rocm::DnnConvolutionDescriptor ConvolutionParamsmiopen::MakeConvolutionDescriptor(miopenDataType_t convDataType) const {
    // According to `ov::op::v1::Convolution` spec, it "computes 1D, 2D or 3D convolution
    // (cross-correlation to be precise)".
    constexpr miopenConvolutionMode_t mode = miopenConvolution;

    // The convolution computation will be done in the specified dataType, which can be
    // potentially different from the input/output tensors.
    const miopenDataType_t datatype = convDataType;

    rocm::DnnConvolutionDescriptor conv_desc;
    conv_desc.set(NumberOfSpatialDims(), paddings_.data(), strides_.data(), dilations_.data(), mode, NumberOfGroups());

    return conv_desc;
}

ConvolutionDescriptorsmiopen::ConvolutionDescriptorsmiopen(const CreationContext& context,
                                                         const ConvolutionParamsmiopen& params,
                                                         const std::vector<miopenDataType_t> half_desc_types)
    : params_{params},
      tensor_element_type_{params_.ElementType()},
      conv_desc_type_{params_.ElementType()},
      input_{params_.MakeInputDescriptor()},
      filter_{params_.MakeFilterDescriptor()},
      output_{params_.MakeOutputDescriptor()},
      conv_{},
      algo_perf_{},
      workspace_size_{0},
      half_desc_types_{half_desc_types} {
    auto& dnnHandle = context.dnnHandle();
    if (context.opBenchOption()) {
        BenchmarkOptimalAlgo(dnnHandle, params_);
    } else {
        GetAlgo(dnnHandle);
    }
}

void ConvolutionDescriptorsmiopen::BenchmarkOptimalAlgo(const rocm::DnnHandle& dnnHandle,
                                                       const ConvolutionParamsmiopen& params) {
    constexpr auto kNumSelectAlgo = 3;
    throwIfError(miopenConvolutionForwardGetWorkSpaceSize(dnnHandle.get(), filter_.get(), input_.get(), conv_.get(), output_.get(),&workspace_size_));
    size_t convForwardAlgorithmMaxCount;
    throwIfError(miopenConvolutionForwardGetSolutionCount(dnnHandle.get(), filter_.get(), input_.get(), conv_.get(), output_.get(),&convForwardAlgorithmMaxCount));
    std::vector<int> timesmiopenAlgosSelected(convForwardAlgorithmMaxCount);
    std::array<miopenConvAlgoPerf_t, kNumSelectAlgo> miopenAlgos{};
    for (auto& algo : miopenAlgos) {
        FindAlgo(dnnHandle);
        algo = algo_perf_;
        OPENVINO_ASSERT(algo_perf_.fwd_algo >= 0);
        OPENVINO_ASSERT(algo_perf_.fwd_algo < convForwardAlgorithmMaxCount);
        timesmiopenAlgosSelected[algo_perf_.fwd_algo] += 1;
    }
    auto maxAlgoIter = std::max_element(timesmiopenAlgosSelected.begin(), timesmiopenAlgosSelected.end());
    auto optimalAlgoId =
        static_cast<miopenConvFwdAlgorithm_t>(std::distance(timesmiopenAlgosSelected.begin(), maxAlgoIter));
    auto optimalAlgo = std::find_if(
        miopenAlgos.begin(), miopenAlgos.end(), [optimalAlgoId](const auto& a) { return a.fwd_algo == optimalAlgoId; });
    algo_perf_ = *optimalAlgo;
}

void ConvolutionDescriptorsmiopen::GetAlgo(const rocm::DnnHandle& dnnHandle) {
    switch (tensor_element_type_) {
        case miopenHalf:
            //std::cout<<"half"<<std::endl;
            for (const auto& half_desc_type : half_desc_types_) {
                if (GetAlgoForConvDataType(dnnHandle, half_desc_type)) {
                    conv_desc_type_ = half_desc_type;
                    return;
                }
            }
            break;
        default:
            //std::cout<<"float"<<std::endl;
            if (GetAlgoForConvDataType(dnnHandle, tensor_element_type_)) return;
    }

    throw_ov_exception("miopen: Unsupported convolution");
}

bool ConvolutionDescriptorsmiopen::GetAlgoForConvDataType(const rocm::DnnHandle& dnnHandle,
                                                         miopenDataType_t convDataType) {
    miopenStatus_t status = miopenStatusSuccess;
    conv_ = params_.MakeConvolutionDescriptor(convDataType);
   
    std::size_t workspace_size = 0;
    status = ::miopenConvolutionForwardGetWorkSpaceSize(dnnHandle.get(),
                                                           filter_.get(),
                                                           input_.get(),
                                                           conv_.get(),
                                                           output_.get(),
                                                           &workspace_size);
    if(status != miopenStatusSuccess)
             throw_ov_exception("MIOpen Failed to get forward workspace size");
    else
    {
        //workspace_size_ = workspace_size;
        //std::cout<<"workspace_size:"<<workspace_size<<std::endl;
        size_t solution_count;
        status = ::miopenConvolutionForwardGetSolutionCount(dnnHandle.get(),
                                                    filter_.get(),
                                                    input_.get(),
                                                    conv_.get(),
                                                    output_.get(),
                                                   &solution_count);
        if(status != miopenStatusSuccess)
            throw_ov_exception("MIOpen failed to get solution count");

        std::vector<miopenConvSolution_t> solutions(solution_count);
        size_t actual_count=0;
        status = ::miopenConvolutionForwardGetSolution(dnnHandle.get(),
                                                     filter_.get(),
                                                     input_.get(),
                                                     conv_.get(),
                                                     output_.get(),
                                                     solution_count,
                                                     &actual_count,
                                                     solutions.data());
        if(status != miopenStatusSuccess)
            throw_ov_exception("MIOpen failed to get solution");
        else{
            // Prefer the solution with the smallest non-zero workspace, falling back to the first
            solution_id_ = solutions[0].solution_id;
            workspace_size_ = solutions[0].workspace_size;
            for (size_t i = 0; i < actual_count; ++i) {
                if (solutions[i].workspace_size > 0 && solutions[i].workspace_size < workspace_size_) {
                    solution_id_ = solutions[i].solution_id;
                    workspace_size_ = solutions[i].workspace_size;
                }
            }
            // Compile the chosen solution so miopenConvolutionForwardImmediate can execute it
            ::miopenConvolutionForwardCompileSolution(dnnHandle.get(),
                                                      filter_.get(),
                                                      input_.get(),
                                                      conv_.get(),
                                                      output_.get(),
                                                      solution_id_);
        } 

        /*
        status = ::miopenConvolutionForwardGetSolutionWorkspaceSize(dnnHandle.get(),
                                                           filter_.get(),
                                                           input_.get(),
                                                           conv_.get(),
                                                           output_.get(),
                                                           solution_id_ ,
                                                           &workspace_size);
        if(status != miopenStatusSuccess)
            throw_ov_exception("MIOpen failed to get solution workspace size");
        else
            workspace_size_ = workspace_size;
        */
    }

    return true;
}

void ConvolutionDescriptorsmiopen::FindAlgo(const rocm::DnnHandle& dnnHandle) {
    switch (tensor_element_type_) {
        case miopenHalf:
            for (const auto& half_desc_type : half_desc_types_) {
                if (FindAlgoForConvDataType(dnnHandle, half_desc_type)) {
                    conv_desc_type_ = half_desc_type;
                    return;
                }
            }
            break;
        default:
            if (FindAlgoForConvDataType(dnnHandle, tensor_element_type_)) return;
    }

    throw_ov_exception("miopen: Unsupported convolution");
}

bool ConvolutionDescriptorsmiopen::FindAlgoForConvDataType(const rocm::DnnHandle& dnnHandle,
                                                          miopenDataType_t convDataType) {
    miopenStatus_t status = miopenStatusSuccess;
    conv_ = params_.MakeConvolutionDescriptor(convDataType);
    const int requestedAlgoCount = 1;
    int returnedAlgoCount = 0;
    #if 0
    status = ::miopenFindConvolutionForwardAlgorithm(dnnHandle.get(),
                                                    input_.get(),
                                                    filter_.get(),
                                                    conv_.get(),
                                                    output_.get(),
                                                    requestedAlgoCount,
                                                    &returnedAlgoCount,
                                                    &algo_perf_);
   
    if ((status != miopenStatusSuccess) || (algo_perf_.status != miopenStatusSuccess) || (returnedAlgoCount <= 0)) {
        return false;
    }

    //throwIfError(::miopenSetConvolutionMathType(conv_.get(), algo_perf_.mathType));

    size_t sizeInBytes = 0;
    throwIfError(::miopenConvolutionForwardGetSolutionCount(
        dnnHandle.get(), input_.get(), filter_.get(), conv_.get(), output_.get(), algo_perf_.algo, &sizeInBytes));
    algo_perf_.memory = sizeInBytes;
    #endif
    return true;
}

void ConvolutionDescriptorsmiopen::FindAlgo(const rocm::DnnHandle& dnnHandle,
                                           rocm::DevicePointer<const void*> inPtr,
                                           rocm::DevicePointer<const void*> filterPtr,
                                           rocm::DevicePointer<void*> outPtr,
                                           rocm::DeviceBuffer<std::byte> workspace) {
    switch (tensor_element_type_) {
        case miopenHalf:
            for (const auto& half_desc_type : half_desc_types_) {
                if (FindAlgoForConvDataType(dnnHandle, inPtr, filterPtr, outPtr, workspace, half_desc_type)) {
                    conv_desc_type_ = half_desc_type;
                    return;
                }
            }
            break;
        default:
            if (FindAlgoForConvDataType(dnnHandle, inPtr, filterPtr, outPtr, workspace, tensor_element_type_)) return;
    }

    throw_ov_exception("miopen: Unsupported convolution");
}

bool ConvolutionDescriptorsmiopen::FindAlgoForConvDataType(const rocm::DnnHandle& dnnHandle,
                                                          rocm::DevicePointer<const void*> inPtr,
                                                          rocm::DevicePointer<const void*> filterPtr,
                                                          rocm::DevicePointer<void*> outPtr,
                                                          rocm::DeviceBuffer<std::byte> workspace,
                                                          miopenDataType_t convDataType) {
    miopenStatus_t status = miopenStatusSuccess;
    conv_ = params_.MakeConvolutionDescriptor(convDataType);
    const int requestedAlgoCount = 1;
    int returnedAlgoCount = 0;
    /*
    status = ::miopenFindConvolutionForwardAlgorithmEx(dnnHandle.get(),
                                                      input_.get(),
                                                      inPtr.get(),
                                                      filter_.get(),
                                                      filterPtr.get(),
                                                      conv_.get(),
                                                      output_.get(),
                                                      outPtr.get(),
                                                      requestedAlgoCount,
                                                      &returnedAlgoCount,
                                                      &algo_perf_,
                                                      workspace.data(),
                                                      workspace.size());
    return (status == miopen_STATUS_SUCCESS) && (algo_perf_.status == miopen_STATUS_SUCCESS) && (returnedAlgoCount > 0);
    */
}
/*
ConvolutionBackpropDataParamsmiopen::ConvolutionBackpropDataParamsmiopen(
    const Convolution::Details::ConvolutionBackwardDataParams& params)
    : number_of_dims_{static_cast<int>(params.NumberOfDims())},
      groups_{static_cast<int>(params.groups_)},
      data_type_{convertDataType<miopenDataType_t>(params.element_type_)} {
    if (params.pads_begin_ != params.pads_end_) {
        throw_ov_exception(
            fmt::format("Asymmetric padding is not supported: padding_before: "
                        "{}, padding_after: {}",
                        params.pads_begin_,
                        params.pads_end_));
    }

    OPENVINO_ASSERT(number_of_dims_ > NON_SPATIAL_DIMS_NUMBER);
    OPENVINO_ASSERT(params.doutput_shape_.size() == number_of_dims_);
    OPENVINO_ASSERT(params.filter_shape_.size() == number_of_dims_);
    OPENVINO_ASSERT(params.dinput_shape_.size() == number_of_dims_);

    const size_t number_of_spatial_dims = NumberOfSpatialDims();
    OPENVINO_ASSERT(params.strides_.size() == number_of_spatial_dims);
    OPENVINO_ASSERT(params.dilations_.size() == number_of_spatial_dims);
    OPENVINO_ASSERT(params.pads_begin_.size() == number_of_spatial_dims);

    std::copy(params.doutput_shape_.begin(), params.doutput_shape_.end(), doutput_shape_.begin());
    std::copy(params.filter_shape_.begin(), params.filter_shape_.end(), filter_shape_.begin());
    std::copy(params.dinput_shape_.begin(), params.dinput_shape_.end(), dinput_shape_.begin());

    std::copy(params.strides_.begin(), params.strides_.end(), strides_.begin());
    std::copy(params.dilations_.begin(), params.dilations_.end(), dilations_.begin());
    std::copy(params.pads_begin_.begin(), params.pads_begin_.end(), paddings_.begin());
}

rocm::DnnTensorDescriptor ConvolutionBackpropDataParamsmiopen::MakeDOutputDescriptor() const {
    return rocm::DnnTensorDescriptor{}.set(
        miopenTensorLayout_t::miopenTensorNCHW, data_type_, number_of_dims_, doutput_shape_.data());
}

rocm::DnnFilterDescriptor ConvolutionBackpropDataParamsmiopen::MakeFilterDescriptor() const {
    return rocm::DnnFilterDescriptor{}.set(
        data_type_, miopenTensorLayout_t::miopenTensorNCHW, number_of_dims_, filter_shape_.data());
}

rocm::DnnTensorDescriptor ConvolutionBackpropDataParamsmiopen::MakeDInputDescriptor() const {
    return rocm::DnnTensorDescriptor{}.set(
        miopenTensorLayout_t::miopenTensorNCHW, data_type_, number_of_dims_, dinput_shape_.data());
}

rocm::DnnConvolutionDescriptor ConvolutionBackpropDataParamsmiopen::MakeConvolutionDescriptor(
    miopenDataType_t convDataType) const {
    // According to `ov::op::v1::Convolution` spec, it "computes 1D, 2D or 3D convolution
    // (cross-correlation to be precise)".
    constexpr miopenConvolutionMode_t mode = miopen_CROSS_CORRELATION;

    // The convolution computation will be done in the specified dataType, which can be
    // potentially different from the input/output tensors.
    const miopenDataType_t datatype = convDataType;

    rocm::DnnConvolutionDescriptor conv_desc;
    conv_desc.set(NumberOfSpatialDims(), paddings_.data(), strides_.data(), dilations_.data(), mode, datatype);

    // Enable computations on Tensor Core hardware which requires at least Volta GPU
    // (compute capability 7.0).
    const miopenMathType_t math_type = miopen_TENSOR_OP_MATH;
    throwIfError(::miopenSetConvolutionMathType(conv_desc.get(), math_type));
    throwIfError(::miopenSetConvolutionGroupCount(conv_desc.get(), groups_));

    return conv_desc;
}

ConvolutionBackpropDataDescriptormiopen::ConvolutionBackpropDataDescriptormiopen(
    const CreationContext& context,
    const ConvolutionBackpropDataParamsmiopen& params,
    const std::vector<miopenDataType_t> half_desc_types)
    : params_{params},
      tensor_element_type_{params_.ElementType()},
      conv_desc_type_{params_.ElementType()},
      filter_desc_{params_.MakeFilterDescriptor()},
      doutput_desc_{params_.MakeDOutputDescriptor()},
      dinput_desc_{params_.MakeDInputDescriptor()},
      conv_{},
      algo_perf_{},
      half_desc_types_{half_desc_types} {
    auto& dnnHandle = context.dnnHandle();
    if (context.opBenchOption()) {
        BenchmarkOptimalAlgo(dnnHandle);
    } else {
        GetAlgo(dnnHandle);
    }
}

void ConvolutionBackpropDataDescriptormiopen::BenchmarkOptimalAlgo(const rocm::DnnHandle& dnnHandle) {
    constexpr auto kNumSelectAlgo = 3;
    int convBackwardDataAlgorithmMaxCount;
    throwIfError(miopenGetConvolutionBackwardDataAlgorithmMaxCount(dnnHandle.get(), &convBackwardDataAlgorithmMaxCount));
    std::vector<int> timesmiopenAlgosSelected(convBackwardDataAlgorithmMaxCount);
    std::array<miopenConvolutionBwdDataAlgoPerf_t, kNumSelectAlgo> miopenAlgos{};
    for (auto& algo : miopenAlgos) {
        FindAlgo(dnnHandle);
        algo = algo_perf_;
        OPENVINO_ASSERT(algo_perf_.algo >= 0);
        OPENVINO_ASSERT(algo_perf_.algo < convBackwardDataAlgorithmMaxCount);
        timesmiopenAlgosSelected[algo_perf_.algo] += 1;
    }
    auto maxAlgoIter = std::max_element(timesmiopenAlgosSelected.begin(), timesmiopenAlgosSelected.end());
    auto optimalAlgoId =
        static_cast<miopenConvolutionBwdDataAlgo_t>(std::distance(timesmiopenAlgosSelected.begin(), maxAlgoIter));
    auto optimalAlgo = std::find_if(
        miopenAlgos.begin(), miopenAlgos.end(), [optimalAlgoId](const auto& a) { return a.algo == optimalAlgoId; });
    algo_perf_ = *optimalAlgo;
}

void ConvolutionBackpropDataDescriptormiopen::GetAlgo(const rocm::DnnHandle& dnnHandle) {
    switch (tensor_element_type_) {
        case miopenHalf:
            for (const auto& half_desc_type : half_desc_types_) {
                if (GetAlgoForConvDataType(dnnHandle, half_desc_type)) {
                    conv_desc_type_ = half_desc_type;
                    return;
                }
            }
            break;
        default:
            if (GetAlgoForConvDataType(dnnHandle, tensor_element_type_)) return;
    }

    throw_ov_exception("miopen: Unsupported convolution");
}

bool ConvolutionBackpropDataDescriptormiopen::GetAlgoForConvDataType(const rocm::DnnHandle& dnnHandle,
                                                                    miopenDataType_t convDataType) {
    miopenStatus_t status = miopen_STATUS_NOT_SUPPORTED;
    conv_ = params_.MakeConvolutionDescriptor(convDataType);
    const int requestedAlgoCount = 1;
    int returnedAlgoCount = 0;
    status = ::miopenGetConvolutionBackwardDataAlgorithm_v7(dnnHandle.get(),
                                                           filter_desc_.get(),
                                                           doutput_desc_.get(),
                                                           conv_.get(),
                                                           dinput_desc_.get(),
                                                           requestedAlgoCount,
                                                           &returnedAlgoCount,
                                                           &algo_perf_);

    if ((status != miopen_STATUS_SUCCESS) || (algo_perf_.status != miopen_STATUS_SUCCESS) || (returnedAlgoCount <= 0)) {
        return false;
    }

    throwIfError(::miopenSetConvolutionMathType(conv_.get(), algo_perf_.mathType));

    size_t sizeInBytes = 0;
    throwIfError(::miopenGetConvolutionBackwardDataWorkspaceSize(dnnHandle.get(),
                                                                filter_desc_.get(),
                                                                doutput_desc_.get(),
                                                                conv_.get(),
                                                                dinput_desc_.get(),
                                                                algo_perf_.algo,
                                                                &sizeInBytes));
    algo_perf_.memory = sizeInBytes;

    return true;
}

void ConvolutionBackpropDataDescriptormiopen::FindAlgo(const rocm::DnnHandle& dnnHandle) {
    switch (tensor_element_type_) {
        case miopenHalf:
            for (const auto half_desc_type : half_desc_types_) {
                if (FindAlgoForConvDataType(dnnHandle, half_desc_type)) {
                    conv_desc_type_ = half_desc_type;
                    return;
                }
            }
            break;
        default:
            if (FindAlgoForConvDataType(dnnHandle, tensor_element_type_)) return;
    }

    throw_ov_exception("miopen: Unsupported convolution");
}

bool ConvolutionBackpropDataDescriptormiopen::FindAlgoForConvDataType(const rocm::DnnHandle& dnnHandle,
                                                                     miopenDataType_t convDataType) {
    miopenStatus_t status = miopen_STATUS_NOT_SUPPORTED;
    conv_ = params_.MakeConvolutionDescriptor(convDataType);
    const int requestedAlgoCount = 1;
    int returnedAlgoCount = 0;
    status = ::miopenFindConvolutionBackwardDataAlgorithm(dnnHandle.get(),
                                                         filter_desc_.get(),
                                                         doutput_desc_.get(),
                                                         conv_.get(),
                                                         dinput_desc_.get(),
                                                         requestedAlgoCount,
                                                         &returnedAlgoCount,
                                                         &algo_perf_);

    if ((status != miopen_STATUS_SUCCESS) || (algo_perf_.status != miopen_STATUS_SUCCESS) || (returnedAlgoCount <= 0)) {
        return false;
    }

    throwIfError(::miopenSetConvolutionMathType(conv_.get(), algo_perf_.mathType));

    size_t sizeInBytes = 0;
    throwIfError(::miopenGetConvolutionBackwardDataWorkspaceSize(dnnHandle.get(),
                                                                filter_desc_.get(),
                                                                doutput_desc_.get(),
                                                                conv_.get(),
                                                                dinput_desc_.get(),
                                                                algo_perf_.algo,
                                                                &sizeInBytes));
    algo_perf_.memory = sizeInBytes;

    return true;
}

void ConvolutionBackpropDataDescriptormiopen::FindAlgo(const rocm::DnnHandle& dnnHandle,
                                                      rocm::DevicePointer<const void*> filterPtr,
                                                      rocm::DevicePointer<const void*> dInPtr,
                                                      rocm::DevicePointer<void*> dOutPtr,
                                                      rocm::DeviceBuffer<std::byte> workspace) {
    switch (tensor_element_type_) {
        case miopenHalf:
            if (FindAlgoForConvDataType(dnnHandle, filterPtr, dInPtr, dOutPtr, workspace, miopenHalf)) return;
            if (FindAlgoForConvDataType(dnnHandle, filterPtr, dInPtr, dOutPtr, workspace, miopenFloat)) return;
            break;
        default:
            if (FindAlgoForConvDataType(dnnHandle, filterPtr, dInPtr, dOutPtr, workspace, tensor_element_type_)) return;
    }

    throw_ov_exception("miopen: Unsupported convolution");
}

bool ConvolutionBackpropDataDescriptormiopen::FindAlgoForConvDataType(const rocm::DnnHandle& dnnHandle,
                                                                     rocm::DevicePointer<const void*> filterPtr,
                                                                     rocm::DevicePointer<const void*> dInPtr,
                                                                     rocm::DevicePointer<void*> dOutPtr,
                                                                     rocm::DeviceBuffer<std::byte> workspace,
                                                                     miopenDataType_t convDataType) {
    miopenStatus_t status = miopen_STATUS_NOT_SUPPORTED;
    conv_ = params_.MakeConvolutionDescriptor(convDataType);
    const int requestedAlgoCount = 1;
    int returnedAlgoCount = 0;
    status = ::miopenFindConvolutionBackwardDataAlgorithmEx(dnnHandle.get(),
                                                           filter_desc_.get(),
                                                           filterPtr.get(),
                                                           doutput_desc_.get(),
                                                           dInPtr.get(),
                                                           conv_.get(),
                                                           dinput_desc_.get(),
                                                           dOutPtr.get(),
                                                           requestedAlgoCount,
                                                           &returnedAlgoCount,
                                                           &algo_perf_,
                                                           workspace.data(),
                                                           workspace.size());
    return (status == miopen_STATUS_SUCCESS) && (algo_perf_.status == miopen_STATUS_SUCCESS) && (returnedAlgoCount > 0);
}
*/
std::shared_ptr<rocm::DnnTensorDescriptor> MakeFusedAddDescriptor(const ov::Shape& shape,
                                                                  ov::element::Type_t element_type) {
    std::array<int, MIOPEN_DIM_MAX> int_shape;
    std::copy(shape.begin(), shape.end(), int_shape.begin());
    auto desc = std::make_shared<rocm::DnnTensorDescriptor>();
    desc->set(miopenTensorLayout_t::miopenTensorNCHW,
              convertDataType<miopenDataType_t>(element_type),
              static_cast<int>(shape.size()),
              int_shape.data());
    return desc;
}

std::shared_ptr<rocm::DnnActivationDescriptor> MakeFusedActivationDescriptor(nodes::ActivationMode mode) {
    auto desc = std::make_shared<rocm::DnnActivationDescriptor>();
    desc->set(convertActivationMode(mode), 0.0, 0.0, 0.0);
    return desc;
}

}  // namespace ov::rocm_gpu::Convolution::Details
