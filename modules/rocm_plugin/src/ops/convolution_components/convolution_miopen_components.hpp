// Copyright (C) 2021-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <rocm_creation_context.hpp>

#include "convolution_components.hpp"
#include "rocm/dnn.hpp"
#define MIOPEN_DIM_MAX  5
namespace ov::rocm_gpu::Convolution::Details {

/**
 * @brief Presents convolution parameters in a form suitable for miopen API.
 */
class ConvolutionParamsmiopen {
public:
    ConvolutionParamsmiopen(const Convolution::Details::ConvolutionParams& params);

    int NumberOfSpatialDims() const { return number_of_dims_ - NON_SPATIAL_DIMS_NUMBER; }
    int NumberOfGroups() const { return groups_;}
    miopenDataType_t ElementType() const { return data_type_; }

    rocm::DnnTensorDescriptor MakeInputDescriptor() const;
    rocm::DnnFilterDescriptor MakeFilterDescriptor() const;
    rocm::DnnTensorDescriptor MakeOutputDescriptor() const;
    rocm::DnnConvolutionDescriptor MakeConvolutionDescriptor(miopenDataType_t convDataType) const;

private:
    const int number_of_dims_;
    const int groups_;
    const miopenDataType_t data_type_;
    using IntArray = std::array<int, MIOPEN_DIM_MAX>;
    IntArray input_shape_;
    IntArray filter_shape_;
    IntArray output_shape_;
    IntArray strides_;
    IntArray dilations_;
    IntArray paddings_;
};

/**
 * @brief Presents convolution parameters in a form suitable for miopen API.
 */
/*
class ConvolutionBackpropDataParamsmiopen {
public:
    ConvolutionBackpropDataParamsmiopen(const Convolution::Details::ConvolutionBackwardDataParams& params);

    int NumberOfSpatialDims() const { return number_of_dims_ - NON_SPATIAL_DIMS_NUMBER; }
    miopenDataType_t ElementType() const { return data_type_; }

    rocm::DnnTensorDescriptor MakeDOutputDescriptor() const;
    rocm::DnnFilterDescriptor MakeFilterDescriptor() const;
    rocm::DnnTensorDescriptor MakeDInputDescriptor() const;
    rocm::DnnConvolutionDescriptor MakeConvolutionDescriptor(miopenDataType_t convDataType) const;

private:
    const int number_of_dims_;
    const int groups_;
    const miopenDataType_t data_type_;
    using IntArray = std::array<int, MIOPEN_DIM_MAX>;
    IntArray doutput_shape_;
    IntArray filter_shape_;
    IntArray dinput_shape_;
    IntArray strides_;
    IntArray dilations_;
    IntArray paddings_;
};
*/
/**
 * @brief Prepares all data required for miopen convolution API invocation.
 */
class ConvolutionDescriptorsmiopen {
public:
    ConvolutionDescriptorsmiopen(const CreationContext& context,
                                const Convolution::Details::ConvolutionParamsmiopen& params,
                                const std::vector<miopenDataType_t> half_desc_types = {miopenHalf,
                                                                                      miopenFloat});

    miopenDataType_t ElementType() const { return tensor_element_type_; }
    miopenDataType_t DescType() const { return conv_desc_type_; }
    const rocm::DnnTensorDescriptor& Input() const { return input_; }
    const rocm::DnnFilterDescriptor& Filter() const { return filter_; }
    const rocm::DnnTensorDescriptor& Output() const { return output_; }
    const rocm::DnnConvolutionDescriptor& Conv() const { return conv_; }
    const miopenConvAlgoPerf_t& Algo() const { return algo_perf_; }
    const std::size_t WorkspaceSize() const {return workspace_size_;}
    const uint64_t SolutionId() const {return solution_id_;}
    void FindAlgo(const rocm::DnnHandle& dnnHandle,
                  rocm::DevicePointer<const void*> inPtr,
                  rocm::DevicePointer<const void*> filterPtr,
                  rocm::DevicePointer<void*> outPtr,
                  rocm::DeviceBuffer<std::byte> workspace);

private:
    bool FindAlgoForConvDataType(const rocm::DnnHandle& dnnHandle,
                                 rocm::DevicePointer<const void*> inPtr,
                                 rocm::DevicePointer<const void*> filterPtr,
                                 rocm::DevicePointer<void*> outPtr,
                                 rocm::DeviceBuffer<std::byte> workspace,
                                 miopenDataType_t convDataType);
    void BenchmarkOptimalAlgo(const rocm::DnnHandle& dnnHandle, const ConvolutionParamsmiopen& params);
    void GetAlgo(const rocm::DnnHandle& dnnHandle);
    bool GetAlgoForConvDataType(const rocm::DnnHandle& dnnHandle, miopenDataType_t convDataType);
    void FindAlgo(const rocm::DnnHandle& dnnHandle);
    bool FindAlgoForConvDataType(const rocm::DnnHandle& dnnHandle, miopenDataType_t convDataType);

private:
    ConvolutionParamsmiopen params_;
    miopenDataType_t tensor_element_type_;
    miopenDataType_t conv_desc_type_;
    rocm::DnnTensorDescriptor input_;
    rocm::DnnFilterDescriptor filter_;
    rocm::DnnTensorDescriptor output_;
    rocm::DnnConvolutionDescriptor conv_;
    miopenConvAlgoPerf_t  algo_perf_;
    std::vector<miopenDataType_t> half_desc_types_;
    std::size_t workspace_size_;
    uint64_t solution_id_;
};

/**
 * @brief Prepares all data required for miopen convolution API invocation.
 */
/*
class ConvolutionBackpropDataDescriptormiopen {
public:
    ConvolutionBackpropDataDescriptormiopen(const CreationContext& context,
                                           const Convolution::Details::ConvolutionBackpropDataParamsmiopen& params,
                                           const std::vector<miopenDataType_t> half_desc_types = {miopenHalf,
                                                                                                 miopenFloat});

    miopenDataType_t ElementType() const { return tensor_element_type_; }
    miopenDataType_t DescType() const { return conv_desc_type_; }
    const rocm::DnnTensorDescriptor& dOutput() const { return doutput_desc_; }
    const rocm::DnnFilterDescriptor& Filter() const { return filter_desc_; }
    const rocm::DnnTensorDescriptor& dInput() const { return dinput_desc_; }
    const rocm::DnnConvolutionDescriptor& Conv() const { return conv_; }
    const miopenConvBwdDataAlgorithm_t & Algo() const { return algo_perf_; }
    void FindAlgo(const rocm::DnnHandle& dnnHandle,
                  rocm::DevicePointer<const void*> filterPtr,
                  rocm::DevicePointer<const void*> dInPtr,
                  rocm::DevicePointer<void*> dOutPtr,
                  rocm::DeviceBuffer<std::byte> workspace);

private:
    bool FindAlgoForConvDataType(const rocm::DnnHandle& dnnHandle,
                                 rocm::DevicePointer<const void*> filterPtr,
                                 rocm::DevicePointer<const void*> dInPtr,
                                 rocm::DevicePointer<void*> dOutPtr,
                                 rocm::DeviceBuffer<std::byte> workspace,
                                 miopenDataType_t convDataType);
    void BenchmarkOptimalAlgo(const rocm::DnnHandle& dnnHandle);
    void GetAlgo(const rocm::DnnHandle& dnnHandle);
    bool GetAlgoForConvDataType(const rocm::DnnHandle& dnnHandle, miopenDataType_t convDataType);
    void FindAlgo(const rocm::DnnHandle& dnnHandle);
    bool FindAlgoForConvDataType(const rocm::DnnHandle& dnnHandle, miopenDataType_t convDataType);

private:
    ConvolutionBackpropDataParamsmiopen params_;
    miopenDataType_t tensor_element_type_;
    miopenDataType_t conv_desc_type_;
    rocm::DnnFilterDescriptor filter_desc_;
    rocm::DnnTensorDescriptor doutput_desc_;
    rocm::DnnTensorDescriptor dinput_desc_;
    rocm::DnnConvolutionDescriptor conv_;
    miopenConvBwdDataAlgorithm_t  algo_perf_;
    std::vector<miopenDataType_t> half_desc_types_;
};
*/
std::shared_ptr<rocm::DnnTensorDescriptor> MakeFusedAddDescriptor(const ov::Shape& shape,
                                                                  ov::element::Type_t element_type);
std::shared_ptr<rocm::DnnActivationDescriptor> MakeFusedActivationDescriptor(nodes::ActivationMode mode);

}  // namespace ov::rocm_gpu::Convolution::Details
