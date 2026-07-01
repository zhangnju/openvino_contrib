// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "convolution_miopen_be.hpp"
#include "convolution_miopen.hpp"

#include <fmt/format.h>

#include <algorithm>
#include <openvino/core/except.hpp>
#include <ops/converters.hpp>

#include "rocm/constant_factory.hpp"
#include "rocm/dnn_be_algo.hpp"

namespace ov {
namespace rocm_gpu {

constexpr int NON_SPATIAL_DIMS_NUMBER = 2;

struct DnnTensorID {
    static constexpr int64_t input = 'x';
    static constexpr int64_t filter = 'w';
    static constexpr int64_t output = 'y';
};

ConvolutionmiopenBE::ConvolutionmiopenBE(const CreationContext& context,
                                       const ov::Node& node,
                                       IndexCollection&& inputIds,
                                       IndexCollection&& outputIds,
                                       const Convolution::Details::ConvolutionParams& params)
    : OperationMIOPEN{context, node, std::move(inputIds), std::move(outputIds)}, params_{params} {
    const miopenDataType_t tensor_element_type = convertDataType<miopenDataType_t>(params.element_type_);

    // Convolution dimension according to op spec (1D, 2D or 3D). 1D should
    // already be turned into 2D at this point.
    const int arrayLength = static_cast<int>(params.input_shape_.size()) - NON_SPATIAL_DIMS_NUMBER;
    OPENVINO_ASSERT((arrayLength == 2) || (arrayLength == 3), "Node name: ", GetName());
    OPENVINO_ASSERT(arrayLength == params.strides_.size(), "Node name: ", GetName());
    OPENVINO_ASSERT(arrayLength == params.dilations_.size(), "Node name: ", GetName());
    OPENVINO_ASSERT(arrayLength == params.padding_before_.size(), "Node name: ", GetName());
    OPENVINO_ASSERT(arrayLength == params.padding_after_.size(), "Node name: ", GetName());

    auto input_desc = MakeTensorDescriptor(DnnTensorID::input, tensor_element_type, params.input_shape_);
    auto filter_desc = MakeTensorDescriptor(DnnTensorID::filter, tensor_element_type, params.filter_shape_);
    auto output_desc = MakeTensorDescriptor(DnnTensorID::output, tensor_element_type, params.output_shape_);

    auto conv_desc = rocm::DnnBEConvolutionDescriptorBuilder()
                         .setMode(miopenConvolutionMode_t::miopenConvolution)
                         .setComputeType(tensor_element_type)
                         .setNumberOfSpatialDimensions(arrayLength)
                         .setPrePaddings(params.padding_before_)
                         .setPostPaddings(params.padding_after_)
                         .setDilations(params.dilations_)
                         .setFilterStrides(params.strides_)
                         .build();

    auto conv_op_desc_builder = rocm::DnnBEOperationConvolutionForwardDescriptorBuilder()
                                    .setXDesc(input_desc)
                                    .setWDesc(filter_desc)
                                    .setYDesc(output_desc)
                                    .setConvDesc(conv_desc);
    if (tensor_element_type == miopenDouble) {
        conv_op_desc_builder.setScalingParams<MIOPEN_TYPE_DOUBLE>(1, 0);
    } else {
        conv_op_desc_builder.setScalingParams<MIOPEN_TYPE_FLOAT>(1, 0);
    }
    auto conv_op_desc = conv_op_desc_builder.build();

    auto dnnHandle = std::make_shared<rocm::DnnHandle>();

    std::array<miopenBackendDescriptor_t, 1> ops{conv_op_desc->get()};
    rocm::DnnBEOperationGraphDescriptorBuilder graphBuilder;
    graphBuilder.setDnnHandle(dnnHandle);
    graphBuilder.setOperations(ops);
    auto graph = graphBuilder.build();

    auto plans = rocm::getAllExecutionPlansFromHeuristics(graph, *dnnHandle);
    if (plans.empty()) {
        throw_ov_exception("miopen BE API: Unsupported convolution");
    }

    std::shared_ptr<rocm::DnnBEExecutionPlan> plan;
    if (context.opBenchOption()) {
        plan = performBenchmarks(context.dnnHandle(), plans);
    } else {
        plan = std::move(plans[0]);
    }

    engine_config_ = plan->getConfigDesc();
    workspace_size_ = plan->getWorkspaceSize();
}

std::shared_ptr<rocm::DnnBEExecutionPlan> ConvolutionmiopenBE::performBenchmarks(
    const rocm::DnnHandle& dnnHandle, std::vector<std::shared_ptr<rocm::DnnBEExecutionPlan>>& plans) {
    auto input = rocm::DefaultStream::stream().malloc(ov::element::Type{params_.element_type_}.size() *
                                                      ov::shape_size(params_.input_shape_));
    auto filter = rocm::DefaultStream::stream().malloc(ov::element::Type{params_.element_type_}.size() *
                                                       ov::shape_size(params_.filter_shape_));
    auto output = rocm::DefaultStream::stream().malloc(ov::element::Type{params_.element_type_}.size() *
                                                       ov::shape_size(params_.output_shape_));
    auto variantPackBuilder = rocm::DnnBEVariantPackBuilder();
    std::array<int64_t, 3> uids = {DnnTensorID::input, DnnTensorID::filter, DnnTensorID::output};
    std::array<const void*, 3> data_ptrs = {input.get(), filter.get(), output.get()};
    variantPackBuilder.setTensorPointers(uids, data_ptrs);

    constexpr const size_t kNumBenchmarks = 100;
    return rocm::performBenchmarks<kNumBenchmarks>(dnnHandle, plans, variantPackBuilder);
}

WorkbufferRequest ConvolutionmiopenBE::GetWorkBufferRequest() const {
    OPENVINO_ASSERT(engine_config_, "Node name: ", GetName());
    if (workspace_size_ < 0) {
        ov::rocm_gpu::throw_ov_exception(fmt::format("Workspace Size Invalid = {}", workspace_size_));
    }
    const size_t size = std::max(static_cast<int64_t>(0), workspace_size_);
    if (size > 0) {
        return {{}, {size}};
    } else {
        return {};
    }
}

void ConvolutionmiopenBE::Execute(const InferenceRequestContext& context,
                                 Inputs inputs,
                                 Outputs outputs,
                                 const Workbuffers& workbuffers) const {
    MiopenLockGuard miopen_lock;
    OPENVINO_ASSERT(inputs.size() == 2, "Node name: ", GetName());
    OPENVINO_ASSERT(outputs.size() == 1, "Node name: ", GetName());

    auto dnnHandle = context.getThreadContext().dnnHandle();
    auto workbuffer = workbuffers.mutable_buffers.empty() ? nullptr : workbuffers.mutable_buffers[0].get();
    std::array<const void*, 3> dataPtrs = {inputs[Convolution::Details::FusedConvolutionIndices::input].get(),
                                           inputs[Convolution::Details::FusedConvolutionIndices::filter].get(),
                                           outputs[Convolution::Details::FusedConvolutionIndices::output].get()};
    const std::array<int64_t, 3> uids = {DnnTensorID::input, DnnTensorID::filter, DnnTensorID::output};
    auto variantPackBuilder = rocm::DnnBEVariantPackBuilder();
    variantPackBuilder.setTensorPointers(uids, dataPtrs);
    variantPackBuilder.setWorkspase(workbuffer);
    const auto variantPack = variantPackBuilder.build();

    const auto plan = rocm::DnnBEExecutionPlanBuilder()
                          .setDnnHandle(context.getThreadContext().dnnHandle())
                          .setEngineConfig(engine_config_)
                          .build();

    throwIfError(::miopenBackendExecute(context.getThreadContext().dnnHandle().get(), plan->get(), variantPack->get()));
}

rocmGraphCompatibility ConvolutionmiopenBE::GetrocmGraphCompatibility() const { return rocmGraphCompatibility::NONE; }

std::shared_ptr<rocm::DnnBETensorDescriptor> ConvolutionmiopenBE::MakeTensorDescriptor(int64_t id,
                                                                                      miopenDataType_t element_type,
                                                                                      const ov::Shape& shape) {
    const int nbDims = shape.size();
    if (nbDims < 4 || nbDims > 5)
        throw_ov_exception(fmt::format("Unexpected number of dimensions for Convolution input/output: {}", nbDims));

    return rocm::DnnBETensorDescriptorBuilder()
        .setUniqueId(id)
        .setDataType(element_type)
        .setShape(shape)
        .setStrides(ov::row_major_strides(shape))
        .setAlignment(rocm::memoryAlignment)
        .setIsVirtual(false)
        .build();
}
}  // namespace rocm_gpu
}  // namespace ov
