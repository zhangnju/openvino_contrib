// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "rocm_infer_request.hpp"

#include "rocm/float16.hpp"
#include <fmt/format.h>

#include <algorithm>
#include <gsl/span_ext>
#include <map>
#include <memory>
#include <string>
#include <utility>

#include "rocm_compiled_model.hpp"
#include "rocm_graph_topology_runner.hpp"
#include "ops/silu_tracking.hpp"
#include "rocm_itt.hpp"
#include "rocm_plugin.hpp"
#include "rocm_profiler.hpp"
#include "rocm_simple_execution_delegator.hpp"
#include "rocm/properties.hpp"
#include "openvino/runtime/make_tensor.hpp"
#include "openvino/runtime/threading/executor_manager.hpp"

namespace ov {
namespace rocm_gpu {
using namespace utils;

using Time = std::chrono::steady_clock;

namespace {
void allocate_tensor_impl(ov::SoPtr<ov::ITensor>& tensor, const ov::element::Type& element_type, const ov::Shape& shape) {
    if (!tensor || tensor->get_element_type() != element_type) {
        tensor = ov::SoPtr<ov::ITensor>{ov::make_tensor(element_type, shape), nullptr};
    } else {
        tensor->set_shape(shape);
    }
}

inline std::unique_ptr<IExecutionDelegator> create_execution_delegator(bool is_profiling_enabled,
                                                                       const SubGraph& subGraph) {
    if (is_profiling_enabled) {
        return std::make_unique<Profiler>(subGraph);
    }
    return std::make_unique<SimpleExecutionDelegator>();
}

}  // namespace

rocmInferRequest::rocmInferRequest(const std::shared_ptr<const CompiledModel>& compiled_model)
    : ov::ISyncInferRequest(compiled_model),
      cancellation_token_{[this] { memory_proxy_.reset(); }},
      executionDelegator_{
          create_execution_delegator(compiled_model->get_property(ov::enable_profiling.name()).as<bool>(),
                                     compiled_model->get_topology_runner().GetSubGraph())},
      is_benchmark_mode_{compiled_model->get_property(ov::rocm_gpu::operation_benchmark.name()).as<bool>()} {
    create_infer_request();
}

void rocmInferRequest::create_infer_request() {
    auto compiled_model = get_rocm_model();
    auto device_id = std::to_string(compiled_model->config_.get_device_id());
    auto request_id = std::to_string(compiled_model->request_id_.fetch_add(1));
    std::string name = "rocm" + device_id + "_" + compiled_model->model_->get_friendly_name() + "_req" + request_id;

    _profilingTask = {
        openvino::itt::handle(name + "_Preprocess"),
        openvino::itt::handle(name + "_Postprocess"),
        openvino::itt::handle(name + "_StartPipline"),
        openvino::itt::handle(name + "_WaitPipline"),
    };

    // Allocate plugin backend specific memory handles
    input_tensors_.resize(get_inputs().size());
    output_tensors_.resize(get_outputs().size());

    // Allocate input/output tensors
    for (const auto& input : get_inputs()) {
        allocate_tensor(input, [input](ov::SoPtr<ov::ITensor>& tensor) {
            // Can add a check to avoid double work in case of shared tensors
            allocate_tensor_impl(tensor,
                                 input.get_element_type(),
                                 input.get_partial_shape().is_dynamic() ? ov::Shape{0} : input.get_shape());
        });
    }
    for (const auto& output : get_outputs()) {
        allocate_tensor(output, [output](ov::SoPtr<ov::ITensor>& tensor) {
            // Can add a check to avoid double work in case of shared tensors
            allocate_tensor_impl(tensor,
                                 output.get_element_type(),
                                 output.get_partial_shape().is_dynamic() ? ov::Shape{0} : output.get_shape());
        });
    }
}

void rocmInferRequest::infer_preprocess() {
    OV_ITT_SCOPED_TASK(itt::domains::rocm_gpu, _profilingTask[PerfStages::Preprocess]);
    executionDelegator_->start_stage();

    convert_batched_tensors();
    check_tensors();

    // Allocate host input tensors
    OPENVINO_ASSERT(get_inputs().size() == input_tensors_.size());
    for (size_t i = 0; i < get_inputs().size(); i++) {
        auto tensor = ov::make_tensor(get_tensor(get_inputs()[i]));
        ov::element::Type element_type = tensor.get_element_type();
        ov::Shape shape = tensor.get_shape();
        if (tensor.is<ov::RemoteTensor>()) {
            OPENVINO_ASSERT(true, "rocm plugin doesn't support remote tensor.");
        } else if (tensor.is_continuous()) {
            // No ROI extraction is needed
            input_tensors_.at(i) =
                std::make_shared<ov::Tensor>(element_type, shape, tensor.data());
        } else {
            OPENVINO_ASSERT(element_type.bitwidth() % 8 == 0,
                            "Template plugin: Unsupported ROI tensor with element type having ",
                            std::to_string(element_type.bitwidth()),
                            " bits size");
            // Perform manual extraction of ROI tensor
            // Basic implementation doesn't take axis order into account `desc.getBlockingDesc().getOrder()`
            // Performance of manual extraction is not optimal, but it is ok for template implementation
            input_tensors_.at(i) = std::make_shared<ov::Tensor>(element_type, shape);
            tensor.copy_to(*input_tensors_[i].get());
        }
    }
    // Allocate host output tensors
    OPENVINO_ASSERT(get_outputs().size() == output_tensors_.size());
    OPENVINO_ASSERT(get_outputs().size() == get_rocm_model()->model_->get_results().size());
    for (size_t i = 0; i < get_outputs().size(); i++) {
        const auto& result = get_rocm_model()->model_->get_results()[i];
        if (result->get_output_partial_shape(0).is_dynamic()) {
            output_tensors_.at(i) = std::make_shared<ov::Tensor>();
            continue;
        }
        auto tensor = ov::make_tensor(get_tensor(get_outputs()[i]));
        ov::element::Type element_type = tensor.get_element_type();
        ov::Shape shape = tensor.get_shape();
        if (tensor.is_continuous() && !tensor.is<ov::RemoteTensor>())
            output_tensors_.at(i) = std::make_shared<ov::Tensor>(element_type, shape, tensor.data());
        else
            output_tensors_.at(i) = std::make_shared<ov::Tensor>(element_type, shape);
    }
    executionDelegator_->stop_stage(PerfStages::Preprocess);
}

void rocmInferRequest::start_pipeline(const ThreadContext& threadContext) {
    try {
        OV_ITT_SCOPED_TASK(itt::domains::rocm_gpu, _profilingTask[PerfStages::StartPipeline])
        executionDelegator_->start_stage();
        auto compiled_model = get_rocm_model();
        memory_proxy_ = compiled_model->memory_pool_->WaitAndGet(cancellation_token_);
        auto& memory = memory_proxy_->Get();
        auto& rocmGraphContext = memory.rocmGraphContext();
        auto& topology_runner = compiled_model->get_topology_runner();
        InferenceRequestContext inferRequestContext{input_tensors_,
                                                    compiled_model->input_index_,
                                                    output_tensors_,
                                                    compiled_model->output_index_,
                                                    threadContext,
                                                    cancellation_token_,
                                                    *executionDelegator_,
                                                    rocmGraphContext,
                                                    is_benchmark_mode_};
        topology_runner.UpdateContext(inferRequestContext, memory);
        // Clear any leftover silu_tracking marks from the previous inference.
        // Normally cleared by SwishOp::Execute, but skipped Swish nodes (rocm_swish_inplace)
        // are removed from exec_sequence_ and never run, so marks may persist.
        // Clearing here ensures each inference starts with a clean state.
        g_silu_applied_buffers.clear();
        topology_runner.Run(inferRequestContext, memory);
        executionDelegator_->stop_stage(PerfStages::StartPipeline);
    } catch (...) {
        // TODO:
        // Log error once logger is available
        memory_proxy_.reset();
        throw;
    }
}

void rocmInferRequest::wait_pipeline(const ThreadContext& threadContext) {
    OV_ITT_SCOPED_TASK(itt::domains::rocm_gpu, _profilingTask[PerfStages::WaitPipeline])
    executionDelegator_->start_stage();
    // TODO: probably all time will be spent in synchonize, out of reach of ThrowIfCanceled
    threadContext.stream().synchronize();
    memory_proxy_.reset();
    executionDelegator_->stop_stage(PerfStages::WaitPipeline);
}

void rocmInferRequest::infer_postprocess() {
    OV_ITT_SCOPED_TASK(itt::domains::rocm_gpu, _profilingTask[PerfStages::Postprocess]);
    executionDelegator_->start_stage();

    OPENVINO_ASSERT(get_outputs().size() == output_tensors_.size());
    OPENVINO_ASSERT(get_outputs().size() == get_rocm_model()->model_->get_results().size());
    for (size_t i = 0; i < get_outputs().size(); i++) {
        const auto& result = get_rocm_model()->model_->get_results()[i];
        auto host_tensor = *output_tensors_[i].get();
        auto tensor = ov::make_tensor(get_tensor(get_outputs()[i]));
        if (result->get_output_partial_shape(0).is_dynamic()) {
            ov::Output<const ov::Node> output{result->output(0).get_node(), result->output(0).get_index()};
            allocate_tensor(output, [host_tensor](ov::SoPtr<ov::ITensor>& tensor) {
                allocate_tensor_impl(tensor, host_tensor.get_element_type(), host_tensor.get_shape());
                auto ov_tensor = ov::make_tensor(tensor);
                host_tensor.copy_to(ov_tensor);
            });
        } else if (!tensor.is_continuous()) {
            host_tensor.copy_to(tensor);
        } else if (tensor.is<ov::RemoteTensor>()) {
            OPENVINO_ASSERT(true, "rocm plugin doesn't support RemoteTensor.");
        }
    }
    executionDelegator_->stop_stage(PerfStages::Postprocess);
    executionDelegator_->process_events();
}

void rocmInferRequest::cancel() {
    cancellation_token_.cancel();
    get_rocm_model()->get_memory_pool()->Interrupt();
}

void rocmInferRequest::infer() {
    OPENVINO_NOT_IMPLEMENTED;
}

std::shared_ptr<const CompiledModel> rocmInferRequest::get_rocm_model() {
    auto& compiled_model = get_compiled_model();
    auto rocm_model = std::dynamic_pointer_cast<const CompiledModel>(compiled_model);
    OPENVINO_ASSERT(rocm_model);
    return rocm_model;
}

void rocmInferRequest::set_tensors_impl(const ov::Output<const ov::Node> port,
                                        const std::vector<ov::SoPtr<ov::ITensor>>& tensors) {
    for (const auto& input : get_inputs()) {
        if (input == port) {
            m_batched_tensors[input.get_tensor_ptr()] = tensors;
            return;
        }
    }
    OPENVINO_THROW("Cannot find input tensors for port ", port);
}

std::vector<ov::SoPtr<ov::IVariableState>> rocmInferRequest::query_state() const {
    OPENVINO_NOT_IMPLEMENTED;
}

std::vector<ov::ProfilingInfo> rocmInferRequest::get_profiling_info() const {
    return executionDelegator_->get_performance_counts();
}
}  // namespace rocm_gpu
}  // namespace ov
