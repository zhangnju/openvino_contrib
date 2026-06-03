// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "rocm_graph_context.hpp"

namespace ov {
namespace rocm_gpu {

void rocmGraphInfo::reset() {
    graph_.reset();
    graphExec_.reset();
    parameterNodes_.clear();
    resultNodes_.clear();
    transferNodes_.clear();
    kernelNodes_.clear();
}

void rocmGraphInfo::add_parameter(const std::string& tensorName,
                                  const rocm::Stream& stream,
                                  rocm::DevicePointer<void*> dst,
                                  const void* src,
                                  std::size_t size) {
    rocm::CaptureInfo captureInfo{stream};
    parameterNodes_.emplace(tensorName, captureInfo.addUploadNode(dst, src, size));
}

void rocmGraphInfo::add_result(const std::string& tensorName,
                               const rocm::Stream& stream,
                               void* dst,
                               rocm::DevicePointer<const void*> src,
                               std::size_t size) {
    rocm::CaptureInfo captureInfo{stream};
    resultNodes_.emplace(tensorName, captureInfo.addDownloadNode(dst, src, size));
}

void rocmGraphInfo::add_transfer(const rocm::Stream& stream,
                                 rocm::DevicePointer<void*> dst,
                                 rocm::DevicePointer<const void*> src,
                                 std::size_t size) {
    rocm::CaptureInfo captureInfo{stream};
    transferNodes_.emplace_back(captureInfo.addTransferNode(dst, src, size));
}

bool rocmGraphInfo::is_initialized() const { return graph_.has_value() && graphExec_.has_value(); }

void rocmGraphInfo::update_capture(const TensorMappingContext& context) {
    for (auto&& [tensorName, node] : parameterNodes_) {
        node.update_src(graphExec_.value(), (context.get_input_tensor(tensorName)->data()));
    }
    for (auto&& [tensorName, node] : resultNodes_) {
        node.update_dst(graphExec_.value(), context.get_output_tensor(tensorName)->data());
    }
}

std::size_t rocmGraphInfo::get_graphs_count() const { return is_initialized() ? 1 : 0; }

void rocmGraphInfo::launch(const rocm::Stream& stream) const { graphExec_.value().launch(stream); }

void rocmGraphPack::reset() {
    graphs_.clear();
    currentGraphIndex_ = 0;
}

void rocmGraphPack::add_parameter(const std::string& tensorName,
                                     const rocm::Stream& stream,
                                     rocm::DevicePointer<void*> dst,
                                     const void* src,
                                     std::size_t size) {
    OPENVINO_ASSERT(currentGraphIndex_ < graphs_.size(), "Graph index/vector size incosistency");
    graphs_[currentGraphIndex_]->add_parameter(tensorName, stream, dst, src, size);
}

void rocmGraphPack::add_result(const std::string& tensorName,
                                  const rocm::Stream& stream,
                                  void* dst,
                                  rocm::DevicePointer<const void*> src,
                                  std::size_t size) {
    OPENVINO_ASSERT(currentGraphIndex_ < graphs_.size(), "Graph index/vector size incosistency");
    graphs_[currentGraphIndex_]->add_result(tensorName, stream, dst, src, size);
}

void rocmGraphPack::add_transfer(const rocm::Stream& stream,
                                 rocm::DevicePointer<void*> dst,
                                 rocm::DevicePointer<const void*> src,
                                 std::size_t size) {
    graphs_[currentGraphIndex_]->add_transfer(stream, dst, src, size);
}

void rocmGraphPack::set_current_graph(const rocm::Graph& graph) {
    OPENVINO_ASSERT(currentGraphIndex_ < graphs_.size(), "Graph index/vector size incosistency");
    graphs_[currentGraphIndex_]->set_current_graph(graph);
}

bool rocmGraphPack::is_initialized() const {
    const auto size = graphs_.size();
    return size != 0 && graphs_[size - 1]->is_initialized();
}

void rocmGraphPack::update_capture(const TensorMappingContext& context) {
    for (currentGraphIndex_ = 0; currentGraphIndex_ < graphs_.size(); ++currentGraphIndex_) {
        graphs_[currentGraphIndex_]->update_capture(context);
    }
}

IrocmGraphInfo& rocmGraphPack::add(std::shared_ptr<IrocmGraphInfo> ptr) {
    currentGraphIndex_ = graphs_.size();
    graphs_.emplace_back(ptr);
    return *graphs_.back();
}

IrocmGraphInfo& rocmGraphPack::get_current_graph() { return *graphs_[currentGraphIndex_]; }

void rocmGraphPack::select_current_graph(std::size_t index) {
    OPENVINO_ASSERT(index < graphs_.size(), "Graph index/vector size incosistency");
    currentGraphIndex_ = index;
}

std::size_t rocmGraphPack::get_params_count() const {
    return std::accumulate(
        graphs_.begin(), graphs_.end(), static_cast<std::size_t>(0), [](auto sum, const auto& graph) {
            return sum + graph->get_params_count();
        });
}

std::size_t rocmGraphPack::get_results_count() const {
    return std::accumulate(
        graphs_.begin(), graphs_.end(), static_cast<std::size_t>(0), [](auto sum, const auto& graph) {
            return sum + graph->get_results_count();
        });
}

std::size_t rocmGraphPack::get_transfers_count() const {
    return std::accumulate(
        graphs_.begin(), graphs_.end(), static_cast<std::size_t>(0), [](auto sum, const auto& graph) {
            return sum + graph->get_transfers_count();
        });
}

std::size_t rocmGraphPack::get_kernels_count() const {
    return std::accumulate(
        graphs_.begin(), graphs_.end(), static_cast<std::size_t>(0), [](auto sum, const auto& graph) {
            return sum + graph->get_kernels_count();
        });
}

std::size_t rocmGraphPack::get_graphs_count() const {
    return std::accumulate(
        graphs_.begin(), graphs_.end(), static_cast<std::size_t>(0), [](auto sum, const auto& graph) {
            return sum + graph->get_graphs_count();
        });
}

void rocmGraphPack::launch(const rocm::Stream& stream) const { graphs_[currentGraphIndex_]->launch(stream); }

}  // namespace rocm_gpu
}  // namespace ov
