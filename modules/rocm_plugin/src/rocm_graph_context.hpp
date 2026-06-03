// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <rocm/graph.hpp>

#include "rocm_tensor_mapping_context.hpp"

namespace ov {
namespace rocm_gpu {

class IrocmGraphInfo {

public:
    virtual ~IrocmGraphInfo() = 0;

    virtual void reset() = 0;

    virtual void add_parameter(const std::string& tensorName,
                               const rocm::Stream& stream,
                               rocm::DevicePointer<void*> dst,
                               const void* src,
                               std::size_t size) = 0;

    virtual void add_result(const std::string& tensorName,
                            const rocm::Stream& stream,
                            void* dst,
                            rocm::DevicePointer<const void*> src,
                            std::size_t size) = 0;

    virtual void add_transfer(const rocm::Stream& stream,
                              rocm::DevicePointer<void*> dst,
                              rocm::DevicePointer<const void*> src,
                              std::size_t size) = 0;

    template <typename... Args>
    void add_kernel(const rocm::Stream& stream, void* kernel, dim3 gridDim, dim3 blockDim, Args&&... args) {
        rocm::CaptureInfo captureInfo{stream};
        get_kernels().emplace_back(captureInfo.addKernelNode(kernel, gridDim, blockDim, std::forward<Args>(args)...));
    }

    template <typename... Args>
    void update_kernel(std::size_t index, Args&&... args) {
        get_kernels()[index].update_params(get_graph_exec().value(), std::forward<Args>(args)...);
    }

    virtual void set_current_graph(const rocm::Graph& graph) = 0;

    virtual bool is_initialized() const = 0;
    virtual bool is_nested() const = 0;

    virtual void update_capture(const TensorMappingContext& context) = 0;

    virtual IrocmGraphInfo& add(std::shared_ptr<IrocmGraphInfo> ptr) = 0;

    virtual IrocmGraphInfo& get_current_graph() = 0;

    virtual void select_current_graph(std::size_t index) = 0;

    virtual std::size_t get_params_count() const = 0;
    virtual std::size_t get_results_count() const = 0;
    virtual std::size_t get_transfers_count() const = 0;
    virtual std::size_t get_kernels_count() const = 0;

    virtual std::size_t get_graphs_count() const = 0;

    virtual void launch(const rocm::Stream& stream) const = 0;

    virtual std::vector<rocm::KernelNode>& get_kernels() = 0;
    virtual std::optional<rocm::GraphExec>& get_graph_exec() = 0;
};

inline IrocmGraphInfo::~IrocmGraphInfo() = default;

class rocmGraphInfo : public IrocmGraphInfo {
public:
    rocmGraphInfo() = default;
    rocmGraphInfo(const rocmGraphInfo&) = delete;
    rocmGraphInfo& operator=(const rocmGraphInfo&) = delete;

    static std::shared_ptr<IrocmGraphInfo> create() { return std::make_shared<rocmGraphInfo>(); }

    void reset() override;

    void add_parameter(const std::string& tensorName,
                       const rocm::Stream& stream,
                       rocm::DevicePointer<void*> dst,
                       const void* src,
                       std::size_t size) override;

    void add_result(const std::string& tensorName,
                    const rocm::Stream& stream,
                    void* dst,
                    rocm::DevicePointer<const void*> src,
                    std::size_t size) override;

    void add_transfer(const rocm::Stream& stream,
                      rocm::DevicePointer<void*> dst,
                      rocm::DevicePointer<const void*> src,
                      std::size_t size) override;

    void set_current_graph(const rocm::Graph& graph) override {
        graph_.emplace(graph);
        graphExec_.emplace(graph);
    }

    bool is_initialized() const override;
    bool is_nested() const override { return false; };

    void update_capture(const TensorMappingContext& context) override;

    IrocmGraphInfo& add(std::shared_ptr<IrocmGraphInfo> ptr) override {
        OPENVINO_THROW("add() called for rocmGraphInfo");
    }

    IrocmGraphInfo& get_current_graph() override { return *this; }

    void select_current_graph(std::size_t index) override {
        OPENVINO_THROW("select_current_graph() called for rocmGraphInfo");
    }

    std::size_t get_params_count() const override { return parameterNodes_.size(); }
    std::size_t get_results_count() const override { return resultNodes_.size(); }
    std::size_t get_transfers_count() const override { return transferNodes_.size(); }
    std::size_t get_kernels_count() const override { return kernelNodes_.size(); }

    std::size_t get_graphs_count() const override;

    void launch(const rocm::Stream& stream) const override;

    std::vector<rocm::KernelNode>& get_kernels() override { return kernelNodes_; };
    std::optional<rocm::GraphExec>& get_graph_exec() override { return graphExec_; };

    const std::map<std::string, rocm::UploadNode>& get_parameter_nodes() const { return parameterNodes_; }
    const std::map<std::string, rocm::DownloadNode>& get_result_nodes() const { return resultNodes_; }

private:
    std::optional<rocm::Graph> graph_{};
    std::optional<rocm::GraphExec> graphExec_{};

    std::map<std::string, rocm::UploadNode> parameterNodes_;
    std::map<std::string, rocm::DownloadNode> resultNodes_;

    std::vector<rocm::TransferNode> transferNodes_;
    std::vector<rocm::KernelNode> kernelNodes_;
};

class rocmGraphPack : public IrocmGraphInfo {
public:
    rocmGraphPack() = default;
    rocmGraphPack(const rocmGraphPack&) = delete;
    rocmGraphPack& operator=(const rocmGraphPack&) = delete;

    static std::shared_ptr<IrocmGraphInfo> create() { return std::make_shared<rocmGraphPack>(); }

    void reset() override;

    void add_parameter(const std::string& tensorName,
                       const rocm::Stream& stream,
                       rocm::DevicePointer<void*> dst,
                       const void* src,
                       std::size_t size) override;

    void add_result(const std::string& tensorName,
                    const rocm::Stream& stream,
                    void* dst,
                    rocm::DevicePointer<const void*> src,
                    std::size_t size) override;

    void add_transfer(const rocm::Stream& stream,
                      rocm::DevicePointer<void*> dst,
                      rocm::DevicePointer<const void*> src,
                      std::size_t size) override;

    void set_current_graph(const rocm::Graph& graph) override;

    bool is_initialized() const override;
    bool is_nested() const override { return true; };

    void update_capture(const TensorMappingContext& context) override;

    IrocmGraphInfo& add(std::shared_ptr<IrocmGraphInfo> ptr) override;

    IrocmGraphInfo& get_current_graph() override;

    void select_current_graph(std::size_t index) override;

    std::size_t get_params_count() const override;
    std::size_t get_results_count() const override;
    std::size_t get_transfers_count() const override;
    std::size_t get_kernels_count() const override;

    std::size_t get_graphs_count() const override;

    void launch(const rocm::Stream& stream) const override;

    std::vector<rocm::KernelNode>& get_kernels() override { return graphs_[currentGraphIndex_]->get_kernels(); };
    std::optional<rocm::GraphExec>& get_graph_exec() override { return graphs_[currentGraphIndex_]->get_graph_exec(); };

private:
    std::vector<std::shared_ptr<IrocmGraphInfo>> graphs_{};
    std::size_t currentGraphIndex_ = 0;
};

using rocmGraphContext = rocmGraphPack;

}  // namespace rocm_gpu
}  // namespace ov
