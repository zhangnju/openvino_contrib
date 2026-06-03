// Copyright (C) 2021-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <optional>

#include <rocm/node_params.hpp>
#include "runtime.hpp"

namespace rocm {

class GraphCapture;
class CaptureInfo;

class Graph : public Handle<hipGraph_t> {
public:
    Graph(unsigned int flags);

    friend bool operator==(const Graph& lhs, const Graph& rhs);

    friend GraphCapture;

private:
    Graph(hipGraph_t graph);

    static hipError_t createFromNative(hipGraph_t* pGraph, hipGraph_t anotherGraph);

    static hipGraph_t createNativeWithFlags(unsigned int flags);
};

bool operator==(const Graph& rhs, const Graph& lhs);

class GraphExec : public Handle<hipGraphExec_t> {
public:
    GraphExec(const Graph& g);

#if defined(rocm_VERSION) && rocm_VERSION >= 12020
    hipGraphExecUpdateResultInfo update(const Graph& g) const;
#else
    hipGraphExecUpdateResult update(const Graph& g) const;
#endif

    void launch(const Stream& stream) const;

    friend bool operator==(const GraphExec& lhs, const GraphExec& rhs);

#if !defined(NDEBUG) || defined(_DEBUG)
private:
    static constexpr std::size_t kErrorStringLen = 1024;
    char errorMsg_[kErrorStringLen];
#endif
};

bool operator==(const GraphExec& lhs, const GraphExec& rhs);

class GraphCapture {
public:
    class GraphCaptureScope {
    public:
        GraphCaptureScope(GraphCapture& graphCapture);

        GraphCaptureScope(const GraphCaptureScope&) = delete;

        GraphCaptureScope& operator=(const GraphCaptureScope&) = delete;

        ~GraphCaptureScope();

    private:
        GraphCapture& graphCapture_;
    };

    GraphCapture(const Stream& capturedStream);

    [[nodiscard]] GraphCaptureScope getScope();

    [[nodiscard]] const Graph& getGraph();

private:
    Stream stream_;
    hipGraph_t rocmGraph_{};
    hipError_t capturedError_{hipSuccess};
    std::optional<Graph> graph_{};
};

class UploadNode {
    friend CaptureInfo;

public:
    void update_src(const GraphExec& exec, const void* src);
    bool operator==(const UploadNode& rhs) const;

private:
    UploadNode(hipGraphNode_t node, rocm::DevicePointer<void*> dst, const void* src, std::size_t size);

    hipGraphNode_t node_;
    rocm::DevicePointer<void*> dst_;
    const void* src_;
    std::size_t size_;
};

class DownloadNode {
    friend CaptureInfo;

public:
    void update_dst(const GraphExec& exec, void* dst);
    bool operator==(const DownloadNode& rhs) const;

private:
    DownloadNode(hipGraphNode_t node, void* dst, rocm::DevicePointer<const void*> src, std::size_t size);

    hipGraphNode_t node_;
    void* dst_;
    rocm::DevicePointer<const void*> src_;
    std::size_t size_;
};

class TransferNode {
    friend CaptureInfo;

public:
    void update_ptrs(const GraphExec& exec, rocm::DevicePointer<void*> dst, rocm::DevicePointer<const void*> src);
    bool operator==(const TransferNode& rhs) const;

private:
    TransferNode(hipGraphNode_t node,
                 rocm::DevicePointer<void*> dst,
                 rocm::DevicePointer<const void*> src,
                 std::size_t size);

    hipGraphNode_t node_;
    rocm::DevicePointer<void*> dst_;
    rocm::DevicePointer<const void*> src_;
    std::size_t size_;
};

bool operator==(const hipKernelNodeParams& lhs, const hipKernelNodeParams& rhs);

class KernelNode {
    friend CaptureInfo;

public:
    template <typename... Args>
    void update_params(const GraphExec& exec, Args&&... args) {
        node_params_.reset_args();
        node_params_.add_args(std::forward<Args>(args)...);
        throwIfError(hipGraphExecKernelNodeSetParams(exec.get(), node_, &node_params_.get_knp()));
    }

    bool operator==(const KernelNode& rhs) const;

private:
    KernelNode(hipGraphNode_t node, rocm::NodeParams&& params);

    hipGraphNode_t node_;
    rocm::NodeParams node_params_;
};

class CaptureInfo {
public:
    CaptureInfo(const Stream& capturedStream);
    UploadNode addUploadNode(rocm::DevicePointer<void*> dst, const void* src, std::size_t size);
    DownloadNode addDownloadNode(void* dst, rocm::DevicePointer<const void*> src, std::size_t size);
    TransferNode addTransferNode(rocm::DevicePointer<void*> dst,
                                 rocm::DevicePointer<const void*> src,
                                 std::size_t size);
    template <typename... Args>
    KernelNode addKernelNode(void* kernel, dim3 gridDim, dim3 blockDim, Args&&... args);

private:
    const Stream& stream_;
    hipGraph_t capturingGraph_;
    hipStreamCaptureStatus captureStatus_;
    const hipGraphNode_t* deps_;
    size_t depCount_;
};

template <typename... Args>
KernelNode CaptureInfo::addKernelNode(void* kernel, dim3 gridDim, dim3 blockDim, Args&&... args) {
    hipGraphNode_t newNode;
    rocm::NodeParams params{kernel, gridDim, blockDim};
    params.add_args(std::forward<Args>(args)...);
    throwIfError(hipGraphAddKernelNode(&newNode, capturingGraph_, deps_, depCount_, &params.get_knp()));
    throwIfError(hipStreamUpdateCaptureDependencies(stream_.get(), &newNode, 1, 1));
    return KernelNode{newNode, std::move(params)};
}

}  // namespace rocm
