// Copyright (C) 2021-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <hip/hip_runtime.h>
#include "graph.hpp"
#include "openvino/core/except.hpp"
#include <fmt/format.h>

namespace rocm {

Graph::Graph(unsigned int flags) :
        Graph { createNativeWithFlags(flags) } {
}

Graph::Graph(hipGraph_t graph) :
        Handle { createFromNative, hipGraphDestroy, graph } {
}

hipError_t Graph::createFromNative(hipGraph_t* pGraph, const hipGraph_t anotherGraph) {
    *pGraph = anotherGraph;
    return hipSuccess;
}

hipGraph_t Graph::createNativeWithFlags(unsigned int flags) {
    hipGraph_t g;
    throwIfError(hipGraphCreate(&g, flags));
    return g;
}

bool operator==(const Graph& rhs, const Graph& lhs) { return rhs.get() == lhs.get(); }

GraphExec::GraphExec(const Graph& g)
#if !defined(NDEBUG) || defined(_DEBUG)
    try
#endif
    : Handle(hipGraphInstantiate,
             hipGraphExecDestroy,
             g.get(),
             static_cast<hipGraphNode_t*>(nullptr),
#if !defined(NDEBUG) || defined(_DEBUG)
             errorMsg_,
             kErrorStringLen)
#else
             static_cast<char*>(nullptr),
             static_cast<size_t>(0ul))
#endif
{
}
#if !defined(NDEBUG) || defined(_DEBUG)
catch (std::exception& e) {
    OPENVINO_THROW(e.what(), ": ", errorMsg_);
}
#endif

#if defined(rocm_VERSION) && rocm_VERSION >= 12020
rocmGraphExecUpdateResultInfo GraphExec::update(const Graph& g) const {
    hipGraphExecUpdateResultInfo res;
    throwIfError(hipGraphExecUpdate(get(), g.get(), &res));
    return res;
}
#else
hipGraphExecUpdateResult GraphExec::update(const Graph& g) const {
    hipGraphExecUpdateResult res;
    throwIfError(hipGraphExecUpdate(get(), g.get(), nullptr, &res));
    return res;
}
#endif

void GraphExec::launch(const Stream& stream) const {
    throwIfError(hipGraphLaunch(get(), stream.get()));
}

bool operator==(const GraphExec& lhs, const GraphExec& rhs) { return rhs.get() == lhs.get(); }

GraphCapture::GraphCaptureScope::GraphCaptureScope(GraphCapture& graphCapture) : graphCapture_{graphCapture} {
    throwIfError(hipStreamBeginCapture(graphCapture_.stream_.get(), hipStreamCaptureModeThreadLocal));
}

GraphCapture::GraphCaptureScope::~GraphCaptureScope() {
    graphCapture_.capturedError_ = hipStreamEndCapture(graphCapture_.stream_.get(), &graphCapture_.rocmGraph_);
}

GraphCapture::GraphCapture(const Stream& capturedStream) :
        stream_ { capturedStream } {
}

GraphCapture::GraphCaptureScope GraphCapture::getScope() {
    graph_.reset();
    rocmGraph_ = nullptr;
    capturedError_ = hipSuccess;
    return GraphCapture::GraphCaptureScope { *this };
}

const Graph& GraphCapture::getGraph() {
    throwIfError(capturedError_);
    if (!graph_ && rocmGraph_ != nullptr) {
        graph_ = std::make_optional<Graph>( Graph{ rocmGraph_ });
    }
    return graph_.value();
}

CaptureInfo::CaptureInfo(const Stream& capturedStream) : stream_{capturedStream} {
    throwIfError(hipStreamGetCaptureInfo_v2(capturedStream.get(), &captureStatus_, nullptr,
            &capturingGraph_, &deps_, &depCount_));
}

UploadNode CaptureInfo::addUploadNode(DevicePointer<void*> dst, const void* src, std::size_t size) {
    hipGraphNode_t newNode;
    throwIfError(hipGraphAddMemcpyNode1D(&newNode, capturingGraph_, deps_, depCount_,
            dst.get(), src, size, hipMemcpyHostToDevice));
    throwIfError(hipStreamUpdateCaptureDependencies(stream_.get(), &newNode, 1, 1));
    return UploadNode{newNode, dst, src, size};
}

DownloadNode CaptureInfo::addDownloadNode(void* dst, DevicePointer<const void*> src,
                                                       std::size_t size) {
    hipGraphNode_t newNode;
    throwIfError(hipGraphAddMemcpyNode1D(&newNode, capturingGraph_, deps_, depCount_,
            dst, src.get(), size, hipMemcpyDeviceToHost));
    throwIfError(hipStreamUpdateCaptureDependencies(stream_.get(), &newNode, 1, 1));
    return DownloadNode{newNode, dst, src, size};
}

TransferNode CaptureInfo::addTransferNode(rocm::DevicePointer<void*> dst,
                                          rocm::DevicePointer<const void*> src,
                                          std::size_t size) {
    hipGraphNode_t newNode;
    throwIfError(hipGraphAddMemcpyNode1D(
        &newNode, capturingGraph_, deps_, depCount_, dst.get(), src.get(), size, hipMemcpyDeviceToDevice));
    throwIfError(hipStreamUpdateCaptureDependencies(stream_.get(), &newNode, 1, 1));
    return TransferNode{newNode, dst, src, size};
}

void UploadNode::update_src(const GraphExec& exec, const void* src) {
    if (src_ != src) {
        throwIfError(hipGraphExecMemcpyNodeSetParams1D(exec.get(), node_,
                dst_.get(), src, size_, hipMemcpyHostToDevice));
        src_ = src;
    }
}

UploadNode::UploadNode(hipGraphNode_t node, DevicePointer<void*> dst, const void* src, std::size_t size)
    : node_{node},
      dst_{dst},
      src_{src},
      size_{size} {
}

void DownloadNode::update_dst(const GraphExec& exec, void* dst) {
    if (dst_ != dst) {
        throwIfError(hipGraphExecMemcpyNodeSetParams1D(exec.get(), node_,
                dst, src_.get(), size_, hipMemcpyDeviceToHost));
        dst_ = dst;
    }
}

DownloadNode::DownloadNode(hipGraphNode_t node, void* dst, DevicePointer<const void*> src, std::size_t size)
    : node_{node}, dst_{dst}, src_{src}, size_{size} {}

void rocm::TransferNode::update_ptrs(const GraphExec& exec,
                                     rocm::DevicePointer<void*> dst,
                                     rocm::DevicePointer<const void*> src) {
    if (dst_ != dst || src_ != src) {
        dst_ = dst;
        src_ = src;
        throwIfError(hipGraphExecMemcpyNodeSetParams1D(
            exec.get(), node_, dst_.get(), src_.get(), size_, hipMemcpyDeviceToDevice));
    }
}

rocm::TransferNode::TransferNode(hipGraphNode_t node,
                                 rocm::DevicePointer<void*> dst,
                                 rocm::DevicePointer<const void*> src,
                                 std::size_t size)
    : node_{node}, dst_{dst}, src_{src}, size_{size} {}

rocm::KernelNode::KernelNode(hipGraphNode_t node, rocm::NodeParams&& params) : node_{node}, node_params_{params} {}

bool UploadNode::operator==(const UploadNode& rhs) const {
    return size_ == rhs.size_ && src_ == rhs.src_ && dst_.get() == rhs.dst_.get() && node_ == rhs.node_;
}

bool DownloadNode::operator==(const DownloadNode& rhs) const {
    return size_ == rhs.size_ && src_.get() == rhs.src_.get() && dst_ == rhs.dst_ && node_ == rhs.node_;
}

bool rocm::TransferNode::operator==(const TransferNode& rhs) const {
    return size_ == rhs.size_ && src_.get() == rhs.src_.get() && dst_.get() == rhs.dst_.get() && node_ == rhs.node_;
}

bool KernelNode::operator==(const KernelNode& rhs) const {
    return node_ == rhs.node_ && node_params_ == rhs.node_params_;
}
}  // namespace rocm
