// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "rocm_op_buffers_extractor.hpp"

#include <fmt/format.h>

#include <error.hpp>
#include <gsl/span_ext>
#include <openvino/op/constant.hpp>
#include <openvino/op/matmul.hpp>
#include <openvino/op/reshape.hpp>
#include <openvino/op/result.hpp>
#include <openvino/op/squeeze.hpp>
#include <openvino/op/tensor_iterator.hpp>
#include <openvino/op/transpose.hpp>
#include <openvino/op/unsqueeze.hpp>
#include <stdexcept>
#include <transformer/nodes/concat_optimized.hpp>
#include <utility>
// Zero-copy VariadicSplit alias support (added without modifying existing functions)
#include "ops/variadic_split_alias.hpp"

namespace ov {
namespace rocm_gpu {

OperationBuffersExtractor::OperationBuffersExtractor(gsl::span<const NodePtr> ordered_nodes,
                                                     bool is_stable_params,
                                                     bool is_stable_results)
    : is_stable_params_{is_stable_params},
      is_stable_results_{is_stable_results},
      num_ordered_nodes_{static_cast<unsigned long>(ordered_nodes.size())} {
    for (int node_idx = 0; node_idx < num_ordered_nodes_; node_idx++) {
        const auto& node = ordered_nodes[node_idx];
        if (IsParameterNode(*node))
            extractParameterTensors(node, node_idx);
        else if (IsResultNode(*node))
            extractResultTensors(node);
        else if (IsConstantNode(*node))
            extractImmutableTensors(node);
        else if (IsConcatOptimizedNode(*node))
            mergeConcatMutableTensors(node, node_idx);
        else if (isReshapeOnlyNode(*node))
            extractReshapeTensors(node, node_idx);
        // Zero-copy VariadicSplit: outputs alias into sub-ranges of input buffer.
        // VariadicSplitAlias is inserted by VariadicSplitZeroCopyPass when
        // ROCM_ZEROCOPY_SPLIT != "0". No GPU kernel is needed; outputs directly
        // point into the input buffer at the appropriate channel offset.
        else if (ov::is_type<nodes::VariadicSplitAlias>(node.get()))
            extractSplitAliasTensors(node, node_idx);
        // In-place Swish: output buffer is an alias of input buffer at offset=0.
        // Creates a new TensorID (child of input TensorID) with zero byte offset,
        // so GPU pointer is identical (in_ptr == out_ptr in SwishOp::Execute).
        // Unlike extractReshapeTensors (shares same TensorID), this creates a child
        // TensorID to allow the output to have its own lifespan extension without
        // interfering with the input's buffer reuse schedule.
        // ROCM_SWISH_INPLACE=0 disables (default: enabled).
        else if (node->get_rt_info().count("rocm_attn_pe_conv") &&
                 node->get_rt_info().count("rocm_pe_4input") &&
                 node->get_input_size() >= 4) {
            // 4-input FGC pe noop (Form B): PRE-ALLOCATE output buffer with lifespan_start = 0.
            //
            // pe_add (in RocmAttentionMatMulOp::Execute) writes pe+attn result to this buffer.
            // AV Execute receives it as outputs[1] (via extra_node_outputs_).
            //
            // Why lifespan_start = 0 ("pre-allocated" mode)?
            // Problem: FGC is produced at FGC_idx > AV_idx. If lifespan_start = FGC_idx,
            // MemorySolver assigns the address slot [AV_idx, FGC_idx) to another tensor.
            // hipMemcpyAsync at AV_idx corrupts that tensor → GPU fault.
            // Fix: extend lifespan_start to 0 (before any node). MemorySolver allocates
            // the buffer at a fixed address valid for [0, last_consumer_idx]. No other
            // buffer uses this address range (MemorySolver guarantees no overlap).
            // Cost: ~307KB × N_attention_modules reserved for entire inference (acceptable).
            extractMutableTensors(node, 0);  // lifespan_start = 0 = pre-allocated
        }
        else if (node->get_rt_info().count("rocm_swish_inplace") &&
                 []{ const char* e=std::getenv("ROCM_SWISH_INPLACE"); return !(e && std::string(e)=="0"); }()) {
            // Get input TensorID
            OPENVINO_ASSERT(node->get_input_size() >= 1);
            const auto& in_name = GetTensorNameInternal(node->inputs().at(0));
            const auto& input_tensor_id = tensor_names_.at(in_name);
            // Create output TensorID as child of input at offset=0 (same GPU address)
            const size_t out_bytes = GetTensorByteSize(node->outputs().at(0));
            auto child_tensor = std::make_shared<TensorID>(next_buffer_id_++);
            child_tensor->SetParent(input_tensor_id, 0);  // offset=0 → same GPU pointer
            mutable_tensor_sizes_[child_tensor->GetId()] = out_bytes;
            // Register output tensor name → child TensorID
            tensor_names_.emplace(GetTensorNameInternal(node->outputs().at(0)), child_tensor);
            // Note: no mutable_buffers_ entry needed (child of existing buffer)
        }
        else if (node->get_rt_info().count("rocm_pe_add_noop")) {
            // pe+attn Add noop: allocate normally for safety.
            // The NoOpAddOp::Execute copies from AV-path input to output.
            // Buffer aliasing attempted separately (currently disabled due to Concat conflicts).
            extractMutableTensors(node, node_idx);
        }
        else
            extractMutableTensors(node, node_idx);
    }
    for (int node_idx = 0; node_idx < num_ordered_nodes_; node_idx++) {
        for (const auto& input : ordered_nodes[node_idx]->inputs()) {
            try {
                const auto tensorName = GetTensorNameInternal(input);
                const BufferID bufferId = tensor_names_.at(tensorName)->GetBuffer().GetId();
                const bool isImmutableBuffer = immutable_buffers_.find(bufferId) != immutable_buffers_.end();
                if (!isImmutableBuffer) {
                    auto& mutableBuffer = mutable_buffers_.at(bufferId);
                    if (node_idx > mutableBuffer.lifespan_end) {
                        mutableBuffer.lifespan_end = node_idx;
                    }
                    if (mutableBuffer.size < GetTensorByteSize(input)) {
                        ThrowBufferSizesAreNotMatchError(input);
                    }
                }
            } catch (std::out_of_range& e) {
                ThrowGraphIsBadFormedError(input);
            }
        }

    }
}

std::vector<TensorID> OperationBuffersExtractor::inputTensorIds(const ov::Node& node) const {
    std::vector<TensorID> result{};
    for (const auto& input : node.inputs()) {
        const auto& tensorId = tensor_names_.at(GetTensorNameInternal(input));
        result.push_back(*tensorId);
    }
    // Append extra (non-graph-edge) inputs registered by registerAttentionExtraInputs()
    // or other post-processing passes. These allow ops like RocmAttentionMatMulOp to
    // access tensors they don't have direct graph edges to (e.g. the full QKV tensor).
    const auto it = extra_node_inputs_.find(node.get_friendly_name());
    if (it != extra_node_inputs_.end()) {
        std::cerr << "[AttnExtra] Injecting " << it->second.size() << " extra inputs for: "
                  << node.get_friendly_name().substr(0,50) << "\n";
        for (const auto& extra_id : it->second) {
            result.push_back(extra_id);
        }
    }
    return result;
}

std::vector<TensorID> OperationBuffersExtractor::outputTensorIds(const ov::Node& node) const {
    if (IsResultNode(node)) return {};
    std::vector<TensorID> result{};
    for (const auto& output : node.outputs()) {
        const auto& tensorId = tensor_names_.at(GetTensorNameInternal(output));
        result.push_back(*tensorId);
    }
    // Append extra outputs (e.g. FGC output buffer for pe fusion)
    auto it = extra_node_outputs_.find(node.get_friendly_name());
    if (it != extra_node_outputs_.end()) {
        for (const auto& extra_id : it->second) {
            result.push_back(extra_id);
        }
    }
    return result;
}

int OperationBuffersExtractor::mutableBufferLifespanStart(BufferID buffer_id) const {
    try {
        return mutable_buffers_.at(buffer_id).lifespan_start;
    } catch (std::out_of_range& e) {
        throw_ov_exception(fmt::format("Buffer id {} is out of range.", buffer_id));
    }
}

int OperationBuffersExtractor::mutableBufferLifespanEnd(BufferID buffer_id) const {
    try {
        return mutable_buffers_.at(buffer_id).lifespan_end;
    } catch (std::out_of_range& e) {
        throw_ov_exception(fmt::format("Buffer id {} is out of range.", buffer_id));
    }
}

std::size_t OperationBuffersExtractor::mutableBufferSize(BufferID buffer_id) const {
    try {
        return mutable_buffers_.at(buffer_id).size;
    } catch (std::out_of_range& e) {
        throw_ov_exception(fmt::format("Buffer id {} is out of range.", buffer_id));
    }
}

gsl::span<const OperationBuffersExtractor::Byte> OperationBuffersExtractor::immutableBuffer(BufferID buffer_id) const {
    try {
        return immutable_buffers_.at(buffer_id);
    } catch (std::out_of_range& e) {
        throw_ov_exception(fmt::format("Buffer id {} is out of range.", buffer_id));
    }
}

std::vector<BufferID> OperationBuffersExtractor::mutableBuffersIds() const {
    std::vector<BufferID> result{};
    for (const auto& pair : mutable_buffers_) {
        result.push_back(pair.first);
    }
    return result;
}

std::vector<BufferID> OperationBuffersExtractor::immutableBuffersIds() const {
    std::vector<BufferID> result{};
    for (const auto& pair : immutable_buffers_) {
        result.push_back(pair.first);
    }
    return result;
}

void OperationBuffersExtractor::mergeConcatMutableTensors(const NodePtr& node, int node_idx) {
    std::vector<std::pair<std::string, TensorID::Ptr>> mergedTensors;
    mergedTensors.reserve(node->inputs().size());
    for (const auto& input : node->inputs()) {
        const auto& tensorName = GetTensorNameInternal(input.get_source_output());
        const auto& tensorId = tensor_names_.at(tensorName);
        // For in-place Swish nodes, tensorId is a child TensorID (has parent).
        // ConcatOptimized needs root TensorIDs, so we follow to the root.
        // This correctly handles the case where Swish output aliases FusedConv output.
        if (&tensorId->GetBuffer() != tensorId.get()) {
            // Child tensor: use root buffer's TensorID (which is already in tensor_names_)
            // The root TensorID corresponds to the FusedConv output buffer.
            // We register a new root-level entry for this tensor to allow merging.
            // Actually for ConcatOptimized, we need the input to be an independent root buffer.
            // If it's a child (aliased), fall back to extractMutableTensors for this concat.
            extractMutableTensors(node, node_idx);
            return;
        }
        mergedTensors.emplace_back(tensorName, tensorId);
    }
    OPENVINO_ASSERT(!mergedTensors.empty());

    std::vector<BufferID> mergedBufferIds;
    std::transform(mergedTensors.begin(), mergedTensors.end(), std::back_inserter(mergedBufferIds), [](const auto& nt) {
        return nt.second->GetBuffer().GetId();
    });

    int minLifespanStart = mutable_buffers_.at(mergedBufferIds.front()).lifespan_start;
    for (const auto& bufferId : mergedBufferIds) {
        const int lifespanStart = mutable_buffers_.at(bufferId).lifespan_start;
        if (lifespanStart < minLifespanStart) {
            minLifespanStart = lifespanStart;
        }
    }

    const auto& output = node->output(0);
    auto mergedTensorByteSize = GetTensorByteSize(output);
    auto parentTensor = std::make_shared<TensorID>(next_buffer_id_);
    next_buffer_id_ += 1;
    tensor_names_.emplace(GetTensorNameInternal(output), parentTensor);

    mutable_buffers_.emplace(std::make_pair(parentTensor->GetBuffer().GetId(),
                                            BufferDesc{minLifespanStart, node_idx, mergedTensorByteSize}));
    for (const auto& bufferId : mergedBufferIds) {
        mutable_buffers_.erase(bufferId);
    }

    size_t totalSize = 0;
    for (const auto& t : mergedTensors) {
        auto& tensor = tensor_names_.at(t.first);
        tensor->SetParent(parentTensor, totalSize);
        totalSize += mutable_tensor_sizes_.at(tensor->GetId());
    }
    mutable_tensor_sizes_[parentTensor->GetId()] = totalSize;
    OPENVINO_ASSERT(mergedTensorByteSize == totalSize);
}

void OperationBuffersExtractor::extractReshapeTensors(const NodePtr& node, int node_idx) {
    try {
        OPENVINO_ASSERT(node->inputs().size() >= 1);
        OPENVINO_ASSERT(node->outputs().size() == 1);
        const auto input = node->inputs().at(0);
        const auto& tensorId = tensor_names_.at(GetTensorNameInternal(input));
        const auto output = node->outputs().at(0);
        tensor_names_.emplace(GetTensorNameInternal(output), tensorId);
    } catch (std::out_of_range&) {
        throw_ov_exception(fmt::format("Failed to extract output buffer for reshape only node '{}'", node->get_name()));
    }
}

void OperationBuffersExtractor::extractMutableTensors(const NodePtr& node, int node_idx) {
    for (const auto& output : node->outputs()) {
        auto tensorByteSize = GetTensorByteSize(output);
        if (getenv("ROCM_TRACE_MEMORY") && tensorByteSize > 1024ULL * 1024 * 1024) {
            fprintf(stderr, "[MEM-BIG] id=%d node=%s op=%s shape=[", next_buffer_id_,
                    node->get_friendly_name().c_str(), node->get_type_info().name);
            for (size_t i = 0; i < output.get_shape().size(); i++)
                fprintf(stderr, "%s%zu", i?",":"", output.get_shape()[i]);
            fprintf(stderr, "] size=%.1fMB\n", tensorByteSize / 1048576.0);
        }
        mutable_tensor_sizes_[next_buffer_id_] = tensorByteSize;
        mutable_buffers_.emplace(std::make_pair(next_buffer_id_, BufferDesc{node_idx, node_idx, tensorByteSize}));
        tensor_names_.emplace(GetTensorNameInternal(output), std::make_shared<TensorID>(next_buffer_id_));
        next_buffer_id_++;
    }
}

void OperationBuffersExtractor::extractParameterTensors(const NodePtr& node, int node_idx) {
    if (node->inputs().size() > 0) {
        OPENVINO_ASSERT(node->get_output_size() > 0);
        auto input = node->inputs().front().get_source_output();
        const auto& tensorId = tensor_names_.at(GetTensorNameInternal(input));
        for (auto& output : node->outputs()) {
            tensor_names_.emplace(GetTensorNameInternal(output), tensorId);
        }
    } else {
        const int lastNodeIdx = is_stable_params_ ? num_ordered_nodes_ : node_idx;
        for (const auto& output : node->outputs()) {
            auto tensorByteSize = GetTensorByteSize(output);
            mutable_tensor_sizes_[next_buffer_id_] = tensorByteSize;
            mutable_buffers_.emplace(
                std::make_pair(next_buffer_id_, BufferDesc{node_idx, lastNodeIdx, tensorByteSize}));
            tensor_names_.emplace(GetTensorNameInternal(output), std::make_shared<TensorID>(next_buffer_id_));
            next_buffer_id_++;
        }
    }
}

void OperationBuffersExtractor::extractResultTensors(const NodePtr& node) {
    if (node->get_output_size() > 0) {
        auto input = node->inputs().front().get_source_output();
        const auto& tensorId = tensor_names_.at(GetTensorNameInternal(input));
        for (auto& output : node->outputs()) {
            tensor_names_.emplace(GetTensorNameInternal(output), tensorId);
        }
    }
    if (is_stable_results_) {
        auto input = node->inputs().front().get_source_output();
        const auto& tensorId = tensor_names_.at(GetTensorNameInternal(input));
        auto resultBuffer = std::find_if(mutable_buffers_.begin(), mutable_buffers_.end(), [&tensorId](const auto& mb) {
            return mb.first == tensorId->GetId();
        });
        if (resultBuffer == mutable_buffers_.end()) {
            throw_ov_exception(fmt::format("Cannot find mutable buffer for Result with name {}", node->get_name()));
        }
        resultBuffer->second.lifespan_end = num_ordered_nodes_;
    }
}

void OperationBuffersExtractor::extractImmutableTensors(const NodePtr& node) {
    auto constant = std::dynamic_pointer_cast<ov::op::v0::Constant>(node);
    const Byte* ptr = reinterpret_cast<const Byte*>(constant->get_data_ptr());
    auto span = gsl::make_span(ptr, GetTensorByteSize(node->output(0)));
    auto tensor = std::make_shared<TensorID>(next_buffer_id_);
    immutable_buffers_.emplace(std::make_pair(tensor->GetId(), span));
    tensor_names_.emplace(GetTensorNameInternal(node->output(0)), tensor);
    next_buffer_id_++;
}

WorkbufferIds OperationBuffersExtractor::processWorkbufferRequest(int node_idx, const WorkbufferRequest& request,
                                                                   int lifespan_extra) {
    WorkbufferIds result{};
    for (auto size : request.immutable_sizes) {
        immutable_workbuffers_.emplace(next_buffer_id_, size);
        result.immutableIds.push_back(next_buffer_id_);
        if (size > 1024*1024)
            fprintf(stderr, "[ImmWB-op] node=%d id=%zu size=%.1fMB\n", node_idx, (size_t)next_buffer_id_, size/1048576.0);
        next_buffer_id_++;
    }
    for (auto size : request.mutable_sizes) {
        // mutable workbuffers share the same memory space with mutable I/O buffers.
        // lifespan_extra > 0 extends the end to prevent early buffer reuse when
        // async GPU kernels still need the buffer after Execute() returns.
        mutable_buffers_.emplace(std::make_pair(next_buffer_id_,
            BufferDesc{node_idx, node_idx + lifespan_extra, size}));
        result.mutableIds.push_back(next_buffer_id_);
        next_buffer_id_++;
    }
    // Pinned host buffers: allocate contiguous slots in the per-request pinned pool.
    // We record byte offsets; the MemoryManager uses pinnedPool() + offset to produce void*.
    for (auto size : request.pinned_sizes) {
        // Align to pointer size for safety
        const size_t aligned = (size + sizeof(void*) - 1) & ~(sizeof(void*) - 1);
        result.pinnedOffsets.push_back(total_pinned_bytes_);
        total_pinned_bytes_ += aligned;
    }
    return result;
}

void OperationBuffersExtractor::initConstantMemory(DeviceMemBlock::Ptr memory_block) const {
    for (const auto& buffer_id : memory_block->bufferIds()) {
        auto span = immutableBuffer(buffer_id);
        void* device_ptr = memory_block->deviceBufferPtr(buffer_id);
        OPENVINO_ASSERT(device_ptr != nullptr);
        throwIfError(::hipMemcpy(device_ptr, span.data(), span.size_bytes(), hipMemcpyHostToDevice));
    }
}

MemoryModel::Ptr OperationBuffersExtractor::createConstantMemoryModel() const {
    ImmutableMemoryModelBuilder constants_block_builder;
    // Process nGraph and add allocations
    for (auto id : immutableBuffersIds()) {
        auto span = immutableBuffer(id);
        constants_block_builder.addAllocation(id, span.size());
    }
    return constants_block_builder.build();
}

MemoryModel::Ptr OperationBuffersExtractor::createMutableMemoryModel() const {
    MemoryModelBuilder mutable_model_builder;
    for (auto id : mutableBuffersIds()) {
        mutable_model_builder.addAllocation(
            id, mutableBufferLifespanStart(id), mutableBufferLifespanEnd(id), mutableBufferSize(id));
    }
    return mutable_model_builder.build();
}

MemoryModel::Ptr OperationBuffersExtractor::createImmutableMemoryModel() const {
    ImmutableMemoryModelBuilder immutable_workbuffer_model_builder;
    const auto& immutable_workbufer_sizes = immutableWorkbufferSizes();
    size_t total_immutable = 0;
    for (const auto& index : immutable_workbufer_sizes) {
        immutable_workbuffer_model_builder.addAllocation(index.first, index.second);
        total_immutable += index.second;
        if (index.second > 1024*1024)
            fprintf(stderr, "[ImmWB] id=%zu size=%.1fMB\n", (size_t)index.first, index.second/1048576.0);
    }
    if (total_immutable > 1024*1024*100)
        fprintf(stderr, "[ImmWB] TOTAL=%.1fMB (%zu entries)\n", total_immutable/1048576.0, immutable_workbufer_sizes.size());
    return immutable_workbuffer_model_builder.build();
}

bool OperationBuffersExtractor::IsParameterNode(const ov::Node& node) {
    return dynamic_cast<const ov::op::v0::Parameter*>(&node) != nullptr;
}

bool OperationBuffersExtractor::IsResultNode(const ov::Node& node) {
    return dynamic_cast<const ov::op::v0::Result*>(&node) != nullptr;
}

bool OperationBuffersExtractor::IsConstantNode(const ov::Node& node) {
    return dynamic_cast<const ov::op::v0::Constant*>(&node) != nullptr;
}

bool OperationBuffersExtractor::IsConcatOptimizedNode(const ov::Node& node) {
    return dynamic_cast<const nodes::ConcatOptimized*>(&node) != nullptr;
}

bool OperationBuffersExtractor::isReshapeOnlyNode(const ov::Node& node) {
    return ov::is_type<const ov::op::v1::Reshape>(&node) || ov::is_type<const ov::op::v0::Squeeze>(&node) ||
           ov::is_type<const ov::op::v0::Unsqueeze>(&node);
}

void OperationBuffersExtractor::ThrowBufferSizesAreNotMatchError(const ov::Input<ov::Node>& input) {
    throw_ov_exception(
        fmt::format("Buffer size of Input #{} of {} node and corresponding "
                    "output #{} of {} node are not equal.",
                    input.get_index(),
                    input.get_node()->get_name(),
                    input.get_source_output().get_index(),
                    input.get_source_output().get_node()->get_name()));
}

void OperationBuffersExtractor::ThrowGraphIsBadFormedError(const ov::Input<ov::Node>& input) {
    throw_ov_exception(
        fmt::format("Provided graph is bad formed. Input #{} of \"{}\" node "
                    "isn't connected to any output",
                    input.get_index(),
                    input.get_node()->get_name()));
}

// ── Attention extra input registration ───────────────────────────────────────
// For attention MatMul nodes tagged with "rocm_attn_qkv_name" rt_info,
// find the QKV tensor in tensor_names_ and register it as an extra input.
// This allows RocmAttentionMatMulOp::Execute() to access inputs.back() as QKV.
#include <openvino/op/reshape.hpp>
void OperationBuffersExtractor::registerAttentionExtraInputs(gsl::span<const NodePtr> ordered_nodes) {
    // Build a reverse map: node_name → node (so we can find the Reshape→QKV)
    std::unordered_map<std::string, NodePtr> name_to_node;
    for (const auto& n : ordered_nodes) {
        name_to_node[n->get_name()] = n;
        name_to_node[n->get_friendly_name()] = n;
    }

    for (const auto& node : ordered_nodes) {
        const auto& rt = node->get_rt_info();
        auto it = rt.find("rocm_attn_qkv_name");
        if (it == rt.end()) continue;
        // Debug: show which node has the tag
        std::cerr << "[AttnExtra] Found rocm_attn_qkv_name on: "
                  << node->get_type_name() << " friendly=" << node->get_friendly_name().substr(0,50) << "\n";

        const std::string qkv_name = it->second.as<std::string>();

        // Find the QKV tensor in tensor_names_
        // The QKV source is a Conv/other node whose output is named qkv_name.
        // We look for it by the output tensor name convention used in tensor_names_.
        // The convention: node_name + "_" + output_index (see GetTensorNameInternal).
        // The QKV tensor is output[0] of the node named qkv_name.
        const std::string tensor_name = qkv_name + "_0";  // output index 0
        auto tn_it = tensor_names_.find(tensor_name);
        if (tn_it == tensor_names_.end()) {
            // Try alternate: the qkv_name might already be the full tensor name
            tn_it = tensor_names_.find(qkv_name);
        }
        if (tn_it == tensor_names_.end()) {
            std::cerr << "[AttnExtra] WARNING: QKV tensor not found: " << qkv_name << "\n";
            continue;
        }

        const TensorID& qkv_id = *tn_it->second;
        // Register as extra input for this attention MatMul node
        extra_node_inputs_[node->get_friendly_name()].push_back(qkv_id);
        std::cerr << "[AttnExtra] Registered QKV tensor for: "
                  << node->get_friendly_name().substr(0, 40) << "\n";

        // pe(V) filter/bias injection: enabled by default (ROCM_FUSE_PE != "0").
        const char* fuse_pe_e = std::getenv("ROCM_FUSE_PE");
        if (fuse_pe_e && std::string(fuse_pe_e) == "0") continue;  // pe disabled
        const auto& rt2 = node->get_rt_info();
        auto pe_it2 = rt2.find("rocm_attn_pe_add");
        if (pe_it2 == rt2.end()) continue;  // Not an AV with pe

        auto pe_conv_name_it = rt2.find("rocm_attn_pe_conv_name");
        if (pe_conv_name_it == rt2.end()) continue;
        const std::string pe_conv_name = pe_conv_name_it->second.as<std::string>();

        // Find the FusedGroupConvolution node by name
        for (const auto& n2 : ordered_nodes) {
            if (n2->get_name() != pe_conv_name && n2->get_friendly_name() != pe_conv_name) continue;
            // FusedGroupConvolution: input[0]=data, input[1]=filter, input[2]=bias
            if (n2->get_input_size() < 3) break;
            const std::string filter_tname = GetTensorNameInternal(n2->inputs().at(1));
            const std::string bias_tname   = GetTensorNameInternal(n2->inputs().at(2));
            auto f_it = tensor_names_.find(filter_tname);
            auto b_it = tensor_names_.find(bias_tname);
            if (f_it != tensor_names_.end() && b_it != tensor_names_.end()) {
                extra_node_inputs_[node->get_friendly_name()].push_back(*f_it->second);  // filter
                extra_node_inputs_[node->get_friendly_name()].push_back(*b_it->second);  // bias
                // Inject FGC output buffer as an EXTRA OUTPUT of AV MatMul (not extra input).
                // This registers AV as the "producer" of fgc_out at AV_idx, so MemorySolver
                // allocates the buffer with lifespan_start = AV_idx (valid at AV Execute time).
                // pe_add writes pe+attn to this buffer → outputs[1] in AV Execute.
                const std::string fgc_out_name = GetTensorNameInternal(n2->outputs().at(0));
                auto fgc_out_it = tensor_names_.find(fgc_out_name);
                if (fgc_out_it != tensor_names_.end()) {
                    extra_node_outputs_[node->get_friendly_name()].push_back(*fgc_out_it->second);
                }
                std::cerr << "[AttnExtra] Registered pe filter+bias+fgcout for AV op: "
                          << node->get_friendly_name().substr(0, 40) << "\n";
            }
            break;
        }
    }
}

// ── Zero-copy VariadicSplit: buffer aliasing ───────────────────────────────────
// For VariadicSplitAlias nodes (inserted by VariadicSplitZeroCopyPass),
// we make each output[i] a child of the input buffer at the channel offset.
// This means:
//   output[0] buffer starts at: input_base + 0
//   output[1] buffer starts at: input_base + split_lens[0] * H * W * elem_bytes
//   output[i] buffer starts at: input_base + sum(split_lens[0..i-1]) * H * W * elem_bytes
//
// The memory planner then ensures the input buffer is large enough and all
// output "buffers" are simply offset views — no GPU copy needed.
// VariadicSplitAliasOp::Execute() is a no-op.
void OperationBuffersExtractor::extractSplitAliasTensors(const NodePtr& node, int node_idx) {
    // Safety check: if any downstream consumer of any output is a ConcatOptimized,
    // fall back to normal allocation (mergeConcatMutableTensors can't handle child tensors).
    for (size_t i = 0; i < node->get_output_size(); ++i) {
        for (const auto& tgt : node->output(i).get_target_inputs()) {
            if (IsConcatOptimizedNode(*tgt.get_node()->shared_from_this())) {
                extractMutableTensors(node, node_idx);
                return;
            }
        }
    }

    // Get input TensorID (must already be in tensor_names_)
    OPENVINO_ASSERT(node->get_input_size() >= 3);
    const auto& in_output = node->input(0).get_source_output();
    const auto& in_tensor_name = GetTensorNameInternal(in_output);
    auto it = tensor_names_.find(in_tensor_name);
    OPENVINO_ASSERT(it != tensor_names_.end(),
        "ZeroCopySplit: input tensor not found in memory map: ", in_tensor_name);
    const auto& input_tensor_id = it->second;

    // Additional safety: the input tensor must be a ROOT buffer (no parent)
    // Otherwise the offset calculation would be relative to an unknown base.
    if (&input_tensor_id->GetBuffer() != input_tensor_id.get()) {
        extractMutableTensors(node, node_idx);
        return;
    }

    // Get split lengths from the constant node (input 2)
    auto lens_const = std::dynamic_pointer_cast<ov::op::v0::Constant>(
        node->input(2).get_source_output().get_node_shared_ptr());
    OPENVINO_ASSERT(lens_const, "ZeroCopySplit: split_lengths not a constant");
    std::vector<int64_t> split_lens;
    { const auto& et=lens_const->get_element_type();
      if(et==ov::element::i64) split_lens=lens_const->cast_vector<int64_t>();
      else if(et==ov::element::i32) for(auto v:lens_const->cast_vector<int32_t>()) split_lens.push_back(v);
      else for(auto v:lens_const->cast_vector<float>()) split_lens.push_back(static_cast<int64_t>(v)); }

    // Get element size and spatial dimensions (H×W) for computing byte offset
    const auto& in_shape = node->get_input_shape(0);  // [N, C, H, W]
    OPENVINO_ASSERT(in_shape.size() == 4, "ZeroCopySplit: expected 4D NCHW input");
    const size_t elem_bytes = node->get_input_element_type(0).size();
    const size_t spatial = in_shape[2] * in_shape[3];  // H * W

    // Extend the input buffer's lifespan to cover all uses (node_idx is the split op)
    // The input buffer must live at least until all outputs are consumed.
    // lifespan_end is updated by the main loop for inputs, so no action needed here.

    // Create child TensorIDs for each output
    size_t channel_offset = 0;
    for (size_t i = 0; i < node->get_output_size(); ++i) {
        const size_t byte_offset = channel_offset * spatial * elem_bytes;
        const size_t out_bytes = static_cast<size_t>(split_lens[i]) * spatial * elem_bytes;

        // Create a new TensorID whose ID is unique but whose buffer is the INPUT's buffer
        // via SetParent. This makes it a virtual "child" tensor at the given offset.
        auto child_tensor = std::make_shared<TensorID>(next_buffer_id_++);
        child_tensor->SetParent(input_tensor_id, static_cast<unsigned>(byte_offset));

        // Register the child tensor size (for memory model, even though it's aliased)
        mutable_tensor_sizes_[child_tensor->GetId()] = out_bytes;

        // Register output tensor name
        const auto& out_name = GetTensorNameInternal(node->output(i));
        tensor_names_.emplace(out_name, child_tensor);

        // Note: we do NOT add this to mutable_buffers_ because it's not an independent
        // allocation — it's a sub-range of the input buffer. The memory model builder
        // will only see the input buffer, not the alias children.

        channel_offset += static_cast<size_t>(split_lens[i]);
    }
}

}  // namespace rocm_gpu
}  // namespace ov
