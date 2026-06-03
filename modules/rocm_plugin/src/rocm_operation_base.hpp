// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <rocm/device_pointers.hpp>
#include <rocm_creation_context.hpp>
#include <rocm_inference_request_context.hpp>
#include <memory>
#include <memory_manager/model/rocm_memory_model.hpp>
#include <string>
#include <string_view>
#include <vector>

#include "rocm_inference_request_context.hpp"
#include "memory_manager/rocm_workbuffers.hpp"
using namespace std;
namespace ov {

class Node;

}

namespace ov {
namespace rocm_gpu {

template <typename T>
using DevicePointer = rocm::DevicePointer<T>;

enum class rocmGraphCompatibility { NONE, FULL, SPECIAL };

class IOperationExec {
public:
    using Inputs = gsl::span<const rocm::DevicePointer<const void*>>;
    using Outputs = gsl::span<const rocm::DevicePointer<void*>>;
    using Buffers = std::vector<rocm::DevicePointer<void*>>;
    enum class WorkbufferStatus { NoInitNeeded, InitNeeded };

    virtual ~IOperationExec() = default;
    virtual void Execute(const InferenceRequestContext& context,
                         Inputs inputTensors,
                         Outputs outputTensors,
                         const Workbuffers& workbuffers) const = 0;

    virtual rocmGraphCompatibility GetrocmGraphCompatibility() const = 0;

    virtual void Capture(InferenceRequestContext& context,
                         Inputs inputTensors,
                         Outputs outputTensors,
                         const Workbuffers& workbuffers) const = 0;
    virtual void ExecuteGraph(InferenceRequestContext& context,
                              Inputs inputTensors,
                              Outputs outputTensors,
                              const Workbuffers& workbuffers) const = 0;
    virtual void InitSharedImmutableWorkbuffers(const Buffers&) = 0;
    virtual WorkbufferRequest GetWorkBufferRequest() const = 0;
    virtual const WorkbufferIds& GetWorkbufferIds() const = 0;
    virtual WorkbufferStatus SetWorkbufferIds(WorkbufferIds&& workbufferIds) = 0;
};

class IOperationMeta {
public:
    struct Category {
        static constexpr std::string_view rocm{"rocm"};
        static constexpr std::string_view rocBLAS{"rocBLAS"};
        static constexpr std::string_view MIOPEN{"MIOPEN"};
        //static constexpr std::string_view cuTENSOR{"cuTENSOR"};
    };

    virtual ~IOperationMeta() = default;
    virtual const std::string_view& GetCategory() const = 0;
    virtual const std::string& GetName() const = 0;
    virtual const std::string& GetTypeName() const = 0;
    virtual const ov::element::Type& GetRuntimePrecision() const = 0;
    virtual gsl::span<const TensorID> GetInputIds() const = 0;
    virtual gsl::span<const TensorID> GetOutputIds() const = 0;
};

class OperationBase : public IOperationExec, public IOperationMeta, public std::enable_shared_from_this<OperationBase> {
public:
    using Ptr = std::shared_ptr<OperationBase>;
    using WeakPtr = std::weak_ptr<OperationBase>;
    using IndexCollection = std::vector<TensorID>;
    OperationBase(const CreationContext& context,
                  const ov::Node& node,
                  IndexCollection&& inputIds,
                  IndexCollection&& outputIds);

    rocmGraphCompatibility GetrocmGraphCompatibility() const override { return rocmGraphCompatibility::NONE; }

    void Capture(InferenceRequestContext& context,
                 Inputs inputTensors,
                 Outputs outputTensors,
                 const Workbuffers& workbuffers) const override {
        Execute(context, inputTensors, outputTensors, workbuffers);
    }
    // For operations with rocmGraphCompatibility::SPECIAL, e.g. TI; the vast majority or operations doesn't use this
    void ExecuteGraph(InferenceRequestContext& context,
                      Inputs inputTensors,
                      Outputs outputTensors,
                      const Workbuffers& workbuffers) const override {}

    WorkbufferRequest GetWorkBufferRequest() const override {
        return {};  // Most operators do not need workbuffers
    }
    void InitSharedImmutableWorkbuffers(const Buffers&) override {}

protected:
    OperationBase(const CreationContext& context,
                  const std::shared_ptr<ov::Node>& node,
                  IndexCollection&& inputIds,
                  IndexCollection&& outputIds)
        : OperationBase(context, *node, std::move(inputIds), std::move(outputIds)) {}
    OperationBase(const CreationContext& context, std::nullptr_t)
        : node_name_{}, type_name_{}, input_ids_{}, output_ids_{} {}

public:
    const std::string_view& GetCategory() const override { return Category::rocm; }
    const std::string& GetName() const override { return node_name_; }
    const std::string& GetTypeName() const override { return type_name_; }
    const ov::element::Type& GetRuntimePrecision() const override { return runtime_precision_; }
    gsl::span<const TensorID> GetInputIds() const override { return input_ids_; }
    gsl::span<const TensorID> GetOutputIds() const override { return output_ids_; }
    const WorkbufferIds& GetWorkbufferIds() const override { return workbuffer_ids_; }
    WorkbufferStatus SetWorkbufferIds(WorkbufferIds&& workbufferIds) override {
        workbuffer_ids_ = workbufferIds;
        return workbuffer_ids_.immutableIds.empty() ? WorkbufferStatus::NoInitNeeded : WorkbufferStatus::InitNeeded;
    }

protected:
    std::string node_name_;
    std::string type_name_;
    ov::element::Type runtime_precision_ = ov::element::dynamic;
    const IndexCollection input_ids_;
    const IndexCollection output_ids_;
    WorkbufferIds workbuffer_ids_;
};

template <decltype(&IOperationMeta::Category::rocm) CategoryString>
class CategorizedOperationBase : public OperationBase {
protected:
    using OperationBase::OperationBase;

public:
    const std::string_view& GetCategory() const override { return *CategoryString; }
};

using OperationMIOPEN = CategorizedOperationBase<&IOperationMeta::Category::MIOPEN>;
using OperationRocBlas = CategorizedOperationBase<&IOperationMeta::Category::rocBLAS>;
//using OperationCuTensor = CategorizedOperationBase<&IOperationMeta::Category::cuTENSOR>;

/**
 * @brief Downcasts a shared node pointer to a ConcreteOperator reference
 */
template <class ConcreteOperator>
ConcreteOperator& downcast(const std::shared_ptr<ov::Node>& node) {
    return dynamic_cast<ConcreteOperator&>(*node.get());
}
}  // namespace rocm_gpu
}  // namespace ov
