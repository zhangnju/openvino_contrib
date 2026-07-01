// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "convolution_components/convolution_miopen_components.hpp"
#include "rocm_operation_base.hpp"
#include <mutex>

namespace ov {
namespace rocm_gpu {

// MIOpen's solver cache is not thread-safe during solver lookup (LazyFindAlgo).
// Once each op's solver is found and cached, Forward calls are thread-safe.
// We always lock — the lock contention is minimal for cached ops since
// LazyFindAlgo returns immediately on subsequent calls.
inline std::mutex& getMiopenMutex() {
    static std::mutex mu;
    return mu;
}

// LazyFindAlgo has its own mutex (getMiopenFindMutex in convolution_miopen_components).
// The Execute-level lock is only needed during the first few calls when LazyFindAlgo
// hasn't completed for all handles. After that, MIOpen Forward is thread-safe.
// Use a counter: lock for first 200 calls, then skip (all algos cached by then).
struct MiopenLockGuard {
    bool locked_;
    MiopenLockGuard() {
        static std::atomic<int> call_count{0};
        int n = call_count.fetch_add(1, std::memory_order_relaxed);
        locked_ = (n < 200);
        if (locked_) getMiopenMutex().lock();
    }
    ~MiopenLockGuard() {
        if (locked_) getMiopenMutex().unlock();
    }
};


/**
 * @brief Implements `ov::op::v1::Convolution` using miopen API
 * which doesn't support asymmetric padding.
 */
class Convolutionmiopen : public OperationMIOPEN {
public:
    Convolutionmiopen(const CreationContext& context,
                     const ov::Node& node,
                     IndexCollection&& inputIds,
                     IndexCollection&& outputIds,
                     const Convolution::Details::ConvolutionParams& params);

    void Execute(const InferenceRequestContext& context,
                 Inputs inputTensors,
                 Outputs outputTensors,
                 const Workbuffers&) const override;

    WorkbufferRequest GetWorkBufferRequest() const override;

    rocmGraphCompatibility GetrocmGraphCompatibility() const override;

private:
    Convolution::Details::ConvolutionDescriptorsmiopen descs_;
};

}  // namespace rocm_gpu
}  // namespace ov
