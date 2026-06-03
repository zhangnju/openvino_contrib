// Copyright (C) 2021-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "runtime.hpp"

namespace rocm {

class Event : public Handle<hipEvent_t> {
public:
    Event() : Handle((static_cast<__host__ hipError_t (*)(hipEvent_t* event)>(hipEventCreate)), hipEventDestroy) {}
    auto&& record(const hipStream_t& stream) {
        throwIfError(hipEventRecord(get(), stream));
        return std::move(*this);
    }
    void synchronize() { throwIfError(hipEventSynchronize(get())); }
    float elapsedSince(const Event& start) const { return createFirstArg(hipEventElapsedTime, start.get(), get()); }
};

}  // namespace rocm
