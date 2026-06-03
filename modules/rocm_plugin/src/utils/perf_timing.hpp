// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <optional>

#include "rocm/event.hpp"

namespace ov::rocm_gpu::utils {
/**
 * @brief class PerformaceTiming measures time between two events
 * and accumulates results from sequential start/stop calls
 */
class PerformaceTiming {
public:
    PerformaceTiming() = default;
    PerformaceTiming(const rocm::Stream& stream)
        : start_{rocm::Event{}} {
        start_->record(stream.get());
    }
    void setStart(const rocm::Stream& stream) {
        start_.emplace(rocm::Event{}.record(stream.get()));
    }
    void setStop(const rocm::Stream& stream) {
        stop_.emplace(rocm::Event{}.record(stream.get()));
    }
    float measure() {
        if (start_.has_value() && stop_.has_value()) {
            auto elapsed = stop_->elapsedSince(*start_);
            if (elapsed != std::numeric_limits<float>::quiet_NaN()) {
                duration_ += stop_->elapsedSince(*start_);
            }
        }
        clear();
        return duration_;
    }
    float duration() const noexcept { return duration_; }
    void clear() {
        start_.reset();
        stop_.reset();
    }

private:
    std::optional<rocm::Event> start_{};
    std::optional<rocm::Event> stop_{};
    float duration_{};
};
}  // namespace ov::rocm_gpu::utils
