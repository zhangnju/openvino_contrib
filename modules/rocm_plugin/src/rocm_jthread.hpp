// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <thread>

namespace ov {
namespace rocm_gpu {

class rocmJThread final {
public:
    template <typename Function, typename... Args>
    explicit rocmJThread(Function&& f, Args&&... args) {
        thread_ = std::thread(std::forward<Function>(f), std::forward<Args>(args)...);
    }
    rocmJThread(rocmJThread&&) noexcept = default;
    rocmJThread& operator=(rocmJThread&&) noexcept = default;
    rocmJThread(const rocmJThread&) noexcept = delete;
    rocmJThread& operator=(const rocmJThread&) noexcept = delete;

    ~rocmJThread() {
        if (thread_.joinable()) {
            thread_.join();
        }
    }

private:
    std::thread thread_;
};

}  // namespace rocm_gpu
}  // namespace ov
