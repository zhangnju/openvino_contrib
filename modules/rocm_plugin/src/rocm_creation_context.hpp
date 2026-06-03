// Copyright (C) 2021-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <rocm_config.hpp>

#include "rocm/blas.hpp"
#include "rocm/dnn.hpp"
#include "rocm/tensor.hpp"

namespace ov {
namespace rocm_gpu {

class CreationContext {
    rocm::Device device_;
    rocm::DnnHandle dnn_handle_;
    bool op_bench_option_;

public:
    explicit CreationContext(rocm::Device d, bool opBenchOption)
        : device_{d.setCurrent()}, op_bench_option_{opBenchOption} {}
    rocm::Device device() const { return device_; }
    const rocm::DnnHandle& dnnHandle() const { return dnn_handle_; }
    bool opBenchOption() const noexcept { return op_bench_option_; }
};

}  // namespace rocm_gpu
}  // namespace ov
