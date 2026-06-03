// Copyright (C) 2021-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "rocm/blas.hpp"
#include "rocm/dnn.hpp"
#include "rocm/tensor.hpp"

namespace ov {
namespace rocm_gpu {

class ThreadContext {
    rocm::Device device_;
    rocm::Stream stream_;
    rocm::DnnHandle dnnHandle_;
    rocm::rocblasHandle rocBlasHandle_;
    //rocm::HipTensorHandle hipTensorHandle_;

public:
    explicit ThreadContext(rocm::Device d) : device_{d.setCurrent()} {
        dnnHandle_.setStream(stream_);
        rocBlasHandle_.setStream(stream_);
    }
    rocm::Device device() const { return device_; }
    const rocm::Stream& stream() const noexcept { return stream_; }
    const rocm::DnnHandle& dnnHandle() const noexcept { return dnnHandle_; }
    const rocm::rocblasHandle& rocBlasHandle() const noexcept { return rocBlasHandle_; }
    //const rocm::HipTensorHandle& hipTensorHandle() const noexcept { return hipTensorHandle_; }
};

}  // namespace rocm_gpu
}  // namespace ov
