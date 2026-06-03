// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "rocm_infer_request.hpp"
#include "openvino/runtime/iasync_infer_request.hpp"
#include "openvino/runtime/iinfer_request.hpp"

namespace ov {
namespace rocm_gpu {

class rocmAsyncInferRequest : public ov::IAsyncInferRequest {
public:
    rocmAsyncInferRequest(const rocmInferRequest::Ptr& request,
                          const std::shared_ptr<ov::threading::ITaskExecutor>& task_executor,
                          const std::shared_ptr<ov::threading::ITaskExecutor>& wait_executor,
                          const std::shared_ptr<ov::threading::ITaskExecutor>& callback_executor);

    ~rocmAsyncInferRequest();
    void cancel() override;
    void infer_thread_unsafe() override;

private:
    rocmInferRequest::Ptr request_;
};

}  // namespace rocm_gpu
}  // namespace ov
