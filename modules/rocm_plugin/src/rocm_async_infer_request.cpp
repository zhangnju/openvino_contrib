// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "rocm_async_infer_request.hpp"
#include "rocm_itt.hpp"
#include "rocm_thread_pool.hpp"

namespace ov {
namespace rocm_gpu {

rocmAsyncInferRequest::rocmAsyncInferRequest(const rocmInferRequest::Ptr& request,
                                             const std::shared_ptr<ov::threading::ITaskExecutor>& task_executor,
                                             const std::shared_ptr<ov::threading::ITaskExecutor>& wait_executor,
                                             const std::shared_ptr<ov::threading::ITaskExecutor>& callback_executor)
    : ov::IAsyncInferRequest(request, task_executor, callback_executor),
      request_(request) {
    // In current implementation we have CPU only tasks and no needs in 2 executors
    // So, by default single stage pipeline is created.
    // This stage executes InferRequest::Infer() using cpuTaskExecutor.
    // But if remote asynchronous device is used the pipeline can by splitted tasks that are executed by cpuTaskExecutor
    // and waiting tasks. Waiting tasks can lock execution thread so they use separate threads from other executor.
    constexpr const auto remoteDevice = true;

    auto rocm_thread_pool = std::dynamic_pointer_cast<rocmThreadPool>(wait_executor);
    if (remoteDevice) {
        m_pipeline = {{task_executor,
                      [this] {
                          OV_ITT_SCOPED_TASK(itt::domains::rocm_gpu, "rocmAsyncInferRequest::infer_preprocess");
                          request_->infer_preprocess();
                      }},
                     {wait_executor,
                      [this, rocm_thread_pool] {
                          auto& threadContext = rocm_thread_pool->get_thread_context();
                          {
                              OV_ITT_SCOPED_TASK(itt::domains::rocm_gpu, "rocmAsyncInferRequest::start_pipeline");
                              request_->start_pipeline(threadContext);
                          }
                          {
                              OV_ITT_SCOPED_TASK(itt::domains::rocm_gpu, "rocmAsyncInferRequest::wait_pipeline");
                              request_->wait_pipeline(threadContext);
                          }
                      }},
                     {task_executor, [this] {
                          OV_ITT_SCOPED_TASK(itt::domains::rocm_gpu, "rocmAsyncInferRequest::infer_postprocess");
                          request_->infer_postprocess();
                      }}};
    }
}

rocmAsyncInferRequest::~rocmAsyncInferRequest() {
    ov::IAsyncInferRequest::stop_and_wait();
}

void rocmAsyncInferRequest::cancel() {
    ov::IAsyncInferRequest::cancel();
    request_->cancel();
}

void rocmAsyncInferRequest::infer_thread_unsafe() {
    start_async_thread_unsafe();
}
}  // namespace rocm_gpu
}  // namespace ov
