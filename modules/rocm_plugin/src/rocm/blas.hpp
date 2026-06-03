// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <rocblas/rocblas.h>

#include "runtime.hpp"

inline std::string rocblasGetErrorString(rocblas_status status) {
    switch (status) {
        case rocblas_status_success:
            return "rocblas Status Success";
        case rocblas_status_invalid_handle:
            return "rocblas Status Not Initialized";
        case rocblas_status_not_implemented:
            return "rocblas Status Not Implemented";
        case rocblas_status_invalid_pointer:
            return "rocblas Status Invalid Pointer";
        case rocblas_status_invalid_size:
            return "rocblas Status Invalid Size";
        case rocblas_status_memory_error:
            return "rocblas Status Memory Error";
        case rocblas_status_internal_error:
            return "rocblas Status Internal Error";
        case rocblas_status_perf_degraded:
            return "rocblas Status Perf Degraded";
        case rocblas_status_size_query_mismatch:
            return "rocblas Status Unmatched start/stop size query";
        case rocblas_status_size_increased:
            return "rocblas Status Queried device memory size increased";
        case rocblas_status_size_unchanged:
            return "rocblas Status Queried device memory size unchanged";
        case rocblas_status_invalid_value:
            return "rocblas Status Queried Passed argument not valid";
        case rocblas_status_continue:
            return "rocnlas status Nothing preventing function to proceed";
        case rocblas_status_check_numerics_fail:
            return "rocblas Status vector/matrix has a NaN/Infinity/denormal value";
        case rocblas_status_excluded_from_build:
            return "rocblas Status Function is not available in build, likely a function requiring Tensile built without Tensile";
        case rocblas_status_arch_mismatch:
            return "rocblas Status Architecture Mismatched";
        default:
            return "rocblas Unknown Status";
    }
}

inline void throwIfError(
    rocblas_status_ err,
    const std::experimental::source_location& location = std::experimental::source_location::current()) {
    if (err != rocblas_status_success) ov::rocm_gpu::throw_ov_exception(rocblasGetErrorString(err), location);
}

inline void logIfError(
    rocblas_status_ err,
    const std::experimental::source_location& location = std::experimental::source_location::current()) {
    if (err != rocblas_status_success) ov::rocm_gpu::logError(rocblasGetErrorString(err), location);
}

namespace rocm {

class rocblasHandle : public Handle<rocblas_handle> {
public:
    rocblasHandle() : Handle((rocblas_create_handle), rocblas_destroy_handle) {}
    void setStream(Stream& stream) { throwIfError(rocblas_set_stream(get(), stream.get())); }
};

}  // namespace rocm
