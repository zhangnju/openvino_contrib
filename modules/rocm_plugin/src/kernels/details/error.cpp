// Copyright (C) 2021-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <fmt/format.h>

#include <error.hpp>
#include <exception>

namespace ov {
namespace rocm_gpu {
namespace kernel {

class Error : public std::exception {
public:
    Error(const std::string& msg, const std::experimental::source_location& location)
        : msg_{fmt::format("{}:{}({}): {}", location.file_name(), location.line(), location.function_name(), msg)} {}

    [[nodiscard]] const char* what() const noexcept override { return msg_.c_str(); }

private:
    std::string msg_;
};

[[gnu::cold]] void throwIfError(hipError_t err, const std::experimental::source_location& location) {
    if (err != hipSuccess) {
        throw Error(hipGetErrorString(err), location);
    }
}

[[gnu::cold, noreturn]] void throw_ov_exception(const std::string& msg,
                                              const std::experimental::source_location& location) {
    throw Error(msg, location);
}

void assertThrow(bool condition, const std::string& msg, const std::experimental::source_location& location) {
    if (!condition) {
        throw Error(msg, location);
    }
}

}  // namespace kernel
}  // namespace rocm_gpu
}  // namespace ov
