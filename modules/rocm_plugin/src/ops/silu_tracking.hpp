// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// Thread-local tracking of output buffers where SiLU has already been applied
// by a FusedConvolutionRocMLIR kernel. SwishOp checks this to skip redundant work.
//
// Usage:
//   After launching a Conv+Bias+SiLU kernel:
//     mark_silu_applied(output_ptr)
//
//   In SwishOp::Execute(), before running SwishOpImpl:
//     if (is_silu_applied(input_ptr)) { clear_silu_mark(input_ptr); return; }

#pragma once
#include <unordered_set>

namespace ov {
namespace rocm_gpu {

// Thread-local because each inference thread has its own execution context.
// Cleared after SwishOp consumes the mark (or at end of inference).

inline thread_local std::unordered_set<const void*> g_silu_applied_buffers;

inline void mark_silu_applied(const void* output_ptr) {
    g_silu_applied_buffers.insert(output_ptr);
}

inline bool is_silu_applied(const void* input_ptr) {
    return g_silu_applied_buffers.count(input_ptr) > 0;
}

inline void clear_silu_mark(const void* ptr) {
    g_silu_applied_buffers.erase(ptr);
}

}  // namespace rocm_gpu
}  // namespace ov
