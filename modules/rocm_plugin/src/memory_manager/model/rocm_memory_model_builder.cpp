// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "rocm_memory_model_builder.hpp"
#include "openvino/core/except.hpp"
#include "memory_manager/model/details/rocm_memory_utils.hpp"

namespace ov {
namespace rocm_gpu {

void MemoryModelBuilder::addAllocation(BufferID id, int producerIndex, int lastConsumerIndex, size_t bsize) {
    OPENVINO_ASSERT(bsize > 0, "Allocation size is zero!");  // Verify that allocation size isn't zero.
    auto res = offsets_.emplace(id, 0);
    OPENVINO_ASSERT(res.second, "ID is not unique!");  // Verify that "id" is unique.
    const int64_t aligned_size = static_cast<int64_t>(applyAllignment(bsize));
    boxes_.emplace_back(ov::MemorySolver::Box{producerIndex, lastConsumerIndex, aligned_size, id});
}

MemoryModel::Ptr MemoryModelBuilder::build() {
    ov::MemorySolver solver{boxes_};
    const size_t blob_size = solver.solve();
    for (auto& pair : offsets_) pair.second = solver.get_offset(pair.first);

    if (getenv("ROCM_TRACE_MEMORY")) {
        // Sort by size descending to show biggest tensors
        auto sorted = boxes_;
        std::sort(sorted.begin(), sorted.end(), [](const auto& a, const auto& b) { return a.size > b.size; });
        fprintf(stderr, "[MEM] MemorySolver: %zu tensors, blob=%.1fMB\n", boxes_.size(), blob_size / 1048576.0);
        fprintf(stderr, "[MEM] Top 20 largest tensors:\n");
        for (int i = 0; i < std::min((int)sorted.size(), 20); i++) {
            fprintf(stderr, "[MEM]   id=%lld start=%d end=%d size=%.1fMB span=%d\n",
                    (long long)sorted[i].id, sorted[i].start, sorted[i].finish,
                    sorted[i].size / 1048576.0, sorted[i].finish - sorted[i].start);
        }
        // Count how many tensors are alive at each step
        int max_step = 0;
        for (auto& b : boxes_) if (b.finish > max_step) max_step = b.finish;
        int64_t peak_alive = 0;
        int peak_step = 0;
        for (int s = 0; s <= max_step; s++) {
            int64_t alive = 0;
            for (auto& b : boxes_) if (b.start <= s && s <= b.finish) alive += b.size;
            if (alive > peak_alive) { peak_alive = alive; peak_step = s; }
        }
        fprintf(stderr, "[MEM] Peak alive: %.1fMB at step %d/%d\n",
                peak_alive / 1048576.0, peak_step, max_step);
    }

    return std::make_shared<MemoryModel>(blob_size, offsets_);
}

}  // namespace rocm_gpu
}  // namespace ov
