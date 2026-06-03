// Copyright (C) 2021-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <algorithm>
#include <cmath>
#include <rocm/runtime.hpp>
#include <map>
#include <vector>

#include "details/rocm_type_traits.hpp"
#include "details/tensor_helpers.hpp"

namespace ov {
namespace rocm_gpu {
namespace kernel {

template <typename TDataType>
struct __align__(1) DetectionOutputResult {
    TDataType img;
    TDataType cls;
    TDataType conf;
    TDataType xmin;
    TDataType ymin;
    TDataType xmax;
    TDataType ymax;
};

class DetectionOutput {
public:
    using Buffers = std::vector<rocm::DevicePointer<void*>>;

    struct Attrs {
        enum CodeType {
            Caffe_PriorBoxParameter_CORNER = 0,
            Caffe_PriorBoxParameter_CENTER_SIZE,
        };

        size_t num_images;
        size_t prior_size;
        size_t num_priors;
        size_t num_loc_classes;
        size_t offset;
        size_t num_results;
        size_t out_total_size;

        size_t num_classes;
        int background_label_id = 0;
        int top_k = -1;
        bool variance_encoded_in_target = false;
        int keep_top_k{};
        CodeType code_type = Caffe_PriorBoxParameter_CORNER;
        bool share_location = true;
        float nms_threshold;
        float confidence_threshold = 0;
        bool clip_after_nms = false;
        bool clip_before_nms = false;
        bool decrease_label_id = false;
        bool normalized = false;
        size_t input_height = 1;
        size_t input_width = 1;
        float objectness_score = 0;
    };

    enum {
        kLocationsWBIdx = 0,
        kConfPredsWBIdx,
        kPriorBboxesWBIdx,
        kPriorVariancesWBIdx,
        kDecodeBboxesWBIdx,
        kTempDecodeBboxesWBIdx,
        kTempScorePerClassPrioIdxs0WBIdx,
        kTempScorePerClassPrioIdxs1WBIdx,
        kPrioBoxIdxsByClassWBIdx,
        kNumDetectionsWBIdx,
        kNumRequiredWB,
        kArmLocationsWBIdx = kNumRequiredWB,
        kNumOptionalWB,
    };

    DetectionOutput(Type_t element_type,
                    size_t max_threads_per_block,
                    size_t location_size,
                    size_t confidence_size,
                    size_t priors_size,
                    size_t arm_confidence_size,
                    size_t arm_location_size,
                    size_t result_size,
                    Attrs attrs);

    void operator()(const rocm::Stream& stream,
                    rocm::DevicePointer<const void*> location,
                    rocm::DevicePointer<const void*> confidence,
                    rocm::DevicePointer<const void*> priors,
                    const void* armLocation,
                    const void* armConfidence,
                    std::vector<rocm::DevicePointer<void*>> mutableWorkbuffers,
                    rocm::DevicePointer<void*> result) const;
    std::vector<size_t> getMutableWorkbufferSizes() const;
    std::vector<size_t> getImmutableWorkbufferSizes() const;
    void initSharedImmutableWorkbuffers(const Buffers& buffers);

protected:
    template <typename TDataType>
    std::vector<size_t> getMutableWorkbufferSizes() const;
    template <typename TDataType>
    void call(const rocm::Stream& stream,
              rocm::DevicePointer<const void*> location,
              rocm::DevicePointer<const void*> confidence,
              rocm::DevicePointer<const void*> priors,
              const void* armConfidence,
              const void* armLocation,
              std::vector<rocm::DevicePointer<void*>> mutableWorkbuffers,
              rocm::DevicePointer<void*> result) const;

private:
    Type_t element_type_{};
    Attrs attrs_;
    Attrs* dattrs_ptr_ = nullptr;

    size_t max_threads_per_block_;

    size_t location_size_;
    size_t confidence_size_;
    size_t priors_size_;
    size_t arm_confidence_size_;
    size_t arm_location_size_;
    size_t result_size_;
};

}  // namespace kernel
}  // namespace rocm_gpu
}  // namespace ov
