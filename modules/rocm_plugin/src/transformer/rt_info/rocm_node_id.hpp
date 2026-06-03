// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/core/node.hpp"
#include "openvino/core/runtime_attribute.hpp"

namespace ov {
namespace rocm_gpu {
namespace rt_info {

void set_node_id(const std::shared_ptr<Node>& node, uint64_t id);

void remove_node_id(const std::shared_ptr<Node>& node);

uint64_t get_node_id(const std::shared_ptr<Node>& node);

/**
 * @ingroup ie_runtime_attr_api
 * @brief rocmNodeId class represents runtime info attribute that marks operation
 * with order id
 */
class rocmNodeId : public RuntimeAttribute {
public:
    OPENVINO_RTTI("rocm_node_id", "0");

    rocmNodeId() = default;

    bool is_copyable() const override { return false; }
};
}  // namespace rt_info
}  // namespace rocm_gpu
}  // namespace ov
