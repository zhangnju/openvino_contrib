#pragma once
#include <openvino/pass/graph_rewrite.hpp>

namespace ov {
namespace rocm_gpu {
namespace pass {

class SwinFlashAttentionFusionPass : public ov::pass::ModelPass {
public:
    OPENVINO_RTTI("SwinFlashAttentionFusionPass", "0");
    bool run_on_model(const std::shared_ptr<ov::Model>& model) override;
};

}  // namespace pass
}  // namespace rocm_gpu
}  // namespace ov
