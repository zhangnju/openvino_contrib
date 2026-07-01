#pragma once
#include <openvino/pass/pass.hpp>
#include <string>

namespace ov { namespace rocm_gpu { namespace pass {

class FlashAttentionTritonFusionPass : public ov::pass::ModelPass {
public:
    OPENVINO_RTTI("FlashAttentionTritonFusionPass", "0");
    explicit FlashAttentionTritonFusionPass(const std::string& arch = "") : arch_(arch) {}
    bool run_on_model(const std::shared_ptr<ov::Model>& model) override;
private:
    std::string arch_;
};

}}} // namespace
