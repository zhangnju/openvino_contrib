#pragma once
#include <openvino/pass/graph_rewrite.hpp>

namespace ov {
namespace rocm_gpu {
namespace pass {

class FusedRTDFusionPass : public ov::pass::ModelPass {
public:
    OPENVINO_RTTI("FusedRTDFusionPass", "0");
    FusedRTDFusionPass(const std::string& arch, int num_cu)
        : arch_(arch), num_cu_(num_cu) {}
    bool run_on_model(const std::shared_ptr<ov::Model>& model) override;
private:
    std::string arch_;
    int num_cu_;
};

}  // namespace pass
}  // namespace rocm_gpu
}  // namespace ov
