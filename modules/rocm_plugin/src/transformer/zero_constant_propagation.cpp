#include "zero_constant_propagation.hpp"

#include <cstdio>
#include <cstdlib>
#include <openvino/op/constant.hpp>
#include <openvino/op/multiply.hpp>
#include <openvino/op/add.hpp>
#include <openvino/core/graph_util.hpp>

namespace ov {
namespace rocm_gpu {
namespace pass {

namespace {
static bool g_trace = false;

static bool is_zero_constant(const std::shared_ptr<ov::Node>& node) {
    auto constant = std::dynamic_pointer_cast<ov::op::v0::Constant>(node);
    if (!constant) return false;
    auto et = constant->get_element_type();
    if (et != ov::element::f32 && et != ov::element::f16 && et != ov::element::i32)
        return false;

    auto data = constant->get_data_ptr<char>();
    size_t nbytes = ov::shape_size(constant->get_shape()) * et.size();
    if (nbytes == 0) return false;

    for (size_t i = 0; i < nbytes; ++i) {
        if (data[i] != 0) return false;
    }
    return true;
}

static std::shared_ptr<ov::op::v0::Constant> make_zero(ov::element::Type et, const ov::Shape& shape) {
    return ov::op::v0::Constant::create(et, shape, {0});
}

} // namespace

bool ZeroConstantPropagation::run_on_model(const std::shared_ptr<ov::Model>& model) {
    if (std::getenv("ROCM_DISABLE_ZERO_PROP")) return false;
    g_trace = std::getenv("ROCM_TRACE_ZERO_PROP") != nullptr;

    bool changed = false;
    int mul_zero = 0, add_zero = 0;
    int iterations = 0;
    const int max_iterations = 10;

    // Iterate: one simplification may expose more (e.g., mul→0 feeds an add)
    bool any_change = true;
    while (any_change && iterations < max_iterations) {
        any_change = false;
        ++iterations;

        for (const auto& node : model->get_ordered_ops()) {
            // Rule 1: Multiply(x, 0) → 0 or Multiply(0, x) → 0
            if (auto mul = std::dynamic_pointer_cast<ov::op::v1::Multiply>(node)) {
                auto in0 = mul->get_input_node_shared_ptr(0);
                auto in1 = mul->get_input_node_shared_ptr(1);
                bool z0 = is_zero_constant(in0);
                bool z1 = is_zero_constant(in1);
                if (z0 || z1) {
                    auto out_shape = mul->get_output_shape(0);
                    auto out_et = mul->get_output_element_type(0);
                    auto zero = make_zero(out_et, out_shape);
                    ov::replace_node(mul, zero);
                    ++mul_zero;
                    any_change = true;
                    changed = true;
                    if (g_trace)
                        fprintf(stderr, "[ZeroProp] mul*0→0: %s shape=%zu\n",
                                mul->get_friendly_name().c_str(), ov::shape_size(out_shape));
                    continue;
                }
            }

            // Rule 2: Add(x, 0) → x or Add(0, x) → x
            if (auto add = std::dynamic_pointer_cast<ov::op::v1::Add>(node)) {
                auto in0 = add->get_input_node_shared_ptr(0);
                auto in1 = add->get_input_node_shared_ptr(1);
                bool z0 = is_zero_constant(in0);
                bool z1 = is_zero_constant(in1);

                if (z0 || z1) {
                    auto non_zero = z0 ? add->input_value(1) : add->input_value(0);
                    auto zero_shape = z0 ? in0->get_shape() : in1->get_shape();
                    auto out_shape = add->get_output_shape(0);

                    // Only simplify if shapes match (no broadcast needed)
                    if (non_zero.get_shape() == out_shape) {
                        add->output(0).replace(non_zero);
                        ++add_zero;
                        any_change = true;
                        changed = true;
                        if (g_trace)
                            fprintf(stderr, "[ZeroProp] x+0→x: %s\n",
                                    add->get_friendly_name().c_str());
                    } else if (z0 && z1) {
                        // Both zero → result is zero
                        auto out_et = add->get_output_element_type(0);
                        auto zero = make_zero(out_et, out_shape);
                        ov::replace_node(add, zero);
                        ++add_zero;
                        any_change = true;
                        changed = true;
                    }
                    continue;
                }
            }
        }
    }

    if (g_trace || mul_zero > 0 || add_zero > 0)
        fprintf(stderr, "[ZeroConstantPropagation] mul*0→0: %d, x+0→x: %d, iterations: %d\n",
                mul_zero, add_zero, iterations);
    return changed;
}

}  // namespace pass
}  // namespace rocm_gpu
}  // namespace ov
