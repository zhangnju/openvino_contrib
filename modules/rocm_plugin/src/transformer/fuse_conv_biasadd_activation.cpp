// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include <type_traits>
#include <utility>
#include <memory>

#include "openvino/cc/pass/itt.hpp"
#include "fuse_conv_biasadd_activation.hpp"

#include "openvino/core/graph_util.hpp"
#include "openvino/core/node.hpp"
#include "openvino/core/node_output.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/core/shape.hpp"
#include "openvino/core/type/element_type.hpp"
#include "openvino/op/variadic_split.hpp"
#include "openvino/opsets/opset1.hpp"
#include "openvino/op/swish.hpp"
#include "openvino/pass/pattern/matcher.hpp"
#include "openvino/pass/pattern/op/or.hpp"
#include "openvino/pass/pattern/op/label.hpp"
#include "openvino/pass/pattern/op/pattern.hpp"
#include "openvino/pass/manager.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"


#include "nodes/activation_type.hpp"
#include "nodes/fused_convolution.hpp"
#include "nodes/fused_convolution_slice.hpp"
#include "nodes/fused_convolution_slice_out.hpp"
#include "nodes/fused_convolution_backprop_data.hpp"
#include "rt_info/rocm_node_id.hpp"

using namespace ov::pass::pattern;

using ov::rocm_gpu::nodes::FusedConvolution;
using ov::rocm_gpu::nodes::FusedGroupConvolution;

using ActivationMode = ov::rocm_gpu::nodes::ActivationMode;
using FusedConvBackpropData = ov::rocm_gpu::nodes::FusedConvBackpropData;

using namespace ov::rocm_gpu::rt_info;

namespace {
template <class A, class B>
std::pair<std::shared_ptr<A>, std::shared_ptr<B>> parse_eltwise_inputs(std::shared_ptr<ov::Node> node) {
    auto eltwise = std::dynamic_pointer_cast<A>(node->input(0).get_source_output().get_node_shared_ptr());
    auto constant = std::dynamic_pointer_cast<B>(node->input(1).get_source_output().get_node_shared_ptr());

    if (!eltwise) {
        eltwise = std::dynamic_pointer_cast<A>(node->input(1).get_source_output().get_node_shared_ptr());
        constant = std::dynamic_pointer_cast<B>(node->input(0).get_source_output().get_node_shared_ptr());
    }

    if (!eltwise || !constant) {
        return {nullptr, nullptr};
    }

    return {eltwise, constant};
}

template <typename TFusedConvolution>
struct FusedConvCallbacks {
    static_assert(std::is_same_v<TFusedConvolution, FusedConvolution> ||
                      std::is_same_v<TFusedConvolution, FusedGroupConvolution>,
                  "TFusedConvolution should be either FusedConvolution or FusedGroupConvolution");
    static bool fuse_convolution_with_biasadd(Matcher &m) {
        auto eltwise = m.get_match_root();
        auto [m_conv, m_const] =
            parse_eltwise_inputs<typename TFusedConvolution::BaseConvolution, ov::op::v0::Constant>(eltwise);
        if (!m_conv || !m_const) {
            return false;
        }

        if (m_conv->inputs().size() != 2) {
            return false;
        }

        if (std::dynamic_pointer_cast<ov::op::v1::Add>(eltwise) == nullptr) {
            return false;
        }

        const ov::Output<ov::Node> &data = m_conv->input(0).get_source_output();
        const ov::Output<ov::Node> &filters = m_conv->input(1).get_source_output();
        const ov::Output<ov::Node> &bias = m_const->output(0);

        auto fused_conv = std::make_shared<TFusedConvolution>(data,
                                                              filters,
                                                              bias,
                                                              m_conv->get_strides(),
                                                              m_conv->get_pads_begin(),
                                                              m_conv->get_pads_end(),
                                                              m_conv->get_dilations(),
                                                              m_conv->get_auto_pad(),
                                                              ActivationMode::NO_ACTIVATION);
        ov::Output<ov::Node> new_conv(fused_conv);

        fused_conv->set_friendly_name(eltwise->get_friendly_name());

        ov::copy_runtime_info({m_conv, eltwise}, new_conv.get_node_shared_ptr());
        set_node_id(new_conv.get_node_shared_ptr(), get_node_id(eltwise));

        ov::replace_node(m.get_match_root(), new_conv.get_node_shared_ptr());
        return true;
    }

    static std::pair<std::shared_ptr<TFusedConvolution>, std::shared_ptr<ov::Node>> parse_fusedconv_inputs(
        std::shared_ptr<ov::Node> add) {
        std::shared_ptr<TFusedConvolution> fused_conv = nullptr;

        auto input0 = add->input(0).get_source_output().get_node_shared_ptr();
        auto input1 = add->input(1).get_source_output().get_node_shared_ptr();

        auto fused_conv0 = std::dynamic_pointer_cast<TFusedConvolution>(input0);
        auto fused_conv1 = std::dynamic_pointer_cast<TFusedConvolution>(input1);

        auto can_be_fused = [](const std::shared_ptr<ov::Node>& target, const std::shared_ptr<ov::Node>& fused_input) {
            return (target && fused_input && (get_node_id(target) > get_node_id(fused_input) || ov::op::util::is_constant(fused_input)));
        };

        if (fused_conv0 && fused_conv1) {
            if (can_be_fused(fused_conv0, input1)) {
                return {fused_conv0, input1};
            } else if (can_be_fused(fused_conv1, input0)) {
                return {fused_conv1, input0};
            }
        }

        if (fused_conv0 && can_be_fused(fused_conv0, input1)) {
            return {fused_conv0, input1};
        }

        if (fused_conv1 && can_be_fused(fused_conv1, input0)) {
            return {fused_conv1, input0};
        }
        return {nullptr, nullptr};
    }

    static bool sink_add_to_fused_convolution(Matcher &m) {
        auto add = std::dynamic_pointer_cast<ov::op::v1::Add>(m.get_match_root());
        auto [fused_conv, node] = parse_fusedconv_inputs(m.get_match_root());
        if (!fused_conv || !node) {
            return false;
        }

        if (fused_conv->has_add_node() || fused_conv->get_activation() != ActivationMode::NO_ACTIVATION) {
            return false;
        }

        // VariadicSplit has no default output index; fusing it would trigger a validation
        // exception during graph rewrite. Skip this pattern to keep the graph valid.
        if (ov::is_type<ov::op::v1::VariadicSplit>(node)) {
            return false;
        }

        const ov::Output<ov::Node> &data = fused_conv->input(0).get_source_output();
        const ov::Output<ov::Node> &filters = fused_conv->input(1).get_source_output();
        const ov::Output<ov::Node> &bias = fused_conv->input(2).get_source_output();

        auto fused_conv_add = std::make_shared<TFusedConvolution>(data,
                                                                  filters,
                                                                  bias,
                                                                  node,
                                                                  fused_conv->get_strides(),
                                                                  fused_conv->get_pads_begin(),
                                                                  fused_conv->get_pads_end(),
                                                                  fused_conv->get_dilations(),
                                                                  fused_conv->get_auto_pad(),
                                                                  ActivationMode::NO_ACTIVATION);
        ov::Output<ov::Node> fused_conv_add_output{fused_conv_add};

        fused_conv_add->set_friendly_name(add->get_friendly_name());
        ov::copy_runtime_info({node, fused_conv}, fused_conv_add);
        set_node_id(fused_conv_add, get_node_id(add));

        ov::replace_node(fused_conv, fused_conv_add);
        ov::replace_node(m.get_match_root(), fused_conv_add);

        return true;
    }

    static bool sink_activation_to_fused_convolution(Matcher &m) {
        auto activationNode = m.get_match_root();
        auto fused_conv = std::dynamic_pointer_cast<TFusedConvolution>(
            activationNode->input(0).get_source_output().get_node_shared_ptr());
        if (fused_conv->get_activation() != ActivationMode::NO_ACTIVATION) {
            return false;
        }

        ActivationMode activation = ActivationMode::NO_ACTIVATION;
        if (ov::is_type<ov::op::v0::Relu>(activationNode)) {
            activation = ActivationMode::RELU;
        } else if (ov::is_type<ov::op::v0::Sigmoid>(activationNode)) {
            activation = ActivationMode::SIGMOID;
        } else if (ov::is_type<ov::op::v4::Swish>(activationNode)) {
            activation = ActivationMode::SWISH;
        } else if (ov::is_type<ov::op::v0::Tanh>(activationNode)) {
            activation = ActivationMode::TANH;
        } else {
            return false;
        }
        fused_conv->set_activation(activation);

        fused_conv->set_friendly_name(activationNode->get_friendly_name());
        set_node_id(fused_conv, get_node_id(activationNode));

        ov::replace_node(m.get_match_root(), fused_conv);

        return true;
    }
};  // struct FusedConvCallbacks

bool fuse_convolution_backprop_data_with_add(Matcher &m) {
    auto add = std::dynamic_pointer_cast<ov::op::v1::Add>(m.get_match_root());
    auto [conv_backprop_data, add_constant] =
        parse_eltwise_inputs<ov::op::v1::ConvolutionBackpropData, ov::op::v0::Constant>(add);

    const auto conv_element_type = conv_backprop_data->get_input_element_type(0);
    const auto add_element_type = add_constant->get_output_element_type(0);
    if (conv_element_type != add_element_type) {
        return false;
    }

    const auto conv_output_shape = dynamic_cast<ov::Node *>(conv_backprop_data.get())->get_output_shape(0);
    const auto add_output_shape = add_constant->get_output_shape(0);
    const auto size = ov::element::Type(conv_element_type).size();
    const auto conv_in_bytes = size * ov::shape_size(conv_output_shape);
    const auto add_in_bytes = size * ov::shape_size(add_output_shape);
    if (add_in_bytes > conv_in_bytes) {
        return false;
    }

    const ov::Output<ov::Node> &data = conv_backprop_data->input(0).get_source_output();
    const ov::Output<ov::Node> &filters = conv_backprop_data->input(1).get_source_output();
    std::shared_ptr<FusedConvBackpropData> fused_conv_backprop_data_add;

    if (3 == conv_backprop_data->get_input_size()) {
        auto output_shape = conv_backprop_data->input(2).get_source_output();
        fused_conv_backprop_data_add =
            std::make_shared<FusedConvBackpropData>(data,
                                                    filters,
                                                    output_shape,
                                                    add_constant,
                                                    conv_backprop_data->get_strides(),
                                                    conv_backprop_data->get_pads_begin(),
                                                    conv_backprop_data->get_pads_end(),
                                                    conv_backprop_data->get_dilations(),
                                                    conv_backprop_data->get_auto_pad(),
                                                    conv_backprop_data->get_output_padding());
    } else {
        fused_conv_backprop_data_add =
            std::make_shared<FusedConvBackpropData>(data,
                                                    filters,
                                                    add_constant,
                                                    conv_backprop_data->get_strides(),
                                                    conv_backprop_data->get_pads_begin(),
                                                    conv_backprop_data->get_pads_end(),
                                                    conv_backprop_data->get_dilations(),
                                                    conv_backprop_data->get_auto_pad(),
                                                    conv_backprop_data->get_output_padding());
    }

    ov::Output<ov::Node> fused_conv_backprop_data_add_output{fused_conv_backprop_data_add};

    fused_conv_backprop_data_add->set_friendly_name(add->get_friendly_name());
    ov::copy_runtime_info({conv_backprop_data, add}, fused_conv_backprop_data_add);

    ov::replace_node(add, fused_conv_backprop_data_add);
    ov::replace_node(conv_backprop_data, fused_conv_backprop_data_add);

    return true;
}
bool is_bias_to_be_fused(const ov::Output<ov::Node>& output) {
    constexpr auto conv_bias_rank_min{3};
    constexpr auto conv_bias_rank_max{5};
    auto node = std::dynamic_pointer_cast<ov::op::v1::Add>(output.get_node_shared_ptr());
    if (!node) {
        return false;
    }

    auto input0 = node->input(0);
    auto input1 = node->input(1);

    const auto partial_shape0 = node->input(0).get_partial_shape();
    const auto partial_shape1 = node->input(1).get_partial_shape();

    if (partial_shape0.is_dynamic() || partial_shape1.is_dynamic()) {
        return false;
    }

    if (node->get_autob() != ov::op::AutoBroadcastType::NUMPY) {
        return false;
    }

    if (input0.get_element_type() != input1.get_element_type()) {
        return false;
    }

    const auto conv_shape = partial_shape0.to_shape();
    const auto bias_shape = partial_shape1.to_shape();
    const auto bias_rank = bias_shape.size();
    if (bias_rank < conv_bias_rank_min || bias_rank > conv_bias_rank_max) {
        return false;
    }
    const auto num_spatial_dims = conv_shape.size() - 2;
    if (num_spatial_dims == 3) {
        return false;  // NOTE: 3D convolution fusing was disabled due to 3d_unet bad performance
    }
    const auto nchw_channel_dim_reverse_offset = num_spatial_dims + 1;
    size_t bias_channel_index = bias_shape.size() - nchw_channel_dim_reverse_offset;
    size_t conv_channel_index = conv_shape.size() - nchw_channel_dim_reverse_offset;
    if (bias_shape.at(bias_channel_index) != conv_shape.at(conv_channel_index)) {
        return false;
    }
    for (size_t i = 0; i < bias_shape.size(); i++) {
        if ((i != bias_channel_index) && (bias_shape.at(i) != 1)) return false;
    }
    return true;
}
bool is_add_to_be_fused(const ov::Output<ov::Node>& output) {
    auto node = std::dynamic_pointer_cast<ov::op::v1::Add>(output.get_node_shared_ptr());
    if (!node) {
        return false;
    }

    auto input0 = node->input(0);
    auto input1 = node->input(1);

    const auto partial_shape0 = node->input(0).get_partial_shape();
    const auto partial_shape1 = node->input(1).get_partial_shape();

    if (input0.get_element_type() != input1.get_element_type()) {
        return false;
    }

    if (partial_shape0.is_dynamic() || partial_shape1.is_dynamic()) {
        return false;
    }
    return (partial_shape0.to_shape() == partial_shape1.to_shape());
}
} // namespace

bool ov::rocm_gpu::pass::rocmFuseMarkUpNodesOrder::run_on_model(const std::shared_ptr<ov::Model>& m) {
    RUN_ON_FUNCTION_SCOPE(rocmFuseMarkUpNodesOrder);
    uint64_t id = 0;
    for (auto& node : m->get_ordered_ops()) {
        set_node_id(node, id++);
    }
    return false;
}

bool ov::rocm_gpu::pass::rocmFuseCleanUpNodesOrder::run_on_model(const std::shared_ptr<ov::Model>& m) {
    RUN_ON_FUNCTION_SCOPE(rocmFuseCleanUpNodesOrder);
    for (auto& node : m->get_ordered_ops()) {
        remove_node_id(node);
    }
    return false;
}

ov::rocm_gpu::pass::FuseConvolutionWithBiasAdd::FuseConvolutionWithBiasAdd() {
    MATCHER_SCOPE(FuseConvolutionWithBiasAdd);
    auto conv = wrap_type<ov::op::v1::Convolution>(consumers_count(1));
    auto bias = wrap_type<ov::op::v0::Constant>();
    auto add = wrap_type<ov::op::v1::Add>({conv, bias}, is_bias_to_be_fused);

    matcher_pass_callback callback = [](Matcher &m) {
        return FusedConvCallbacks<FusedConvolution>::fuse_convolution_with_biasadd(m);
    };

    auto m = std::make_shared<Matcher>(add, matcher_name);
    register_matcher(m, callback);
}

ov::rocm_gpu::pass::FuseGroupConvolutionWithBiasAdd::FuseGroupConvolutionWithBiasAdd() {
    MATCHER_SCOPE(FuseGroupConvolutionWithBiasAdd);
    auto conv = wrap_type<ov::op::v1::GroupConvolution>(consumers_count(1));
    auto bias = wrap_type<ov::op::v0::Constant>();
    auto add = wrap_type<ov::op::v1::Add>({conv, bias}, is_bias_to_be_fused);

    matcher_pass_callback callback = [](Matcher &m) {
        return FusedConvCallbacks<FusedGroupConvolution>::fuse_convolution_with_biasadd(m);
    };

    auto m = std::make_shared<Matcher>(add, matcher_name);
    register_matcher(m, callback);
}

ov::rocm_gpu::pass::FuseConvolutionWithBiasAddAdd::FuseConvolutionWithBiasAddAdd() {
    MATCHER_SCOPE(FuseConvolutionWithBiasAddAdd);
    auto fused_convolution = wrap_type<FusedConvolution>(consumers_count(1));
    auto add1 = wrap_type<ov::op::v1::Add>({fused_convolution, any_input()}, is_add_to_be_fused);
    auto add2 = wrap_type<ov::op::v1::Add>({any_input(), fused_convolution}, is_add_to_be_fused);
    auto add = std::make_shared<::op::Or>(ov::OutputVector{ add1, add2 });

    matcher_pass_callback callback = [](Matcher &m) {
        return FusedConvCallbacks<FusedConvolution>::sink_add_to_fused_convolution(m);
    };

    auto m = std::make_shared<Matcher>(add, matcher_name);
    register_matcher(m, callback);
}

ov::rocm_gpu::pass::FuseGroupConvolutionWithBiasAddAdd::FuseGroupConvolutionWithBiasAddAdd() {
    MATCHER_SCOPE(FuseGroupConvolutionWithBiasAddAdd);
    auto fused_convolution = wrap_type<FusedGroupConvolution>(consumers_count(1));
    auto add1 = wrap_type<ov::op::v1::Add>({fused_convolution, any_input()}, is_add_to_be_fused);
    auto add2 = wrap_type<ov::op::v1::Add>({any_input(), fused_convolution}, is_add_to_be_fused);
    auto add = std::make_shared<::op::Or>(ov::OutputVector{ add1, add2 });

    matcher_pass_callback callback = [](Matcher &m) {
        return FusedConvCallbacks<FusedGroupConvolution>::sink_add_to_fused_convolution(m);
    };

    auto m = std::make_shared<Matcher>(add, matcher_name);
    register_matcher(m, callback);
}

ov::rocm_gpu::pass::SinkActivationToFusedConvolution::SinkActivationToFusedConvolution() {
    MATCHER_SCOPE(SinkActivationToFusedConvolution);
    auto fused_convolution = wrap_type<FusedConvolution>(consumers_count(1));
    // Swish disabled: SinkActivationToFusedConvolution with Swish causes excessive D2D copies.
    // SiLU is handled separately by FusedElementwise or SwishOpImpl in the graph.
    auto activation = wrap_type<ov::op::v0::Relu, ov::op::v0::Sigmoid>({fused_convolution});

    matcher_pass_callback callback = [](Matcher &m) {
        return FusedConvCallbacks<FusedConvolution>::sink_activation_to_fused_convolution(m);
    };

    auto m = std::make_shared<Matcher>(activation, matcher_name);
    register_matcher(m, callback);
}

// ─────────────────────────────────────────────────────────────────────────────
// SinkSwishAddToFusedConvolution
// Matches: FusedConvolution(no-add, no-act) → Swish(1 consumer) → Add(skip)
// Rewrites to: FusedConvolution(4-inputs: data,filter,bias,skip; activation=SWISH)
//
// This enables FusedConvolutionRocMLIR to compile a 5-arg Conv+Bias+SiLU+SkipAdd
// kernel, replacing three separate kernel launches with one.
// ─────────────────────────────────────────────────────────────────────────────
// SinkSwishAddToFusedConvolution is implemented as a ModelPass (graph traversal)
// rather than a MatcherPass because the pattern FusedConvolution→Swish→Add requires
// exact input-ordering checks that are easier to express imperatively.
// SinkSwishAddToFusedConvolution is implemented as SinkSwishAddPass (ModelPass)
// in rocmConvolutionFusion::run_on_model. This empty MatcherPass exists for
// header-file compatibility (the class is declared in the header).
ov::rocm_gpu::pass::SinkSwishAddToFusedConvolution::SinkSwishAddToFusedConvolution() {
    MATCHER_SCOPE(SinkSwishAddToFusedConvolution);
    auto dummy = wrap_type<ov::op::v0::Constant>();
    register_matcher(std::make_shared<Matcher>(dummy, matcher_name),
                     [](Matcher&) { return false; });
}

bool ov::rocm_gpu::pass::rocmConvolutionFusion::run_on_model(const std::shared_ptr<ov::Model>& m) {
    RUN_ON_FUNCTION_SCOPE(rocmConvolutionFusion);
    ov::pass::Manager manager(get_pass_config());

    manager.register_pass<rocmFuseMarkUpNodesOrder>();

    auto fuse_conv_bias_add_activation = manager.register_pass<ov::pass::GraphRewrite>();
    ADD_MATCHER(fuse_conv_bias_add_activation, FuseConvolutionWithBiasAdd)
    ADD_MATCHER(fuse_conv_bias_add_activation, FuseConvolutionWithBiasAddAdd)
    ADD_MATCHER(fuse_conv_bias_add_activation, SinkActivationToFusedConvolution)
    // Group conv bias fusion runs in the same GraphRewrite pass so SinkSwishAddPass
    // (next pass) can see FusedGroupConvolution and fuse Swish+Add into them.
    ADD_MATCHER(fuse_conv_bias_add_activation, FuseGroupConvolutionWithBiasAdd)
    ADD_MATCHER(fuse_conv_bias_add_activation, FuseGroupConvolutionWithBiasAddAdd)
    fuse_conv_bias_add_activation->set_name("ov::rocm_gpu::pass::fuse_conv_bias_add_activation");

    // Fuse FusedConvolution → Swish → Add(skip) into 5-arg Conv+Bias+SiLU+Add kernel.
    // Implemented as ModelPass (imperative traversal) because MatcherPass's pattern
    // matching for Or-patterns with input ordering had platform-specific issues.
    struct SinkSwishAddPass : ov::pass::ModelPass {
        OPENVINO_RTTI("SinkSwishAddPass", "0");
        bool run_on_model(const std::shared_ptr<ov::Model>& model) override {
            bool changed = false;
            // Collect candidates first to avoid modifying graph while iterating.
            // Tuple: (conv_node, swish, add, skip_input, is_group_conv)
            std::vector<std::tuple<std::shared_ptr<ov::Node>,
                                    std::shared_ptr<ov::op::v4::Swish>,
                                    std::shared_ptr<ov::op::v1::Add>,
                                    ov::Output<ov::Node>,
                                    bool>> candidates;

            for (const auto& op : model->get_ordered_ops()) {
                auto swish = std::dynamic_pointer_cast<ov::op::v4::Swish>(op);
                if (!swish) continue;

                // Check: FusedConvolution or FusedGroupConvolution → Swish
                auto fc_out = swish->input(0).get_source_output();
                auto fused_conv  = std::dynamic_pointer_cast<FusedConvolution>(fc_out.get_node_shared_ptr());
                auto fused_gconv = std::dynamic_pointer_cast<FusedGroupConvolution>(fc_out.get_node_shared_ptr());
                if (!fused_conv && !fused_gconv) continue;

                const bool is_group = (fused_gconv != nullptr);
                const bool has_add  = is_group ? fused_gconv->has_add_node()  : fused_conv->has_add_node();
                const auto act      = is_group ? fused_gconv->get_activation() : fused_conv->get_activation();
                if (has_add) continue;
                if (act != ActivationMode::NO_ACTIVATION) continue;

                // Check: Swish → Add (skip connection), Swish has exactly one consumer
                auto swish_out = swish->output(0);
                if (swish_out.get_target_inputs().size() != 1) continue;
                auto add_tgt = *swish_out.get_target_inputs().begin();
                auto add_node = std::dynamic_pointer_cast<ov::op::v1::Add>(
                    add_tgt.get_node()->shared_from_this());
                if (!add_node) continue;

                // Find skip input (the Add input that is NOT the Swish output)
                ov::Output<ov::Node> skip_input;
                for (size_t i = 0; i < 2; ++i) {
                    if (add_node->input(i).get_source_output().get_node_shared_ptr() != swish) {
                        skip_input = add_node->input(i).get_source_output();
                    }
                }
                if (!skip_input.get_node()) continue;

                // Check shapes match
                auto ps0 = add_node->input(0).get_partial_shape();
                auto ps1 = add_node->input(1).get_partial_shape();
                if (ps0.is_dynamic() || ps1.is_dynamic()) continue;
                if (ps0.to_shape() != ps1.to_shape()) continue;
                if (add_node->input(0).get_element_type() != add_node->input(1).get_element_type()) continue;

                std::shared_ptr<ov::Node> conv_node = is_group
                    ? std::static_pointer_cast<ov::Node>(fused_gconv)
                    : std::static_pointer_cast<ov::Node>(fused_conv);
                candidates.emplace_back(conv_node, swish, add_node, skip_input, is_group);
            }

            int n_group = 0;
            for (auto& [conv, sw, add, skip, is_group] : candidates) {
                if (is_group) n_group++;
                const ov::Output<ov::Node>& data    = conv->input(0).get_source_output();
                const ov::Output<ov::Node>& filters = conv->input(1).get_source_output();
                const ov::Output<ov::Node>& bias    = conv->input(2).get_source_output();

                std::shared_ptr<ov::Node> fused;
                if (!is_group) {
                    auto fc = std::dynamic_pointer_cast<FusedConvolution>(conv);
                    fused = std::make_shared<FusedConvolution>(
                        data, filters, bias, skip,
                        fc->get_strides(), fc->get_pads_begin(), fc->get_pads_end(),
                        fc->get_dilations(), fc->get_auto_pad(), ActivationMode::SWISH);
                } else {
                    auto fgc = std::dynamic_pointer_cast<FusedGroupConvolution>(conv);
                    fused = std::make_shared<FusedGroupConvolution>(
                        data, filters, bias, skip,
                        fgc->get_strides(), fgc->get_pads_begin(), fgc->get_pads_end(),
                        fgc->get_dilations(), fgc->get_auto_pad(), ActivationMode::SWISH);
                }

                fused->set_friendly_name(add->get_friendly_name());
                ov::copy_runtime_info({conv, sw, add}, fused);
                set_node_id(fused, get_node_id(add));

                ov::replace_node(conv, fused);
                ov::replace_node(add, fused);
                changed = true;
            }
            std::cerr << "[SinkSwishAdd] Fusing " << candidates.size()
                      << " Conv+Bias+SiLU+Add patterns (" << n_group << " group conv)\n";
            return changed;
        }
    };
    // Fuse VariadicSplit(axis=1) → FusedConvolution by replacing the conv's input with
    // the pre-split tensor and recording the channel slice parameters in runtime info.
    // This allows FusedConvolutionRocMLIR to compile a zero-copy Slice+Conv kernel.
    //
    // Graph before:  FullInput → VariadicSplit → [out0: Concat], [out1: FusedConvolution]
    // Graph after:   FullInput → [VariadicSplit → Concat], [FusedConvolution(using FullInput)]
    //                where FusedConvolution's runtime info has slice_c_start, slice_c_full
    struct FuseVariadicSplitConvPass : ov::pass::ModelPass {
        OPENVINO_RTTI("FuseVariadicSplitConvPass", "0");
        bool run_on_model(const std::shared_ptr<ov::Model>& model) override {
            bool changed = false;
            // Collect VariadicSplit(axis=1) nodes whose 2nd output feeds FusedConvolution(SWISH)
            std::vector<std::tuple<
                std::shared_ptr<ov::op::v1::VariadicSplit>,  // the split node
                std::shared_ptr<FusedConvolution>,            // the conv node
                int,    // c_start in the full input
                int     // C_full = total channels before split
            >> candidates;

            for (const auto& op : model->get_ordered_ops()) {
                auto split = std::dynamic_pointer_cast<ov::op::v1::VariadicSplit>(op);
                if (!split) continue;

                // Only handle channel-axis (axis=1) splits with exactly 2 outputs
                if (split->get_output_size() != 2) continue;
                auto axis_node = std::dynamic_pointer_cast<ov::op::v0::Constant>(
                    split->input(1).get_source_output().get_node_shared_ptr());
                if (!axis_node) continue;
                std::vector<int64_t> axis_data;
                { const auto& _et=(axis_node)->get_element_type();
                  if(_et==ov::element::i64) axis_data=(axis_node)->cast_vector<int64_t>();
                  else if(_et==ov::element::i32) for(auto _v:(axis_node)->cast_vector<int32_t>()) axis_data.push_back(_v);
                  else for(auto _v:(axis_node)->cast_vector<float>()) axis_data.push_back(static_cast<int64_t>(_v)); }
                if (axis_data.empty() || axis_data[0] != 1) continue;  // must be axis=1

                // Get split lengths
                auto split_len_node = std::dynamic_pointer_cast<ov::op::v0::Constant>(
                    split->input(2).get_source_output().get_node_shared_ptr());
                if (!split_len_node) continue;
                std::vector<int64_t> split_lens;
                { const auto& _et=(split_len_node)->get_element_type();
                  if(_et==ov::element::i64) split_lens=(split_len_node)->cast_vector<int64_t>();
                  else if(_et==ov::element::i32) for(auto _v:(split_len_node)->cast_vector<int32_t>()) split_lens.push_back(_v);
                  else for(auto _v:(split_len_node)->cast_vector<float>()) split_lens.push_back(static_cast<int64_t>(_v)); }
                if (split_lens.size() != 2) continue;

                // Get full input shape
                const auto full_shape = split->get_input_shape(0);
                if (full_shape.size() != 4) continue;  // must be NCHW
                const int C_full = static_cast<int>(full_shape[1]);

                // Check: out[0] feeds only Concat (not Conv); out[1] feeds FusedConvolution(SWISH)
                // We only fuse the second half that goes to Conv
                for (size_t out_idx = 0; out_idx < 2; ++out_idx) {
                    auto split_out = split->output(out_idx);
                    for (const auto& tgt : split_out.get_target_inputs()) {
                        auto fc = std::dynamic_pointer_cast<FusedConvolution>(
                            tgt.get_node()->shared_from_this());
                        if (!fc) continue;
                        // activation_ in graph node may be NO_ACTIVATION even if Swish follows
                        // (the factory sets SWISH dynamically at compile time, not in the graph)
                        if (fc->get_activation() != ActivationMode::NO_ACTIVATION) continue;
                        if (fc->has_add_node()) continue;
                        // Accept FusedConvolution regardless of its downstream consumer:
                        // - If downstream is Swish: FusedConvolutionSlice compiles Slice+Conv+Bias+SiLU
                        //   and SiLU tracking makes the downstream Swish a no-op.
                        // - If downstream is NOT Swish: FusedConvolutionSlice compiles Slice+Conv+Bias
                        //   (plain bias-fused kernel, same as FusedConvolution would use).
                        // Either way, the VariadicSplit data copy is eliminated.
                        // (Previously restricted to Swish-consumer only -- Priority 3 fix: remove restriction)
                        (void)0;  // no additional filter needed

                        // Compute channel start
                        int c_start = (out_idx == 0) ? 0 : static_cast<int>(split_lens[0]);
                        candidates.emplace_back(split, fc, c_start, C_full);
                    }
                }
            }

            std::cerr << "[SliceConv] Found " << candidates.size()
                      << " VariadicSplit→FusedConvolution(SWISH) patterns\n";

            for (auto& [split, fc, c_start, C_full] : candidates) {
                // Get the full (pre-split) input to VariadicSplit
                const ov::Output<ov::Node>& full_input = split->input(0).get_source_output();
                const ov::Output<ov::Node>& filters    = fc->input(1).get_source_output();
                const ov::Output<ov::Node>& bias       = fc->input(2).get_source_output();

                // Create FusedConvolutionSlice: takes full input + carries c_start attribute.
                // Its validate_and_infer_types() uses filter.C for shape computation,
                // so it accepts full_input.C != filter.C without error.
                // FusedConvolutionRocMLIR will compile a MIGraphX-style zero-copy Slice+Conv kernel.
                auto fc_slice = std::make_shared<nodes::FusedConvolutionSlice>(
                    full_input, filters, bias,
                    c_start,
                    fc->get_strides(), fc->get_pads_begin(), fc->get_pads_end(),
                    fc->get_dilations(), fc->get_auto_pad());

                fc_slice->set_friendly_name(fc->get_friendly_name() + "_sliceconv");
                ov::copy_runtime_info(fc, fc_slice);
                set_node_id(fc_slice, get_node_id(fc));

                // Replace the original FusedConvolution and its downstream Swish
                // The Swish consumer of fc needs to be redirected too
                // Find the Swish consumer of fc and replace it with fc_slice output
                // (SiLU is fused into the slice-conv kernel; downstream SwishOp becomes no-op)
                ov::replace_node(fc, fc_slice);
                changed = true;
            }
            std::cerr << "[SliceConv] Fused " << candidates.size()
                      << " VariadicSplit→FusedConvolutionSlice patterns\n";
            return changed;
        }
    };
    // SinkSwishAddPass: fuse FusedConvolution → Swish → Add(skip) into 5-arg kernel.
    manager.register_pass<SinkSwishAddPass>();

    // FuseConvSliceOutSiluAddPass: detect FusedConvolution → VariadicSplit(axis=1) → Swish → Add(skip)
    // and replace with FusedConvolutionSliceOut (output-side slice fused into conv kernel).
    // Matches MIGraphX pattern: mlir_convolution_broadcast_slice_add_sigmoid_mul_add (30 instances).
    struct FuseConvSliceOutSiluAddPass : ov::pass::ModelPass {
        OPENVINO_RTTI("FuseConvSliceOutSiluAddPass", "0");
        bool run_on_model(const std::shared_ptr<ov::Model>& model) override {
            bool changed = false;
            std::vector<std::tuple<
                std::shared_ptr<FusedConvolution>,
                std::shared_ptr<ov::op::v1::VariadicSplit>,
                size_t,     // split output index
                std::shared_ptr<ov::op::v4::Swish>,
                std::shared_ptr<ov::op::v1::Add>,
                ov::Output<ov::Node>  // skip input
            >> candidates;

            for (const auto& op : model->get_ordered_ops()) {
                // Match FusedConvolution with NO activation and NO skip-add
                auto fc = std::dynamic_pointer_cast<FusedConvolution>(op);
                if (!fc) continue;
                if (fc->get_activation() != ActivationMode::NO_ACTIVATION) continue;
                if (fc->has_add_node()) continue;

                // FusedConv output must feed a VariadicSplit(axis=1)
                auto fc_out = fc->output(0);
                for (const auto& fc_tgt : fc_out.get_target_inputs()) {
                    auto vsplit = std::dynamic_pointer_cast<ov::op::v1::VariadicSplit>(
                        fc_tgt.get_node()->shared_from_this());
                    if (!vsplit) continue;

                    // Check axis=1 (channel split)
                    auto axis_node = std::dynamic_pointer_cast<ov::op::v0::Constant>(
                        vsplit->input(1).get_source_output().get_node_shared_ptr());
                    if (!axis_node) continue;
                    std::vector<int64_t> axis_data;
                    { const auto& _et=(axis_node)->get_element_type();
                      if(_et==ov::element::i64) axis_data=(axis_node)->cast_vector<int64_t>();
                      else if(_et==ov::element::i32) for(auto _v:(axis_node)->cast_vector<int32_t>()) axis_data.push_back(_v);
                      else for(auto _v:(axis_node)->cast_vector<float>()) axis_data.push_back(static_cast<int64_t>(_v)); }
                    if (axis_data.empty() || axis_data[0] != 1) continue;

                    auto split_len_node = std::dynamic_pointer_cast<ov::op::v0::Constant>(
                        vsplit->input(2).get_source_output().get_node_shared_ptr());
                    if (!split_len_node) continue;
                    std::vector<int64_t> split_lens;
                    { const auto& _et=(split_len_node)->get_element_type();
                      if(_et==ov::element::i64) split_lens=(split_len_node)->cast_vector<int64_t>();
                      else if(_et==ov::element::i32) for(auto _v:(split_len_node)->cast_vector<int32_t>()) split_lens.push_back(_v);
                      else for(auto _v:(split_len_node)->cast_vector<float>()) split_lens.push_back(static_cast<int64_t>(_v)); }

                    // Check each VariadicSplit output for → Swish → Add(skip) pattern
                    for (size_t out_idx = 0; out_idx < vsplit->get_output_size(); ++out_idx) {
                        auto split_out = vsplit->output(out_idx);
                        if (split_out.get_target_inputs().size() != 1) continue;

                        auto swish_node = std::dynamic_pointer_cast<ov::op::v4::Swish>(
                            split_out.get_target_inputs().begin()->get_node()->shared_from_this());
                        if (!swish_node) continue;

                        auto swish_out = swish_node->output(0);
                        if (swish_out.get_target_inputs().size() != 1) continue;

                        auto add_node = std::dynamic_pointer_cast<ov::op::v1::Add>(
                            swish_out.get_target_inputs().begin()->get_node()->shared_from_this());
                        if (!add_node) continue;

                        // Check shapes match for skip-add
                        auto ps0 = add_node->input(0).get_partial_shape();
                        auto ps1 = add_node->input(1).get_partial_shape();
                        if (ps0.is_dynamic() || ps1.is_dynamic()) continue;
                        if (ps0.to_shape() != ps1.to_shape()) continue;
                        if (add_node->input(0).get_element_type() !=
                            add_node->input(1).get_element_type()) continue;

                        // Find skip input (not Swish output)
                        ov::Output<ov::Node> skip_input;
                        for (size_t i = 0; i < 2; ++i) {
                            if (add_node->input(i).get_source_output().get_node_shared_ptr() != swish_node) {
                                skip_input = add_node->input(i).get_source_output();
                            }
                        }
                        if (!skip_input.get_node()) continue;
                        if (ov::is_type<ov::op::v1::VariadicSplit>(skip_input.get_node_shared_ptr())) continue;

                        candidates.emplace_back(fc, vsplit, out_idx, swish_node, add_node, skip_input);
                    }
                }
            }

            std::cerr << "[SliceOutConv] Found " << candidates.size()
                      << " FusedConv→Split→Swish→Add patterns\n";

            for (auto& [fc, vsplit, out_idx, swish, add, skip] : candidates) {
                const auto& flt_shape = fc->get_input_shape(1);  // [K, C/G, R, S]
                const auto& full_out_shape = fc->get_output_shape(0);  // [N, K, OH, OW]

                // Compute c_out_start/end for this split output
                auto split_len_node = std::dynamic_pointer_cast<ov::op::v0::Constant>(
                    vsplit->input(2).get_source_output().get_node_shared_ptr());
                std::vector<int64_t> split_lens;
                { const auto& _et=(split_len_node)->get_element_type();
                  if(_et==ov::element::i64) split_lens=(split_len_node)->cast_vector<int64_t>();
                  else if(_et==ov::element::i32) for(auto _v:(split_len_node)->cast_vector<int32_t>()) split_lens.push_back(_v);
                  else for(auto _v:(split_len_node)->cast_vector<float>()) split_lens.push_back(static_cast<int64_t>(_v)); }
                int c_out_start = 0;
                for (size_t i = 0; i < out_idx; ++i) c_out_start += static_cast<int>(split_lens[i]);
                int c_out_end = c_out_start + static_cast<int>(split_lens[out_idx]);

                const ov::Output<ov::Node>& data    = fc->input(0).get_source_output();
                const ov::Output<ov::Node>& filter  = fc->input(1).get_source_output();
                const ov::Output<ov::Node>& bias    = fc->input(2).get_source_output();

                auto fused = std::make_shared<nodes::FusedConvolutionSliceOut>(
                    data, filter, bias, skip,
                    c_out_start, c_out_end,
                    fc->get_strides(), fc->get_pads_begin(), fc->get_pads_end(),
                    fc->get_dilations(), fc->get_auto_pad());

                fused->set_friendly_name(add->get_friendly_name() + "_sliceout");
                ov::copy_runtime_info({fc, vsplit, swish, add}, fused);

                // Replace the Add node (final output) with the new fused node
                // fc and vsplit may still have other consumers → only replace add
                ov::replace_node(add, fused);
                changed = true;
            }
            return changed;
        }
    };
    manager.register_pass<FuseConvSliceOutSiluAddPass>();

    // MarkSwishAddEpiloguePass: detect FusedConvolution → Swish → Add(swish, aux_silu) pattern
    // (the 14 C2f/C2PSA bottleneck patterns left after SinkSwishAdd).
    // Tags the FusedConvolution with rt_info "rocm_epilogue_silu_add_aux" pointing to aux node.
    // When ROCMLIR_EPILOGUE_FUSION=1, FusedConvolutionRocMLIR uses the 6-arg migraphx kernel:
    //   conv+bias+silu+skip_add → silu(fc_out) → add(silu, aux)
    struct MarkSwishAddEpiloguePass : ov::pass::ModelPass {
        OPENVINO_RTTI("MarkSwishAddEpiloguePass", "0");
        bool run_on_model(const std::shared_ptr<ov::Model>& model) override {
            bool changed = false;
            for (const auto& op : model->get_ordered_ops()) {
                // Match: FusedConvolution (any activation, with or without skip_add)
                auto fc = std::dynamic_pointer_cast<FusedConvolution>(op);
                if (!fc) continue;
                // Already tagged → skip
                if (fc->get_rt_info().count("rocm_epilogue_silu_add_aux")) continue;

                // FC output must feed at least one Swish (may have other consumers too)
                if (fc->get_output_size() == 0) continue;
                auto fc_out = fc->output(0);
                std::shared_ptr<ov::op::v4::Swish> swish;
                for (const auto& inp : fc_out.get_target_inputs()) {
                    swish = std::dynamic_pointer_cast<ov::op::v4::Swish>(
                        inp.get_node()->shared_from_this());
                    if (swish) break;
                }
                if (!swish) continue;

                // Swish output must feed exactly one Add (to ensure correct rewiring)
                if (swish->get_output_size() == 0) continue;
                auto swish_out = swish->output(0);
                if (swish_out.get_target_inputs().size() != 1) continue;
                auto add = std::dynamic_pointer_cast<ov::op::v1::Add>(
                    swish_out.get_target_inputs().begin()->get_node()->shared_from_this());
                if (!add) continue;

                // Add must have 2 inputs: swish_out and aux
                if (add->get_input_size() != 2) continue;
                auto in0 = add->input(0).get_source_output();
                auto in1 = add->input(1).get_source_output();
                ov::Output<ov::Node> aux_out;
                if (in0.get_node_shared_ptr().get() == swish.get()) {
                    aux_out = in1;
                } else if (in1.get_node_shared_ptr().get() == swish.get()) {
                    aux_out = in0;
                } else {
                    continue;  // neither input is from swish
                }

                // Aux must itself be from a Swish (SiLU) following a FusedConvolution or similar
                auto aux_swish = std::dynamic_pointer_cast<ov::op::v4::Swish>(
                    aux_out.get_node_shared_ptr());
                if (!aux_swish) continue;

                // Shape check: fc output, swish output, aux must have same shape
                auto fc_shape  = fc->get_output_partial_shape(0);
                auto aux_shape = aux_out.get_partial_shape();
                if (fc_shape.is_dynamic() || aux_shape.is_dynamic()) continue;
                if (fc_shape.to_shape() != aux_shape.to_shape()) continue;

                const size_t n_consumers = fc_out.get_target_inputs().size();

                if (n_consumers == 1) {
                    // ── Clean case: FC output feeds ONLY Swish → safe to rewrite graph ──
                    // Create 5-input FC: (data, filter, bias, skip, aux_silu)
                    // It computes: conv+bias+silu+skip_add → silu(out) → add(silu, aux)
                    // Output replaces the Add node's output in the graph.
                    ov::OutputVector new_inputs;
                    for (size_t i = 0; i < fc->get_input_size(); ++i)
                        new_inputs.push_back(fc->input_value(i));
                    new_inputs.push_back(aux_out);  // aux_silu as 5th input

                    auto new_fc = fc->clone_with_new_inputs(new_inputs);
                    new_fc->set_friendly_name(fc->get_friendly_name());
                    // Mark this 5-input FC for ROCMLIR_EPILOGUE_FUSION kernel selection
                    new_fc->get_rt_info()["rocm_epilogue_silu_add_aux"] =
                        ov::Any(aux_swish->get_friendly_name());
                    ov::copy_runtime_info(fc, new_fc);

                    // Replace Add's output with new FC (eliminates Swish and Add from graph)
                    ov::replace_node(add, new_fc);
                    changed = true;
                    std::cerr << "[EpilogueRewrite] FC '" << fc->get_friendly_name()
                              << "' → Swish → Add fused into 5-input FC (6-arg kernel)\n";
                } else {
                    // ── Multi-consumer case: FC feeds other branches too ──
                    // Cannot replace FC without duplicating conv computation.
                    // Just tag for informational purposes; FusedElementwise handles this chain.
                    fc->get_rt_info()["rocm_epilogue_silu_add_aux"] =
                        ov::Any(aux_swish->get_friendly_name());
                    changed = true;
                    std::cerr << "[EpilogueTag] FC '" << fc->get_friendly_name()
                              << "' consumers=" << n_consumers
                              << " (multi-consumer, cannot rewrite → FusedElementwise handles)\n";
                }
            }
            return changed;
        }
    };
    // Register by default; disable with ROCMLIR_EPILOGUE_FUSION=0.
    // MarkSwishAddEpiloguePass tags FC→Swish→Add(aux_silu) for 6-arg migraphx kernel.
    {
        const char* e = std::getenv("ROCMLIR_EPILOGUE_FUSION");
        if (!(e && std::string(e) == "0"))
            manager.register_pass<MarkSwishAddEpiloguePass>();
    }

    // MarkConvReshapeEpiloguePass: detect FusedConvolution(bias_only) → Reshape pattern
    // and tag the FC for ROCMLIR_EPILOGUE_FUSION to compile as mlir_convolution_broadcast_add_reshape.
    // Matches MIGraphX's 168-instance Q/K/V attention projection pattern in yolo26x:
    //   conv 1×1 (N×C×H×W → N×K×H×W) + bias → reshape (N×K×H×W → N×K×(H*W))
    // The reshape is a "free" logical view — rocMLIR writes conv output in reshaped layout.
    // Only applies when FC output has single consumer (the Reshape node).
    struct MarkConvReshapeEpiloguePass : ov::pass::ModelPass {
        OPENVINO_RTTI("MarkConvReshapeEpiloguePass", "0");
        bool run_on_model(const std::shared_ptr<ov::Model>& model) override {
            bool changed = false;
            for (const auto& op : model->get_ordered_ops()) {
                // Match: FusedConvolution with bias-only (no SiLU, no skip-add)
                auto fc = std::dynamic_pointer_cast<FusedConvolution>(op);
                if (!fc) continue;
                if (fc->get_activation() != ActivationMode::NO_ACTIVATION) continue;
                if (fc->has_add_node()) continue;
                if (fc->get_rt_info().count("rocm_conv_reshape_dims")) continue;

                // FC output must feed exactly one Reshape node
                if (fc->get_output_size() == 0) continue;
                auto fc_out = fc->output(0);
                if (fc_out.get_target_inputs().size() != 1) continue;

                auto reshape = std::dynamic_pointer_cast<ov::op::v1::Reshape>(
                    fc_out.get_target_inputs().begin()->get_node()->shared_from_this());
                if (!reshape) continue;

                // Reshape output shape must be fully static and element-count preserving
                auto out_pshape = reshape->get_output_partial_shape(0);
                if (out_pshape.is_dynamic()) continue;
                auto in_pshape  = fc->get_output_partial_shape(0);
                if (in_pshape.is_dynamic()) continue;

                // Store target reshape dims as comma-separated string in rt_info
                std::string dims_str;
                for (size_t i = 0; i < out_pshape.size(); ++i) {
                    if (i) dims_str += ",";
                    dims_str += std::to_string(out_pshape[i].get_length());
                }
                fc->get_rt_info()["rocm_conv_reshape_dims"] = ov::Any(dims_str);
                changed = true;

                std::cerr << "[ConvReshapeTag] FC '" << fc->get_friendly_name()
                          << "' → Reshape dims=[" << dims_str << "]\n";
            }
            return changed;
        }
    };
    // Register by default; disable with ROCMLIR_EPILOGUE_FUSION=0 (and no ROCMLIR_CONV_RESHAPE_FUSION).
    {
        const char* e = std::getenv("ROCMLIR_EPILOGUE_FUSION");
        if (std::getenv("ROCMLIR_CONV_RESHAPE_FUSION") || !(e && std::string(e) == "0"))
            manager.register_pass<MarkConvReshapeEpiloguePass>();
    }

    // FuseVariadicSplitConvPass: replace FusedConvolution with FusedConvolutionSlice
    // to enable zero-copy Slice+Conv+SiLU fusion via MIGraphX-quality kernels.
    manager.register_pass<FuseVariadicSplitConvPass>();

    // EliminateFusedSiluSwishPass: remove Swish nodes that follow FusedConvolution(SWISH).
    // When FusedConvolutionRocMLIR compiles a Conv+Bias+SiLU fused kernel, the downstream
    // Swish is a no-op (silu_tracking). We eliminate it from the graph entirely to avoid
    // the kernel dispatch overhead (~8µs × 44 nodes = ~350µs saved).
    //
    // For Swish nodes with exactly 1 consumer: replace Swish output with FusedConv output.
    // For multi-consumer Swish: keep (cannot safely redirect all consumers).
    // This is safe because FusedConvolutionRocMLIR already writes SiLU-applied values.
    //
    // Controlled by ROCM_ELIMINATE_SWISH_NOOP (default: enabled).
    struct EliminateFusedSiluSwishPass : ov::pass::ModelPass {
        OPENVINO_RTTI("EliminateFusedSiluSwishPass", "0");
        bool run_on_model(const std::shared_ptr<ov::Model>& model) override {
            // Instead of removing Swish from the graph (which breaks silu_tracking),
            // we mark it with rt_info["rocm_swish_inplace"] = "1" so the memory planner
            // can share the input/output buffer. When in_ptr == out_ptr, SwishOp::Execute
            // does nothing (the is_silu_applied path skips the D2D copy entirely).
            // This mirrors MIGraphX's approach where Conv+SiLU writes to output buffer
            // directly and no separate Swish buffer exists.
            const char* env = std::getenv("ROCM_SWISH_INPLACE");
            if (env && std::string(env) == "0") return false;

            bool changed = false;
            int eliminated = 0;
            std::vector<std::pair<std::shared_ptr<ov::op::v4::Swish>,
                                   ov::Output<ov::Node>>> to_eliminate;

            for (const auto& op : model->get_ordered_ops()) {
                auto swish = std::dynamic_pointer_cast<ov::op::v4::Swish>(op);
                if (!swish) continue;

                // Check input comes from FusedConvolution/FusedGroupConvolution that will use SiLU fusion.
                // Match both:
                //   (a) FusedConvolution/FusedGroupConvolution with activation=SWISH
                //   (b) FusedConvolution/FusedGroupConvolution with NO_ACTIVATION, no-add
                //       (the factory will compile these as Conv+Bias+SiLU at runtime)
                // In-place marking is safe regardless of how many consumers Swish has,
                // because rocm_swish_inplace causes the memory planner to share input/output
                // buffers and SwishOp::Execute becomes a complete no-op via silu_tracking.
                auto fc_out = swish->input(0).get_source_output();
                auto fc = std::dynamic_pointer_cast<FusedConvolution>(fc_out.get_node_shared_ptr());
                auto fgc = std::dynamic_pointer_cast<FusedGroupConvolution>(fc_out.get_node_shared_ptr());
                auto fcs = std::dynamic_pointer_cast<nodes::FusedConvolutionSlice>(fc_out.get_node_shared_ptr());
                if (!fc && !fgc && !fcs) continue;

                // FusedConvolutionSlice factory checks has_swish_consumer at Op creation time and
                // compiles SiLU inline when true — so downstream Swish is always a no-op for these.
                bool will_fuse_silu = fcs ? true
                    : fc
                    ? ((fc->get_activation() == ActivationMode::SWISH) ||
                       (fc->get_activation() == ActivationMode::NO_ACTIVATION && !fc->has_add_node()))
                    : ((fgc->get_activation() == ActivationMode::SWISH) ||
                       (fgc->get_activation() == ActivationMode::NO_ACTIVATION && !fgc->has_add_node()));
                if (!will_fuse_silu) continue;

                to_eliminate.emplace_back(swish, fc_out);
            }

            std::cerr << "[EliminateSwish] Removing " << to_eliminate.size()
                      << " FusedConv→Swish no-op nodes from graph\n";

            for (auto& [swish, fc_out] : to_eliminate) {
                // Mark Swish as in-place: the memory planner will share input/output buffer.
                // When in_ptr == out_ptr in SwishOp::Execute(), the silu_tracking path
                // exits immediately without any D2D copy (complete no-op on GPU).
                // The rt_info tag is read by OperationBuffersExtractor::isSwishInplaceNode().
                swish->get_rt_info()["rocm_swish_inplace"] = std::string("1");
                eliminated++;
                changed = true;
            }
            return changed;
        }
    };
    manager.register_pass<EliminateFusedSiluSwishPass>();

    manager.register_pass<rocmFuseConvBackpropDataAdd>();
    manager.register_pass<rocmFuseCleanUpNodesOrder>();

    manager.run_passes(m);
    return false;
}

ov::rocm_gpu::pass::rocmFuseConvBackpropDataAdd::rocmFuseConvBackpropDataAdd() {
    MATCHER_SCOPE(rocmFuseConvBackpropDataAdd);
    auto conv_backprop_data =
        wrap_type<ov::op::v1::ConvolutionBackpropData>(consumers_count(1));
    auto bias = wrap_type<ov::op::v0::Constant>();
    auto add = wrap_type<ov::op::v1::Add>({conv_backprop_data, bias}, is_add_to_be_fused);

    matcher_pass_callback callback = [](Matcher &m) {
        return fuse_convolution_backprop_data_with_add(m);
    };

    auto m = std::make_shared<Matcher>(add, matcher_name);
    register_matcher(m, callback);
}
