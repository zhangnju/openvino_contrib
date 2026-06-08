// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/cc/pass/itt.hpp"
#include "rocm_graph_transformer.hpp"

#include <fmt/format.h>

#include "openvino/core/preprocess/pre_post_process.hpp"
#include "openvino/pass/manager.hpp"
#include "openvino/op/gru_sequence.hpp"
#include "openvino/op/rnn_sequence.hpp"
#include "openvino/op/lstm_sequence.hpp"
#include "transformations/common_optimizations/common_optimizations.hpp"
#include "transformations/common_optimizations/nop_elimination.hpp"
#include "transformations/common_optimizations/shuffle_channels_fusion.hpp"
#include "transformations/convert_precision.hpp"
#include "transformations/fp16_compression/convert_compression_only_to_legacy.hpp"
#include "transformations/fp16_compression/mark_decompression_convert_constant_folding.hpp"
#include "transformations/init_node_info.hpp"
#include "transformations/op_conversions/bidirectional_sequences_decomposition.hpp"
#include "transformations/op_conversions/convert_mod.hpp"
#include "transformations/op_conversions/convert_reduce_to_pooling.hpp"
#include "transformations/op_conversions/convert_sequences_to_tensor_iterator.hpp"
#include "transformations/op_conversions/convert_ti_to_sequences.hpp"
#include "transformer/convolution_asym_padding_transformation.hpp"
#include "transformer/fuse_conv_biasadd_activation.hpp"

#include "bidirectional_lstm_sequence_composition.hpp"
#include "concat_transformation.hpp"
#include "detection_output_fix_input_types_transformation.hpp"
#include "fuse_matmul_add.hpp"
#include "matmul_transformations.hpp"
#include "reduce_transformation.hpp"
#include "remove_duplicated_results_transformation.hpp"
#include "remove_redundant_convert_transformation.hpp"
#include "transformations/op_conversions/convert_divide.hpp"
#include "transformations/op_conversions/convert_interpolate1_to_interpolate4.hpp"
#include "transformations/op_conversions/convert_subtract.hpp"
#include "transformations/op_conversions/convert_gelu.hpp"
#include "transformations/op_conversions/gelu7_downgrade.hpp"
#include "transformations/op_conversions/mvn6_decomposition.hpp"
#include "transformations/op_conversions/hswish_decomposition.hpp"
#include "transformations/common_optimizations/reshape_prelu.hpp"
#include "transformer/rocmlir_conv_decompose_transformation.hpp"
#include "transformer/elementwise_fusion_transformation.hpp"
#include "transformer/variadic_split_zero_copy.hpp"
#include "transformer/rocm_attention_fusion.hpp"

using namespace ov::rocm_gpu;

void GraphTransformer::transform(const rocm::Device& device,
                                 std::shared_ptr<ov::Model>& model,
                                 const Configuration& config) const {
    auto inference_precision = config.get_inference_precision();
    if (inference_precision == ov::element::f16 && !isHalfSupported(device)) {
        inference_precision = ov::element::f32;
    }

    auto upscale_precision = [&]() -> bool {
        return !isHalfSupported(device) || inference_precision == ov::element::f32;
    };
    auto downscale_precision = [&]() -> bool {
        return isHalfSupported(device) && inference_precision == ov::element::f16;
    };

    precisions_map fp_convert_precision_map = {
        {ov::element::f64, ov::element::f32},
        {ov::element::i64, ov::element::i32},  // Most ROCm kernels only support i32 for integer ops
    };
    type_to_fuse_map empty_fuse_map = {};
    if (upscale_precision()) {
        fp_convert_precision_map.insert(std::make_pair(ov::element::f16, ov::element::f32));
    } else if (downscale_precision()) {
        fp_convert_precision_map.insert(std::make_pair(ov::element::f32, ov::element::f16));
    }

    auto pass_config = std::make_shared<ov::pass::PassConfig>();
    ov::pass::Manager pass_manager{pass_config};

    pass_config->enable<ov::pass::ConvertInterpolate1ToInterpolate4>();
    pass_config->disable<ov::pass::MVN6Decomposition>();
    // NOTE: Elementwise decompositions are now disabled because generally their straightforward versions
    // are executed faster on rocm/MIOPEN.
    // However this is not valid for the case with broadcasting of very large shapes (e.g. {{1024, 1024, 384, 2}, {1}})
    // on rocm, for them decomposed MIOPEN versions are faster.
    // TODO: Consider as possible optimisations: enabling these decompositions for large shapes, creating MIOPEN versions
    // for these operations, implementing in-place logic in rocm GPU plugin for these operations.
    pass_config->disable<ov::pass::ConvertSubtract>();
    pass_config->disable<ov::pass::ConvertDivide>();
    pass_config->disable<ov::pass::ConvertMod>();
    pass_config->disable<ov::pass::Gelu7Downgrade>();
    pass_config->disable<ov::pass::ConvertGELU>();
    pass_config->disable<ov::pass::HSwishDecomposition>();
    pass_config->disable<ov::pass::ConvertReduceMaxToPooling>();
    pass_config->disable<ov::pass::ConvertReduceMeanToPooling>();
    pass_config->disable<ov::pass::ConvertReduceSumToPooling>();
    pass_config->disable<ov::pass::ShuffleChannelsFusion>();

    // Skip decomposition for LSTMSequence and GRUSequence
    pass_config->disable<ov::pass::BidirectionalLSTMSequenceDecomposition>();
    pass_config->disable<ov::pass::BidirectionalGRUSequenceDecomposition>();
    // TODO: Uncomment when support for RNNSequence will be added
    //pass_config->disable<ov::pass::BidirectionalRNNSequenceDecomposition>();

    [[maybe_unused]] const auto& originOps = model->get_ordered_ops();
    [[maybe_unused]] const auto& originOpsSize = originOps.size();

    pass_manager.register_pass<ov::pass::InitNodeInfo>();
    pass_manager.register_pass<ov::pass::ConvertPrecision>(fp_convert_precision_map, empty_fuse_map, true, false);
    pass_manager.register_pass<ov::pass::CommonOptimizations>();
    pass_manager.register_pass<ov::pass::ReshapePRelu>();
    // Do we actually need this transformations in plugin?
    // Having duplicated results seems to be rare case in real world.
    // But currently it affects the rocmInferRequest which implementation
    // relies on number of outputs of original model
    // pass_manager.register_pass<ov::rocm_gpu::pass::RemoveDuplicatedResultsTransformation>();
    pass_manager.register_pass<ov::rocm_gpu::pass::RemoveRedundantConvertTransformation>();
    pass_manager.register_pass<ov::rocm_gpu::pass::BidirectionalSequenceComposition>();
    pass_manager.register_pass<ov::pass::ConvertSequenceToTensorIterator>();

    // Sequences supported by the plugin shouldn't be converted to TensorIterator.
    auto is_sequence_primitive_supported = [](const std::shared_ptr<const ov::Node> &node) -> bool {
        if (std::dynamic_pointer_cast<const ov::op::v5::RNNSequence>(node)) {
            return false;
        } else if (const auto &gru_seq = std::dynamic_pointer_cast<const ov::op::v5::GRUSequence>(node)) {
            return (gru_seq->get_clip() == 0.0f &&
                    gru_seq->get_activations() == std::vector<std::string>{"sigmoid", "tanh"} &&
                    (gru_seq->get_input_size() != 1 || gru_seq->get_hidden_size() != 1) &&
                    (gru_seq->get_direction() != ov::op::RecurrentSequenceDirection::REVERSE) &&
                    (gru_seq->get_direction() != ov::op::RecurrentSequenceDirection::BIDIRECTIONAL));
        } else if (const auto &lstm_seq = std::dynamic_pointer_cast<const ov::op::v5::LSTMSequence>(node)) {
            return (lstm_seq->get_clip() == 0.0f &&
                    lstm_seq->get_activations() == std::vector<std::string>{"sigmoid", "tanh", "tanh"} &&
                    lstm_seq->get_activations_alpha() == std::vector<float>{1.0f, 1.0f, 1.0f} &&
                    lstm_seq->get_activations_beta() == std::vector<float>{0.0f, 0.0f, 0.0f} &&
                    (lstm_seq->get_input_size() != 1 || lstm_seq->get_hidden_size() != 1) &&
                    (lstm_seq->get_direction() != ov::op::RecurrentSequenceDirection::REVERSE));
        }
        return false;
    };

    pass_config->set_callback<ov::pass::ConvertRNNSequenceToTensorIterator,
                              ov::pass::ConvertGRUSequenceToTensorIterator,
                              ov::pass::ConvertLSTMSequenceToTensorIterator>(
            [is_sequence_primitive_supported](const std::shared_ptr<const ov::Node> &node) -> bool {
                return is_sequence_primitive_supported(node);
            });

    // NOTE: RocMLIRConvDecompose (K×C group decomposition) was designed to work around
    // a MIOpen GemmFwd1x1 bug on gfx950 (CDNA). It converts 3×3 Convolution into
    // GroupConvolution with G=K*C groups, but this produces shapes with thousands of
    // groups (e.g. G=18432) that neither rocMLIR nor MIOpen can handle stably.
    // On gfx1201 (RDNA 4) rocMLIR handles all conv shapes directly — disable these passes.
    // To re-enable for gfx950 only, guard with: if (device.gcnArch().find("gfx950") != npos)
#ifdef ENABLE_ROCMLIR
    // pass_manager.register_pass<ov::rocm_gpu::pass::RocMLIRConvDecompose>();
    // pass_manager.register_pass<ov::rocm_gpu::pass::RocMLIRGroupConvDecompose>();
#endif

    pass_manager.register_pass<ov::rocm_gpu::pass::ConvolutionAsymPaddingTransformation>();
    pass_manager.register_pass<ov::rocm_gpu::pass::GroupConvolutionAsymPaddingTransformation>();
    pass_manager.register_pass<ov::rocm_gpu::pass::rocmConvolutionFusion>();
    pass_manager.register_pass<ov::rocm_gpu::pass::ConvolutionBackpropDataAsymPaddingTransformation>();
    pass_manager.register_pass<ov::rocm_gpu::pass::GroupConvolutionBackpropDataAsymPaddingTransformation>();
    pass_manager.register_pass<ov::rocm_gpu::pass::FusedConvBackpropDataAsymPaddingTransformation>();
    pass_manager.register_pass<ov::rocm_gpu::pass::TransposeMatMulTransformation>();
    pass_manager.register_pass<ov::rocm_gpu::pass::FullyConnectedTransformation>();
    pass_manager.register_pass<ov::rocm_gpu::pass::ConcatTransformation>();
    pass_manager.register_pass<ov::rocm_gpu::pass::ReduceTransformation>();
    pass_manager.register_pass<ov::rocm_gpu::pass::DetectionOutputFixInputTypesTransformation>();

    // Fuse chains of elementwise ops (Swish, Mul, Add, Sigmoid, etc.) into a single
    // FusedElementwise kernel launch. Run AFTER conv fusion (which handles Conv+SiLU)
    // and AFTER NopElimination. The pass eliminates ~150+ individual kernel launches
    // in yolo26x by reading each intermediate tensor only once.
    // NOTE: ElementwiseFusionPass correctly handles pe(V) Add nodes (attention pe Add) via
    // a fixed aux-input calculation for binary-op chain roots.
    pass_manager.register_pass<ov::rocm_gpu::pass::ElementwiseFusionPass>();

    // Fuse Attention MatMul patterns (Reshape+Split+Transpose+MatMul) via MIGraphX MLIR.
    // Replaces rocBLAS Q*K^T (0.644ms/iter) with fused kernel (0.020ms/iter) = 32× speedup.
    // Controlled by ROCM_FUSE_ATTENTION env var (default: enabled on gfx12xx fp16).
    // pe(V) conv fusion: enabled on gfx1201 (RDNA4) where Form B is verified stable.
    // gfx1100 (RDNA3): rocMLIR depthwise conv for pe generates MIOpen fallback error on gfx1100,
    //   so pe fusion is kept disabled until rocMLIR gfx1100 depthwise support is confirmed.
    // gfx950 (CDNA3/wave64): pe Add absorbed by ElementwiseFusionPass instead.
    {
        const std::string arch = device.props().gcnArchName;
        const bool enable_pe_fusion = arch.find("gfx1201") != std::string::npos;
        pass_manager.register_pass<ov::rocm_gpu::pass::RocmAttentionFusionPass>(arch, enable_pe_fusion);
    }

    // Zero-copy VariadicSplit: replace channel-axis VariadicSplit with VariadicSplitAlias.
    // The memory planner (rocm_op_buffers_extractor) assigns alias outputs as sub-ranges
    // of the input buffer, eliminating GPU data copies entirely (as MIGraphX does with
    // its load[offset] mechanism). Enabled when ROCM_ZEROCOPY_SPLIT != "0" (default: on).
    pass_manager.register_pass<ov::rocm_gpu::pass::VariadicSplitZeroCopyPass>();

    // Do we actually need to eliminate broadcast one more time at the end?
    pass_manager.register_pass<ov::pass::NopElimination>();

    pass_manager.run_passes(model);

    return;
}
