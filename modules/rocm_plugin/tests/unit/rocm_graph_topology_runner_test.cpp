// Copyright (C) 2020-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <rocm_graph_topology_runner.hpp>
#include <rocm_simple_execution_delegator.hpp>
#include <ops/parameter.hpp>
#include <ops/result.hpp>

#include "test_networks.hpp"

using namespace ov::rocm_gpu;
using namespace testing;

class rocmGraphTopologyRunnerTest : public Test {
protected:
    static std::map<std::string, std::size_t> PopulateInputIndices(std::shared_ptr<ov::Model> model) {
        std::map<std::string, std::size_t> inputIndices;
        for (const auto& parameter : model->get_parameters()) {
            const auto& parameter_index = model->get_parameter_index(parameter);
            inputIndices.emplace(ParameterOp::GetInputTensorName(*parameter), parameter_index);
        }
        return inputIndices;
    }

    static std::map<std::string, std::size_t> PopulateOutputIndices(std::shared_ptr<ov::Model> model) {
        std::map<std::string, std::size_t> outputIndices;
        for (auto& result : model->get_results()) {
            const auto& result_index = model->get_result_index(result->input_value(0));
            for (const auto& outputName : ResultOp::GetOutputTensorName(*result)) {
                outputIndices.emplace(outputName, result_index);
            }
        }
        return outputIndices;
    }

    static std::vector<std::shared_ptr<ov::Tensor>> PopulateTensors(const std::vector<ov::Output<ov::Node>>& nodes) {
        std::vector<std::shared_ptr<ov::Tensor>> ret;
        for (const auto& node : nodes)
            ret.push_back(std::make_shared<ov::Tensor>(node.get_element_type(), node.get_shape()));
        return ret;
    }

    std::shared_ptr<ov::Model> model_{create_matmul_test_model()};
    CreationContext creationContext_{{}, false};
    ThreadContext threadContext_{{}};
    CancellationToken cancellationToken_{};
    rocmGraphContext rocmGraphContext_{};
    rocmGraphTopologyRunner runner_{creationContext_, model_};
    SimpleExecutionDelegator simpleExecutionDelegator_{};
    std::vector<std::shared_ptr<ov::Tensor>> inputTensors_{PopulateTensors(model_->inputs())};
    std::vector<std::shared_ptr<ov::Tensor>> outputTensors_{PopulateTensors(model_->outputs())};
    std::map<std::string, std::size_t> inputIndeces_{PopulateInputIndices(model_)};
    std::map<std::string, std::size_t> outputIndeces_{PopulateOutputIndices(model_)};
    InferenceRequestContext inferRequestContext_{inputTensors_,
                                                 inputIndeces_,
                                                 outputTensors_,
                                                 outputIndeces_,
                                                 threadContext_,
                                                 cancellationToken_,
                                                 simpleExecutionDelegator_,
                                                 rocmGraphContext_,
                                                 false};
    DeviceMemBlock deviceMemBlock_{runner_.GetSubGraph().memoryManager()->mutableTensorsMemoryModel()};
};

TEST_F(rocmGraphTopologyRunnerTest, InstantiateGraphExec) {
    runner_.UpdateContext(inferRequestContext_, deviceMemBlock_);
    EXPECT_TRUE(inferRequestContext_.getrocmGraphContext().is_initialized());
}

TEST_F(rocmGraphTopologyRunnerTest, BasicRun) {
    EXPECT_NO_THROW(runner_.UpdateContext(inferRequestContext_, deviceMemBlock_));
    EXPECT_NO_THROW(runner_.Run(inferRequestContext_, deviceMemBlock_));
}

TEST_F(rocmGraphTopologyRunnerTest, CheckGraphExecIsInstantiatedOnce) {
    runner_.UpdateContext(inferRequestContext_, deviceMemBlock_);
    const auto oldrocmGraphContext = &inferRequestContext_.getrocmGraphContext();
    runner_.UpdateContext(inferRequestContext_, deviceMemBlock_);
    EXPECT_EQ(&inferRequestContext_.getrocmGraphContext(), oldrocmGraphContext);
}

TEST_F(rocmGraphTopologyRunnerTest, CheckMemcpyNodesArePopulated) {
    runner_.UpdateContext(inferRequestContext_, deviceMemBlock_);
    EXPECT_GT(inferRequestContext_.getrocmGraphContext().get_params_count(), 0);
    EXPECT_GT(inferRequestContext_.getrocmGraphContext().get_results_count(), 0);
}

TEST_F(rocmGraphTopologyRunnerTest, CheckMemcpyNodesAreUpdated) {
    runner_.UpdateContext(inferRequestContext_, deviceMemBlock_);
    rocmGraphContext_.select_current_graph(0);
    const auto& oldCurrentGraph = rocmGraphContext_.get_current_graph();
    ASSERT_FALSE(oldCurrentGraph.is_nested());

    const auto& oldInfo = dynamic_cast<const rocmGraphInfo&>(oldCurrentGraph);
    const auto oldParamNodes = std::map<std::string, rocm::UploadNode>{oldInfo.get_parameter_nodes()};
    const auto oldResultNodes = std::map<std::string, rocm::DownloadNode>{oldInfo.get_result_nodes()};

    std::vector<std::shared_ptr<ov::Tensor>> inputTensors{PopulateTensors(model_->inputs())};
    std::vector<std::shared_ptr<ov::Tensor>> outputTensors{PopulateTensors(model_->outputs())};
    InferenceRequestContext inferRequestContext{inputTensors,
                                                inputIndeces_,
                                                outputTensors,
                                                outputIndeces_,
                                                threadContext_,
                                                cancellationToken_,
                                                simpleExecutionDelegator_,
                                                rocmGraphContext_,
                                                false};
    runner_.UpdateContext(inferRequestContext, deviceMemBlock_);

    rocmGraphContext_.select_current_graph(0);
    const auto& newCurrentGraph = rocmGraphContext_.get_current_graph();
    ASSERT_FALSE(newCurrentGraph.is_nested());

    const auto& newInfo = dynamic_cast<const rocmGraphInfo&>(newCurrentGraph);
    const auto& newParamNodes = newInfo.get_parameter_nodes();
    const auto& newResultNodes = newInfo.get_result_nodes();

    EXPECT_NE(newParamNodes, oldParamNodes);
    EXPECT_NE(newResultNodes, oldResultNodes);
}

TEST_F(rocmGraphTopologyRunnerTest, CheckMemcpyNodesAreNotUpdatedIfPointersUnchanged) {
    runner_.UpdateContext(inferRequestContext_, deviceMemBlock_);
    rocmGraphContext_.select_current_graph(0);
    const auto& oldCurrentGraph = rocmGraphContext_.get_current_graph();
    ASSERT_FALSE(oldCurrentGraph.is_nested());

    const auto& oldInfo = dynamic_cast<const rocmGraphInfo&>(oldCurrentGraph);
    const auto oldParamNodes = std::map<std::string, rocm::UploadNode>{oldInfo.get_parameter_nodes()};
    const auto oldResultNodes = std::map<std::string, rocm::DownloadNode>{oldInfo.get_result_nodes()};

    InferenceRequestContext inferRequestContext{inputTensors_,
                                                inputIndeces_,
                                                outputTensors_,
                                                outputIndeces_,
                                                threadContext_,
                                                cancellationToken_,
                                                simpleExecutionDelegator_,
                                                rocmGraphContext_,
                                                false};
    runner_.UpdateContext(inferRequestContext, deviceMemBlock_);
    rocmGraphContext_.select_current_graph(0);
    const auto& newCurrentGraph = rocmGraphContext_.get_current_graph();
    ASSERT_FALSE(newCurrentGraph.is_nested());

    const auto& newInfo = dynamic_cast<const rocmGraphInfo&>(newCurrentGraph);
    const auto& newParamNodes = newInfo.get_parameter_nodes();
    const auto& newResultNodes = newInfo.get_result_nodes();

    EXPECT_EQ(newParamNodes, oldParamNodes);
    EXPECT_EQ(newResultNodes, oldResultNodes);
}
