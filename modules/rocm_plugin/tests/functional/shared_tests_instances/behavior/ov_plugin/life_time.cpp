// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "behavior/ov_plugin/life_time.hpp"

#include <rocm_test_constants.hpp>

using namespace ov::test::behavior;

namespace {

INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTests,
                         OVHoldersTest,
                         ::testing::Values(ov::test::utils::DEVICE_ROCM),
                         OVHoldersTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTests,
                         OVHoldersTestOnImportedNetwork,
                         ::testing::Values(ov::test::utils::DEVICE_ROCM, "HETERO:rocm"),
                         OVHoldersTestOnImportedNetwork::getTestCaseName);

}  // namespace
