# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

# import torch_npu
from pathlib import Path
import unittest
import torch
import torch.nn as nn
import numpy as np
import sys
import os

from unittest.mock import MagicMock, patch

from omni_planner.omni_planner import OmniPlanner
from omni_planner.config import Config
from omni_planner.cluster_status import ClusterStatus

#torch.set_printoptions(threshold=float("inf"))  # Print all values
class TestOmniPlanner(unittest.TestCase):
    def setUp(self):
        self.planner = OmniPlanner("./tests/config_test.yaml")
    
    def tearDown(self):
        OmniPlanner.cleanup()
        if hasattr(self, 'planner'):
            del self.planner

    def test_omni_planner_plan(self):
        # Example input: 3 tokens, 4 experts each, with importance scores
        input_token = torch.tensor([
            [0, 1, 2, 3],  # Token 1
            [1, 0, 3, 2],  # Token 2
            [3, 2, 1, 0]   # Token 3
        ], dtype=torch.float32).npu()

        input_expert_id = torch.tensor([
            [0, 1, 2, 3],  # Token 1 expert
            [1, 0, 3, 2],  # Token 2 expert
            [3, 2, 1, 0]   # Token 3 expert
        ], dtype=torch.long).npu()

        input_expert_score = torch.tensor([
            [0.4, 0.3, 0.2, 0.1],  # Token 1 expert score
            [0.4, 0.2, 0.2, 0.2],  # Token 2 expert score
            [0.7, 0.11, 0.1, 0.09]   # Token 3 expert score
        ], dtype=torch.float32).npu()

        token, token_expert_ids, token_scores = self.planner.plan(layer_idx_moe=0,
                                                                  tokens=input_token,
                                                                  token_expert_ids=input_expert_id,
                                                                  token_expert_scores=input_expert_score)

        print("Input mapping:")
        print(input_token, input_expert_id, input_expert_score)

        print("\nOptimized mapping:")
        print(token, token_expert_ids, token_scores)

        except_expert_ids = torch.tensor([[ 0,  1,  2,  -1],
        [ 1,  0,  3, 2],
        [ 3,  2,  -1,  -1]], dtype=torch.long).npu()
        print(token_expert_ids)

        self.assertTrue(torch.equal(token_expert_ids, except_expert_ids))

    def test_get_deepseek_v3_moe_layer_idx(self):
        self.assertEqual(OmniPlanner.get_deepseek_v3_moe_layer_idx("model.layers.5.mlp.experts"), 2)
        self.assertEqual(OmniPlanner.get_deepseek_v3_moe_layer_idx("model.layers.10.mlp.experts"), 7)

    def test_initialization(self):
        self.assertIsInstance(self.planner.config, Config)
        self.assertIsNotNone(self.planner.expert_mapping)
        self.assertIsInstance(self.planner.cluster_status, ClusterStatus)
        self.assertEqual(len(self.planner.optimizers), 2)
        self.assertIn("AdaRouter", [type(instance).__name__ for instance in self.planner.optimizers])

    def test_total_deployed_experts(self):
        expected_experts = 256
        np.testing.assert_array_equal(self.planner.total_deployed_experts, expected_experts)

    def test_is_expert_on_current_rank(self):
        exists, position = self.planner.is_expert_on_current_rank(0, 0, 0, 3)
        self.assertTrue(exists)
        self.assertEqual(position, 0)

        exists, position = self.planner.is_expert_on_current_rank(0, 100, 0, 3)
        self.assertFalse(exists)
        self.assertEqual(position, 64)

    def test_get_num_of_redundant_experts(self):
        num_of_redundant_experts = self.planner.get_num_of_redundant_experts(0, 63, 0)
        self.assertEqual(num_of_redundant_experts, 1)

    

class TestOmniPlanner_with_heat_optimizer(unittest.TestCase):
    def setUp(self):
        pass
    
    def tearDown(self):
        pass
    
    def test_get_max_num_deployed_expert_per_rank_0(self):
        planner = OmniPlanner("./tests/config_heat.yaml",world_size=64,num_devices_per_host=16)
        self.assertEqual(planner.get_max_num_deployed_expert_per_rank(), 5)
    
    def test_get_max_num_deployed_expert_per_rank_1(self):
        planner = OmniPlanner("./tests/config_heat.yaml",rank=1,world_size=64,num_devices_per_host=16)
        self.assertEqual(planner.get_max_num_deployed_expert_per_rank(), 5)

class TestEmptyPattern(unittest.TestCase):
    def setUp(self):
        # 创建一个空pattern的OmniPlanner实例用于测试
        self.planner = OmniPlanner("./tests/config_empty.yaml")
    
    def tearDown(self):
        OmniPlanner.cleanup()
        del self.planner

    @unittest.skip("singleton instance can not change config.")
    def test_empty_pattern(self):
        self.assertEqual(self.planner.placement_pattern, None)
        self.assertEqual(self.planner.is_expert_on_current_rank(0, 16, 1, 16), (True, 0))

if __name__ == '__main__':
    unittest.main()