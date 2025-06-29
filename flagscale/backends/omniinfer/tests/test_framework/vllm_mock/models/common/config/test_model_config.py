# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

import os
import unittest
from dataclasses import is_dataclass

#TODO: 添加更多的测试用例   
class TestModelAdditionalConfig(unittest.TestCase):
    
    def test_basic_load_json_config(self):
        """测试基本配置加载功能"""
        os.environ['MODEL_EXTRA_CFG_PATH'] = 'test_config.json'
        from model_config import init_model_extra_config
        config = init_model_extra_config()
        
        # 验证配置结构
        self.assertTrue(is_dataclass(config))
        self.assertTrue(is_dataclass(config.parall_config))
        self.assertTrue(is_dataclass(config.operator_opt_config))
        
        # 验证配置值
        self.assertEqual(config.parall_config.dense_mlp_tp_size, 2)
        self.assertEqual(config.parall_config.embedding_tp_size, 3)
        self.assertEqual(config.parall_config.self_attention_dp_size, 4)
        self.assertEqual(config.parall_config.self_attention_tp_size, 5)
        self.assertEqual(config.parall_config.mlaprolog_dp_size, 6)
        self.assertEqual(config.parall_config.mlaprolog_tp_size, 7)
        self.assertEqual(config.parall_config.fa_dp_size, 8)
        self.assertEqual(config.parall_config.fa_tp_size, 9)
        self.assertEqual(config.parall_config.mlaepilog_dp_size, 10)
        self.assertEqual(config.parall_config.mlaepilog_tp_size, 11)
        self.assertEqual(config.parall_config.moe_tp_size, 12)
        self.assertEqual(config.parall_config.moe_ep_size, 13)
        self.assertEqual(config.parall_config.lm_dp_size, 14)
        self.assertEqual(config.parall_config.lm_tp_size, 15)
        self.assertEqual(config.parall_config.pp_size, 16)
        self.assertEqual(config.parall_config.redundancy_expert_num, 17)
        self.assertEqual(config.parall_config.dp_size, 18)
        
        self.assertEqual(config.operator_opt_config.enable_kv_rmsnorm_rope_cache, False)
        self.assertEqual(config.operator_opt_config.prefill_dispatch_combine, False)
        self.assertEqual(config.operator_opt_config.enable_node_mlp, True)
        self.assertEqual(config.operator_opt_config.moe_multi_stream_tune, True)
        self.assertEqual(config.operator_opt_config.best_ep, True)
        self.assertEqual(config.operator_opt_config.enable_pd_separated, True)
        self.assertEqual(config.operator_opt_config.merge_qkv, True)
        self.assertEqual(config.operator_opt_config.two_stage_comm, True)
        self.assertEqual(config.operator_opt_config.use_chunked_prefill, True)
        self.assertEqual(config.operator_opt_config.use_w8a8_dynamic_quant, False)
        self.assertEqual(config.operator_opt_config.use_copy_blocks_op, True)
        self.assertEqual(config.operator_opt_config.fused_experts_v2, True)
        self.assertEqual(config.operator_opt_config.gmm_nz, True)
        self.assertEqual(config.operator_opt_config.moe_dispatch_combine, True)
        self.assertEqual(config.operator_opt_config.use_omni_placement, True)
        self.assertEqual(config.operator_opt_config.omni_placement_config_path, ".")
        self.assertEqual(config.operator_opt_config.enable_fusion_spec, True)
        self.assertEqual(config.operator_opt_config.enable_alltoall, True)
        self.assertEqual(config.operator_opt_config.enable_moe_expert_parallel, False)
        self.assertEqual(config.operator_opt_config.decode_gear_list, [17])
        
    def test_default_config_when_no_json(self):
         # 准备测试数据
        from model_config import init_model_extra_config, envs
        envs.MODEL_EXTRA_CFG_PATH = ""
        config = init_model_extra_config()

        # 验证配置结构
        self.assertTrue(is_dataclass(config))
        self.assertTrue(is_dataclass(config.parall_config))
        self.assertTrue(is_dataclass(config.operator_opt_config))

        self.assertEqual(config.parall_config.dense_mlp_tp_size, 1)
        self.assertEqual(config.parall_config.embedding_tp_size, 1)
        self.assertEqual(config.parall_config.self_attention_dp_size, 1)
        self.assertEqual(config.parall_config.self_attention_tp_size, 1)
        self.assertEqual(config.parall_config.mlaprolog_dp_size, 1)
        self.assertEqual(config.parall_config.mlaprolog_tp_size, 1)
        self.assertEqual(config.parall_config.fa_dp_size, 1)
        self.assertEqual(config.parall_config.fa_tp_size, 1)
        self.assertEqual(config.parall_config.mlaepilog_dp_size, 1)
        self.assertEqual(config.parall_config.mlaepilog_tp_size, 1)
        self.assertEqual(config.parall_config.moe_tp_size, 1)
        self.assertEqual(config.parall_config.moe_ep_size, 1)
        self.assertEqual(config.parall_config.lm_dp_size, 1)
        self.assertEqual(config.parall_config.lm_tp_size, 1)
        self.assertEqual(config.parall_config.pp_size, 1)
        self.assertEqual(config.parall_config.redundancy_expert_num, 0)
        self.assertEqual(config.parall_config.dp_size, 1)

        self.assertEqual(config.operator_opt_config.enable_kv_rmsnorm_rope_cache, True)
        self.assertEqual(config.operator_opt_config.prefill_dispatch_combine, True)
        self.assertEqual(config.operator_opt_config.enable_node_mlp, False)
        self.assertEqual(config.operator_opt_config.moe_multi_stream_tune, False)
        self.assertEqual(config.operator_opt_config.best_ep, False)
        self.assertEqual(config.operator_opt_config.enable_pd_separated, False)
        self.assertEqual(config.operator_opt_config.merge_qkv, False)
        self.assertEqual(config.operator_opt_config.two_stage_comm, False)
        self.assertEqual(config.operator_opt_config.use_chunked_prefill, False)
        self.assertEqual(config.operator_opt_config.use_w8a8_dynamic_quant, True)
        self.assertEqual(config.operator_opt_config.use_copy_blocks_op, False)
        self.assertEqual(config.operator_opt_config.fused_experts_v2, False)
        self.assertEqual(config.operator_opt_config.gmm_nz, False)
        self.assertEqual(config.operator_opt_config.moe_dispatch_combine, True)
        self.assertEqual(config.operator_opt_config.use_omni_placement, False)
        self.assertEqual(config.operator_opt_config.omni_placement_config_path, None)
        self.assertEqual(config.operator_opt_config.enable_fusion_spec, False)
        self.assertEqual(config.operator_opt_config.enable_alltoall, False)
        self.assertEqual(config.operator_opt_config.enable_moe_expert_parallel, True)
        self.assertEqual(config.operator_opt_config.decode_gear_list, [16])

if __name__ == '__main__':
    unittest.main()
