# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

from dataclasses import dataclass, field
import json
from vllm.logger import logger
import omni.adaptors.vllm.envs as envs

@dataclass
class ModelParallelConfig:
    dense_mlp_tp_size: int = 1
    embedding_tp_size: int = 1
    
    self_attention_dp_size: int = 1
    self_attention_tp_size: int = 1

    mlaprolog_dp_size: int = 1
    mlaprolog_tp_size: int = 1

    fa_dp_size: int = 1
    fa_tp_size: int = 1

    mlaepilog_dp_size: int = 1
    mlaepilog_tp_size: int = 1
    
    moe_tp_size: int = 1
    moe_ep_size: int = 1
    lm_dp_size: int = 1
    lm_tp_size: int = 1
    pp_size: int = 1
    dp_size: int = 1
    
    redundancy_expert_num: int = 0
    
@dataclass    
class ModelProfilingConfig:
    pass
 
@dataclass
class ModelPrecisionDiffConfig:
    pass
 
@dataclass
class ModelOperatorOptConfig:
    enable_kv_rmsnorm_rope_cache: bool = True
    prefill_dispatch_combine: bool = True
    prefill_enable_mla_alltoall: bool = False
    enable_node_mlp: bool = False
    moe_multi_stream_tune: bool = False
    best_ep: bool = False
    enable_pd_separated: bool = False
    merge_qkv: bool = False
    two_stage_comm: bool = False
    use_chunked_prefill: bool = False
    use_w8a8_dynamic_quant: bool = True
    use_copy_blocks_op: bool = False
    fused_experts_v2: bool = False
    gmm_nz: bool = False
    moe_dispatch_combine: bool = True
    use_omni_placement: bool = False
    omni_placement_config_path:str = None
    enable_fusion_spec: bool = False
    enable_alltoall: bool = False
    enable_moe_expert_parallel: bool = True
    use_a3_high_performance_cann: bool = True
    use_super_kernel: bool = False
    use_mlaprolog: bool = False
    opt_w2_scale_cast: bool = False
    decode_gear_list: list[int] = field(default_factory=lambda: [16])
    
    def __post_init__(self):
        # Check the dependencies of use_omni_placement and omni_placement_config_path
        if self.use_omni_placement and not self.omni_placement_config_path:
            raise ValueError(
                "When use_omni_placement=True, omni_placement_config_path must be provided!"
            )
 
@dataclass      
class ModelExtraConfig:
    parall_config: ModelParallelConfig = field(default_factory=ModelParallelConfig)
    profiling_config: ModelProfilingConfig = field(default_factory=ModelProfilingConfig)
    precision_diff_config: ModelPrecisionDiffConfig = field(default_factory=ModelPrecisionDiffConfig)
    operator_opt_config: ModelOperatorOptConfig = field(default_factory=ModelOperatorOptConfig)
    model_extra_cfg_path: str = ""
    

def init_model_extra_config() -> ModelExtraConfig:
    model_config = ModelExtraConfig()
    model_extra_cfg_path = envs.MODEL_EXTRA_CFG_PATH

    try:
        with open(model_extra_cfg_path, 'r') as f:
            config_data = json.load(f)
        # Recursively create nested objects
        parall_config = ModelParallelConfig(**config_data['model_parallel_config'])
        operator_opt_configonfig = ModelOperatorOptConfig(**config_data['operator_optimizition_config'])
        model_config = ModelExtraConfig(
                parall_config=parall_config,
                operator_opt_config=operator_opt_configonfig,
                model_extra_cfg_path=model_extra_cfg_path)
    except FileNotFoundError:
        logger.warning(f"[WARNING] Config file not found: {model_extra_cfg_path}, using default configuration.")
    except json.JSONDecodeError as e:
        raise RuntimeError(f"[ERROR] Invalid JSON format in config file: {e}")
    except KeyError as e:
        raise RuntimeError(f"[ERROR] Missing required key in config data: {e}")
    except TypeError as e:
        raise RuntimeError(f"[ERROR] Config structure mismatch or incorrect field types: {e}")
    except Exception as e:
        raise RuntimeError(f"[ERROR] Unexpected error while loading model extra config: {e}")

    return model_config

model_extra_config = init_model_extra_config()
