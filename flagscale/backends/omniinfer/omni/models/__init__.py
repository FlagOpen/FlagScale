# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

from omni.adaptors.vllm.patches import model_patch 
from vllm import ModelRegistry
import os

import os
if os.getenv("PROFILING_NAMELIST", None):
    print("<<<Profiler patch environmental variable is enabled, applying profiler patches.")
    from omni.adaptors.vllm.patches.profiler_patches import apply_profiler_patches


def register_model():

    ModelRegistry.register_model(
        "DeepseekV2ForCausalLM",
        "omni.models.deepseek.deepseek_v2:CustomDeepseekV2ForCausalLM")

    ModelRegistry.register_model(
        "DeepseekV3ForCausalLM",
        "omni.models.deepseek.deepseek_v3:DeepseekV3ForCausalLM")
 
    ModelRegistry.register_model(
        "DeepSeekMTPModel",
        "omni.models.deepseek.deepseek_mtp:DeepseekV3MTP")