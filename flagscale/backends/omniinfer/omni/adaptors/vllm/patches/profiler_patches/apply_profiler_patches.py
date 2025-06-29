# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

import importlib
import sys
import os
from pathlib import Path
import logging
import yaml
from .prof_wrapper import torchnpu_prof_wrapper, timer_prof_wrapper, viztracer_prof_wrapper


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

wrapper_dict = {
    "torchnpu": torchnpu_prof_wrapper, 
    "timer": timer_prof_wrapper, 
    "viztracer": viztracer_prof_wrapper
}

# Parse config from namelist, apply profiler monkey patch
def apply_patches(namelist_path: str):
    try:
        namelist_file = Path(__file__).parent / namelist_path

        # Load namelist
        with namelist_file.open('r') as f:
            config = yaml.safe_load(f)

        profiler_type = config.get('type')
        if not (profiler_type=='torchnpu' or 
                profiler_type=='timer' or 
                profiler_type=='viztracer'):
            logger.error(f"<<<type of namelist invalid, should be one of torchnpu/timer/viztracer")
            raise RuntimeError("<<<type of namelist invalid, should be one of torchnpu/timer/viztracer")
        logger.info(f"<<<Applying {profiler_type} profiler patches from {namelist_path}")
        wrapper_method = wrapper_dict[profiler_type]
        
        base_params = config.get("base_params")

        # Extract target modules and methods
        targets: List[Tuple[str, Optional[str]]] = []
        for target in config.get('targets', []):
            module_name = target.get('module')
            class_name = None
            if ":" in module_name:
                module_name, class_name = module_name.split(":")
            function_name = target.get('function_name')
            if module_name:
                targets.append((module_name, class_name, function_name))
            else:
                logger.warning(f"<<<Skipping target with missing 'module': {target}")

        if not targets:
            logger.warning(f"<<<No valid targets found in {namelist_path}")
            return

        for module_name, class_name, function_name in targets:
            logger.info(f"<<<Patching {module_name}.{function_name or 'all methods'}")
            try:
                original_module = importlib.import_module(module_name)
                if class_name:
                    try:
                        target_class = getattr(original_module, class_name)
                        try:
                            original_function = getattr(target_class, function_name)
                            wrapped_function = wrapper_method(original_function, base_params)
                            setattr(target_class, function_name, wrapped_function)
                            logger.info(f"<<<<{module_name}.{class_name}.{function_name} is wrapped")
                        except AttributeError:
                            logger.warning(
                                f"<<<Function '{function_name}' not found in class '{class_name}' "
                                f"of module '{module_name}'"
                            )
                            continue
                    except AttributeError:
                        logger.warning(f"<<<Class '{class_name}' not found in module '{module_name}'")
                        continue
                else:
                    try:
                        original_function = getattr(original_module, function_name)
                        wrapped_function = wrapper_method(original_function, base_params)
                        setattr(original_module, function_name, wrapped_function)
                        logger.info(f"<<<<{module_name}.{function_name} is wrapped")
                    except AttributeError:
                        logger.warning(f"<<<Function '{function_name}' not found in module '{module_name}'")
                        continue
            except ImportError as e:
                logger.warning(f"<<<Failed to import module '{module_name}': {str(e)}")
                continue
            except Exception as e:
                logger.warning(
                    f"<<<Unexpected error while wrapping {module_name}.{class_name or ''}."
                    f"{function_name}: {str(e)}"
                )
                continue

    except (FileNotFoundError, ImportError, AttributeError, RuntimeError, yaml.YAMLError) as e:
        logger.error(f"<<<Failed to apply model patches: {e}")
        raise


profiling_namelist = os.getenv("PROFILING_NAMELIST", None)
if os.path.isfile(profiling_namelist):
    apply_patches(profiling_namelist)
else:
    logger.error(f"'{profiling_namelist}' does not exist.")
    raise FileNotFoundError(f"'{profiling_namelist}' does not exist.")