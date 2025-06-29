# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

import functools
import os
import torch_npu
import yaml
from datetime import datetime
import inspect
import logging

# Helper to wrap methods with tracing
def torchnpu_prof_wrapper(original_method, params):
    import torch
    import torch_npu
    save_path           = params.get('save_path', 'profiling_output/')
    profiler_level      = params.get('profiler_level', 'Level0')
    export_type         = params.get('export_type', 'Text')
    msprof_tx           = params.get('msprof_tx', False)
    aic_metrics         = params.get('aic_metrics', 'PipeUtilization')
    l2_cache            = params.get('l2_cache', False)
    op_attr             = params.get('op_attr', False)
    data_simplification = params.get('data_simplification', False)
    record_op_args      = params.get('record_op_args', False)
    activities          = params.get('activities', ['NPU', 'CPU'])
    with_stack          = params.get('with_stack', False)
    record_shapes       = params.get('record_shapes', False)
    profile_memory      = params.get('profile_memory', False)
    with_flops          = params.get('with_flops', False)
    with_modules        = params.get('with_modules', False)
    sched               = params.get('schedule', 
                                        {'wait': 1, 
                                        'warmup': 1, 
                                        'active': 1, 
                                        'repeat': 1, 
                                        'skip_first': 10})

    activity_list = []
    if "NPU" in activities:
        activity_list.append(torch_npu.profiler.ProfilerActivity.NPU)
    if "CPU" in activities:
        activity_list.append(torch_npu.profiler.ProfilerActivity.CPU)


    if profiler_level == "Level0":
        profiler_level = torch_npu.profiler.ProfilerLevel.Level0
    elif profiler_level == "Level2":
        profiler_level = torch_npu.profiler.ProfilerLevel.Level2
    else:
        profiler_level = torch_npu.profiler.ProfilerLevel.Level1

    try:
        aic_metrics = getattr(torch_npu.profiler.AiCMetrics, aic_metrics)
    except ImportError as e:
        logger.warning(f"<<<Failed to import module '{aic_metrics}': {str(e)}, "
                       f"use PipeUtilization instead."
        )
        aic_metrics = torch_npu.profiler.AiCMetrics.PipeUtilization

    experimental_config = torch_npu.profiler._ExperimentalConfig(
        profiler_level=profiler_level,
        aic_metrics=aic_metrics, 
        l2_cache=bool(l2_cache), 
        msprof_tx=bool(msprof_tx), 
        op_attr=bool(op_attr), 
        data_simplification=bool(data_simplification), 
        record_op_args=bool(record_op_args))

    profiler_config = {
        "activities":activity_list,
        "with_stack":bool(with_stack),
        "record_shapes":bool(record_shapes),
        "profile_memory":bool(profile_memory),
        "with_flops":bool(with_flops),
        "with_modules":bool(with_modules),
        "experimental_config":experimental_config,
        "schedule":torch_npu.profiler.schedule(
            wait=sched["wait"], 
            warmup=sched["warmup"], 
            active=sched["active"], 
            repeat=sched["repeat"], 
            skip_first=sched["skip_first"]),
        "on_trace_ready":torch_npu.profiler.tensorboard_trace_handler(save_path)
    }
    if inspect.iscoroutinefunction(original_method):
        logging.info(f"<<<INFO: {original_method.__qualname__} is async function, use async wrapper")
        @functools.wraps(original_method)
        async def async_wrapper(self, *args, **kwargs):
            with torch_npu.profiler.profile(**profiler_config) as prof:
                result = await original_method(self, *args, **kwargs)
                torch.npu.synchronize()
                prof.step()
            return result
        return async_wrapper
    else:
        logging.info(f"<<<INFO: {original_method.__qualname__} is sync function, use sync wrapper")
        @functools.wraps(original_method)
        def wrapper(self, *args, **kwargs):
            with torch_npu.profiler.profile(**profiler_config) as prof:
                result = original_method(self, *args, **kwargs)
                torch.npu.synchronize()
                prof.step()
            return result
        return wrapper


def timer_prof_wrapper(original_method, params):
    if inspect.iscoroutinefunction(original_method):
        logging.info(f"<<<INFO: {original_method.__qualname__} is async function, use async wrapper")
        @functools.wraps(original_method)
        async def async_wrapper(self, *args, **kwargs):
            import time
            import logging
            st = time.time()
            result = await original_method(self, *args, **kwargs)
            duration = time.time() - st
            logging.info(f"<<<Duration of {original_method.__qualname__} is {duration*1000} ms.")
            return result
        return async_wrapper
    else:
        logging.info(f"<<<INFO: {original_method.__qualname__} is sync function, use sync wrapper")
        @functools.wraps(original_method)
        def wrapper(self, *args, **kwargs):
            import time
            import logging
            st = time.time()
            result = original_method(self, *args, **kwargs)
            duration = time.time() - st
            logging.info(f"<<<Duration of {original_method.__qualname__} is {duration*1000} ms.")
            return result
        return wrapper


def viztracer_prof_wrapper(original_method, params):
    # viztracer --combine ./viztracer_output/*.json --output_file ./viztracer_output/combined.json
    from viztracer import VizTracer
    import time
    import uuid
    save_dir         = params.get("save_dir", "./viztracer_output")
    max_stack_depth  = params.get("max_stack_depth", -1)
    tracer_entries   = params.get("tracer_entries", 500000)
    output_format    = params.get("output_format", "json")
    log_async        = params.get("log_async", True)
    pid_suffix       = params.get("pid_suffix", True)

    if inspect.iscoroutinefunction(original_method):
        logging.info(f"<<<INFO: {original_method.__qualname__} is async function, use async wrapper")
        @functools.wraps(original_method)
        async def async_wrapper(self, *args, **kwargs):
            tracer = VizTracer(
                output_file=os.path.join(save_dir,  f"trace_{time.time()}_{uuid.uuid4().hex[:8]}.json"), 
                tracer_entries=tracer_entries,
                max_stack_depth=max_stack_depth,
                log_async=log_async,
                pid_suffix=pid_suffix,
            )
            tracer.start()
            try:
                result = await original_method(self, *args, **kwargs)
            finally:
                tracer.stop()
                tracer.save()
            return result
        return async_wrapper
    else:
        logging.info(f"<<<INFO: {original_method.__qualname__} is sync function, use sync wrapper")
        @functools.wraps(original_method)
        def wrapper(self, *args, **kwargs):
            tracer = VizTracer(
                output_file=os.path.join(save_dir,  f"trace_{time.time()}_{uuid.uuid4().hex[:8]}.json"), 
                tracer_entries=tracer_entries,
                max_stack_depth=max_stack_depth,
                log_async=log_async,
                pid_suffix=pid_suffix,
            )
            tracer.start()
            try:
                result = original_method(self, *args, **kwargs)
            finally:
                tracer.stop()
                tracer.save()
            return result
        return wrapper