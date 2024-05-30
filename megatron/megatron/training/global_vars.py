# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

"""Megatron global variables."""

import os
import sys
import torch

from megatron.training import dist_signal_handler
from megatron.core import Timers
from megatron.training.tokenizer import build_tokenizer
from .microbatches import build_num_microbatches_calculator
from .microbatches_hetero import build_num_microbatches_calculator_hetero
from .hetero_context import HeteroContext

_GLOBAL_ARGS = None
_GLOBAL_NUM_MICROBATCHES_CALCULATOR = None
_GLOBAL_TOKENIZER = None
_GLOBAL_TENSORBOARD_WRITER = None
_GLOBAL_WANDB_WRITER = None
_GLOBAL_ONE_LOGGER = None
_GLOBAL_ADLR_AUTORESUME = None
_GLOBAL_TIMERS = None
_GLOBAL_SIGNAL_HANDLER = None
_GLOBAL_HETERO_CONTEXT = None
_GLOBAL_DEVICE_TYPE = None

def get_args():
    """Return arguments."""
    _ensure_var_is_initialized(_GLOBAL_ARGS, 'args')
    return _GLOBAL_ARGS


def get_num_microbatches():
    return _GLOBAL_NUM_MICROBATCHES_CALCULATOR.get()


def get_current_global_batch_size():
    return _GLOBAL_NUM_MICROBATCHES_CALCULATOR.get_current_global_batch_size()


def update_num_microbatches(consumed_samples, consistency_check=True):
    _GLOBAL_NUM_MICROBATCHES_CALCULATOR.update(consumed_samples,
                                               consistency_check)


def get_tokenizer():
    """Return tokenizer."""
    _ensure_var_is_initialized(_GLOBAL_TOKENIZER, 'tokenizer')
    return _GLOBAL_TOKENIZER


def get_tensorboard_writer():
    """Return tensorboard writer. It can be None so no need
    to check if it is initialized."""
    return _GLOBAL_TENSORBOARD_WRITER


def get_wandb_writer():
    """Return wandb writer. It can be None so no need
    to check if it is initialized."""
    return _GLOBAL_WANDB_WRITER


def get_one_logger():
    """Return one logger. It can be None so no need
    to check if it is initialized."""
    return _GLOBAL_ONE_LOGGER

def get_adlr_autoresume():
    """ADLR autoresume object. It can be None so no need
    to check if it is initialized."""
    return _GLOBAL_ADLR_AUTORESUME


def get_timers():
    """Return timers."""
    _ensure_var_is_initialized(_GLOBAL_TIMERS, 'timers')
    return _GLOBAL_TIMERS


def get_signal_handler():
    _ensure_var_is_initialized(_GLOBAL_SIGNAL_HANDLER, 'signal handler')
    return _GLOBAL_SIGNAL_HANDLER


def get_hetero_context():
    """Return heterogenous context."""
    _ensure_var_is_initialized(_GLOBAL_HETERO_CONTEXT, 'hetero context')
    return _GLOBAL_HETERO_CONTEXT


def _set_signal_handler():
    global _GLOBAL_SIGNAL_HANDLER
    _ensure_var_is_not_initialized(_GLOBAL_SIGNAL_HANDLER, 'signal handler')
    _GLOBAL_SIGNAL_HANDLER = dist_signal_handler.DistributedSignalHandler().__enter__()


def set_global_variables(args, build_tokenizer=True):
    """Set args, tokenizer, tensorboard-writer, adlr-autoresume, and timers."""

    assert args is not None

    _ensure_var_is_not_initialized(_GLOBAL_ARGS, 'args')
    set_args(args)

    _build_num_microbatches_calculator(args)
    if build_tokenizer:
        _ = _build_tokenizer(args)
    _set_adlr_autoresume(args)
    _set_timers(args)

    if args.exit_signal_handler:
        _set_signal_handler()


def set_global_writers(args):
    """Set tensorboard-writer and wandb writer.

    Note that this function should be called after calling finish_mpu_init.
    This is because we can know which rank is the last one after the rank mapping in finish_mpu_init.
    """

    assert args is not None

    _ensure_var_is_initialized(_GLOBAL_ARGS, 'args')

    from .utils import is_last_rank
    if is_last_rank(): 
        _set_tensorboard_writer(args)
        _set_one_logger(args)

    # build wandb writers for all processes in the dp group of the last rank 
    from megatron.core import mpu 
    size = torch.distributed.get_world_size(mpu.get_model_parallel_group())
    ranks_tensor = torch.tensor([0 for _ in range(size)], dtype=torch.int, device='cuda')
    if is_last_rank():
        ranks_list = torch.distributed.get_process_group_ranks(mpu.get_model_parallel_group())
        ranks_tensor = torch.tensor(ranks_list, dtype=torch.int, device='cuda') 
    torch.distributed.all_reduce(ranks_tensor)
    if torch.distributed.get_rank() in ranks_tensor.tolist(): 
        _set_wandb_writer(args)


def set_args(args):
    global _GLOBAL_ARGS
    _GLOBAL_ARGS = args


def _build_num_microbatches_calculator(args):

    global _GLOBAL_NUM_MICROBATCHES_CALCULATOR
    _ensure_var_is_not_initialized(_GLOBAL_NUM_MICROBATCHES_CALCULATOR,
                                   'num microbatches calculator')

    if args.hetero_mode != "dp":
        _GLOBAL_NUM_MICROBATCHES_CALCULATOR = build_num_microbatches_calculator(
            args)
    else:
        _GLOBAL_NUM_MICROBATCHES_CALCULATOR = build_num_microbatches_calculator_hetero(
            args)


def _build_tokenizer(args):
    """Initialize tokenizer."""
    global _GLOBAL_TOKENIZER
    _ensure_var_is_not_initialized(_GLOBAL_TOKENIZER, 'tokenizer')
    _GLOBAL_TOKENIZER = build_tokenizer(args)
    return _GLOBAL_TOKENIZER


def rebuild_tokenizer(args):
    global _GLOBAL_TOKENIZER
    _GLOBAL_TOKENIZER = None
    return _build_tokenizer(args)


def _set_tensorboard_writer(args):
    """Set tensorboard writer."""
    global _GLOBAL_TENSORBOARD_WRITER
    _ensure_var_is_not_initialized(_GLOBAL_TENSORBOARD_WRITER,
                                   'tensorboard writer')

    if hasattr(args, 'tensorboard_dir') and \
       args.tensorboard_dir:
        try:
            from torch.utils.tensorboard import SummaryWriter
            print('> setting tensorboard ...')
            _GLOBAL_TENSORBOARD_WRITER = SummaryWriter(
                log_dir=args.tensorboard_dir,
                max_queue=args.tensorboard_queue_size)
        except ModuleNotFoundError:
            print('WARNING: TensorBoard writing requested but is not '
                  'available (are you using PyTorch 1.1.0 or later?), '
                  'no TensorBoard logs will be written.', flush=True)


def _set_wandb_writer(args):
    global _GLOBAL_WANDB_WRITER
    _ensure_var_is_not_initialized(_GLOBAL_WANDB_WRITER,
                                   'wandb writer')
    if getattr(args, 'wandb_project', ''):
        if args.wandb_exp_name == '':
            raise ValueError("Please specify the wandb experiment name!")

        import wandb
        if args.wandb_save_dir:
            save_dir = args.wandb_save_dir
        else:
            # Defaults to the save dir.
            save_dir = os.path.join(args.save, 'wandb')
        rank = torch.distributed.get_rank()
        name = f'{args.wandb_exp_name}-rank{rank}'
        group = args.wandb_exp_name
        wandb_kwargs = {
            'dir': save_dir,
            'name': name,
            'group': group,
            'project': args.wandb_project,
            'mode': 'offline',
            'config': vars(args)}
        os.makedirs(wandb_kwargs['dir'], exist_ok=True)
        wandb.init(**wandb_kwargs)
        _GLOBAL_WANDB_WRITER = wandb


def _set_one_logger(args):
    global _GLOBAL_ONE_LOGGER
    _ensure_var_is_not_initialized(_GLOBAL_ONE_LOGGER, 'one logger')

    if args.enable_one_logger:
        try:
            from one_logger.core import OneLogger
            config = {
               'project': args.one_logger_project,
               'entity': args.one_logger_entity,
               'name': args.one_logger_run_name
            }
            one_logger = OneLogger(config=config)
            _GLOBAL_ONE_LOGGER = one_logger
        except BaseException:
            print('WARNING: one_logger package is required to enable e2e metrics '
                  'tracking. Try pip install '
                  '--index-url=https://sc-hw-artf.nvidia.com/api/pypi/hwinf-ml-pypi/simple'
                  ' one_logger to install it')

def _set_adlr_autoresume(args):
    """Initialize ADLR autoresume."""
    global _GLOBAL_ADLR_AUTORESUME
    _ensure_var_is_not_initialized(_GLOBAL_ADLR_AUTORESUME, 'adlr autoresume')

    if args.adlr_autoresume:
        if args.rank == 0:
            print('enabling autoresume ...', flush=True)
        sys.path.append(os.environ.get('SUBMIT_SCRIPTS', '.'))
        try:
            from userlib.auto_resume import AutoResume
        except BaseException:
            print('ADLR autoresume is not available, exiting ...')
            sys.exit()

        _GLOBAL_ADLR_AUTORESUME = AutoResume


def _set_timers(args):
    """Initialize timers."""
    global _GLOBAL_TIMERS
    _ensure_var_is_not_initialized(_GLOBAL_TIMERS, 'timers')
    _GLOBAL_TIMERS = Timers(args.timing_log_level, args.timing_log_option)


def set_hetero_context(args):
    """Initialize heterogenous context."""
    global _GLOBAL_HETERO_CONTEXT
    _ensure_var_is_not_initialized(_GLOBAL_HETERO_CONTEXT, 'hetero context')
    _GLOBAL_HETERO_CONTEXT = HeteroContext(args)


def _ensure_var_is_initialized(var, name):
    """Make sure the input variable is not None."""
    assert var is not None, '{} is not initialized.'.format(name)


def _ensure_var_is_not_initialized(var, name):
    """Make sure the input variable is not None."""
    assert var is None, '{} is already initialized.'.format(name)


def set_device_type(args):
    """Initialize customized device type."""
    global _GLOBAL_DEVICE_TYPE
    _ensure_var_is_not_initialized(_GLOBAL_DEVICE_TYPE, 'device type')
    assert args.device_type is not None
    _GLOBAL_DEVICE_TYPE = args.device_type

    # Add patches package of device_type to sys.path
    base_base_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    path_hard = os.path.join(base_base_path,"hardwares") 
    path = os.path.join(path_hard,args.device_type)
    assert os.path.exists(path), "Path {} does not exist.".format(path) 
    assert os.path.isdir(path), "Path {} is not a directory.".format(path)
    sys.path.append(path)
    
    # Apply the following patch during the import time
    import patches

