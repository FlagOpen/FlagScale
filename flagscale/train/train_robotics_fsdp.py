import dataclasses
import logging
import platform
import os
import etils.epath as epath
import torch
import numpy as np
import tqdm_loggable.auto as tqdm
import wandb
import ipdb
import random
import functools
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import StateDictType, FullStateDictConfig, ShardedStateDictConfig
from torch.distributed.fsdp import ShardingStrategy
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy
import torch.distributed as dist
from torch.profiler import profile, record_function, ProfilerActivity, schedule, tensorboard_trace_handler

import sys
if "/share/project/lyx/openpi/src" in sys.path:
    sys.path.remove("/share/project/lyx/openpi/src")
if "/share/project/lyx/robotics_final/src" in sys.path:
    sys.path.remove("/share/project/lyx/robotics_final/src")

# sys.path.append("/share/project/fengyupu/robotics_final_copy/src")
sys.path.append("/share/project/hcr/repos/robotics_fsdp/robotics_final_copy/src")

import robotics.models.model as _model
# import openpi.shared.array_typing as at
import robotics.training.utils as training_utils
import robotics.training.checkpoints as _checkpoints
import robotics.training.config as _config
import robotics.training.data_loader as _data_loader
import robotics.training.optimizer as _optimizer
import robotics.training.utils as training_utils
# import robotics.training.weight_loaders as _weight_loaders


def init_ddp(config: _config.TrainConfig):
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    random.seed(config.seed)
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    torch.distributed.init_process_group(backend='nccl', init_method='env://')
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True
    return local_rank


def init_logging():
    """Custom logging format for better readability."""
    level_mapping = {"DEBUG": "D", "INFO": "I", "WARNING": "W", "ERROR": "E", "CRITICAL": "C"}

    class CustomFormatter(logging.Formatter):
        def format(self, record):
            record.levelname = level_mapping.get(record.levelname, record.levelname)
            return super().format(record)

    formatter = CustomFormatter(
        fmt="%(asctime)s.%(msecs)03d [%(levelname)s] %(message)-80s (%(process)d:%(filename)s:%(lineno)s)",
        datefmt="%H:%M:%S",
    )

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        handler = logging.StreamHandler()
        logger.addHandler(handler)
    logger.handlers[0].setFormatter(formatter)

def init_wandb(config: _config.TrainConfig, *, resuming: bool, log_code: bool = False, enabled: bool = True):
    if not enabled:
        wandb.init(mode="disabled")
        return

    ckpt_dir = config.checkpoint_dir
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory {ckpt_dir} does not exist.")
    if resuming:
        run_id = (ckpt_dir / "wandb_id.txt").read_text().strip()
        wandb.init(id=run_id, resume="must", project=config.project_name)
    else:
        wandb.init(
            name=config.exp_name,
            config=dataclasses.asdict(config),
            project=config.project_name,
        )
        (ckpt_dir / "wandb_id.txt").write_text(wandb.run.id)

    if log_code:
        wandb.run.log_code(epath.Path(__file__).parent.parent)


def is_llm(name: str) -> bool:
    """Check if the model name corresponds to a large language model."""
    return name.startswith("llm.") and ("mlp.0" in name or "0.weight" in name or "0.bias" in name)

def is_vit(name: str) -> bool:
    """Check if the model name corresponds to a vision transformer."""
    return name.startswith("img.")

def is_action_expert(name: str) -> bool:
    """Check if the model name corresponds to a large language model."""
    return name.startswith("llm.") and ("mlp.1" in name or "1.weight" in name or "1.bias" in name)

# Policy 3: 只包装包含可训练参数（requires_grad=True）的模块。
def freeze_aware_wrap_policy(module, recurse: bool, nonwrapped_numel: int) -> bool:
    has_trainable_params = any(p.requires_grad for p in module.parameters(recurse=False) if p.requires_grad)
    return has_trainable_params

def init_train_state(config: _config.TrainConfig, *, resume: bool, local_rank: int):
    model = config.model.create()
    model = config.weight_loader.load(model).to(
        device=f"cuda:{local_rank}", dtype=getattr(torch, config.training_dtype))
    
    # if config.freeze_vlm:
    for n, p in model.named_parameters():
        if is_vit(n) or is_llm(n):
            print(f"is vit/llm: {n}")
            # p.requires_grad = False

    # FSDP
    from torch.distributed.fsdp.wrap import (
        size_based_auto_wrap_policy,
        transformer_auto_wrap_policy,
        lambda_auto_wrap_policy
    )
    from robotics.models.qwen import Qwen2_5_VLDecoderLayer

    # Combine multiple policies: wrap transformer layers AND large layers
    def combined_auto_wrap_policy(module, recurse, nonwrapped_numel):
        # Exclusion rules: Never wrap these module types
        module_name = module.__class__.__name__

        # Exclude vision transformer components that have in-place operations
        # vision_modules_to_exclude = [
        #     # 'Qwen2VLVisionModel',
        #     # 'Qwen2VLVisionEmbeddings',
        #     # 'Qwen2VLVisionEncoder',
        #     # 'Qwen2VLVisionPatchEmbed',
        #     'Qwen2RMSNorm',  # This is likely what's causing the in-place operation error
        # ]

        # if any(excluded in module_name for excluded in vision_modules_to_exclude):
        #     return False

        # Exclude entire vision model to avoid in-place operation issues
        # if hasattr(module, '__class__'):
        #     class_name = module.__class__.__name__
        #     module_path = module.__class__.__module__

        #     # Exclude all Qwen2_5_VL vision components
        #     if 'qwen2_5_vl' in module_path.lower() and any(x in class_name.lower() for x in ['vision', 'rmsnorm', 'norm']):
        #         if local_rank == 0:
        #             print(f"EXCLUDING vision module: {class_name} from {module_path}")
        #         return False

        #     # Also try excluding by checking for the specific problematic operation
        #     if hasattr(module, 'weight') and hasattr(module, 'forward'):
        #         # This catches normalization layers that do in-place-like operations
        #         if 'norm' in class_name.lower():
        #             if local_rank == 0:
        #                 print(f"EXCLUDING norm module: {class_name}")
        #             return False

        # Policy 1: Wrap transformer layers
        transformer_policy = functools.partial(
            transformer_auto_wrap_policy,
            transformer_layer_cls={Qwen2_5_VLDecoderLayer}
        )

        # Policy 2: Wrap layers with 10K+ parameters to ensure linear layers get wrapped
        size_policy = functools.partial(
            size_based_auto_wrap_policy,
            min_num_params=10_000  # Low threshold to catch linear layers like state_proj
        )

        # Return True if either policy wants to wrap
        return transformer_policy(module, recurse, nonwrapped_numel) or size_policy(module, recurse, nonwrapped_numel)

    print(f"init_train_state model: {model}")

    # for name, param in model.named_parameters():
    #     print(f"name: {name}, param.shape: {param.shape}")

    # auto_wrap_policy = combined_auto_wrap_policy
    # auto_wrap_policy = functools.partial(size_based_auto_wrap_policy, min_num_params=100_000_000)
    auto_wrap_policy = functools.partial(transformer_auto_wrap_policy, transformer_layer_cls={Qwen2_5_VLDecoderLayer})
    # auto_wrap_policy = freeze_aware_wrap_policy
    model = FSDP(
        model,
        auto_wrap_policy=auto_wrap_policy,
        device_id=local_rank,
        use_orig_params=True,
        sync_module_states=True,
        forward_prefetch=True,
        sharding_strategy=ShardingStrategy._HYBRID_SHARD_ZERO2,
        # sharding_strategy=ShardingStrategy.NO_SHARD,
    )

    # for name, param in model.named_parameters():
    #     print(f"after fsdp name: {name}, param.shape: {param.shape}")

    # print(f"Before optimizer: llm.layers.0.post_attention_layernorm.weight.shape = {model.llm.layers[0].post_attention_layernorm.weight.shape}")

    # Store reference to FSDP wrapper in the underlying model for safe_forward
    # model.module._fsdp_wrapper = model

    # optimizer = torch.optim.AdamW(model.parameters(), lr=2.5e-5, weight_decay=0.01)
    # optimizer, scheduler, clip_gradient_norm = _optimizer.create_optimizer(config.optimizer, config.lr_schedule, model.parameters())
    optimizer, scheduler, clip_gradient_norm = _optimizer.create_optimizer(config.optimizer, config.lr_schedule, [p for p in model.parameters() if p.requires_grad])
    print(f"opt_params: {len([p for p in model.parameters() if p.requires_grad])}/{len([p for p in model.parameters()])}")

    # print(f"After optimizer: llm.layers.0.post_attention_layernorm.weight.shape = {model.llm.layers[0].post_attention_layernorm.weight.shape}")

    train_state = training_utils.TrainState(
        step=0,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        clip_gradient_norm=clip_gradient_norm,
        ema_decay=None,
        ema_params=None,
    )
    return train_state

def train_step(
    config: _config.TrainConfig,
    state: training_utils.TrainState,
    batch: tuple[_model.Observation, _model.Actions],
    profiler=None,
    enable_profiling=False,
):
    enable_profiling = False
    if enable_profiling and profiler is not None:
        with profiler:
            with record_function("train_step"):
                return _train_step_internal(config, state, batch)
    else:
        return _train_step_internal(config, state, batch)


def _train_step_internal(
    config: _config.TrainConfig,
    state: training_utils.TrainState,
    batch: tuple[_model.Observation, _model.Actions],
):
    model = state.model
    optimizer = state.optimizer
    scheduler = state.scheduler
    clip_gradient_norm = state.clip_gradient_norm
    ema_decay = state.ema_decay
    ema_params = state.ema_params

    model.train()
    observation, actions = batch
    optimizer.zero_grad()

    chunked_loss = model(
        observation.to(dtype=getattr(torch, config.training_dtype), device="cuda"),
        actions.to(dtype=getattr(torch, config.training_dtype), device="cuda")
    )
    loss = torch.mean(chunked_loss)
    loss.backward()

    from torch.distributed.fsdp.fully_sharded_data_parallel import FullyShardedDataParallel as FSDP
    if isinstance(model, FSDP):
        model.clip_grad_norm_(max_norm=clip_gradient_norm)
    else:
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_gradient_norm)
    optimizer.step()
    scheduler.step()
    # print(f"Step {state.step}: lr={scheduler.get_last_lr()[0]:.8f}, loss={loss.item():.4f}")

    # EMA update #TODO: check if this is correct，先不用EMA了
    # if ema_decay is not None and ema_params is not None:
    #     with torch.no_grad():
    #         for name, param in model.named_parameters():
    #             if name in ema_params:
    #                 ema_params[name].mul_(ema_decay).add_(param.data, alpha=1 - ema_decay)
    #             else:
    #                 ema_params[name] = param.data.clone()

    def is_kernel(name, param):
        return (
            param.ndim > 1 and
            not any(x in name for x in ["bias", "scale", "pos_embedding", "input_embedding"])
        )

    # kernel_params = [p for n, p in model.named_parameters() if is_kernel(n, p)]
    # param_norm = float(torch.norm(torch.stack([torch.norm(p.detach()) for p in kernel_params])))
    # grad_norm = float(torch.norm(torch.stack([torch.norm(p.grad.detach()) for p in kernel_params if p.grad is not None])))

    # 更新训练状态
    new_state = dataclasses.replace(
        state,
        step=state.step + 1,
    )

    info = {
        "loss": float(loss.detach().cpu()),
        # "grad_norm": grad_norm,
        # "param_norm": param_norm,
    }
    return new_state, info


def main(config: _config.TrainConfig):
    local_rank = init_ddp(config)
    master_rank = dist.get_rank() == 0
    if master_rank:
        init_logging()
        logging.info(f"Running on: {platform.node()}")
        if config.batch_size % torch.cuda.device_count() != 0:
            raise ValueError(
                f"Batch size {config.batch_size} must be divisible by the number of devices {torch.cuda.device_count()}."
            )

        checkpoint_dir, resuming = _checkpoints.initialize_checkpoint_dir(
            config.checkpoint_dir,
            keep_period=config.keep_period,
            overwrite=config.overwrite,
            resume=config.resume,
        )

        # init_wandb(config, resuming=resuming, enabled=config.wandb_enabled)
    data_loader = _data_loader.create_data_loader(config, shuffle=True)
    data_iter = iter(data_loader)
    batch = next(data_iter)

    # logging.info(f"Initialized data loader:\n{training_utils.array_tree_to_info(batch)}")

    # Log images from first batch to sanity check.
    # images_to_log = [
    #     wandb.Image(np.concatenate([np.array(img[i]) for img in batch[0].images.values()], axis=1))
    #     for i in range(min(5, len(next(iter(batch[0].images.values())))))
    # ]
    # wandb.log({"camera_views": images_to_log}, step=0)
    train_state = init_train_state(config, resume=config.resume, local_rank=local_rank)
    torch.cuda.synchronize()
    # Ensure all processes are synchronized after FSDP initialization
    dist.barrier()

    # Debug: Check model parameters
    # if master_rank:
    #     total_params = sum(p.numel() for p in train_state.model.parameters())
    #     logging.info(f"Model initialized with {total_params:,} parameters")

    # logging.info(f"Initialized train state:\n{training_utils.array_tree_to_info(train_state.params)}")

    if config.resume:
        train_state = _checkpoints.restore_state(config.checkpoint_dir, train_state)

    start_step = int(train_state.step)
    pbar = tqdm.tqdm(
        range(start_step, config.num_train_steps),
        initial=start_step,
        total=config.num_train_steps,
        dynamic_ncols=True,
    )

    # Initialize profiler for performance analysis
    profiler = None
    enable_profiling = True # os.environ.get("ENABLE_PROFILING", "false").lower() == "true"

    if enable_profiling and master_rank:
        # Create profiler logs directory for each node
        profiler_log_dir = f"./profiler_logs/fsdp_training_2nodes/rank_{dist.get_rank()}"
        os.makedirs(profiler_log_dir, exist_ok=True)
        logging.info(f"Profiler logs will be saved to: {os.path.abspath(profiler_log_dir)}")

        profiler = profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            schedule=schedule(
                wait=0,      # Start profiling immediately
                warmup=0,    # No warmup needed
                active=5,    # Profile for 5 steps
                repeat=1     # Repeat 1 time
            ),
            on_trace_ready=tensorboard_trace_handler(profiler_log_dir),
            record_shapes=True,
            with_stack=True,
            profile_memory=True,
        )
        profiler.start()
        logging.info(f"Profiler started successfully on rank {dist.get_rank()}")
    elif master_rank:
        logging.info("Profiler not enabled")

    infos = []
    for step in pbar:
        # torch.autograd.set_detect_anomaly(True)
        train_state, info = train_step(config, train_state, batch, profiler, enable_profiling)
        infos.append(info)
        if master_rank and step % config.log_interval == 0:
            stacked_infos = {k: np.stack([info[k] for info in infos], axis=0) for k in infos[0]}
            reduced_info = {k: float(v.mean()) for k, v in stacked_infos.items()}
            info_str = ", ".join(f"{k}={v:.4f}" for k, v in reduced_info.items())
            pbar.write(f"Step {step}: {info_str}")
            # wandb.log(reduced_info, step=step)
            infos = []
        batch = next(data_iter)

        if isinstance(train_state.model, FSDP):
            if (step % config.save_interval == 0 and step > start_step) or step == config.num_train_steps - 1:
                # with FSDP.state_dict_type(train_state.model, StateDictType.SHARDED_STATE_DICT, ShardedStateDictConfig(offload_to_cpu=True)):
                with FSDP.state_dict_type(train_state.model, StateDictType.FULL_STATE_DICT, FullStateDictConfig(offload_to_cpu=True, rank0_only=True)):
                    torch.save(train_state.model.state_dict(), os.path.join(config.checkpoint_dir, f"model_step_{step}.pt"))
        elif master_rank and (step % config.save_interval == 0 and step > start_step) or step == config.num_train_steps - 1:
            # _checkpoints.save_state(config.checkpoint_dir, train_state, data_loader)
            # save_dir = os.path.join(self.args.save_dir, f"epoch_{epoch}")
            # os.makedirs(save_dir, exist_ok=True)
            if isinstance(train_state.model, FSDP):
                with FSDP.state_dict_type(train_state.model, StateDictType.FULL_STATE_DICT, FullStateDictConfig(offload_to_cpu=True, rank0_only=True)):
                    torch.save(train_state.model.state_dict(), os.path.join(config.checkpoint_dir, f"model_step_{step}.pt"))
            else:
                torch.save(train_state.model.state_dict(), os.path.join(config.checkpoint_dir, f"model_step_{step}.pt"))
        dist.barrier()

    # Stop profiler if it was running
    if enable_profiling and profiler is not None and master_rank:
        profiler.stop()

    # logging.info("Waiting for checkpoint manager to finish")
    # checkpoint_manager.wait_until_finished()


if __name__ == "__main__":
    main(_config.cli())
