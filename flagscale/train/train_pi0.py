import argparse
import os
import pathlib
import platform
import random

import etils.epath as epath
import numpy as np
import torch
import torch.distributed as dist

import wandb

from flagscale.runner.utils import logger
# from megatron.energon import WorkerConfig, get_loader, get_train_dataset
# from tools.datasets.qwenvl.data.dataset_helpers_robotics import TaskEncoder
from flagscale.models.pi0.modeling_pi0 import PI0Policy, PI0PolicyConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata

def init_ddp(config):
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


def init_wandb(config, *, resuming: bool, log_code: bool = False, enabled: bool = True):
    if not enabled:
        wandb.init(mode="disabled")
        return

    ckpt_dir = pathlib.Path(config.checkpoint_dir)
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory {ckpt_dir} does not exist.")
    if resuming:
        run_id = (ckpt_dir / "wandb_id.txt").read_text().strip()
        wandb.init(id=run_id, resume="must", project=config.project_name)
    else:
        wandb.init(name=config.exp_name, config=vars(config), project=config.project_name)
        (ckpt_dir / "wandb_id.txt").write_text(wandb.run.id)

    if log_code:
        wandb.run.log_code(epath.Path(__file__).parent.parent)


def main(config):
    device = torch.device("cuda")

    local_rank = init_ddp(config)
    if dist.get_rank() == 0 and local_rank == 0:
        logger.info(f"Running on: {platform.node()}")
        if config.batch_size % torch.cuda.device_count() != 0:
            raise ValueError(
                f"Batch size {config.batch_size} must be divisible by the number of devices {torch.cuda.device_count()}."
            )

        checkpoint_dir = config.checkpoint_dir
        resuming = config.resume
        overwrite = config.ckpt_overwrite

        init_wandb(config, resuming=resuming, enabled=config.wandb_enabled)
    # In this case with the standard configuration for Diffusion Policy, it is equivalent to this:
    delta_timestamps = {
        # Load the previous image and state at -0.1 seconds before current frame,
        # then load current image and state corresponding to 0.0 second.
        "observation.image": [-0.1, 0.0],
        "observation.state": [-0.1, 0.0],
        # Load the previous action (-0.1), the next action to be executed (0.0),
        # and 14 future actions with a 0.1 seconds spacing. All these actions will be
        # used to supervise the policy.
        "action": [-0.1, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4],
    }
    dataset = LeRobotDataset(config.data_path, delta_timestamps=delta_timestamps)
    model_config = PI0PolicyConfig.from_pretrained(config.checkpoint_dir)
    policy = PI0Policy.from_pretrained(
        model_path=config.checkpoint_dir,
        tokenizer_path=config.tokenizer_path,
        stat_path=config.stat_path,
        config=model_config,
    )
    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-4)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        num_workers=4,
        batch_size=64,
        shuffle=True,
        pin_memory=device.type != "cpu",
        drop_last=True,
    )
    step = 0
    done = False
    while not done:
        for batch in dataloader:
            batch = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
            loss, _ = policy.forward(batch)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        if step % config.log_freq == 0:
            logger.info(f"step: {step} loss: {loss.item():.3f}")
            logger.info(f"train_pi0.py batch: {type(batch)}")
            logger.info(f"train_pi0.py batch: {batch}")
            # logger.info(f"train_pi0.py batch: {batch.keys()}")
        step += 1
        if step >= config.training_steps:
            done = True
            break
    policy.save_pretrained(config.output_directory)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoint_path_not_set")
    parser.add_argument("--project-name", type=str, default="default_project")
    parser.add_argument("--exp-name", type=str, default="default_exp")
    parser.add_argument("--data-path", type=str, default="energon data_path not set")
    parser.add_argument("--tokenizer-path", type=str, default="tokenizer_path not set")
    parser.add_argument("--state-key", type=str, default="state_key not set")
    parser.add_argument("--action-key", type=str, default="action_key not set")
    parser.add_argument("--action-token-key", type=str, default="action_token_key not set")
    parser.add_argument("--stat-path", type=str, default="stat_path not set")
    parser.add_argument("--output-directory", type=str, default="output_directory not set")
    

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--tensor-model-parallel-size", type=int, default=1)
    parser.add_argument("--pipeline-model-parallel-size", type=int, default=1)
    parser.add_argument("--context-parallel-size", type=int, default=1)
    parser.add_argument("--train-steps", type=int, default=10000)
    parser.add_argument("--log-freq", type=int, default=100)
    

    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--ckpt-overwrite", action="store_true")
    parser.add_argument("--wandb-enabled", action="store_true")

    config = parser.parse_args()

    logger.info(f"train_pi0.py config: {config}")
    # logger.info(f"train_pi0.py config: {vars(config)}")
    main(config)
