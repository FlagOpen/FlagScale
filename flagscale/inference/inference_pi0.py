import time

import torch

from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.factory import make_policy

from flagscale.inference.arguments import parse_config

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def inference(config):
    config_llm = config.llm
    config_dataset = config.dataset
    config_data_loader = config.data_loader
    config_generate = config.generate

    lerobot_dataset = LeRobotDataset(
        config_dataset.dataset_repo_id,
        episodes=config_dataset.episodes,
        video_backend=config_dataset.video_backend,
    )

    dataloader = torch.utils.data.DataLoader(
        lerobot_dataset,
        num_workers=config_data_loader.num_workers,
        batch_size=config_data_loader.batch_size,
    )

    cfg = PreTrainedConfig.from_pretrained(config_llm.model_path)
    cfg.pretrained_path = config_llm.model_path
    policy = make_policy(cfg, ds_meta=lerobot_dataset.meta)
    if config_llm.compile_model:
        policy = torch.compile(policy, mode=config_llm.compile_mode)

    for _ in range(config_generate.num_inference):
        batch = next(iter(dataloader))
        dtype = eval(config_dataset.get("dtype", "torch.float32"))
        for k in batch:
            if isinstance(batch[k], torch.Tensor):
                batch[k] = batch[k].to(device=DEVICE, dtype=dtype)

        images, img_masks = policy.prepare_images(batch)
        state = policy.prepare_state(batch)
        lang_tokens, lang_masks = policy.prepare_language(batch)

        t_s = time.time()
        actions = policy.model.sample_actions(
            images, img_masks, lang_tokens, lang_masks, state, noise=None
        )
        print(f"sample_actions() latency: {(time.time() - t_s)*1000:.2f} ms")
        original_action_dim = policy.config.action_feature.shape[0]
        actions = actions[:, :, :original_action_dim]
    print("actions: ", actions.shape)


if __name__ == "__main__":
    config = parse_config()
    inference(config)
