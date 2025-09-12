import argparse

from typing import Union

from omegaconf import DictConfig, ListConfig, OmegaConf

from flagscale.engine.diffusion_engine import DiffusionEngine
from flagscale.runner.utils import logger


def parse_config() -> Union[DictConfig, ListConfig]:
    """Parse the configuration file"""

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config-path", type=str, required=True, help="Path to the configuration YAML file"
    )
    args = parser.parse_args()

    config = OmegaConf.load(args.config_path)
    # TODO(yupu): Any checks?

    return config


def inference(cfg: DictConfig) -> None:
    """Run the diffusion model inference

    Args:
        cfg: The configuration file
    """

    model_cfg = cfg.get("diffusion", {})
    transforms_cfg = cfg.get("transforms", {})
    engine = DiffusionEngine(model_cfg, transforms_cfg)

    generate_cfg = cfg.get("generate", {})
    outputs = engine.generate(**generate_cfg)

    engine.save(outputs)


if __name__ == "__main__":
    cfg = parse_config()
    assert isinstance(cfg, DictConfig)  # To make pyright happy
    inference(cfg)
    logger.info("Diffusion model inference completed")
