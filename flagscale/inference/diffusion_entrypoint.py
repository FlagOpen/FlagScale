import argparse

from typing import Union

from omegaconf import DictConfig, ListConfig, OmegaConf

from flagscale.inference.inference_engine import InferenceEngine
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
    """Run the model inference

    Args:
        cfg: The parsed configuration
    """

    engine = InferenceEngine(cfg.get("engine", {}))

    generate_cfg = cfg.get("generate", {})
    outputs = engine.generate(**generate_cfg)

    engine.save(outputs)


if __name__ == "__main__":
    parsed_cfg = parse_config()
    assert isinstance(parsed_cfg, DictConfig)  # To make pyright happy
    inference(parsed_cfg)
    logger.info("Inference model inference completed")
