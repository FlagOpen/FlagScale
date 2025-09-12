import os

from typing import Any, Union

from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from omegaconf import DictConfig
from typing_extensions import Optional

from flagscale.models.adapters import BaseAdapter, create_adapter
from flagscale.transforms import TransformManager, create_transforms_from_config

# TODO(yupu): implmeent `DiffusionEngineConfig`, `DiffusionEngineOutput`
# TODO(yupu): supports all kinds of outputs
# e.g. `StableDiffusionPipelineOutput`
# read more from https://github.com/huggingface/diffusers/blob/main/src/diffusers/utils/outputs.py#L40


class DiffusionEngine:
    """A engine that focuses on the diffusion model inference."""

    def __init__(self, config_dict: DictConfig, transforms_cfg: DictConfig) -> None:
        """Initialize the diffusion engine.

        Args:
            config_dict: The config dict. It consists of 2 parts:
                - model_config: model related config
                - engine_config: engine related config
            transforms_cfg: The transforms config dict.
        """

        # TODO(yupu):
        # 1) Build pipeline from engine_config
        # 2) Build adapter over the core module (e.g., pipeline.unet)
        # 3) Register optional caps on adapter (simple callables)
        # 4) Select transforms from engine_args/config (pre/compile/post)
        # 5) Plan and apply transforms on adapter.transformers()

        self.validate_config(config_dict, transforms_cfg)

        self.pipeline = self.load(self.model_name, **self.model_config)

        self.adapter: BaseAdapter = create_adapter(
            self.engine_config.get("adapter", None), self.pipeline
        )

        transforms = create_transforms_from_config(transforms_cfg)
        if transforms:
            manager = TransformManager(transforms)
            manager.apply(self.adapter)

    def validate_config(self, config_dict: DictConfig, transforms_cfg: DictConfig) -> None:
        """Validate the config dict

        The config dict consists of 2 parts:
        - model_config: model related config
        - engine_config: engine related config

        The minimum required config is:
        - model_config.model:
        - engine_config.results_path

        Args:
            config_dict: The config dict.
            transforms_cfg: The transforms config dict.
        """

        self.model_config = config_dict.get("model_config", {})
        if self.model_config is None or self.model_config.get("model", None) is None:
            raise ValueError("model_config.model is required")

        self.model_name = self.model_config.model
        self.model_config.pop("model")

        self.engine_config = config_dict.get("engine_config", {})
        if self.engine_config is None or self.engine_config.get("results_path", None) is None:
            raise ValueError("engine_config.results_path is required")
        self.results_path = self.engine_config.results_path
        self.engine_config.pop("results_path")

        self.transforms_cfg = transforms_cfg

        self.transforms_cfg = transforms_cfg

        self.device = self.model_config.get("device", False)
        self.check_param(self.device, "device")

        self.enable_xformers = self.model_config.get("enable_xformers", False)
        self.check_param(self.enable_xformers, "enable_xformers")

        self.enable_model_cpu_offload = self.model_config.get("enable_model_cpu_offload", False)
        self.check_param(self.enable_model_cpu_offload, "enable_model_cpu_offload")

        self.enable_sequential_cpu_offload = self.model_config.get(
            "enable_sequential_cpu_offload", False
        )
        self.check_param(self.enable_sequential_cpu_offload, "enable_sequential_cpu_offload")

        self.enable_attention_slicing = self.model_config.get("enable_attention_slicing", False)
        self.check_param(self.enable_attention_slicing, "enable_attention_slicing")

        self.enable_vae_slicing = self.model_config.get("enable_vae_slicing", False)
        self.check_param(self.enable_vae_slicing, "enable_vae_slicing")
        self.enable_vae_tiling = self.model_config.get("enable_vae_tiling", False)
        self.check_param(self.enable_vae_tiling, "enable_vae_tiling")

        self.fuse_qkv_projections = self.model_config.get("fuse_qkv_projections", False)
        self.check_param(self.fuse_qkv_projections, "fuse_qkv_projections")
        self.components = self.model_config.get("components", False)
        self.check_param(self.components, "components")

        self.to_cuda = self.model_config.get("to_cuda", True)
        self.check_param(self.to_cuda, "to_cuda")

    def check_param(self, model_param, param_name):
        if not isinstance(model_param, bool):
            raise ValueError(f"the value of {param_name}  must be boolean (True or False)")

    def generate(self, **kwargs) -> Any:
        """Generate the output."""

        outputs = self.pipeline(**kwargs)
        return outputs

    # TODO(yupu): load custom models
    def load(
        self, pretrained_model_name_or_path: Optional[Union[str, os.PathLike]], **kwargs
    ) -> DiffusionPipeline:
        """Load the pipeline.

        Args:
            pretrained_model_name_or_path: The pretrained model name or path.

        Returns:
            The pipeline.
        """

        pipeline = DiffusionPipeline.from_pretrained(pretrained_model_name_or_path, **kwargs)
        # TODO(yupu): Read from config
        if self.to_cuda:
            pipeline.to("cuda")
        if self.device:
            print(pipeline.device)
        if self.enable_xformers:
            pipeline.enable_xformers_memory_efficient_attention()
        if self.enable_model_cpu_offload:
            pipeline.enable_model_cpu_offload()
        if self.enable_sequential_cpu_offload:
            pipeline.enable_sequential_cpu_offload()
        if self.enable_attention_slicing:
            pipeline.enable_attention_slicing()
        if self.enable_vae_slicing:
            pipeline.enable_vae_slicing()
        if self.enable_vae_tiling:
            pipeline.enable_vae_tiling()
        if self.fuse_qkv_projections:
            pipeline.fuse_qkv_projections()
        if self.components:
            print(pipeline.components)
        return pipeline

    # TODO(yupu): save all kinds of outputs, and maybe move to adapter
    def save(self, outputs) -> bool:
        """Save the output."""

        os.makedirs(self.results_path, exist_ok=True)
        image = outputs.images[0]
        image.save(os.path.join(self.results_path, "result.png"))

        return True
