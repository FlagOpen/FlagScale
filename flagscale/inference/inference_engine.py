import importlib
import os

from dataclasses import dataclass
from typing import Any, List, Tuple, Union

import torch
import torch.nn as nn

from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.utils import export_to_video
from omegaconf import DictConfig

from flagscale.inference.runtime_context import RuntimeContext
from flagscale.inference.utils import parse_torch_dtype
from flagscale.transforms import create_transformations_from_config


def _check_required_fields(config_dict: DictConfig, required_fields: List[str]) -> None:
    """Check if the required fields are in the config dict."""
    if not config_dict:
        raise ValueError("config_dict is empty")
    for field in required_fields:
        if field not in config_dict:
            raise ValueError(f"Required field {field} is not in the config dict.")


@dataclass
class InferenceEngineConfig:
    model_config: DictConfig = None
    model_name: str = None
    loader: str = None

    engine_config: DictConfig = None
    results_path: str = None
    output_format: str = None
    state_scopes: List[str] = None

    transforms_cfg: DictConfig = None

    def __init__(self, config_dict: DictConfig) -> None:
        """Load and validate the config dict

        Args:
            config_dict (DictConfig, **required**): The configuration for the inference engine.
        """

        # ==========================================
        #               MODEL CONFIG
        # ==========================================
        self.model_config = config_dict.get("model_config", {})
        _check_required_fields(self.model_config, ["loader", "model"])
        # TODO(yupu): Should we distinguish between the model name and the model path?
        self.model_name = self.model_config.model
        self.loader = self.model_config.loader

        # ==========================================
        #               ENGINE CONFIG
        # ==========================================
        self.engine_config = config_dict.get("engine_config", {})
        _check_required_fields(self.engine_config, ["results_path", "output_format"])
        self.results_path = self.engine_config.results_path
        self.output_format = self.engine_config.output_format
        self.state_scopes = self.engine_config.get("state_scopes", None)

        # ==========================================
        #               TRANSFORMS CONFIG
        # ==========================================
        self.transforms_cfg = self.engine_config.get("transformations", {})


class InferenceEngine:
    """A engine for model inference."""

    def __init__(self, config_dict: DictConfig) -> None:
        """Initialize the inference engine.

        Args:
            config_dict: The config dict. It consists of 2 parts:
                - model_config: config for loading the model
                    - model (str, **required**): the model name or path
                    - loader (str, **required**): the loader for the model,
                        available options: "diffusers", "transformers", "custom", "auto"
                    - other model loading parameters
                - engine_config: engine related config
                    - results_path (str, **required**): the path to save the results
                    - transformations: the transformations to apply to the model
                    - other engine related parameters
        """
        self.config = InferenceEngineConfig(config_dict)

        self.model_or_pipeline, self.backbone = self.load()
        self.apply_transformations()

    def load(self) -> Union[DiffusionPipeline, nn.Module]:
        """Load the model or pipeline.

        Returns:
            The model or pipeline, depending on the loader.
        """

        if self.config.loader == "diffusers":
            return self.load_diffusers_pipeline(self.config.model_name, **self.config.model_config)
        elif self.config.loader == "transformers":
            raise NotImplementedError("Transformers loader is not implemented")
        elif self.config.loader == "custom":
            raise NotImplementedError("Custom loader is not implemented")
        elif self.config.loader == "auto":
            raise NotImplementedError("Auto loader is not implemented")
        else:
            raise ValueError(f"Unsupported loader: {self.config.loader}")

    def apply_transformations(self) -> None:
        """Apply the transformations to the model or pipeline

        `Transformation` will be applied in the EXACT order as specified in the config.
        """

        # TODO(yupu): run preflight/supports check for each transformation
        transformations = create_transformations_from_config(self.config.transforms_cfg)
        for t in transformations:
            success = t.apply(self.backbone)
            if not success:
                raise ValueError(f"Failed to apply transformation: {t}")

    def generate(self, **kwargs) -> Any:
        """Generate the output."""

        # Enforce return_dict=True for easier output saving
        kwargs["return_dict"] = True

        # Build a single torch.Generator from DictConfig: {seed: int, device?: str}
        def _build_generator(gen_cfg: DictConfig, default_device: str) -> torch.Generator:
            device = str(gen_cfg.get("device")) if gen_cfg.get("device") else default_device
            if gen_cfg.get("seed") is None:
                raise ValueError("generator.seed is required in config")
            seed = int(gen_cfg.get("seed"))
            return torch.Generator(device).manual_seed(seed)

        default_device = getattr(self.model_or_pipeline, "device", torch.device("cpu"))
        if isinstance(default_device, torch.device):
            default_device = str(default_device)
        if "generator" in kwargs and kwargs["generator"] is not None:
            kwargs["generator"] = _build_generator(kwargs["generator"], default_device)

        # TODO(yupu): get num_timesteps from the kwargs
        with RuntimeContext().session():
            outputs = self.model_or_pipeline(**kwargs)
            return outputs

    def save(self, outputs) -> bool:
        """Save the output."""

        os.makedirs(self.config.results_path, exist_ok=True)

        if self.config.output_format == "image":
            output_path = os.path.join(self.config.results_path, "output.png")
            # TODO(yupu): Support multiple images
            if hasattr(outputs, "images") and len(outputs.images) > 0:
                image = outputs.images[0]
            else:
                raise NotImplementedError("Not implemented yet")
            image.save(output_path)
        elif self.config.output_format == "video":
            output_path = os.path.join(self.config.results_path, "output.mp4")
            # TODO(yupu): Support multiple videos
            if hasattr(outputs, "frames") and len(outputs.frames) > 0:
                video = outputs.frames[0]
            else:
                raise NotImplementedError("Not implemented yet")
            export_to_video(video, output_path, fps=15)
        else:
            raise ValueError(f"Unsupported output format: {self.config.output_format}")

        return True

    def load_diffusers_pipeline(
        self, pretrained_model_name_or_path: str, **kwargs
    ) -> Tuple[DiffusionPipeline, nn.Module]:
        """Load the DiffusionPipeline and locate the backbone.

        Args:
            pretrained_model_name_or_path (str, **required**): The name or path of the pretrained model.
            **kwargs: The kwargs for the `DiffusionPipeline.from_pretrained` method.
                kwargs could contain methods for enabling/disabling certain features of the pipeline.
                e.g. enable_xformers_memory_efficient_attention, enable_model_cpu_offload, etc.

        Returns:
            A tuple of the DiffusionPipeline and the backbone.
            - pipeline: The DiffusionPipeline.
            - backbone: The backbone of the DiffusionPipeline.
        """

        def _resolve_class(dotted_path: str):
            mod_name, _, cls_name = dotted_path.rpartition(".")
            if not mod_name or not cls_name:
                raise ValueError(f"Invalid class path: '{dotted_path}'")
            return getattr(importlib.import_module(mod_name), cls_name)

        def _normalize_dtype_inplace(d, key: str = "torch_dtype") -> None:
            if key in d:
                parsed = parse_torch_dtype(d.get(key))
                if parsed is not None:
                    d[key] = parsed

        def _kwargs_from_cfg(cfg, default_id: str) -> dict:
            assert isinstance(cfg, DictConfig) or cfg is None
            kwargs_local = {} if cfg is None else {k: cfg.get(k) for k in cfg}
            if "pretrained_model_name_or_path" not in kwargs_local:
                kwargs_local["pretrained_model_name_or_path"] = default_id
            _normalize_dtype_inplace(kwargs_local)
            return kwargs_local

        forbidden_keys = {"model", "loader", "device", "pipeline", "components"}
        sanitized_kwargs = {k: v for k, v in kwargs.items() if k not in forbidden_keys}
        _normalize_dtype_inplace(sanitized_kwargs)

        components_cfg = kwargs.get("components")
        if components_cfg:
            for comp_name, comp_spec in components_cfg.items():
                class_path = comp_spec.get("class")
                if not class_path or not isinstance(class_path, str):
                    raise ValueError(
                        f"Component '{comp_name}' requires a 'class' string (e.g., 'diffusers.AutoencoderKL')."
                    )
                cls = _resolve_class(class_path)
                fp_cfg = comp_spec.get("from_pretrained", {})
                fp_kwargs = _kwargs_from_cfg(fp_cfg, pretrained_model_name_or_path)
                sanitized_kwargs[comp_name] = cls.from_pretrained(**fp_kwargs)

        pipeline_cfg = kwargs.get("pipeline")
        pipeline_fp_args = {}
        if pipeline_cfg and pipeline_cfg.get("class"):
            pipeline_cls = _resolve_class(pipeline_cfg["class"])
            fp_cfg = pipeline_cfg.get("from_pretrained", {})
            pipeline_fp_args = _kwargs_from_cfg(fp_cfg, pretrained_model_name_or_path)
        else:
            pipeline_cls = DiffusionPipeline

        pipeline_kwargs = dict(sanitized_kwargs)
        if pipeline_fp_args:
            pipeline_kwargs.update(pipeline_fp_args)
        if "pretrained_model_name_or_path" in pipeline_kwargs:
            pipeline_kwargs.pop("pretrained_model_name_or_path")
        pipeline = pipeline_cls.from_pretrained(pretrained_model_name_or_path, **pipeline_kwargs)

        device = kwargs.get("device", None)
        if device:
            pipeline.to(device)

        # TODO(yupu): Messy, refactor this
        known_methods_wo_args = [
            "enable_xformers_memory_efficient_attention",
            "enable_model_cpu_offload",
            "enable_sequential_cpu_offload",
            "enable_attention_slicing",
            "enable_vae_slicing",
            "enable_vae_tiling",
            "fuse_qkv_projections",
        ]

        for method in known_methods_wo_args:
            if (
                method in kwargs
                and hasattr(pipeline, method)
                and callable(getattr(pipeline, method))
            ):
                getattr(pipeline, method)()

        # Simple heuristic to find the module to apply the transformations to
        if hasattr(pipeline, "unet"):
            backbone = pipeline.unet
            assert isinstance(
                backbone, nn.Module
            ), f"unet should be a `nn.Module`, but got {type(backbone)}"
        elif hasattr(pipeline, "transformer"):
            backbone = pipeline.transformer
            assert isinstance(
                backbone, nn.Module
            ), f"transformer should be a `nn.Module`, but got {type(backbone)}"
        else:
            raise ValueError(f"Failed to find the backbone of the DiffusionPipeline: {pipeline}")

        return pipeline, backbone
