import importlib
import os

from dataclasses import dataclass
from typing import Any, List, Tuple, Union

import torch.nn as nn

from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.utils import export_to_video
from omegaconf import DictConfig

from flagscale.inference.runtime_context import RuntimeContext
from flagscale.inference.utils import parse_torch_dtype
from flagscale.transforms import create_default_transformations, create_transformations_from_config


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
        # print(
        #     f"self.model_or_pipeline: {self.model_or_pipeline}, self.backbone: {self.backbone}"
        # )
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
            for name, mod in t.targets(self.backbone):
                print(f"Applying transformation: {t} on {name}")
                success = t.apply(mod)
                if not success:
                    raise ValueError(f"Failed to apply transformation: {t} on {name}")
        # assert False

        # default_transformations = create_default_transformations()
        # for t in default_transformations:
        #     success = t.apply(self.backbone)
        #     if not success:
        #         raise ValueError(f"Failed to apply transformation: {t}")

    def generate(self, **kwargs) -> Any:
        """Generate the output."""

        # TODO(yupu): get num_timesteps from the kwargs
        with RuntimeContext(self.config.state_scopes, num_timesteps=50).session():
            outputs = self.model_or_pipeline(**kwargs)
            return outputs

    # TODO(yupu): save all kinds of outputs, and maybe move to adapter
    def save(self, outputs) -> bool:
        """Save the output."""

        os.makedirs(self.config.results_path, exist_ok=True)

        if self.config.output_format == "image":
            image = outputs.images[0]
            image.save(os.path.join(self.config.results_path, "result.png"))
        elif self.config.output_format == "video":
            export_to_video(
                outputs.frames[0], os.path.join(self.config.results_path, "output.mp4"), fps=15
            )
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

        # Sanitize kwargs forwarded to Pipeline.from_pretrained
        forbidden_keys = {"model", "loader", "device", "pipeline_class", "components"}
        sanitized_kwargs = {k: v for k, v in kwargs.items() if k not in forbidden_keys}

        dtype_value = sanitized_kwargs.get("torch_dtype", None)

        parsed_dtype = parse_torch_dtype(dtype_value)
        if parsed_dtype is not None:
            sanitized_kwargs["torch_dtype"] = parsed_dtype
        else:
            # Remove invalid/unsupported dtype strings to avoid errors
            if "torch_dtype" in sanitized_kwargs:
                sanitized_kwargs.pop("torch_dtype")

        # Optionally build and inject subcomponents (e.g., vae)
        components_cfg = kwargs.get("components")
        if components_cfg:
            for comp_name, comp_spec in components_cfg.items():
                class_path = comp_spec.get("class") if hasattr(comp_spec, "get") else None
                if not class_path or not isinstance(class_path, str):
                    raise ValueError(
                        f"Component '{comp_name}' requires a 'class' string (e.g., 'diffusers.AutoencoderKL')."
                    )
                mod_name, _, cls_name = class_path.rpartition(".")
                if not mod_name:
                    raise ValueError(
                        f"Invalid component class path '{class_path}' for '{comp_name}'."
                    )
                cls = getattr(importlib.import_module(mod_name), cls_name)

                fp_args = comp_spec.get("from_pretrained", {}) if hasattr(comp_spec, "get") else {}
                if not isinstance(fp_args, dict):
                    try:
                        fp_args = dict(fp_args)
                    except Exception:
                        raise TypeError(
                            f"'from_pretrained' for component '{comp_name}' must be a mapping."
                        )
                fp_args.setdefault("pretrained_model_name_or_path", pretrained_model_name_or_path)
                # Normalize dtype if provided as string
                if "torch_dtype" in fp_args:
                    parsed_cdtype = parse_torch_dtype(fp_args.get("torch_dtype"))
                    if parsed_cdtype is not None:
                        fp_args["torch_dtype"] = parsed_cdtype
                    else:
                        fp_args.pop("torch_dtype", None)

                component_instance = cls.from_pretrained(**fp_args)
                sanitized_kwargs[comp_name] = component_instance

        # Resolve pipeline class if provided
        pipeline_class_path = kwargs.get("pipeline_class")
        if isinstance(pipeline_class_path, str) and pipeline_class_path:
            p_mod, _, p_cls = pipeline_class_path.rpartition(".")
            if not p_mod:
                raise ValueError(f"Invalid pipeline_class path '{pipeline_class_path}'.")
            pipeline_cls = getattr(importlib.import_module(p_mod), p_cls)
        else:
            pipeline_cls = DiffusionPipeline

        pipeline = pipeline_cls.from_pretrained(pretrained_model_name_or_path, **sanitized_kwargs)

        device = kwargs.get("device", None)
        if device:
            pipeline.to(device)

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
