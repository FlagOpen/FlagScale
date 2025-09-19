from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import torch

from PIL import Image
from transformers import (
    AutoProcessor,
    PretrainedConfig,
    PreTrainedModel,
    Qwen2_5_VLConfig,
    Qwen2_5_VLForConditionalGeneration,
)
from transformers.feature_extraction_utils import BatchFeature

from .action_head.flow_matching_action_head import (
    FlowmatchingActionHead,
    FlowmatchingActionHeadConfig,
)


@dataclass
class RoboBrainRoboticsConfig(PretrainedConfig):
    backbone_cfg: dict = field(init=False, metadata={"help": "Backbone configuration."})

    action_head_cfg: dict = field(init=False, metadata={"help": "Action head configuration."})

    # action_horizon: int = field(default=64, metadata={"help": "Action horizon."})
    # action_dim: int = field(default=32, metadata={"help": "Action dimension."})
    # compute_dtype: str = field(default="float32", metadata={"help": "Compute dtype."})

    training: bool = field(
        default=False, metadata={"help": "Whether the model is in training mode."}
    )
    pretrained_vlm_model_path: str = field(
        default="Local/path/of/the/model", metadata={"help": "Local path of the model."}
    )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        for key, value in kwargs.items():
            setattr(self, key, value)


class RoboBrainRobotics(PreTrainedModel):
    _supports_flash_attn_2 = True

    def __init__(self, config: RoboBrainRoboticsConfig):
        super().__init__(config)

        self.pretrained_vlm_model_path = getattr(
            config, 'pretrained_vlm_model_path', "path/to/pretrained_vlm_model_path"
        )
        training = getattr(config, 'training', False)

        # vlm
        self.processor = AutoProcessor.from_pretrained("path/to/processor", padding_side="left")
        if training:
            self.backbone = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                self.pretrained_vlm_model_path,
                torch_dtype=config.torch_dtype,
                attn_implementation=(
                    config.attn_implementation if hasattr(config, 'attn_implementation') else "sdpa"
                ),
            )
        else:
            backbone_cfg = Qwen2_5_VLConfig.from_dict(config.backbone_cfg)
            self.backbone = Qwen2_5_VLForConditionalGeneration(backbone_cfg)

        # action expert
        action_head_cfg = FlowmatchingActionHeadConfig(**config.action_head_cfg)
        self.action_head = FlowmatchingActionHead(action_head_cfg)

    def prepare_input(
        self,
        instruction: List[str],
        image: list[Dict[str, torch.Tensor] | Dict[str, Image.Image]],
        state: Optional[torch.Tensor],
        action: Optional[torch.Tensor] = None,
    ):
        if isinstance(list(image[0].values())[0], torch.Tensor):
            for item in image:
                for key, value in item.items():
                    img_numpy = value.permute(1, 2, 0).cpu().numpy() * 255.0
                    item[key] = Image.fromarray(img_numpy.astype(np.uint8))

        batch_images = []
        for item in image:
            for key, value in item.items():
                batch_images.append(value)

        texts = []
        for item in instruction:
            conversations = [
                {
                    "role": "user",
                    "content": [
                        *[{"type": "image", "image": "None"} for _ in range(len(image[0]))],
                        {"type": "text", "text": item},
                    ],
                }
            ]
            prompt = self.processor.tokenizer.apply_chat_template(
                conversations,
                chat_template=self.processor.chat_template,
                tokenize=False,
                return_dict=False,
                add_generation_prompt=True,
            )
            texts.append(prompt)

        backbone_inputs = self.processor(
            text=texts, images=batch_images, return_tensors="pt", padding=True, truncation=True
        )
        backbone_inputs = backbone_inputs.to(self.device)

        if action is None:
            action_inputs = BatchFeature(
                data={
                    "embodiment_id": torch.tensor([0]).repeat(len(texts)),
                    "state": state.unsqueeze(1),
                    "state_mask": torch.ones_like(state, dtype=torch.bool),
                }
            )
        else:
            action_inputs = BatchFeature(
                data={
                    "embodiment_id": torch.tensor([0]).repeat(len(texts)),
                    "state": state.unsqueeze(1),
                    "state_mask": torch.ones_like(state, dtype=torch.bool),
                    "action": action,
                    "action_mask": torch.ones_like(action, dtype=torch.bool),
                }
            )

        return backbone_inputs, action_inputs

    def forward(
        self,
        instruction: List[str],
        image: list[Dict[str, torch.Tensor] | Dict[str, Image.Image]],
        state: Optional[torch.Tensor],
        action: Optional[torch.Tensor],
    ):
        backbone_inputs, action_inputs = self.prepare_input(instruction, image, state, action)

        backbone_features = self.backbone(
            **backbone_inputs, output_hidden_states=True, return_dict=True
        ).hidden_states[-1]

        backbone_outputs = BatchFeature(
            data={
                "backbone_features": backbone_features,
                "backbone_attention_mask": backbone_inputs["attention_mask"],
            }
        )  # [B, T2, hidden_size]
        action_head_outputs = self.action_head(backbone_outputs, action_inputs)
        return action_head_outputs

    def get_action(
        self,
        instruction: List[str],
        image: list[Dict[str, torch.Tensor] | Dict[str, Image.Image]],
        state: Optional[torch.Tensor],
    ):
        import time

        start_time = time.time()
        backbone_inputs, action_inputs = self.prepare_input(instruction, image, state)
        print(f"backbone_inputs: {backbone_inputs}")
        print(f"input_ids: {backbone_inputs['input_ids'].shape}")

        print(f"image_encoder time: {(time.time() - start_time) * 1000:.1f} ms")
        start_time = time.time()

        backbone_features = self.backbone(
            **backbone_inputs, output_hidden_states=True, return_dict=True
        ).hidden_states[-1]
        print(f"observation_forward time: {(time.time() - start_time) * 1000:.1f} ms")
        start_time = time.time()

        backbone_outputs = BatchFeature(
            data={
                "backbone_features": backbone_features,
                "backbone_attention_mask": backbone_inputs["attention_mask"],
            }
        )  # [B, T2, hidden_size]

        action_head_outputs = self.action_head.get_action(backbone_outputs, action_inputs)
        print(f"action_forward time: {(time.time() - start_time) * 1000:.1f} ms")
        start_time = time.time()
        return action_head_outputs
