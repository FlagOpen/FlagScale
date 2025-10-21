import json
import logging

from typing import List

import numpy as np
import PIL
import torch

from torchvision import transforms

from megatron.energon import DefaultTaskEncoder
from tools.datasets.qwenvl.data.energon.chatml_pi0 import ChatMLSamplePI0

dataset_logger = logging.getLogger(__name__)


class TaskEncoder(
    DefaultTaskEncoder[ChatMLSamplePI0, ChatMLSamplePI0, ChatMLSamplePI0, ChatMLSamplePI0]
):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.vision_root = config.vision_root
        return

    def encode_sample(self, sample: ChatMLSamplePI0) -> dict:
        conversation = (
            json.loads(sample.conversation)
            if isinstance(sample.conversation, (str, bytes))
            else sample.conversation
        )
        # For PI0 token <image> is useless, the position of image embeddings are fixed
        task = [conversation["conversations"][0]["value"].replace("<image>", "")]

        imgs = []
        for i in sample.imgs:
            image = PIL.Image.open(i)
            image_tensor = transforms.ToTensor()(image)
            imgs.append(image_tensor)

        state_paths = sample.metadata['state'][self.config.state_key]
        state = np.load(state_paths)[0]
        if state.shape[0] < self.config.action_horizon:
            pad_width = self.config.action_horizon - state.shape[0]
            state = np.pad(state, (0, pad_width), mode='constant')
        elif state.shape[0] > self.config.action_horizon:
            state = state[: self.config.action_horizon]
        state = torch.from_numpy(state)

        action_paths = sample.metadata['action'][self.config.action_key]
        action = np.load(action_paths)
        if action.shape[1] < self.config.action_horizon:
            pad_width = self.config.action_horizon - action.shape[1]
            action = np.pad(action, ((0, 0), (0, pad_width)), mode='constant')
        elif action.shape[1] > self.config.action_horizon:
            action = action[:, : self.config.action_horizon]
        action = torch.from_numpy(action)

        batch = {
            'task': task,
            'observation.images.camera0': imgs[0].flip(dims=[0])[None,].to(torch.float32),
            'observation.images.camera1': imgs[1].flip(dims=[0])[None,].to(torch.float32),
            'observation.images.camera2': imgs[2].flip(dims=[0])[None,].to(torch.float32),
            'observation.state': state[None,].to(torch.float32),
            'action': action[None].to(torch.float32),
        }
        return batch

    def batch(self, samples: List[dict]) -> dict:
        rsp = samples[0]
        for s in samples[1:]:
            rsp["task"].extend(s["task"])
            rsp["observation.images.camera0"] = torch.cat(
                [rsp["observation.images.camera0"], s["observation.images.camera0"]], dim=0
            )
            rsp["observation.images.camera1"] = torch.cat(
                [rsp["observation.images.camera1"], s["observation.images.camera1"]], dim=0
            )
            rsp["observation.images.camera2"] = torch.cat(
                [rsp["observation.images.camera2"], s["observation.images.camera2"]], dim=0
            )
            rsp["observation.state"] = torch.cat(
                [rsp["observation.state"], s["observation.state"]], dim=0
            )
            rsp["action"] = torch.cat([rsp["action"], s["action"]], dim=0)
        return rsp

    def encode_batch(self, samples: dict) -> dict:
        return samples
