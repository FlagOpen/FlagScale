# Copyright (c) 2024 Alibaba PAI and Nvidia Megatron-LM Team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import pprint
import dataclasses
import json
import logging
import math
import os
import re
import sys
import traceback
import time

from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union
from datetime import datetime

import numpy as np
import PIL
from PIL import Image
import torch

from torchvision import transforms as T

from megatron.energon import Batch, DefaultTaskEncoder, VQASample
from megatron.training import get_args
from megatron.training.global_vars import get_tokenizer
from megatron.training.tokenizer.tokenizer import _Qwen2VLTokenizer
from megatron.core.parallel_state import (
    get_tensor_model_parallel_rank,
    get_pipeline_model_parallel_world_size,
    get_pipeline_model_parallel_rank,
)
from flagscale.train.models.qwen2_5_vl.tensor_parallel import broadcast_data
from transformers import AutoTokenizer
from tools.datasets.qwenvl.data.energon.chatml_robotics import ChatMLSampleRobotics
from tools.datasets.qwenvl.data.image_processing import get_visual_transform
import logging

dataset_logger = logging.getLogger(__name__)
FIRST_MAX_PADDING = True
LAST_LARGE_IMG = False
CLEAR_CACHE_ITERATION = 200000
IGNORE_IDX = -100
MAX_IMG_THRESHHOLD = 5000
TOKENIZER = None


def get_tokenizer_safe(
    tokenizer_path="/share/project/section/Qwen/Qwen2.5-VL-3B-Instruct", extra_vocab_size=293
):
    global TOKENIZER
    if TOKENIZER is None:
        TOKENIZER = _Qwen2VLTokenizer(tokenizer_path, extra_vocab_size)
    return TOKENIZER


# Type for intermediate batch, after batch()
@dataclass
class ImageTaskSample:
    __key__: str
    __subflavors__: Dict

    imgs: List[np.ndarray]  # (c, h, w)
    videos: List[np.ndarray]  # (c, h, w)

    image_thw_grids: np.ndarray
    video_thw_grids: np.ndarray
    image_input_mask: np.ndarray
    video_input_mask: np.ndarray
    second_per_grid_ts: np.ndarray  # (n_videos, )

    text: np.ndarray
    target: np.ndarray

    state: np.ndarray
    actions: np.ndarray


# Typing for the resulting batch data after encode_batch()
@dataclass
class VQATaskBatch(Batch):
    __keys__: List[str]
    __subflavors__: List[Dict]
    # (num_tiles, c, h, w)
    imgs: torch.Tensor
    videos: torch.Tensor
    image_thw_grids: torch.Tensor
    video_thw_grids: torch.Tensor
    image_input_mask: torch.Tensor
    video_input_mask: torch.Tensor
    second_per_grid_ts: torch.Tensor  # (n_videos, ), read from metadata?

    # (n, seq_len)
    text: torch.Tensor
    # (n, seq_len)
    target: torch.Tensor

    state: torch.Tensor
    actions: torch.Tensor


class InternalWarning(Warning): ...


def convert_to_qwen2vl_content(
    user_input: str, image_pattern: str = "<image>", video_pattern: str = "<video>"
):
    """
    Split user input into format Qwen2VL tokenizer accepts.
    """
    if not isinstance(user_input, str):
        # 如果输入不是字符串，说明它可能已经被处理过了，立即报错
        raise TypeError(
            f"convert_to_qwen2vl_content was called with a non-string input of type {type(user_input)}. Input: {user_input}"
        )
    pattern = r"({image}|{video})".format(image=image_pattern, video=video_pattern)
    contents = []
    cur = 0
    mm_idx = defaultdict(int)
    for matched in re.finditer(pattern, user_input):
        start, end = matched.span()
        text = user_input[cur:start]
        if text:
            contents.append({"type": "text", "text": text})

        contents.append(
            {
                "type": matched.string[start:end][1:-1],
                matched.string[start:end][1:-1]: str(mm_idx[matched.string[start:end][1:-1]]),
            }
        )

        cur = end
        mm_idx[matched.string[start:end][1:-1]] += 1

    if cur < len(user_input):
        contents.append({"type": "text", "text": user_input[cur : len(user_input)]})
        num_images = sum(1 for item in contents if item["type"] == "image")

    return contents


class TaskEncoder(
    DefaultTaskEncoder[Union[VQASample, ChatMLSampleRobotics], ImageTaskSample, VQATaskBatch, dict]
):
    """A simple task encoder for captioning."""

    # ACTION_TOKEN_START_ID = 151665
    ACTION_TOKEN_START_ID = 149595
    ACTION_TOKEN_END_ID = ACTION_TOKEN_START_ID + 2048

    def __init__(self, config):
        super().__init__()
        # Usually: eepose, eepose, action_eepose_token
        self.state_key = config.state_key
        self.action_key = config.action_key
        self.action_token_key = config.action_token_key
        print("Initializing TaskEncoder with state_key:", self.state_key)
        print("Initializing TaskEncoder with action_key:", self.action_key)
        print("Initializing TaskEncoder with action_token_key:", self.action_token_key)

        # self.args = get_args()
        self.tp_size = config.tensor_model_parallel_size
        self.cp_size = config.context_parallel_size
        self.sequence_parallel = True
        self.tokenizer_path = config.tokenizer_path
        self.extra_vocab_size = 293
        # self.tokenizer = AutoTokenizer.from_pretrained("/share/project/section/Qwen/Qwen2.5-VL-3B-Instruct")
        self.tokenizer = get_tokenizer_safe(self.tokenizer_path, self.extra_vocab_size)

        self.temporal_patch_size = 2
        self.merge_size = 2
        self.patch_size = 14

        self.seq_len = 700

        self.vision_root = ""
        self.enable_variable_seq_lengths = True
        self.image_max_pixels = 768 * 768
        self.image_min_pixels = 32 * 32
        # 预缓存常用token IDs - 避免重复查找
        self._token_cache = self._build_token_cache()

        # 预缓存action tokens - 批量生成
        self._action_token_cache = self._build_action_token_cache()

        assert self.vision_root is not None, "Please give the vision root."

    def encode_sample(self, sample: Union[VQASample, ChatMLSampleRobotics]):
        if isinstance(sample, VQASample):
            is_llava_training = (
                sample.__subflavors__["is_llava_training"]
                if "is_llava_training" in sample.__subflavors__
                else False
            )
            if is_llava_training:
                raise NotImplementedError("Sample format not supported")
            else:
                yield self.encode_vqa(sample)
        elif isinstance(sample, ChatMLSampleRobotics):
            yield self.encode_chatml(sample)
        else:
            raise NotImplementedError("Sample format not supported")

    def _flatten_visual_inputs(self, visuals, is_image: bool = True):
        """
        visuals: list of visual inputs, each input is a tensor of shape (c, h, w)
        """
        flattened = []
        thw_grids = []
        for visual in visuals:
            if is_image:
                resized_height, resized_width = visual.shape[-2:]
                patches = np.tile(np.array(visual), (self.temporal_patch_size, 1, 1, 1))
            else:
                assert len(visual) % self.temporal_patch_size == 0
                patches = np.array(visual)
                resized_height, resized_width = patches.shape[-2:]

            channel = patches.shape[1]
            grid_t = patches.shape[0] // self.temporal_patch_size
            grid_h, grid_w = (resized_height // self.patch_size, resized_width // self.patch_size)
            patches = patches.reshape(
                grid_t,
                self.temporal_patch_size,
                channel,
                grid_h // self.merge_size,
                self.merge_size,
                self.patch_size,
                grid_w // self.merge_size,
                self.merge_size,
                self.patch_size,
            )
            patches = patches.transpose(0, 3, 6, 4, 7, 2, 1, 5, 8)
            flatten_patches = patches.reshape(
                grid_t * grid_h * grid_w,
                channel * self.temporal_patch_size * self.patch_size * self.patch_size,
            )
            flattened.append(flatten_patches)
            thw_grids.append((grid_t, grid_h, grid_w))
        return flattened, np.array(thw_grids)

    def _preprocess_image(
        self, image: PIL.Image, image_max_pixels: int = 768 * 768, image_min_pixels: int = 32 * 32
    ) -> PIL.Image:
        """
        Pre-processes a single image.
        """
        if (image.width * image.height) > image_max_pixels:
            resize_factor = math.sqrt(image_max_pixels / (image.width * image.height))
            width, height = int(image.width * resize_factor), int(image.height * resize_factor)
            image = image.resize((width, height))

        if (image.width * image.height) < image_min_pixels:
            resize_factor = math.sqrt(image_min_pixels / (image.width * image.height))
            width, height = int(image.width * resize_factor), int(image.height * resize_factor)
            image = image.resize((width, height))

        if image.mode != "RGB":
            image = image.convert("RGB")

        if min(image.width, image.height) < 28:
            width, height = max(image.width, 28), max(image.height, 28)
            image = image.resize((width, height), resample=Image.Resampling.NEAREST)

        if image.width / image.height > 200:
            width, height = image.height * 180, image.height
            image = image.resize((width, height), resample=Image.Resampling.NEAREST)

        if image.height / image.width > 200:
            width, height = image.width, image.width * 180
            image = image.resize((width, height), resample=Image.Resampling.NEAREST)

        return image

    def _safe_encode(self, text):
        """简化的安全编码"""
        try:
            return self.tokenizer.encode(text, add_special_tokens=False)
        except TypeError:
            return self.tokenizer.encode(text)

    def decode_token_ids_to_readable(self, token_ids, max_tokens=100):
        """将token IDs转换为可读格式 - 临时查找版本"""
        print("=== 对话复原（前{}个tokens）===".format(max_tokens))

        result_text = ""
        boa_id = self._token_cache.get('boa')
        eoa_id = self._token_cache.get('eoa')
        for i, token_id in enumerate(token_ids[:max_tokens]):
            # 先检查常用的特殊tokens
            if token_id == self._token_cache['im_start']:
                result_text += "\n<|im_start|>"
            elif token_id == self._token_cache['im_end']:
                result_text += "<|im_end|>\n"
            elif boa_id is not None and token_id == boa_id:
                result_text += "<boa>"
            elif eoa_id is not None and token_id == eoa_id:
                result_text += "<eoa>"
            elif token_id == self._token_cache['image_pad']:
                result_text += "<|image_pad|>"
            elif token_id == self._token_cache['newline']:
                result_text += "\\n"
            elif token_id == self._token_cache['space']:
                result_text += " "
            elif token_id == self._token_cache['user']:
                result_text += "user"
            elif token_id == self._token_cache['assistant']:
                result_text += "assistant"
            elif token_id == self._token_cache['system']:
                result_text += "system"
            else:
                # 临时查找其他tokens
                found_token = None
                for text, tid in self.tokenizer.vocab.items():
                    if tid == token_id:
                        found_token = text
                        break

                if found_token:
                    result_text += found_token
                else:
                    result_text += f"[UNK_{token_id}]"

        print(result_text)
        return result_text

    def _build_token_cache(self):
        """一次性缓存所有常用token IDs"""
        cache_start = time.time()

        token_cache = {
            'im_start': self.tokenizer.vocab["<|im_start|>"],
            'im_end': self.tokenizer.vocab["<|im_end|>"],
            'user': self.tokenizer.vocab["user"],
            'assistant': self.tokenizer.vocab["assistant"],
            'system': self.tokenizer.vocab["system"],
            'vision_start': self.tokenizer.vocab.get("<|vision_start|>"),
            'vision_end': self.tokenizer.vocab.get("<|vision_end|>"),
            'image_pad': self.tokenizer.vocab.get("<|image_pad|>"),
            'video_pad': self.tokenizer.vocab.get("<|video_pad|>"),
            'newline': self._safe_encode("\n")[0],
            'space': self._safe_encode(" ")[0],
            'boa': self.tokenizer.vocab.get("<boa>", 151665),
            'eoa': self.tokenizer.vocab.get("<eoa>", 151666),
            'action_split': self.tokenizer.vocab.get("<action_split>", 151667),
        }

        cache_end = time.time()
        print(f"Token cache built in {(cache_end - cache_start) * 1000:.2f} ms")
        return token_cache

    def _build_action_token_cache(self):
        """预缓存所有可能的action tokens"""
        # cache_start = time.time()

        action_cache = {}
        # action token ID范围是0-2047
        for action_id in range(2048):
            token_string = f"<action_token_{action_id}>"
            token_id = self.tokenizer.vocab.get(token_string, 149595 + action_id)
            if token_id is not None:
                action_cache[action_id] = token_id

        # cache_end = time.time()
        # print(f"Action token cache built in {(cache_end - cache_start) * 1000:.2f} ms with {len(action_cache)} tokens")
        return action_cache

    def build_conversation_tokens(self, conversation, action_tokens_list):
        """使用缓存避免词汇表查找"""
        build_start = time.time()

        final_token_ids = []

        # 使用缓存的token IDs - 不需要查找
        im_start_id = self._token_cache['im_start']
        im_end_id = self._token_cache['im_end']
        newline_id = self._token_cache['newline']
        user_id = self._token_cache['user']
        assistant_id = self._token_cache['assistant']
        system_id = self._token_cache['system']
        image_pad_id = self._token_cache['image_pad']
        vision_start_id = self._token_cache['vision_start']
        vision_end_id = self._token_cache.get('vision_end')
        # space_id = self._token_cache['space']

        conversation_loop_start = time.time()
        for turn_idx, turn in enumerate(conversation):
            role = turn["role"]
            content = turn["content"]
            action_tokens = (
                action_tokens_list[turn_idx] if turn_idx < len(action_tokens_list) else []
            )

            # 开始标记
            final_token_ids.append(im_start_id)

            if role == "system":
                final_token_ids.append(system_id)
                final_token_ids.append(newline_id)
                if content.strip():
                    text_ids = self._safe_encode(content)
                    final_token_ids.extend(text_ids)

            elif role == "user":
                final_token_ids.append(user_id)
                final_token_ids.append(newline_id)

                # 处理用户内容
                if isinstance(content, list):
                    for item in content:
                        if item["type"] == "text":
                            if item["text"].strip():
                                text_ids = self._safe_encode(item["text"])
                                final_token_ids.extend(text_ids)
                        elif item["type"] == "image":
                            if vision_start_id:
                                final_token_ids.append(vision_start_id)
                            # 使用缓存的image_pad_id
                            final_token_ids.append(image_pad_id)
                            if vision_end_id:
                                final_token_ids.append(vision_end_id)
                else:
                    # 纯文本内容
                    if content.strip():
                        text_ids = self._safe_encode(content)
                        final_token_ids.extend(text_ids)

            elif role == "assistant":
                final_token_ids.append(assistant_id)
                final_token_ids.append(newline_id)

                # 使用缓存的action tokens
                if action_tokens and len(action_tokens) > 0:
                    boa_id = self._token_cache['boa']
                    eoa_id = self._token_cache['eoa']
                    action_split_id = self._token_cache['action_split']
                    # final_token_ids.append(boa_id)
                    for i, action_id in enumerate(action_tokens):
                        # 从缓存中获取token ID
                        if action_id == -1:
                            # 如果遇到哨兵值，就添加真正的分隔符ID
                            final_token_ids.append(action_split_id)
                        else:
                            correct_token_id = self._action_token_cache.get(action_id)
                            if correct_token_id is None:
                                raise ValueError(f"Action token {action_id} not found in cache.")
                            final_token_ids.append(correct_token_id)
                    # final_token_ids.append(eoa_id)
                else:
                    # 普通assistant内容
                    if content.strip():
                        text_ids = self._safe_encode(content)
                        final_token_ids.extend(text_ids)

            # 结束标记
            final_token_ids.append(im_end_id)
            final_token_ids.append(newline_id)

        # conversation_loop_end = time.time()
        # print(f"    Optimized conversation loop time: {(conversation_loop_end - conversation_loop_start) * 1000:.2f} ms")

        # 数组转换
        result = np.array(final_token_ids, dtype=np.int64)
        # print("\n" + "="*50)
        # self.decode_token_ids_to_readable(result, max_tokens=100)

        # build_end = time.time()
        # print(f"    Optimized build_conversation_tokens total time: {(build_end - build_start) * 1000:.2f} ms")

        return result

    def encode_chatml(self, sample: ChatMLSampleRobotics):
        # Process images
        if sample.imgs is not None and len(sample.imgs) > 0:
            imgs = []
            for img in sample.imgs:
                img_path = os.path.join(self.vision_root, img)
                try:
                    image = PIL.Image.open(img_path)
                    image = self._preprocess_image(
                        image=image,
                        image_max_pixels=self.image_max_pixels,
                        image_min_pixels=self.image_min_pixels,
                    )
                    imgs.append(image)
                except Exception as e:
                    raise ValueError(
                        f"Failed to open image: {img_path}. Error: {e} of sample[{sample.__key__}]"
                    )
            imgs_info = self.tokenizer.processor.image_processor(imgs, return_tensors="np")
            flattened_imgs = imgs_info["pixel_values"]
            image_thw_grids = imgs_info["image_grid_thw"]
        else:
            flattened_imgs = []
            image_thw_grids = []

        # Process videos
        if sample.videos is not None and len(sample.videos) > 0:
            videos = [
                [PIL.Image.open(os.path.join(self.vision_root, frame)) for frame in video]
                for video in sample.videos
            ]
            for i, video in enumerate(videos):
                videos[i] = video[: len(video) // 2 * 2]
            videos_info = self.tokenizer.processor.image_processor(
                images=None, videos=videos, return_tensors="pt"
            )
            flattened_videos = videos_info["pixel_values_videos"]
            video_thw_grids = videos_info["video_grid_thw"]
        else:
            flattened_videos = []
            video_thw_grids = []

        conversation = (
            json.loads(sample.conversation)
            if isinstance(sample.conversation, (str, bytes))
            else sample.conversation
        )

        second_per_grid_ts = [1 / 2.0] * len(video_thw_grids)
        if "conversations" in conversation:
            second_per_grid_ts = conversation.get("second_per_grid_ts", second_per_grid_ts)
            second_per_grid_ts = [float(i) for i in second_per_grid_ts]
            conversation = conversation["conversations"]

        role_key = "from" if "from" in conversation[0] else "role"
        content_key = "value" if "from" in conversation[0] else "content"

        converted_conversation = []
        action_tokens_list = []  # Store action tokens for each turn
        if len(conversation) % 2 == 0:
            converted_conversation.append(
                {"role": "system", "content": "You are a helpful assistant."}
            )
            action_tokens_list.append([])
        else:
            dataset_logger.warning(
                f"The sample [{sample.__key__}] has an odd number of conversation turns. The first turn will be used as a system prompt, but this may be incorrect. Please check the sample."
            )
            converted_conversation.append(
                {"role": "system", "content": conversation[0][content_key]}
            )
            action_tokens_list.append([])
            conversation = conversation[1:]

        EXPECTED_ROLE = ["human", "gpt"]
        for turn_idx, turn in enumerate(conversation):
            role = turn[role_key]
            if role != EXPECTED_ROLE[turn_idx % len(EXPECTED_ROLE)]:
                raise InternalWarning(
                    f"Expect conversation organized in order: [sys] human gpt human gpt..., but got role '{role}' in turn {turn_idx}"
                )
            content = turn[content_key]
            current_action_tokens = []

            if role == "human":
                role = "user"
                content = convert_to_qwen2vl_content(content)

            elif role == "gpt":
                role = "assistant"
                if "<action_token>" in content:
                    current_action_tokens = []
                    action_tokens_loaded = False
                    action_token_paths = None
                    # 检查是否有直接提供的 action_token 字段
                    # print(sample.metadata)
                    # print(isinstance(sample.metadata, dict))
                    if hasattr(sample, 'metadata') and isinstance(sample.metadata, dict):
                        action_token_paths = sample.metadata.get(self.action_token_key)
                        # print("Action_token_path",action_token_paths)

                    if action_token_paths is not None:

                        # 处理单个文件（字符串）或多个文件（列表）
                        if isinstance(action_token_paths, str):
                            action_token_paths = [action_token_paths]
                        elif not isinstance(action_token_paths, list):
                            dataset_logger.warning(
                                f"Sample [{sample.__key__}]: Unexpected action_token type: {type(action_token_paths)}"
                            )
                            action_token_paths = []

                        # 计算 <action_token> 的数量来验证
                        action_token_count = content.count("<action_token>")

                        if len(action_token_paths) != action_token_count:
                            dataset_logger.warning(
                                f"Sample [{sample.__key__}]: action_token count mismatch. "
                                f"Found {action_token_count} <action_token> tags but {len(action_token_paths)} files."
                            )

                        # 按顺序加载所有 action token 文件
                        for i, action_token_path in enumerate(action_token_paths):
                            full_action_token_path = os.path.join(
                                self.vision_root, action_token_path
                            )

                            if os.path.exists(full_action_token_path):
                                try:
                                    loaded_tokens = (
                                        np.load(full_action_token_path).flatten().tolist()
                                    )
                                    tokens = [int(token) for token in loaded_tokens]
                                    current_action_tokens.extend(tokens)
                                    action_tokens_loaded = True

                                    # 如果需要分隔符
                                    if i < len(action_token_paths) - 1:
                                        current_action_tokens.append(-1)  # 使用 -1 作为分隔标记
                                        # pass

                                except Exception as e:
                                    dataset_logger.warning(
                                        f"Failed to load action token file: {full_action_token_path}. Error: {e}"
                                    )
                                    current_action_tokens = []
                                    action_tokens_loaded = False
                                    break
                            else:
                                dataset_logger.warning(
                                    f"Action token file not found: {full_action_token_path}"
                                )
                                action_tokens_loaded = False
                                break
                    else:
                        dataset_logger.debug(
                            f"Sample [{sample.__key__}]: No action_eepose_token in metadata."
                        )

                    # 处理结果
                    if action_tokens_loaded and current_action_tokens:
                        # 成功加载 action tokens，清空文本内容
                        content = ""
                        dataset_logger.debug(
                            f"Sample [{sample.__key__}]: Loaded {len(current_action_tokens)} action tokens"
                        )
                    else:
                        # 没有成功加载 action tokens
                        should_have_action_tokens = (
                            hasattr(sample, 'metadata')
                            and isinstance(sample.metadata, dict)
                            and self.action_token_key in sample.metadata
                        )

                        if should_have_action_tokens:
                            # 样本应该有 action tokens 但加载失败
                            dataset_logger.error(
                                f"Sample [{sample.__key__}]: Failed to load action tokens despite having '{self.action_token_key}' in metadata. "
                                f"File path might be incorrect: {sample.metadata.get(self.action_token_key)}. "
                                f"Converting to regular text generation."
                            )
                        else:
                            # 样本本来就没有 action tokens，这是正常的混合数据集情况
                            dataset_logger.debug(
                                f"Sample [{sample.__key__}]: No action tokens available, treating as regular text generation."
                            )
                        # 移除 <action_token> 标记，保留其他内容
                        content = content.replace("<action_token>", "").replace(
                            "<action_split>", ""
                        )
                        # 清理多余的空白字符
                        content = re.sub(r'\s+', ' ', content).strip()

                        # 如果清理后内容为空，提供默认回复
                        if not content:
                            content = "I understand the task."

                        current_action_tokens = []

            converted_conversation.append({"role": role, "content": content})
            action_tokens_list.append(current_action_tokens)

        conversation = converted_conversation

        # input_ids = self.build_conversation_tokens(converted_conversation, action_tokens_list)
        input_ids = self.build_conversation_tokens(converted_conversation, [])

        target = input_ids.copy()
        pad_token_id = IGNORE_IDX

        # Calculate system prompt length and set its mask
        if converted_conversation[0]["role"] == "system":
            system_tokens = self.build_conversation_tokens(
                [converted_conversation[0]], [action_tokens_list[0]]
            )
            system_prompt_prefix = len(system_tokens)
            target[:system_prompt_prefix] = pad_token_id
            start_idx = 1
        else:
            system_prompt_prefix = 0
            start_idx = 0

        offset = system_prompt_prefix

        for turn_idx in range(start_idx, len(converted_conversation)):
            turn = converted_conversation[turn_idx]
            action_tokens = (
                action_tokens_list[turn_idx] if turn_idx < len(action_tokens_list) else []
            )

            # Calculate the token length of the current turn
            turn_tokens = self.build_conversation_tokens([turn], [action_tokens])
            n_tokens = len(turn_tokens)

            if turn["role"] == "user":
                # Mask all user input
                target[offset : offset + n_tokens] = pad_token_id
            elif turn["role"] == "assistant":
                # Mask the assistant's prompt prefix, but not the generated content (text or action tokens)
                assistant_generation_prefix = 3  # <im_start>assistant\n
                target[offset : offset + assistant_generation_prefix] = pad_token_id

            offset += n_tokens

        merge_length = self.merge_size**2
        image_token_id, video_token_id = self.tokenizer.encode(["<|image_pad|>", "<|video_pad|>"])

        image_token_indices = np.where(input_ids == image_token_id)[0]
        assert len(image_token_indices) == len(
            image_thw_grids
        ), f"The sample [{sample.__key__}] has {len(image_thw_grids)} images, but {len(image_token_indices)} image placeholders!"
        video_token_indices = np.where(input_ids == video_token_id)[0]
        assert len(video_token_indices) == len(
            video_thw_grids
        ), f"The sample [{sample.__key__}] has {len(video_thw_grids)} videos, but {len(video_token_indices)} video placeholders!"
        image_thw_grids, video_thw_grids = np.array(image_thw_grids, dtype=np.int64), np.array(
            video_thw_grids, dtype=np.int64
        )

        target_length = (
            input_ids.shape[0]
            - image_thw_grids.shape[0]
            + image_thw_grids.prod(axis=-1).sum() // merge_length
            - video_thw_grids.shape[0]
            + video_thw_grids.prod(axis=-1).sum() // merge_length
        )
        # if target_length > self.seq_len:
        #     dataset_logger.warning(
        #         f"Sample id [{sample.__key__}] has a long sequence with length {target_length}, which will be cutoff to the max length {self.seq_len} in the batching function."
        #     )
        final_input_ids = np.zeros(target_length, dtype=input_ids.dtype)
        final_input_masks = final_input_ids.copy()

        image_idx, video_idx = 0, 0
        indices = np.sort(np.concatenate([image_token_indices, video_token_indices]))
        cur_x, cur_y = 0, 0
        for idx in indices:
            token_id = input_ids[idx]
            if token_id == image_token_id:
                size = image_thw_grids[image_idx].prod() // merge_length
                image_idx += 1
            elif token_id == video_token_id:
                size = video_thw_grids[video_idx].prod() // merge_length
                video_idx += 1
            final_input_ids[cur_y : cur_y + idx - cur_x] = input_ids[cur_x:idx]
            final_input_masks[cur_y : cur_y + idx - cur_x] = target[cur_x:idx]
            cur_y += idx - cur_x
            final_input_ids[cur_y : cur_y + size] = token_id
            final_input_masks[cur_y : cur_y + size] = pad_token_id
            cur_y += size
            cur_x = idx + 1

        if cur_x < len(input_ids):
            final_input_ids[cur_y:] = input_ids[cur_x:]
            final_input_masks[cur_y:] = target[cur_x:]
        # print('=== 处理完成后的final_input_ids ===')
        # print('final_input_ids shape:', final_input_ids.shape)
        # print('final_input_ids:', final_input_ids[-100])

        # target = np.roll(final_input_masks, shift=-1)
        target = final_input_masks
        # target[-1] = pad_token_id

        if (target == pad_token_id).all():
            raise InternalWarning(
                f"Sample id [{sample.__key__}] has all masked labels. The data is invalid and will be dropped!"
            )
        # # # # --- 开始添加/修改调试代码 ---
        # print(f"DEBUG FINAL CHECK FOR sample [{sample.__key__}]:")

        # # # 找到所有 action token 在 final_input_ids 中的位置
        # action_token_indices = np.where(
        #     (final_input_ids >= self.ACTION_TOKEN_START_ID) &
        #     (final_input_ids < self.ACTION_TOKEN_END_ID)
        # )[0]

        # # # DEBUG PRINT: Show final input_ids and target for verification
        # if len(action_token_indices) > 0:
        #     print("  --- Action Token Verification ---")
        #     # 打印第一个和最后几个 action token 进行抽查
        #     indices_to_check = list(action_token_indices[:3]) + list(action_token_indices[-3:])

        #     for idx in sorted(list(set(indices_to_check))): # sorted 和 set 防止重复打印
        #         input_token = final_input_ids[idx]
        #         prev_input_token = final_input_ids[idx - 1]
        #         target_for_prev_token = target[idx]

        #         print(f"  - At index {idx-1}:")
        #         print(f"      Input Token: {prev_input_token}")
        #         print(f"      Target:      {target_for_prev_token}  <-- This should be the token to predict")
        #         print(f"  - At index {idx}:")
        #         print(f"      Input Token: {input_token}  <-- This is the action token")

        #         if input_token == target_for_prev_token:
        #             print("      ✅ CHECK PASSED: Target correctly set to predict the action token.")
        #         else:
        #             print(f"      ❌ CHECK FAILED: Target is {target_for_prev_token}, but should be {input_token}.")
        #     print("  ---------------------------------")
        # else:
        #     print("  - No action tokens found in this sample for verification.")

        image_input_mask = final_input_ids == self.tokenizer.image_token_id
        video_input_mask = final_input_ids == self.tokenizer.video_token_id

        if hasattr(sample, 'metadata') and isinstance(sample.metadata, dict):
            # if 'state' in sample.metadata:
            #     state_paths = sample.metadata['state'][self.state_key]
            #     state = np.load(state_paths)[0]
            # else:
            #     state = np.zeros((32,), dtype=np.float32)
            state = np.zeros((32,), dtype=np.float32)
            action_paths = sample.metadata['action'][self.action_key]

            if state.shape[0] < 32:
                pad_width = 32 - state.shape[0]
                state = np.pad(state, (0, pad_width), mode='constant')
            elif state.shape[0] > 32:
                state = state[:32]
            action = np.load(action_paths)
            if action.shape[1] < 32:
                pad_width = 32 - action.shape[1]
                action = np.pad(action, ((0, 0), (0, pad_width)), mode='constant')
            elif action.shape[1] > 32:
                action = action[:, :32]

        return ImageTaskSample(
            __key__=sample.__key__,
            __subflavors__=sample.__subflavors__,
            imgs=flattened_imgs,
            videos=flattened_videos,
            image_thw_grids=image_thw_grids,
            video_thw_grids=video_thw_grids,
            second_per_grid_ts=np.array(second_per_grid_ts, dtype=np.float32),
            image_input_mask=image_input_mask,
            video_input_mask=video_input_mask,
            text=final_input_ids,
            target=target,
            state=state,
            actions=action,
        )

    def batch(self, samples: List[ImageTaskSample]) -> VQATaskBatch:
        imgs = [
            s.imgs for s in samples if isinstance(s.imgs, np.ndarray) and s.imgs.size > 0
        ]  # len(bs) pixel-value
        if len(imgs) > 0:
            imgs = torch.cat([torch.from_numpy(img) for img in imgs])
        else:
            imgs = torch.empty(
                [0, 3 * self.temporal_patch_size * self.patch_size * self.patch_size],
                dtype=torch.float32,
            )

        image_thw_grids = [thw_grids for s in samples for thw_grids in s.image_thw_grids]
        if len(image_thw_grids) > 0:
            image_thw_grids = torch.from_numpy(np.array(image_thw_grids)).long()
            assert image_thw_grids.prod(dim=-1).sum() == imgs.shape[0]
        else:
            image_thw_grids = torch.empty([0, 3], dtype=torch.long)

        videos = [
            s.videos for s in samples if isinstance(s.videos, np.ndarray) and s.videos.size > 0
        ]
        if len(videos) > 0:
            videos = torch.cat([torch.from_numpy(video) for video in videos])
        else:
            videos = torch.empty(
                [0, 3 * self.temporal_patch_size * self.patch_size * self.patch_size],
                dtype=torch.float32,
            )

        second_per_grid_ts = [
            second_per_grid for s in samples for second_per_grid in s.second_per_grid_ts
        ]
        if len(second_per_grid_ts) > 0:
            second_per_grid_ts = torch.from_numpy(np.array(second_per_grid_ts)).float()
        else:
            second_per_grid_ts = torch.empty([0], dtype=torch.float32)

        video_thw_grids = [thw_grids for s in samples for thw_grids in s.video_thw_grids]
        if len(video_thw_grids) > 0:
            video_thw_grids = torch.from_numpy(np.array(video_thw_grids)).long()
            assert video_thw_grids.prod(dim=-1).sum() == videos.shape[0]
        else:
            video_thw_grids = torch.empty([0, 3], dtype=torch.long)

        global LAST_LARGE_IMG, MAX_IMG_THRESHHOLD
        # MODIFIED START: Restore original cache clearing logic
        # if self.args.curr_iteration > 0 and self.args.curr_iteration % CLEAR_CACHE_ITERATION == 0:
        #     FIRST_MAX_PADDING = True

        if image_thw_grids.prod(axis=-1).sum() // 4 > MAX_IMG_THRESHHOLD:
            MAX_IMG_THRESHHOLD = image_thw_grids.prod(axis=-1).sum() // 4
            # FIRST_MAX_PADDING = True
            LAST_LARGE_IMG = True

        # if not self.enable_variable_seq_lengths:
        #     max_seq_len = self.seq_len
        # else:
        #     max_seq_len = max(len(s.text) for s in samples)
        #     max_seq_len = max(max_seq_len, self.seq_len)
        max_seq_len = max(len(s.text) for s in samples)

        if self.cp_size > 1 or self.sequence_parallel:
            max_seq_len = math.ceil(max_seq_len / (self.tp_size * self.cp_size)) * (
                self.tp_size * self.cp_size
            )
        text_mat = np.full((len(samples), max_seq_len), self.tokenizer.pad_token_id, dtype=np.int64)
        target_mat = np.full((len(samples), max_seq_len), IGNORE_IDX, dtype=np.int64)

        image_input_masks = np.zeros_like(text_mat, dtype=bool)
        video_input_masks = np.zeros_like(text_mat, dtype=bool)
        state_mat = []
        actions_mat = []
        for i, s in enumerate(samples):
            text_len = min(max_seq_len, len(s.text))
            target_len = min(max_seq_len, len(s.target))
            text_mat[i, :text_len] = np.array(s.text)[:text_len]
            if s.image_input_mask is not None:
                image_input_masks[i, :text_len] = np.array(s.image_input_mask)[:text_len]
            if s.video_input_mask is not None:
                video_input_masks[i, :text_len] = np.array(s.video_input_mask)[:text_len]
            target_mat[i, :target_len] = np.array(s.target)[:target_len]
            state_mat.append(s.state)
            actions_mat.append(s.actions)

        batch = VQATaskBatch(
            __keys__=[s.__key__ for s in samples],
            __subflavors__=[s.__subflavors__ for s in samples],
            imgs=imgs,
            videos=videos,
            image_thw_grids=image_thw_grids,
            video_thw_grids=video_thw_grids,
            second_per_grid_ts=second_per_grid_ts,
            image_input_mask=torch.from_numpy(image_input_masks),
            video_input_mask=torch.from_numpy(video_input_masks),
            text=torch.from_numpy(text_mat),
            target=torch.from_numpy(target_mat),
            state=torch.from_numpy(np.array(state_mat)),
            actions=torch.from_numpy(np.array(actions_mat)),
        )

        return batch

    def encode_batch(self, batch: VQATaskBatch) -> dict:
        raw = dataclasses.asdict(batch)
        del raw["__subflavors__"]
        return raw


def print_error_handler(exc: Exception, key: Optional[str], debug=False):
    if not debug and isinstance(exc, InternalWarning):
        return
    print(
        f"The following exception occurred in the dataloader for sample {key} and is skipped",
        file=sys.stderr,
    )
    traceback.print_exc()


def get_rope_index(
    input_ids: Optional[torch.LongTensor] = None,
    image_grid_thw: Optional[torch.LongTensor] = None,
    video_grid_thw: Optional[torch.LongTensor] = None,
    second_per_grid_ts: Optional[torch.Tensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Calculate the 3D rope index based on image and video's temporal, height and width in LLM.

    Explanation:
        Each embedding sequence contains vision embedding and text embedding or just contains text embedding.

        For pure text embedding sequence, the rotary position embedding has no difference with modern LLMs.
        Examples:
            input_ids: [T T T T T], here T is for text.
            temporal position_ids: [0, 1, 2, 3, 4]
            height position_ids: [0, 1, 2, 3, 4]
            width position_ids: [0, 1, 2, 3, 4]

        For vision and text embedding sequence, we calculate 3D rotary position embedding for vision part
        and 1D rotary position embedding for text part.
        Examples:
            Temporal (Time): 3 patches, representing different segments of the video in time.
            Height: 2 patches, dividing each frame vertically.
            Width: 2 patches, dividing each frame horizontally.
            We also have some important parameters:
            fps (Frames Per Second): The video's frame rate, set to 1. This means one frame is processed each second.
            tokens_per_second: This is a crucial parameter. It dictates how many "time-steps" or "temporal tokens" are conceptually packed into a one-second interval of the video. In this case, we have 25 tokens per second. So each second of the video will be represented with 25 separate time points. It essentially defines the temporal granularity.
            temporal_patch_size: The number of frames that compose one temporal patch. Here, it's 2 frames.
            interval: The step size for the temporal position IDs, calculated as tokens_per_second * temporal_patch_size / fps. In this case, 25 * 2 / 1 = 50. This means that each temporal patch will be have a difference of 50 in the temporal position IDs.
            input_ids: [V V V V V V V V V V V V T T T T T], here V is for vision.
            vision temporal position_ids: [0, 0, 0, 0, 50, 50, 50, 50, 100, 100, 100, 100]
            vision height position_ids: [0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1]
            vision width position_ids: [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
            text temporal position_ids: [101, 102, 103, 104, 105]
            text height position_ids: [101, 102, 103, 104, 105]
            text width position_ids: [101, 102, 103, 104, 105]
            Here we calculate the text start position_ids as the max vision position_ids plus 1.

    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
            it.
        image_grid_thw (`torch.LongTensor` of shape `(num_images, 3)`, *optional*):
            The temporal, height and width of feature shape of each image in LLM.
        video_grid_thw (`torch.LongTensor` of shape `(num_videos, 3)`, *optional*):
            The temporal, height and width of feature shape of each video in LLM.
        second_per_grid_ts (`torch.Tensor` of shape `(num_videos)`, *optional*):
            The time interval (in seconds) for each grid along the temporal dimension in the 3D position IDs.
        attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

    Returns:
        position_ids (`torch.LongTensor` of shape `(3, batch_size, sequence_length)`)
        mrope_position_deltas (`torch.Tensor` of shape `(batch_size)`)
    """
    tokenizer = get_tokenizer_safe()
    spatial_merge_size = 2
    image_token_id = tokenizer.image_token_id
    video_token_id = tokenizer.video_token_id
    vision_start_token_id = tokenizer.vision_start_token_id
    tokens_per_second = 2
    if second_per_grid_ts is not None:
        second_per_grid_ts = second_per_grid_ts.cpu()

    mrope_position_deltas = []
    if image_grid_thw is not None or video_grid_thw is not None:
        total_input_ids = input_ids
        if attention_mask is None:
            attention_mask = torch.ones_like(total_input_ids)
        position_ids = torch.ones(
            3,
            input_ids.shape[0],
            input_ids.shape[1],
            dtype=input_ids.dtype,
            device=input_ids.device,
        )
        image_index, video_index = 0, 0
        attention_mask = attention_mask.to(total_input_ids.device)
        for i, input_ids in enumerate(total_input_ids):
            input_ids = input_ids[attention_mask[i] == 1]
            image_nums, video_nums = 0, 0
            vision_start_indices = torch.argwhere(input_ids == vision_start_token_id).squeeze(1)
            vision_tokens = input_ids[vision_start_indices + 1]
            image_nums = (vision_tokens == image_token_id).sum()
            video_nums = (vision_tokens == video_token_id).sum()
            input_tokens = input_ids.tolist()
            llm_pos_ids_list: list = []
            st = 0
            remain_images, remain_videos = image_nums, video_nums
            for _ in range(image_nums + video_nums):
                if image_token_id in input_tokens and remain_images > 0:
                    ed_image = input_tokens.index(image_token_id, st)
                else:
                    ed_image = len(input_tokens) + 1
                if video_token_id in input_tokens and remain_videos > 0:
                    ed_video = input_tokens.index(video_token_id, st)
                else:
                    ed_video = len(input_tokens) + 1
                if ed_image < ed_video:
                    t, h, w = (
                        image_grid_thw[image_index][0],
                        image_grid_thw[image_index][1],
                        image_grid_thw[image_index][2],
                    )
                    second_per_grid_t = 0
                    image_index += 1
                    remain_images -= 1
                    ed = ed_image

                else:
                    t, h, w = (
                        video_grid_thw[video_index][0],
                        video_grid_thw[video_index][1],
                        video_grid_thw[video_index][2],
                    )
                    if second_per_grid_ts is not None:
                        second_per_grid_t = second_per_grid_ts[video_index]
                    else:
                        second_per_grid_t = 1.0
                    video_index += 1
                    remain_videos -= 1
                    ed = ed_video
                llm_grid_t, llm_grid_h, llm_grid_w = (
                    t.item(),
                    h.item() // spatial_merge_size,
                    w.item() // spatial_merge_size,
                )
                text_len = ed - st

                st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)

                range_tensor = torch.arange(llm_grid_t).view(-1, 1)
                expanded_range = range_tensor.expand(-1, llm_grid_h * llm_grid_w)

                time_tensor = expanded_range * second_per_grid_t * tokens_per_second

                time_tensor_long = time_tensor.long()
                t_index = time_tensor_long.flatten()

                h_index = (
                    torch.arange(llm_grid_h)
                    .view(1, -1, 1)
                    .expand(llm_grid_t, -1, llm_grid_w)
                    .flatten()
                )
                w_index = (
                    torch.arange(llm_grid_w)
                    .view(1, 1, -1)
                    .expand(llm_grid_t, llm_grid_h, -1)
                    .flatten()
                )
                llm_pos_ids_list.append(
                    torch.stack([t_index, h_index, w_index]) + text_len + st_idx
                )
                st = ed + llm_grid_t * llm_grid_h * llm_grid_w

            if st < len(input_tokens):
                st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                text_len = len(input_tokens) - st
                llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)

            llm_positions = torch.cat(llm_pos_ids_list, dim=1).reshape(3, -1)
            position_ids[..., i, attention_mask[i] == 1] = llm_positions.to(position_ids.device)
            mrope_position_deltas.append(llm_positions.max() + 1 - len(total_input_ids[i]))
        mrope_position_deltas = torch.tensor(
            mrope_position_deltas, device=input_ids.device
        ).unsqueeze(1)
        return position_ids, mrope_position_deltas
    else:
        if attention_mask is not None:
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            position_ids = position_ids.unsqueeze(0).expand(3, -1, -1).to(input_ids.device)
            max_position_ids = position_ids.max(0, keepdim=False)[0].max(-1, keepdim=True)[0]
            mrope_position_deltas = max_position_ids + 1 - attention_mask.shape[-1]
        else:
            position_ids = (
                torch.arange(input_ids.shape[1], device=input_ids.device)
                .view(1, 1, -1)
                .expand(3, input_ids.shape[0], -1)
            )
            mrope_position_deltas = torch.zeros(
                [input_ids.shape[0], 1], device=input_ids.device, dtype=input_ids.dtype
            )

        return position_ids, mrope_position_deltas


def get_ltor_masks_and_position_ids(
    input_ids,
    image_thw_grids,
    video_thw_grids,
    target,
    pad_token,
    second_per_grid_ts,
    ignore_index=None,
):
    """Build masks and position id for left to right model."""
    # Position ids. [3 X bs X seqlen]
    position_ids, _ = get_rope_index(
        input_ids=input_ids,
        image_grid_thw=image_thw_grids,
        video_grid_thw=video_thw_grids,
        second_per_grid_ts=second_per_grid_ts,
        attention_mask=input_ids != pad_token,
    )

    # Loss mask.
    loss_mask = torch.ones(target.size(), dtype=torch.float, device=input_ids.device)
    loss_mask[target == pad_token] = 0.0  # mask paddings
    if ignore_index is not None:
        loss_mask[target == ignore_index] = 0.0  # mask prompts

    # Attention mask.
    attention_mask = None

    return attention_mask, loss_mask, position_ids


def get_batch(data_iterator):
    """Generate a batch"""
    imgs = None
    tokens = None
    labels = None
    loss_mask = None
    attention_mask = None
    position_ids = None

    # Broadcast data.
    torch.cuda.nvtx.range_push("get_data")
    if data_iterator is not None:
        data = next(data_iterator)
        # pad_token_id = get_tokenizer().pad_token_id
        pad_token_id = IGNORE_IDX
        # while (data["target"] == pad_token_id).all() or (data["target"].shape[-1] < 986 or data["target"].shape[-1] > 1000): # for debug
        while (data["target"] == pad_token_id).all():
            logging.getLogger(__name__).warning(
                "The current data is invalid because the target is all pad_token_id! Get next data to avoid fail, but it's better to check the data!"
            )
            data = next(data_iterator)
    else:
        data = None

    data_text = data["text"]

    target = data["target"]
    # shape: num_tiles x c x h x w
    imgs = data["imgs"]

    # shape: num_tiles x c x h x w
    videos = data["videos"]

    # shape: n_image_samples
    image_thw_grids = data["image_thw_grids"]

    state = data["state"]
    actions = data["actions"]

    # global LAST_LARGE_IMG
    # if LAST_LARGE_IMG:
    #     torch.cuda.empty_cache()
    #     LAST_LARGE_IMG=False
    # if image_thw_grids.prod(axis=-1).sum() // 4 > 3000:
    #     torch.cuda.empty_cache()
    #     LAST_LARGE_IMG = True
    # args = get_args()
    # if data_text.shape[-1] == args.max_padding_length and get_pipeline_model_parallel_rank() == 0:
    #     torch.cuda.empty_cache()
    # shape: n_video_samples
    video_thw_grids = data["video_thw_grids"]
    # shape: n_video_samples
    second_per_grid_ts = data['second_per_grid_ts']

    image_input_mask = data["image_input_mask"]
    video_input_mask = data["video_input_mask"]
    torch.cuda.nvtx.range_pop()

    torch.cuda.nvtx.range_push("index tokens")
    tokenizer = get_tokenizer_safe()

    tokens = data_text.long().contiguous()
    labels = target.contiguous()

    assert tokens.shape == labels.shape, f"tokens: {tokens.shape} != labels: {labels.shape}"
    torch.cuda.nvtx.range_pop()

    # NOTE: no sequence packing in LLM inputs
    torch.cuda.nvtx.range_push("get_ltor_masks_and_position_ids")
    attention_mask, loss_mask, position_ids = get_ltor_masks_and_position_ids(
        tokens, image_thw_grids, video_thw_grids, labels, IGNORE_IDX, second_per_grid_ts
    )
    torch.cuda.nvtx.range_pop()

    return {
        "tokens": tokens,
        "labels": labels,
        "loss_mask": loss_mask,
        "attention_mask": attention_mask,
        "position_ids": position_ids,
        "imgs": imgs,
        "videos": videos,
        "image_thw_grids": image_thw_grids,
        "video_thw_grids": video_thw_grids,
        "image_input_mask": image_input_mask,
        "video_input_mask": video_input_mask,
        "state": state,
        "actions": actions,
    }
