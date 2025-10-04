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
from tools.datasets.qwenvl.data.energon.chatml_unified import ChatMLSample
from tools.datasets.qwenvl.data.image_processing import get_visual_transform
import logging
import time

dataset_logger = logging.getLogger(__name__)
FIRST_MAX_PADDING_FLAG = True
LAST_LARGE_IMG = False
CLEAR_CACHE_ITERATION = 200000
IGNORE_IDX = -100
MAX_IMG_THRESHHOLD = 5000

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

class InternalWarning(Warning): ...

def convert_to_qwen2vl_content(
    user_input: str, image_pattern: str = "<image>", video_pattern: str = "<video>"
):
    """
    Split user input into format Qwen2VL tokenizer accepts.
    """
    if not isinstance(user_input, str):
        # 如果输入不是字符串，说明它可能已经被处理过了，立即报错
        raise TypeError(f"convert_to_qwen2vl_content was called with a non-string input of type {type(user_input)}. Input: {user_input}")
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
    # 只有当图片数量超过一定阈值时才打印，避免刷屏
    if num_images > 2: 
        print(f"DEBUG_CONVERT: Input text: '{user_input[:100]}...'")
        print(f"DEBUG_CONVERT: Generated {num_images} image items. Full content list: {contents}")

    return contents

class TaskEncoder(
    DefaultTaskEncoder[Union[VQASample, ChatMLSample], ImageTaskSample, VQATaskBatch, dict]
):
    """A simple task encoder for captioning."""

    
    def __init__(self):
        super().__init__()

        self.args = get_args()
        self.tp_size = self.args.tensor_model_parallel_size
        self.cp_size = self.args.context_parallel_size
        self.sequence_parallel = self.args.sequence_parallel

        self.tokenizer = get_tokenizer()
        # self.action_token_start_id = self.tokenizer.vocab['<action_token_0>']

        self.temporal_patch_size = self.args.temporal_patch_size
        self.merge_size = self.args.spatial_merge_size
        self.patch_size = self.args.patch_size

        self.seq_len = self.args.max_padding_length

        self.vision_root = self.args.vision_root
        assert self.vision_root is not None, "Please give the vision root."


    def encode_sample(self, sample: Union[VQASample, ChatMLSample]):
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
        elif isinstance(sample, ChatMLSample):
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
        self, image: PIL.Image, image_max_pixels: int = 768*768, image_min_pixels: int = 32*32
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
    ###############
    def build_conversation_tokens(self, conversation, action_tokens_list):
        """
        手动构建对话的token序列，绕过apply_chat_template的分词问题
        """
        final_token_ids = []
        
        # 获取特殊token IDs
        im_start_id = self.tokenizer.vocab["<|im_start|>"]
        im_end_id = self.tokenizer.vocab["<|im_end|>"]
        # newline_id = self.tokenizer.encode("\n", add_special_tokens=False)[0]
        newline_id = self._safe_encode("\n")[0]
        user_id = self.tokenizer.vocab["user"]
        assistant_id = self.tokenizer.vocab["assistant"]
        system_id = self.tokenizer.vocab["system"]
        
        for turn_idx, turn in enumerate(conversation):
            role = turn["role"]
            content = turn["content"]
            action_tokens = action_tokens_list[turn_idx] if turn_idx < len(action_tokens_list) else []
            
            # 开始标记
            final_token_ids.append(im_start_id)
            
            if role == "system":
                final_token_ids.append(system_id)
                final_token_ids.append(newline_id)
                if content.strip():
                    # text_ids = self.tokenizer.encode(content, add_special_tokens=False)
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
                                # text_ids = self.tokenizer.encode(item["text"], add_special_tokens=False)
                                text_ids = self._safe_encode(item["text"])

                                final_token_ids.extend(text_ids)
                        elif item["type"] == "image":
                            # 添加图像占位符
                            image_pad_id = self.tokenizer.vocab["<|image_pad|>"]
                            final_token_ids.append(image_pad_id)
                else:
                    # 纯文本内容
                    if content.strip():
                        text_ids = self._safe_encode(content)
                        final_token_ids.extend(text_ids)
                        
            elif role == "assistant":
                final_token_ids.append(assistant_id)
                final_token_ids.append(newline_id)
                
                # *** 关键部分：处理action tokens ***
                if action_tokens and len(action_tokens) > 0:
                    # 直接使用convert_tokens_to_ids获取正确的token IDs
                    print(f"DEBUG_BUILD: Building tokens for assistant turn with {len(action_tokens)} actions.")
                    for i, action_id in enumerate(action_tokens):
                        token_string = f"<action_token_{action_id}>"
                        correct_token_id = self.tokenizer.vocab.get(token_string)
                        if correct_token_id is None:
                            # 如果在词汇表中找不到这个 action_token，就抛出错误，这能帮助我们发现词汇表是否正确加载
                            raise ValueError(f"Action token '{token_string}' not found in tokenizer vocabulary.")
                        final_token_ids.append(correct_token_id)
                        if i < 3: # 只打印前3个，防止刷屏
                         print(f"DEBUG_BUILD:   > Appended action: {token_string} -> ID: {correct_token_id}")
                        
                        # 在action tokens之间添加空格
                        # if i < len(action_tokens) - 1:
                        #     space_id = self.tokenizer.encode(" ", add_special_tokens=False)[0]
                        #     final_token_ids.append(space_id)
                        if i < len(action_tokens) - 1:
                            try:
                                space_id = self._safe_encode(" ")[0]
                            except:
                                space_id = 220  # 常见的空格token ID
                            final_token_ids.append(space_id)
                else:
                    # 普通assistant内容
                    if content.strip():
                        # text_ids = self.tokenizer.encode(content, add_special_tokens=False)
                        text_ids = self._safe_encode(content)
                        final_token_ids.extend(text_ids)
            
            # 结束标记
            final_token_ids.append(im_end_id)
            final_token_ids.append(newline_id)
        
        return np.array(final_token_ids, dtype=np.int64)
    def encode_chatml(self, sample: ChatMLSample):
        # MODIFIED START: Support multiple images and action tokens
        # Original image and video processing
        # imgs = [get_visual_transform(os.path.join(self.vision_root, img))[0] for img in sample.imgs]
        # videos = [
        #     [get_visual_transform(os.path.join(self.vision_root, frame))[0] for frame in video]
        #     for video in sample.videos
        # ]
        # # NOTE: make n_frames even foreach video
        # for i, video in enumerate(videos):
        #     videos[i] = video[: len(video) // 2 * 2]
        #
        # flattened_imgs, image_thw_grids = self._flatten_visual_inputs(imgs, is_image=True)
        # flattened_videos, video_thw_grids = self._flatten_visual_inputs(videos, is_image=False)

        # Process images
        encode_start = time.time()
        if sample.imgs is not None and len(sample.imgs) > 0:
            imgs = []
            for img in sample.imgs:
                img_path = os.path.join(self.vision_root, img)
                try:
                    image = PIL.Image.open(img_path)
                    image = self._preprocess_image(
                        image=image,
                        image_max_pixels=self.args.image_max_pixels,
                        image_min_pixels=self.args.image_min_pixels,
                    )
                    imgs.append(image)
                except Exception as e:
                    raise ValueError(f"Failed to open image: {img_path}. Error: {e} of sample[{sample.__key__}]")
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

        conversation = json.loads(sample.conversation) if isinstance(sample.conversation, (str, bytes)) else sample.conversation

        second_per_grid_ts = [1 / 2.0] * len(video_thw_grids)
        if "conversations" in conversation:
            second_per_grid_ts = conversation.get("second_per_grid_ts", second_per_grid_ts)
            second_per_grid_ts = [float(i) for i in second_per_grid_ts]
            conversation = conversation["conversations"]
        
        if 'pretrain-0.tar/6' in sample.__key__ or 'pretrain-0.tar/0' in sample.__key__:
             print(f"\nDEBUG_CONVERSATION: ---- Full conversation for sample [{sample.__key__}] ----")
             # 使用 pprint.pprint 来格式化输出
             pprint.pprint(conversation)
             print(f"DEBUG_CONVERSATION: ---------------------------------------------------------\n")

        role_key = "from" if "from" in conversation[0] else "role"
        content_key = "value" if "from" in conversation[0] else "content"

        converted_conversation = []
        action_tokens_list = []  # 存储每个turn的action tokens
        if len(conversation) % 2 == 0:
            converted_conversation.append(
                {"role": "system", "content": "You are a helpful assistant."}
            )
            action_tokens_list.append([])
        else:
            dataset_logger.warning(
                f"The sample [{sample.__key__}] has odd number of conversation turns, and we will use the first turn as system prompt. BUT this may be wrong. Please check the sample."
            )
            converted_conversation.append(
                {"role": "system", "content": conversation[0][content_key]}
            )
            action_tokens_list.append([])
            conversation = conversation[1:]

        EXPECTED_ROLE = ["human", "gpt"]
        print(f"\nDEBUG_LOOP: Sample [{sample.__key__}] -- BEFORE loop. converted_conversation len: {len(converted_conversation)}")
        for turn_idx, turn in enumerate(conversation):
            print(f"DEBUG_LOOP:   Turn [{turn_idx}] START. Original role: {turn.get(role_key)}. converted_conversation len: {len(converted_conversation)}")
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
            # elif role == "gpt":
            #     role = "assistant"
            #     if "<action_token>" in content:
            #         # 加载action tokens
            #         if sample.imgs and len(sample.imgs) > 0:
            #             first_img_path = sample.imgs[0]
            #             action_token_path_derived = first_img_path.replace('/images/', '/action_token/')
                        
            #             import re
            #             match = re.search(r'(\d+)\.jpg$', action_token_path_derived)
            #             if match:
            #                 frame_number = match.group(1)
            #                 base_path = re.sub(r'[^/]+$', '', action_token_path_derived)
            #                 action_token_filename = f'token_{frame_number}.npy'
            #                 action_token_path_derived = os.path.join(base_path, action_token_filename)
            #                 full_action_token_path = os.path.join(self.vision_root, action_token_path_derived)

            #                 if os.path.exists(full_action_token_path):
            #                     try:
            #                         loaded_tokens = np.load(full_action_token_path).flatten().tolist()
            #                         current_action_tokens = [int(token) for token in loaded_tokens]
            #                         # *** 关键变化：清空content，避免被错误tokenize ***
            #                         content = ""
            #                     except Exception as e:
            #                         dataset_logger.warning(f"Found action file but failed to load: {full_action_token_path}. Error: {e}")
            #                         current_action_tokens = []
            #                 else:
            #                     current_action_tokens = []
            #             else:
            #                 current_action_tokens = []
            #         else:
            #             current_action_tokens = []
            elif role == "gpt":
                role = "assistant"
                # 标记一下我们正在处理一个潜在的action_token样本
                is_potential_action_sample = "<action_token>" in content
                
                if is_potential_action_sample:
                    print(f"DEBUG_ACTION: Sample [{sample.__key__}] - Found '<action_token>' in gpt content. Starting search...")

                    if sample.imgs and len(sample.imgs) > 0:
                        first_img_path = sample.imgs[0]
                        print(f"DEBUG_ACTION:   > Using first image path: {first_img_path}")
                        
                        action_token_path_derived = first_img_path.replace('/images/', '/action_token/')
                        print(f"DEBUG_ACTION:   > Path after replacing '/images/': {action_token_path_derived}")

                        import re
                        match = re.search(r'(\d+)\.jpg$', action_token_path_derived)

                        if match:
                            frame_number = match.group(1)
                            print(f"DEBUG_ACTION:   > Regex matched! Found frame number: {frame_number}")

                            base_path = re.sub(r'[^/]+$', '', action_token_path_derived)
                            action_token_filename = f'token_{frame_number}.npy'
                            # 注意：这里 action_token_path_derived 被重新赋值了，这是正确的
                            action_token_path_derived = os.path.join(base_path, action_token_filename)
                            print(f"DEBUG_ACTION:   > Constructed relative token path: {action_token_path_derived}")
                            
                            full_action_token_path = os.path.join(self.vision_root, action_token_path_derived)
                            print(f"DEBUG_ACTION:   > Constructed full token path to check: {full_action_token_path}")

                            if os.path.exists(full_action_token_path):
                                print(f"DEBUG_ACTION:   > SUCCESS! File exists. Attempting to load...")
                                try:
                                    loaded_tokens = np.load(full_action_token_path).flatten().tolist()
                                    current_action_tokens = [int(token) for token in loaded_tokens]
                                    print(f"DEBUG_ACTION:     > Loaded {len(current_action_tokens)} tokens: {current_action_tokens[:10]}...") # 打印前10个token预览
                                    # *** 关键变化：清空content，避免被错误tokenize ***
                                    content = ""
                                except Exception as e:
                                    # 这部分虽然您说没出现，但保留是好习惯
                                    dataset_logger.warning(f"Found action file but failed to load: {full_action_token_path}. Error: {e}")
                                    current_action_tokens = []
                            else:
                                # 这是最关键的失败信息之一
                                print(f"DEBUG_ACTION:   > FAILED! File does not exist at path: {full_action_token_path}")
                                current_action_tokens = []
                        else:
                            # 关键失败信息
                            print(f"DEBUG_ACTION:   > FAILED! Regex did not match on path: {action_token_path_derived}")
                            current_action_tokens = []
                    else:
                        # 关键失败信息
                        print(f"DEBUG_ACTION:   > FAILED! 'sample.imgs' is empty or None, cannot derive path.")
                        current_action_tokens = []
            print(f"DEBUG_APPEND: Sample [{sample.__key__}], Turn [{turn_idx}], Role [{role}]")
            if len(current_action_tokens) > 0:
                print(f"DEBUG_APPEND:   > Preparing to append a NON-EMPTY list: {current_action_tokens[:5]}...")
            else:
                print(f"DEBUG_APPEND:   > Preparing to append an EMPTY list.")

            converted_conversation.append({"role": role, "content": content})
            action_tokens_list.append(current_action_tokens)
            print(f"DEBUG_LOOP:   Turn [{turn_idx}] END. Appended role: {role}. converted_conversation len: {len(converted_conversation)}")
            # 打印最后一个添加的元素，看看它是什么
            pprint.pprint(converted_conversation[-1])

            # 在 append 之后，我们检查一下 action_tokens_list 的最后一个元素
        print(f"DEBUG_LOOP: Sample [{sample.__key__}] -- AFTER loop. Final converted_conversation len: {len(converted_conversation)}")
        print("DEBUG_LOOP: Final converted_conversation structure:")
        pprint.pprint([{'role': t['role']} for t in converted_conversation]) # 只打印角色结构
        print("-" * 50)
        conversation = converted_conversation

        is_problematic_sample = False
        for turn in converted_conversation:
            if turn['role'] == 'user' and isinstance(turn['content'], list):
                num_images_in_turn = sum(1 for item in turn['content'] if item.get('type') == 'image')
                # 我们只关心那些看起来会出错的样本
                if num_images_in_turn == 3: # 根据你的错误日志，问题出在图片数为3的样本
                    is_problematic_sample = True
                    break

        if is_problematic_sample:
            print(f"\nDEBUG_BUILD_IO: ---- Analyzing Sample [{sample.__key__}] ----")
            print(f"DEBUG_BUILD_IO: Input to build_conversation_tokens (converted_conversation):")
            # 为了日志整洁，我们只打印结构
            for i, turn in enumerate(converted_conversation):
                content_type = type(turn['content'])
                content_preview = ""
                if isinstance(turn['content'], list):
                    content_preview = f"list with {len(turn['content'])} items"
                else:
                    content_preview = f"'{str(turn['content'])[:30]}...'"
                print(f"  Turn {i}, Role: {turn['role']}, Content: {content_preview}")

        #############
        input_ids = self.build_conversation_tokens(converted_conversation, action_tokens_list)

        # input_ids = self.tokenizer.apply_chat_template(
        #     conversation, tokenize=True, return_tensors="np"
        # )[0]
        if is_problematic_sample:
            image_pad_id = self.tokenizer.vocab.get("<|image_pad|>")
            count_in_output = np.sum(np.array(input_ids) == image_pad_id)
            
            print(f"DEBUG_BUILD_IO: Output from build_conversation_tokens (input_ids):")
            print(f"  - Total length: {len(input_ids)}")
            print(f"  - Count of '<|image_pad|>' (ID: {image_pad_id}): {count_in_output}")
            print(f"DEBUG_BUILD_IO: -----------------------------------------------\n")

        target = input_ids.copy()
        pad_token_id = IGNORE_IDX
        
        # 计算系统提示长度并设置mask
        if converted_conversation[0]["role"] == "system":
            system_tokens = self.build_conversation_tokens([converted_conversation[0]], [action_tokens_list[0]])
            system_prompt_prefix = len(system_tokens)
            target[:system_prompt_prefix] = pad_token_id
            start_idx = 1
        else:
            system_prompt_prefix = 0
            start_idx = 0
        
        # 为每个turn设置mask
        offset = system_prompt_prefix
        for turn_idx in range(start_idx, len(converted_conversation)):
            turn = converted_conversation[turn_idx]
            action_tokens = action_tokens_list[turn_idx] if turn_idx < len(action_tokens_list) else []
            
            # 计算当前turn的token长度
            turn_tokens = self.build_conversation_tokens([turn], [action_tokens])
            n_tokens = len(turn_tokens)
            
            if turn["role"] == "user":
                # 用户输入全部mask
                target[offset : offset + n_tokens] = pad_token_id
            elif turn["role"] == "assistant":
                # 助手开始部分mask，生成内容不mask
                assistant_generation_prefix = 3  # <im_start>assistant\n
                target[offset : offset + assistant_generation_prefix] = pad_token_id
            
        offset += n_tokens
        # system_prompt_prefix = len(
        #     self.tokenizer.apply_chat_template([conversation[0]], tokenize=True)
        # )
        # assistant_generation_prefix = 3  # <im_start>assistant\n
        # pad_token_id = IGNORE_IDX
        # target[:system_prompt_prefix] = pad_token_id
        # offset = system_prompt_prefix
        # for turn_idx, turn in enumerate(conversation[1:]):
        #     turn_tokens = self.tokenizer.apply_chat_template(
        #         [turn], tokenize=True, return_tensors="np"
        #     )[0]
        #     turn_content = turn_tokens[system_prompt_prefix:]
        #     n_tokens = len(turn_content)
        #     if (target[offset : offset + n_tokens] != turn_content).any():
        #         raise InternalWarning("Encode Error")

        #     if turn["role"] == "user":
        #         target[offset : offset + n_tokens] = pad_token_id
        #     elif turn["role"] == "assistant":
        #         target[offset : offset + assistant_generation_prefix] = pad_token_id
        #     offset += n_tokens

        merge_length = self.merge_size**2
        image_token_id, video_token_id = self.tokenizer.encode(["<|image_pad|>", "<|video_pad|>"])

        image_token_indices = np.where(input_ids == image_token_id)[0]
        assert len(image_token_indices) == len(
            image_thw_grids
        ), f"The sample [{sample.__key__}] with {len(image_thw_grids)} images in the sample, but {len(image_token_indices)} image placeholders!"
        video_token_indices = np.where(input_ids == video_token_id)[0]
        assert len(video_token_indices) == len(
            video_thw_grids
        ), f"With {len(video_thw_grids)} videos in the sample, but {len(video_token_indices)} video placeholders!"
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
        if target_length > self.seq_len:
            dataset_logger.warning(
                f"Sample id [{sample.__key__}] has long sequence with length {target_length}, cutoff to max [self.seq_len+64={self.seq_len}] in batch function..."
            )
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

        target = np.roll(final_input_masks, shift=-1)
        target[-1] = pad_token_id

        if (target == pad_token_id).all():
            raise InternalWarning(
                f"Sample id [{sample.__key__}] with all masked label, the data is invalid! Dropped!"
            )
        ACTION_TOKEN_START_ID = 151936
        #action token 总数是 2048
        ACTION_TOKEN_END_ID = ACTION_TOKEN_START_ID + 2048  

        action_token_indices = np.where(
            (final_input_ids >= ACTION_TOKEN_START_ID) &
            (final_input_ids < ACTION_TOKEN_END_ID)
        )[0]
        
        if len(action_token_indices) > 0:
            for idx in action_token_indices:
                if idx > 0:
                    target[idx - 1] = final_input_ids[idx]

        image_input_mask = final_input_ids == self.tokenizer.image_token_id
        video_input_mask = final_input_ids == self.tokenizer.video_token_id
        if hasattr(self, '_debug_count'):
            self._debug_count += 1
        else:
            self._debug_count = 1

        if self._debug_count <= 3:  # 只调试前3个样本
            print(f"\n🔍 Sample {sample.__key__} - Debug {self._debug_count}")
            print("Input IDs (前20个):")
            for i in range(min(20, len(final_input_ids))):
                token_id = final_input_ids[i]
                try:
                    token_text = self.tokenizer.decode([token_id], skip_special_tokens=False)
                    target_status = "IGNORE" if target[i] == -100 else str(target[i])
                    print(f"  {i:2d}: {token_id:6d} -> {repr(token_text):15s} (target: {target_status})")
                except:
                    print(f"  {i:2d}: {token_id:6d} -> <DECODE_ERROR>")
            
            # 检查action tokens
            ACTION_TOKEN_START_ID = 151936
            action_positions =  np.where(
            (final_input_ids >= ACTION_TOKEN_START_ID) &
            (final_input_ids < ACTION_TOKEN_END_ID)
        )[0]
            if len(action_positions) > 0:
                print(f"Action tokens found at positions: {action_positions}")
                for pos in action_positions[:5]:  # 只显示前5个
                    action_id = final_input_ids[pos] - ACTION_TOKEN_START_ID
                    print(f"  位置 {pos}: action_token_{action_id}")
            print("-" * 50)
        encode_end = time.time()
        print(f"LZY: encode_chatml use time: {(encode_end - encode_start) * 1000} ms")

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
        )

    def encode_vqa(self, sample: VQASample):
        augment = (
            sample.__subflavors__["augmentation"]
            if "augmentation" in sample.__subflavors__
            else False
        )
        has_video = (
            sample.__subflavors__["has_video"] if "has_video" in sample.__subflavors__ else False
        )

        if has_video:
            raise NotImplementedError("You should use sharegpt dataset to train with videos.")
        else:
            imgs = get_visual_transform(sample.image)
            flatten_patches, thw_grids = self._flatten_visual_inputs(imgs, is_image=True)

        assert "<image>" in sample.context
        if isinstance(sample.answers, list):
            answer_list = sample.answers
            weight_list = np.array(sample.answer_weights).astype(np.float32)
            weight_list = weight_list / np.sum(weight_list)
            answer_idx = np.random.choice(weight_list.shape[0], 1, p=weight_list)[0]
            answer = answer_list[answer_idx]
        else:
            answer = sample.answers

        conversation = [
            {"role": "user", "content": convert_to_qwen2vl_content(sample.context)},
            {"role": "assistant", "content": answer},
        ]

        user_inputs = self.tokenizer.apply_chat_template(conversation[:-1], tokenize=False)
        text = self.tokenizer.apply_chat_template(conversation, tokenize=False)

        merge_length = self.merge_size**2
        image_token = "<|image_pad|>"
        assert len(thw_grids) == 1, "Only one image per sample is supported!"
        index = 0
        while image_token in text:
            grid_t, grid_h, grid_w = thw_grids[index]
            l = grid_t * grid_h * grid_w
            text = text.replace(image_token, "<|placeholder|>" * (l // merge_length), 1)
            user_inputs = user_inputs.replace(
                image_token, "<|placeholder|>" * (l // merge_length), 1
            )
            index += 1
        text = text.replace("<|placeholder|>", image_token)
        user_inputs = user_inputs.replace("<|placeholder|>", image_token)

        input_ids = self.tokenizer.tokenize(text)
        user_input_ids = self.tokenizer.tokenize(user_inputs)
        if len(input_ids) > self.seq_len:
            raise InternalWarning(f"Long sequence with length {len(input_ids)} found, dropped...")

        target = np.array(input_ids[1:] + [IGNORE_IDX])
        if len(user_input_ids) >= len(input_ids):
            raise InternalWarning(f"Sample not supported, dropped...")
        if not (np.array(user_input_ids) == np.array(input_ids[: len(user_input_ids)])).all():
            raise InternalWarning(f"Sample not supported, dropped...")
        target[: len(user_input_ids) - 1] = IGNORE_IDX

        img_token_id = self.tokenizer.image_token_id
        image_input_mask = np.array(input_ids) == img_token_id

        return ImageTaskSample(
            __key__=sample.__key__,
            __subflavors__=sample.__subflavors__,
            imgs=flatten_patches,
            videos=list(),
            image_thw_grids=thw_grids,
            video_thw_grids=torch.empty([0, 3], dtype=torch.long),
            image_input_mask=image_input_mask,
            video_input_mask=None,
            second_per_grid_ts=np.zeros(0, dtype=np.float32),
            text=input_ids,
            target=target,
        )

    def batch(self, samples: List[ImageTaskSample]) -> VQATaskBatch:
        imgs = [s.imgs for s in samples if isinstance(s.imgs, np.ndarray) and s.imgs.size > 0]
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

        videos = [s.videos for s in samples if isinstance(s.videos, np.ndarray) and s.videos.size > 0]
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

        global FIRST_MAX_PADDING_FLAG, LAST_LARGE_IMG, MAX_IMG_THRESHHOLD
        # MODIFIED START: Restore original cache clearing logic
        if self.args.curr_iteration > 0 and self.args.curr_iteration % CLEAR_CACHE_ITERATION == 0:
            FIRST_MAX_PADDING_FLAG = True

        if image_thw_grids.prod(axis=-1).sum() // 4 > MAX_IMG_THRESHHOLD:
            MAX_IMG_THRESHHOLD = image_thw_grids.prod(axis=-1).sum() // 4
            FIRST_MAX_PADDING_FLAG = True
            LAST_LARGE_IMG = True

        # if self.args.image_max_pixels == 12845056 and image_thw_grids.prod(axis=-1).sum() // 4 > 16384:
        #     FIRST_MAX_PADDING_FLAG = True
        # if self.args.image_max_pixels == 589824 and image_thw_grids.prod(axis=-1).sum() // 4 > 5000:
        #     FIRST_MAX_PADDING_FLAG = True
        # if self.args.image_max_pixels > 589824 and self.args.image_max_pixels < 12845056 and image_thw_grids.prod(axis=-1).sum() // 4 > 10000:
        #     FIRST_MAX_PADDING_FLAG = True
        # # MODIFIED END

        if not self.args.enable_variable_seq_lengths:
            max_seq_len = self.seq_len
        else:
            if FIRST_MAX_PADDING_FLAG:
                max_seq_len = self.seq_len
                FIRST_MAX_PADDING_FLAG = False
            else:
                max_seq_len = max(len(s.text) for s in samples)
                max_seq_len = min(max_seq_len, self.seq_len)
        if self.cp_size > 1 or self.sequence_parallel:
            max_seq_len = math.ceil(max_seq_len / (self.tp_size * self.cp_size)) * (
                self.tp_size * self.cp_size
            )
        text_mat = np.full((len(samples), max_seq_len), self.tokenizer.pad_token_id, dtype=np.int64)
        target_mat = np.full((len(samples), max_seq_len), IGNORE_IDX, dtype=np.int64)

        image_input_masks = np.zeros_like(text_mat, dtype=bool)
        video_input_masks = np.zeros_like(text_mat, dtype=bool)
        for i, s in enumerate(samples):
            text_len = min(max_seq_len, len(s.text))
            target_len = min(max_seq_len, len(s.target))
            text_mat[i, :text_len] = np.array(s.text)[:text_len]
            if s.image_input_mask is not None:
                image_input_masks[i, :text_len] = np.array(s.image_input_mask)[:text_len]
            if s.video_input_mask is not None:
                video_input_masks[i, :text_len] = np.array(s.video_input_mask)[:text_len]
            target_mat[i, :target_len] = np.array(s.target)[:target_len]

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