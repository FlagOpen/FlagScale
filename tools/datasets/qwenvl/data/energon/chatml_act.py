# Copied from https://github.com/alibaba/Pai-Megatron-Patch/blob/8949a6647cbf6b39837ad3dd911fa4aa0726895b/megatron_patch/data/energon/chatml.py.

import pickle
import re
import warnings

from dataclasses import dataclass
from typing import List, Union

import torch

from webdataset.autodecode import Decoder, imagehandler

from megatron.energon.av import AVWebdatasetDecoder
from megatron.energon.epathlib.epath import EPath
from megatron.energon.flavors.base_dataset import Sample
from megatron.energon.flavors.webdataset import DefaultDecoderWebdatasetFactory


@dataclass
class ChatMLSample(Sample):
    """multi-turn complex samples with images and videos"""

    imgs: List[str]
    videos: List[List[str]]
    conversation: str  # JSON string of GPT-format conversations
    action_token: Union[str, List[str], None] = None  



# class NestedImagesHandler:
#     def __init__(self, imagespec):
#         """Create an image handler.

#         :param imagespec: short string indicating the type of decoding
#         """
#         self.extensions = ["jpgs", "videos"]
#         self.extensions_mapping = {"jpgs": "jpg", "videos": "jpg"}
#         self.image_handler = imagehandler(imagespec)

#     def __call__(self, key, data):
#         """Perform nested image decoding.

#         :param key: file name extension
#         :param data: binary data
#         """
#         extension = re.sub(r".*[.]", "", key)
#         if extension.lower() not in self.extensions:
#             return None
#         data = pickle.loads(data)
#         key = self.extensions_mapping[extension]
#         if extension.lower() == "jpgs":
#             data = [self.image_handler(key, d) for d in data]
#         else:
#             data = [[self.image_handler(key, d) for d in video] for video in data]
#         return data


class NestedImagesPathHandler:
    def __init__(self, imagespec):
        """Create an image handler.

        :param imagespec: short string indicating the type of decoding
        """
        self.extensions = ["jpgs", "videos", "action_tokens"]  # 添加 action_tokens
        # self.extensions = ["jpgs", "videos"]
        self.extensions_mapping = {"jpgs": "jpg", "videos": "jpg", "action_tokens": "action_token"}
        # self.extensions_mapping = {"jpgs": "jpg", "videos": "jpg"}

    def __call__(self, key, data):
        """Perform nested image decoding.

        :param key: file name extension
        :param data: binary data
        """
        extension = re.sub(r".*[.]", "", key)
        if extension.lower() not in self.extensions:
            return None
        try:
            data = pickle.loads(data)
        except Exception as e:
            # 如果解码失败，返回 None，这样字段就不会被设置
            print(f"Warning: Failed to decode {extension}: {e}")
            return None
        
        # 对于 action_tokens，直接返回路径数据，不需要特殊处理
        if extension.lower() == "action_tokens":
            return data
        
        return data
        # data = pickle.loads(data)
        # return data


# During training, data is automatically decoded to from default webdataset to 'ChatMLSample' when loaded using energon-dataloader,
# and this is not done during preparation!!!
# After decoding, the data is passed into the TaskEncoder for further processing.
class ChatMLWebdataset(DefaultDecoderWebdatasetFactory[ChatMLSample]):
    __sample_type__ = ChatMLSample

    def __init__(
        self,
        path: EPath,
        *,
        auto_decode: bool = True,
        image_decode="torchrgb",
        ignore_decoder_errors: bool = False,
        av_decode="AVDecoder",
        video_decode_audio: bool = False,
        **kwargs,
    ):
        super().__init__(
            path,
            auto_decode=auto_decode,
            image_decode=image_decode,
            ignore_decoder_errors=ignore_decoder_errors,
            av_decode=av_decode,
            video_decode_audio=video_decode_audio,
            **kwargs,
        )
        if auto_decode:
            self._decoder = Decoder([NestedImagesPathHandler(self.image_decode)])
        
    def _decode_sample(self, sample_dict):
        """Override to handle mixed datasets with/without action_tokens"""
        # 手动解码，更好地控制每个字段的处理
        decoded_sample = {}
        
        # 解码基础字段
        for key, value in sample_dict.items():
            if key in ["jpgs", "videos"]:
                try:
                    decoded_sample[key] = pickle.loads(value)
                except Exception as e:
                    print(f"Warning: Failed to decode {key}: {e}")
                    decoded_sample[key] = []
            elif key == "action_tokens":
                # action_tokens 字段可能存在也可能不存在
                try:
                    decoded_sample[key] = pickle.loads(value)
                except Exception as e:
                    print(f"Warning: Failed to decode action_tokens: {e}")
                    decoded_sample[key] = None
            elif key == "json":
                try:
                    decoded_sample[key] = value.decode('utf-8') if isinstance(value, bytes) else value
                except Exception as e:
                    print(f"Warning: Failed to decode json: {e}")
                    decoded_sample[key] = "{}"
            else:
                decoded_sample[key] = value
        
        # 创建 ChatMLSample 实例
        sample = ChatMLSample(
            __key__=decoded_sample.get("__key__", "unknown"),
            imgs=decoded_sample.get("jpgs", []),
            videos=decoded_sample.get("videos", []),
            conversation=decoded_sample.get("json", "{}"),
            action_token=decoded_sample.get("action_tokens", None)  # 可能为 None
        )
