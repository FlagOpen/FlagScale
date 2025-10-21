# Adopted from https://github.com/alibaba/Pai-Megatron-Patch/blob/8949a6647cbf6b39837ad3dd911fa4aa0726895b/megatron_patch/data/energon/chatml.py.
# Copied from https://github.com/alibaba/Pai-Megatron-Patch/blob/8949a6647cbf6b39837ad3dd911fa4aa0726895b/megatron_patch/data/energon/chatml.py.

import json
import pickle
import re
import warnings

from dataclasses import dataclass
from typing import Any, Dict, List, Union

import torch

from webdataset.autodecode import Decoder, imagehandler

from megatron.energon.av import AVWebdatasetDecoder
from megatron.energon.epathlib.epath import EPath
from megatron.energon.flavors.base_dataset import Sample
from megatron.energon.flavors.webdataset import DefaultDecoderWebdatasetFactory


@dataclass
class ChatMLSampleRobotics(Sample):
    """multi-turn complex samples with images and videos"""

    imgs: List[str]
    videos: List[List[str]]
    conversation: str  # JSON string of GPT-format conversations
    metadata: Dict[str, Any] = None


class NestedImagesPathHandler:
    def __init__(self, imagespec):
        """Create an image handler.

        :param imagespec: short string indicating the type of decoding
        """
        self.extensions = ["jpgs", "videos", "metadata"]

    def __call__(self, key, data):
        """Perform nested image decoding.

        :param key: file name extension
        :param data: binary data
        """
        extension = re.sub(r".*[.]", "", key)
        if extension.lower() not in self.extensions:
            return None

        # 现在只处理图像和视频
        if extension.lower() in ["jpgs", "videos"]:
            try:
                return pickle.loads(data)
            except Exception as e:
                print(f"Warning: Failed to decode {extension}: {e}")
                return None
        elif extension.lower() == "metadata":
            try:
                # 首先将字节串解码为 UTF-8 字符串，然后用 json.loads 解析
                return json.loads(data.decode('utf-8'))
            except Exception as e:
                print(f"Warning: Failed to decode metadata json: {e}")
                return None

        return None  # 其他未知情况返回 None


# During training, data is automatically decoded to from default webdataset to 'ChatMLSampleRobotics' when loaded using energon-dataloader,
# and this is not done during preparation!!!
# After decoding, the data is passed into the TaskEncoder for further processing.
class ChatMLWebdatasetRobotics(DefaultDecoderWebdatasetFactory[ChatMLSampleRobotics]):
    __sample_type__ = ChatMLSampleRobotics

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
            # self._decoder.handlers.insert(0, NestedImagesPathHandler(self.image_decode))
            self._decoder = Decoder([NestedImagesPathHandler(self.image_decode)])
