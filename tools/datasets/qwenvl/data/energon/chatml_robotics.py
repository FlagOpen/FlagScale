# Adopted from https://github.com/alibaba/Pai-Megatron-Patch/blob/8949a6647cbf6b39837ad3dd911fa4aa0726895b/megatron_patch/data/energon/chatml.py.

import logging
import pickle
import re

from dataclasses import dataclass
from typing import List, Union

from webdataset.autodecode import Decoder

from megatron.energon.epathlib.epath import EPath
from megatron.energon.flavors.base_dataset import Sample
from megatron.energon.flavors.webdataset import DefaultDecoderWebdatasetFactory

logger = logging.getLogger(__name__)


@dataclass
class ChatMLSample(Sample):
    """multi-turn complex samples with images and videos"""

    imgs: List[str]
    videos: List[List[str]]
    action_qpos: str
    action_eepose: str
    state_qpos: str
    state_eepose: str
    conversation: str  # JSON string of GPT-format conversations
    action_token: Union[str, List[str], None] = None


class NestedImagesPathHandler:
    def __init__(self, imagespec):
        """Create an image handler.

        :param imagespec: short string indicating the type of decoding
        """
        self.extensions = [
            "jpgs",
            "videos",
            "action_tokens",
            "action_qpos",
            "action_eepose",
            "state_qpos",
            "state_eepose",
        ]  # 添加 action_tokens
        # self.extensions = ["jpgs", "videos"]
        self.extensions_mapping = {
            "jpgs": "jpg",
            "videos": "jpg",
            "action_tokens": "action_token",
            "action_qpos": "action_qpos",
            "action_eepose": "action_eepose",
            "state_qpos": "state_qpos",
            "state_eepose": "state_eepose",
        }
        # self.extensions_mapping = {"jpgs": "jpg", "videos": "jpg"}

    def __call__(self, key, data):
        """Perform nested image decoding.

        :param key: file name extension
        :param data: binary data
        """
        extension = re.sub(r".*[.]", "", key)
        if extension.lower() not in self.extensions:
            return None
        if extension.lower() == "action_tokens":
            try:
                return pickle.loads(data)
            except (pickle.UnpicklingError, EOFError):
                try:
                    return data.decode('utf-8')
                except Exception as e_decode:
                    logger.error(
                        f"Warning: Failed to decode action_token as a raw string after pickle failed: {e_decode}"
                    )
                    return None
            except Exception as e_other:
                logger.error(
                    f"Warning: An unexpected error occurred while decoding action_token: {e_other}"
                )
                return None

        elif extension.lower() in ["jpgs", "videos", "actions"]:
            try:
                return pickle.loads(data)
            except Exception as e:
                logger.error(f"Warning: Failed to decode {extension}: {e}")
                return None

        return None


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
