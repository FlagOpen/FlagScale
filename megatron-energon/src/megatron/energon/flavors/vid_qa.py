# Copyright (c) 2024, NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import dataclass
from typing import List, Optional

import torch

from megatron.energon.flavors.base_dataset import Sample
from megatron.energon.flavors.base_webdataset import DefaultDecoderWebdataset, VideoData


@dataclass
class VidQASample(Sample):
    """Sample type for video question answering."""

    #: The video data containing the image and audio info.
    video: VideoData
    #: The context/question for the image.
    context: str
    #: The possible answers. Not set for testing.
    answers: Optional[List[str]] = None
    #: The weights of the possible answers. Optionally available.
    answer_weights: Optional[torch.Tensor] = None


class VidQAWebdataset(DefaultDecoderWebdataset[VidQASample]):
    __sample_type__ = VidQASample
