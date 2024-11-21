# Copyright (c) 2024, NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import dataclass

import torch

from megatron.energon.flavors.base_dataset import Sample
from megatron.energon.flavors.base_webdataset import DefaultDecoderWebdataset


@dataclass
class CaptioningSample(Sample):
    """Sample type for image captioning."""

    #: The input image tensor in the shape (C, H, W)
    image: torch.Tensor

    #: The caption string
    caption: str


class CaptioningWebdataset(DefaultDecoderWebdataset[CaptioningSample]):
    __sample_type__ = CaptioningSample
