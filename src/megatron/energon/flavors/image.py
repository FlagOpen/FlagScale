# Copyright (c) 2024, NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import dataclass

import torch

from megatron.energon.flavors.base_dataset import Sample
from megatron.energon.flavors.base_webdataset import DefaultDecoderWebdataset


@dataclass
class ImageSample(Sample):
    """Sample type for an image, e.g. for image reconstruction."""

    #: The input image tensor in the shape (C, H, W)
    image: torch.Tensor


class ImageWebdataset(DefaultDecoderWebdataset[ImageSample]):
    __sample_type__ = ImageSample
