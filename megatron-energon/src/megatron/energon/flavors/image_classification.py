# Copyright (c) 2024, NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import dataclass
from typing import Optional

import torch

from megatron.energon.flavors.base_dataset import Sample
from megatron.energon.flavors.base_webdataset import DefaultDecoderWebdataset


@dataclass
class ImageClassificationSample(Sample):
    """Sample type for classifying an image."""

    #: The input image tensor in the shape (C, H, W)
    image: torch.Tensor
    #: The class label of the image
    label: Optional[int] = None
    #: The class label of the image
    label_name: Optional[str] = None


class ImageClassificationWebdataset(DefaultDecoderWebdataset[ImageClassificationSample]):
    __sample_type__ = ImageClassificationSample
    type = ImageClassificationSample
