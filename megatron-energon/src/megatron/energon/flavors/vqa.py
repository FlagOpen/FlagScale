# Copyright (c) 2024, NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import dataclass
from typing import List, Optional

import torch

from megatron.energon.flavors.base_dataset import Sample
from megatron.energon.flavors.base_webdataset import DefaultDecoderWebdataset


@dataclass
class VQASample(Sample):
    """Sample type for visual question answering."""

    #: The input image tensor in the shape (C, H, W)
    image: torch.Tensor
    #: The context/question for the image
    context: str

    #: The possible answers. Not set for testing.
    answers: Optional[List[str]] = None
    #: The weights of the possible answers. Optionally available.
    answer_weights: Optional[torch.Tensor] = None


class VQAWebdataset(DefaultDecoderWebdataset[VQASample]):
    __sample_type__ = VQASample
