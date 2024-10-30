# Copyright (c) 2024, NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import dataclass
from typing import List, Optional

import torch

from megatron.energon.flavors.base_dataset import Sample
from megatron.energon.flavors.base_webdataset import DefaultDecoderWebdataset


@dataclass
class MultiChoiceVQASample(Sample):
    """Sample type for visual question answering."""

    #: The input image tensor in the shape (C, H, W)
    image: torch.Tensor
    #: The context/question for the image
    context: str

    #: The candidate answers.
    choices: Optional[List[str]] = None
    #: The index of the correct answer.
    correct_choice_idx: int = 0


class MultiChoiceVQAWebdataset(DefaultDecoderWebdataset[MultiChoiceVQASample]):
    __sample_type__ = MultiChoiceVQASample
