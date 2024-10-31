# Copyright (c) 2024, NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import dataclass
from typing import List, Optional

import torch

from megatron.energon.flavors.base_dataset import Sample
from megatron.energon.flavors.base_webdataset import DefaultDecoderWebdataset


@dataclass
class VQAOCRSample(Sample):
    """Sample type for visual question answering."""

    #: The input image tensor in the shape (C, H, W)
    image: torch.Tensor

    #: The context/question for the image (VQA)
    context: str
    #: The text contained in the image (OCR)
    text: str

    #: The possible answers. Not set for testing. (VQA)
    answers: Optional[List[str]] = None
    #: The weights of the possible answers. Optionally available. (VQA)
    answer_weights: Optional[torch.Tensor] = None
    #: The bounding boxes of the words in the image (N, 4|5<x, y, w, h[, confidence]>) (OCR)
    words_boxes: Optional[torch.Tensor] = None
    #: The text contained in each word (N,) (OCR)
    words_text: Optional[List[str]] = None


class VQAOCRWebdataset(DefaultDecoderWebdataset[VQAOCRSample]):
    __sample_type__ = VQAOCRSample
