# Copyright (c) 2024, NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import dataclass
from typing import List, Optional, Union

import torch

from megatron.energon.flavors.base_dataset import Sample
from megatron.energon.flavors.base_webdataset import DefaultDecoderWebdataset


@dataclass
class OCRSample(Sample):
    """Sample type for optical character recognition."""

    #: The input image tensor in the shape (C, H, W)
    image: torch.Tensor
    #: The text contained in the image
    text: str
    #: The bounding boxes of the blocks in the image float(N, 4|5<x, y, w, h>)
    block_boxes: Optional[torch.Tensor] = None
    #: The classes of the blocks in the image int(N, 1<block_class>)
    block_classes: Optional[Union[torch.Tensor, List[str]]] = None
    #: The text contained in each block (N,)
    block_text: Optional[List[str]] = None
    #: The bounding boxes of the lines in the image float(N, 4|5<x, y, w, h[, confidence]>)
    lines_boxes: Optional[torch.Tensor] = None
    #: The text contained in each line (N,)
    lines_text: Optional[List[str]] = None
    #: The bounding boxes of the words in the image float(N, 4|5<x, y, w, h[, confidence]>)
    words_boxes: Optional[torch.Tensor] = None
    #: The text contained in each word (N,)
    words_text: Optional[List[str]] = None
    #: The bounding boxes of the chars in the image float(N, 4|5<x, y, w, h[, confidence]>)
    chars_boxes: Optional[torch.Tensor] = None
    #: The character contained in each char (N,)
    chars_text: Optional[List[str]] = None


class OCRWebdataset(DefaultDecoderWebdataset[OCRSample]):
    __sample_type__ = OCRSample
