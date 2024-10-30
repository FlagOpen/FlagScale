# Copyright (c) 2024, NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import dataclass
from typing import List, Union

import torch

from megatron.energon.flavors.base_dataset import Sample
from megatron.energon.flavors.base_webdataset import DefaultDecoderWebdataset


@dataclass
class InterleavedSample(Sample):
    """Sample type for interleaved media such as text with images."""

    #: The interleaved media (either torch.tensor for an image, or str for text)
    sequence: List[Union[torch.Tensor, str]]


class InterleavedWebdataset(DefaultDecoderWebdataset[InterleavedSample]):
    __sample_type__ = InterleavedSample
