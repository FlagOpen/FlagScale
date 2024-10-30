# Copyright (c) 2024, NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import dataclass
from typing import List, Optional

import torch

from megatron.energon.flavors.base_dataset import Sample
from megatron.energon.flavors.base_webdataset import DefaultDecoderWebdataset


@dataclass
class SimilarityInterleavedSample(Sample):
    """Sample type for interleaved media such as text with images, but without image-text alignment.
    That alignment has to be assigned from the similarity matrix."""

    #: The images of the sequence
    images: List[torch.Tensor]
    #: The texts of the sequence
    texts: List[str]

    #: Similarity matrix between image and text entries in the sequence
    similarity_matrix: Optional[torch.Tensor] = None

    #: The index within texts representing the sentence that this image is matched to
    matched_text_indices: Optional[List[int]] = None


class SimilarityInterleavedWebdataset(DefaultDecoderWebdataset[SimilarityInterleavedSample]):
    __sample_type__ = SimilarityInterleavedSample
