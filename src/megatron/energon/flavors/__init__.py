# Copyright (c) 2024, NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause

from megatron.energon.flavors.base_dataset import BaseCoreDataset, Sample, SavableDataset
from megatron.energon.flavors.base_webdataset import (
    BaseWebdataset,
    DefaultDecoderWebdataset,
    DefaultGenericWebdataset,
    VideoData,
)
from megatron.energon.flavors.captioning import CaptioningSample, CaptioningWebdataset
from megatron.energon.flavors.crude import CrudeSample, CrudeWebdataset
from megatron.energon.flavors.image import ImageSample, ImageWebdataset
from megatron.energon.flavors.image_classification import (
    ImageClassificationSample,
    ImageClassificationWebdataset,
)
from megatron.energon.flavors.interleaved import InterleavedSample, InterleavedWebdataset
from megatron.energon.flavors.multichoice_vqa import MultiChoiceVQASample, MultiChoiceVQAWebdataset
from megatron.energon.flavors.ocr import OCRSample, OCRWebdataset
from megatron.energon.flavors.similarity_interleaved import (
    SimilarityInterleavedSample,
    SimilarityInterleavedWebdataset,
)
from megatron.energon.flavors.text import TextSample, TextWebdataset
from megatron.energon.flavors.vid_qa import VidQASample, VidQAWebdataset
from megatron.energon.flavors.vqa import VQASample, VQAWebdataset
from megatron.energon.flavors.vqa_and_ocr import VQAOCRWebdataset

__all__ = [
    "BaseCoreDataset",
    "BaseWebdataset",
    "CaptioningSample",
    "CaptioningWebdataset",
    "CrudeSample",
    "CrudeWebdataset",
    "DefaultGenericWebdataset",
    "DefaultDecoderWebdataset",
    "ImageClassificationSample",
    "ImageClassificationWebdataset",
    "ImageSample",
    "ImageWebdataset",
    "InterleavedSample",
    "InterleavedWebdataset",
    "MultiChoiceVQASample",
    "MultiChoiceVQAWebdataset",
    "OCRSample",
    "OCRWebdataset",
    "Sample",
    "SavableDataset",
    "SimilarityInterleavedSample",
    "SimilarityInterleavedWebdataset",
    "TextSample",
    "TextWebdataset",
    "VQASample",
    "VQAWebdataset",
    "VQAOCRWebdataset",
    "VideoData",
    "VidQASample",
    "VidQAWebdataset",
]
