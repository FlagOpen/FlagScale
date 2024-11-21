# Copyright (c) 2024, NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause

from enum import Enum
from typing import TypeVar

import PIL.Image
import torch

TTransform = TypeVar("TTransform", bound=torch.nn.Module)


class Sampling(Enum):
    BILINEAR = PIL.Image.BILINEAR
    NEAREST = PIL.Image.NEAREST
