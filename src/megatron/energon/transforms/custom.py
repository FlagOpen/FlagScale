# Copyright (c) 2024, NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause

import torch

from megatron.energon.transforms.mappers import TransformMapper


class CustomTransform(TransformMapper[torch.nn.Module]):
    """Abstract class for additional custom transforms.
    Inherit from this and override at least `apply_transform`.
    """
