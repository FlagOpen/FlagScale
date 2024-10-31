# Copyright (c) 2024, NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause


class SkipSample(Exception):
    """Exception to raise in the map_fn to skip a sample."""
