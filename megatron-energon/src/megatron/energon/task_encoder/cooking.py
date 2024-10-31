# Copyright (c) 2024, NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import dataclass
from typing import Callable, Optional

from megatron.energon.flavors.base_dataset import Sample
from megatron.energon.flavors.crude import CrudeSample


@dataclass
class Cooker:
    """A cooker transforms a crude sample (simple dict) into a
    specific sample type inheriting from `Sample`.

    The `cook` method performs the transformation,
    the other fields are used to select the samples which this cooker
    can transform. If no filters are provided, the cooker will
    transform any sample.
    """

    cook: Callable[[dict], Sample]

    # If multiple of the following conditions are provided
    # then the sample must satisfy all of them.
    is_subflavor: Optional[str] = None
    has_subflavors: Optional[dict] = None
    condition: Optional[Callable[[dict], bool]] = None

    def is_match(self, crude_sample: CrudeSample) -> bool:
        if self.is_subflavor is not None:
            if crude_sample["__subflavor__"] != self.is_subflavor:
                return False

        if self.has_subflavors is not None:
            # Checks if the dict entries provided as a filter all match
            # the ones in the sample. The sample may have additional entries.
            for k, v in self.has_subflavors.items():
                if (
                    k not in crude_sample["__subflavors__"]
                    or crude_sample["__subflavors__"][k] != v
                ):
                    return False

        if self.condition is not None:
            if not self.condition(crude_sample):
                return False

        return True


def basic_sample_keys(crude_sample: dict) -> dict:
    """A convenience helper to extract the basic keys from a crude sample,
    which you will always need to forward to the cooked sample."""

    return {k: v for k, v in crude_sample.items() if k in Sample.__dataclass_fields__.keys()}
