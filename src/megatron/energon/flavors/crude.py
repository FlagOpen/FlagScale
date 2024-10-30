# Copyright (c) 2024, NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Union

import webdataset

from megatron.energon.epathlib import EPath
from megatron.energon.flavors.base_dataset import SavableDataset
from megatron.energon.flavors.base_webdataset import (
    BaseWebdataset,
    DefaultGenericWebdataset,
    ImageDecoder,
)
from megatron.energon.flavors.webdataset.structs import FilteredSample


class CrudeSample(dict):
    """Generic sample type to be processed later."""


class CrudeWebdataset(DefaultGenericWebdataset[CrudeSample]):
    """The CrudeWebdataset is used to load crude / raw samples and
    decode them in the user code using so-called cookers.

    See the documentation under "Crude Data" for more information.
    """

    __sample_type__ = CrudeSample

    def __init__(
        self,
        path: EPath,
        *,
        subflavor: Optional[str] = None,
        subflavors: Optional[Dict[str, Any]] = None,
        part_filter: Optional[Union[str, List[str], Callable[[str], bool]]] = None,
        image_decode: Optional[ImageDecoder] = "torchrgb",
        ignore_decoder_errors: bool = False,
        **kwargs,
    ):
        # The sample is already a dict, but we extend it with the subflavor
        self.subflavor = subflavor
        self.subflavors = subflavors or {}
        self.image_decode = image_decode
        self.ignore_decoder_errors = ignore_decoder_errors

        self._sample_loader = lambda sample: {
            **sample,
            "__subflavor__": self.subflavor,
            "__subflavors__": self.subflavors,
        }

        # A simple part filter can be specified as a list of accepted extensions
        if isinstance(part_filter, list):
            parts = set(part_filter)
            part_filter = lambda part: part in parts
        else:
            assert part_filter is None

        # We skip the parent class __init__ and call the BaseWebdataset.__init__ directly
        BaseWebdataset.__init__(self, path, **kwargs, part_filter=part_filter)

    def _decode_error_handler(self, exc: Exception) -> bool:
        if self.ignore_decoder_errors:
            return True
        raise exc

    def _process_samples(
        self, dataset: SavableDataset[FilteredSample]
    ) -> SavableDataset[CrudeSample]:
        from megatron.energon.wrappers import IterMapDataset, MapDataset

        if self.image_decode:
            # With image decoding:
            decoder = webdataset.decode(self.image_decode, handler=self._decode_error_handler)
            dataset = IterMapDataset(
                dataset,
                decoder,
                error_handler=self.error_handler,
                stateless_iter_fn=True,
                worker_config=self.worker_config,
            )

        return MapDataset(
            dataset,
            self._load_sample,
            error_handler=self.error_handler,
            stateless_map_fn=True,
            worker_config=self.worker_config,
        )

    def _load_sample(self, sample: FilteredSample) -> CrudeSample:
        return self.__sample_type__(**self._sample_loader(sample))

    def config(self) -> Dict[str, Any]:
        return {
            **super().config(),
            "image_decode": self.image_decode,
            "ignore_decoder_errors": self.ignore_decoder_errors,
        }
