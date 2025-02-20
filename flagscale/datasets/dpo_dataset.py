# Copyright (c) 2024, BAAI. All rights reserved.

import logging
import os
import time
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy
import torch

from megatron.core.datasets.gpt_dataset import (
    GPTDataset,
    GPTDatasetConfig,
    _get_ltor_masks_and_position_ids,
)
from megatron.core.datasets.indexed_dataset import (
    IndexedDataset,
    get_bin_path,
    get_idx_path,
)
from megatron.core.datasets.utils import Split

logger = logging.getLogger(__name__)


@dataclass
class DPODatasetConfig(GPTDatasetConfig):
    """Configuration object for Megatron Core DPO datasets"""

    pass


class DPODataset(GPTDataset):
    """The base GPT dataset

    Args:
        indexed_dataset (IndexedDataset): The IndexedDataset around which to build the GPTDataset

        dataset_path (Optional[str]): The real path on disk to the dataset, for bookkeeping

        indexed_indices (numpy.ndarray): The set of the documents indices to expose

        num_samples (Optional[int]): The number of samples to draw from the indexed dataset. When None, build as many samples as correspond to one epoch.

        index_split (Split): The indexed_indices Split

        config (GPTDatasetConfig): The config
    """

    def __init__(
        self,
        indexed_dataset: IndexedDataset,
        dataset_path: Optional[str],
        indexed_indices: numpy.ndarray,
        num_samples: Optional[int],
        index_split: Split,
        config: DPODatasetConfig,
    ) -> None:
        self.config = config
        self.chosen_loss_mask_dataset = None
        self.chosen_ref_log_probs_dataset = None
        self.rejected_dataset = None
        self.rejected_loss_mask_dataset = None
        self.rejected_ref_log_probs_dataset = None

        super().__init__(
            indexed_dataset,
            dataset_path,
            indexed_indices,
            num_samples,
            index_split,
            config,
        )

        self._build_pairs_dataset()

    def _build_single_dataset(self, path_prefix) -> IndexedDataset:
        idx_path = get_idx_path(path_prefix)
        bin_path = get_bin_path(path_prefix)
        assert os.path.exists(idx_path) and os.path.exists(bin_path), \
            f"Dataset {path_prefix} not existed."

        return IndexedDataset(
            path_prefix, multimodal=False, mmap=self.config.mmap_bin_files)

    def _build_pairs_dataset(self) -> None:
        """
          Load Pairs IndexedDataset
        """
        chosen_prefix = "chosen_text_document"
        chosen_loss_mask_prefix = "chosen_loss_mask_document"
        chosen_ref_log_probs_prefix = "chosen_ref_log_probs_document"
        rejected_prefix = "rejected_text_document"
        rejected_loss_mask_prefix = "rejected_loss_mask_document"
        rejected_ref_log_probs_prefix = "rejected_ref_log_probs_document"

        assert self.dataset_path.endswith(chosen_prefix), f"Dataset prefixes should follow naming rules."

        path_prefix = self.dataset_path[: -len(chosen_prefix)] + chosen_loss_mask_prefix
        self.chosen_loss_mask_dataset = self._build_single_dataset(path_prefix)

        path_prefix = self.dataset_path[: -len(chosen_prefix)] + chosen_ref_log_probs_prefix
        self.chosen_ref_log_probs_dataset = self._build_single_dataset(path_prefix)

        path_prefix = self.dataset_path[: -len(chosen_prefix)] + rejected_prefix
        self.rejected_dataset = self._build_single_dataset(path_prefix)

        path_prefix = self.dataset_path[: -len(chosen_prefix)] + rejected_loss_mask_prefix
        self.rejected_loss_mask_dataset = self._build_single_dataset(path_prefix)

        path_prefix = self.dataset_path[: -len(chosen_prefix)] + rejected_ref_log_probs_prefix
        self.rejected_ref_log_probs_dataset = self._build_single_dataset(path_prefix)

        assert self.chosen_loss_mask_dataset and self.chosen_ref_log_probs_dataset and \
            self.rejected_dataset and self.rejected_loss_mask_dataset and self.rejected_ref_log_probs_dataset, \
            f"Some datasets not existed."

    def __getitem__(self, idx: Optional[int]) -> Dict[str, torch.Tensor]:
        """Abstract method implementation

        Args:
            idx (Optioal[int]): The index into the dataset

        Returns:
            Dict[str, torch.Tensor]: The sample information wrapped in a dictionary
        """
        text, _ = self._query_document_sample_shuffle_indices(idx)

        text = torch.from_numpy(text).long()
        chosen = text[:-1].contiguous()
        chosen_labels = text[1:].contiguous()

        chosen_attention_mask, _, chosen_position_ids = _get_ltor_masks_and_position_ids(
            chosen,
            self.config.tokenizer.eod,
            self.config.reset_position_ids,
            self.config.reset_attention_mask,
            self.config.eod_mask_loss,
            self.config.create_attention_mask,
        )

        # aux datasets
        chosen_loss_mask, _ = self._query_document_sample_shuffle_indices_aux_dataset(
            self.chosen_loss_mask_dataset, idx
        )
        chosen_loss_mask = torch.from_numpy(chosen_loss_mask).float()[1:].contiguous()

        chosen_ref_log_probs, _ = self._query_document_sample_shuffle_indices_aux_dataset(
            self.chosen_ref_log_probs_dataset, idx, dtype=numpy.float32
        )
        chosen_ref_log_probs = torch.from_numpy(chosen_ref_log_probs).float()[1:].contiguous()

        text, _ = self._query_document_sample_shuffle_indices_aux_dataset(
            self.rejected_dataset, idx
        )
        text = torch.from_numpy(text).long()
        rejected = text[:-1].contiguous()
        rejected_labels = text[1:].contiguous()
        rejected_attention_mask, _, rejected_position_ids = _get_ltor_masks_and_position_ids(
            rejected,
            self.config.tokenizer.eod,
            self.config.reset_position_ids,
            self.config.reset_attention_mask,
            self.config.eod_mask_loss,
            self.config.create_attention_mask,
        )

        rejected_loss_mask, _ = self._query_document_sample_shuffle_indices_aux_dataset(
            self.rejected_loss_mask_dataset, idx
        )
        rejected_loss_mask = torch.from_numpy(rejected_loss_mask).float()[1:].contiguous()

        rejected_ref_log_probs, _ = self._query_document_sample_shuffle_indices_aux_dataset(
            self.rejected_ref_log_probs_dataset, idx, dtype=numpy.float32
        )
        rejected_ref_log_probs = torch.from_numpy(rejected_ref_log_probs).float()[1:].contiguous()

        return {
                "chosen": chosen,
                "chosen_labels": chosen_labels,
                "chosen_attention_mask": chosen_attention_mask,
                "chosen_loss_mask": chosen_loss_mask,
                "chosen_position_ids": chosen_position_ids,
                "chosen_ref_log_probs": chosen_ref_log_probs,
                "rejected": rejected,
                "rejected_labels": rejected_labels,
                "rejected_attention_mask": rejected_attention_mask,
                "rejected_loss_mask": rejected_loss_mask,
                "rejected_position_ids": rejected_position_ids,
                "rejected_ref_log_probs": rejected_ref_log_probs,
        } 

    def _query_document_sample_shuffle_indices_aux_dataset(
        self, aux_dataset, idx: int, dtype=numpy.int64
    ) -> Tuple[numpy.ndarray, numpy.ndarray]:
        """Get the aux ids and document ids for a given index

        Args:
            aux_dataset (int): The aux dataset
            idx (int): The index into the dataset

        Returns:
            Tuple[numpy.ndarray, numpy.ndarray]: The text ids and document ids
        """
        if aux_dataset is None:
            return (None, None)

        # Do the shuffle mapping
        idx = self.shuffle_index[idx]

        # Get the beginning and end documents and offsets
        doc_index_beg, doc_index_beg_offset = self.sample_index[idx]
        doc_index_end, doc_index_end_offset = self.sample_index[idx + 1]

        document_ids = []
        sample_parts = []

        # Sample spans a single document
        if doc_index_beg == doc_index_end:
            # Add the document id
            document_ids.append(self.document_index[doc_index_beg])

            # Add the entire sample
            sample_parts.append(
                aux_dataset.get(
                    self.document_index[doc_index_beg],
                    offset=doc_index_beg_offset,
                    length=doc_index_end_offset - doc_index_beg_offset + 1,
                )
            )

        # Sample spans multiple documents
        else:
            for i in range(doc_index_beg, doc_index_end + 1):
                # Add the document id
                document_ids.append(self.document_index[i])

                # Add the sample part
                offset = 0 if i > doc_index_beg else doc_index_beg_offset
                length = None if i < doc_index_end else doc_index_end_offset + 1
                sample_parts.append(
                    aux_dataset.get(
                        self.document_index[i], offset=offset, length=length
                    )
                )

        return (
            numpy.array(numpy.concatenate(sample_parts), dtype=dtype),
            numpy.array(document_ids, dtype=numpy.int64),
        )
