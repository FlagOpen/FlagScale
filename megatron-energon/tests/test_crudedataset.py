# Copyright (c) 2024, NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause

"""This module defines tests for crude datasets."""

import logging
import pickle
import sys
import tempfile
import unittest
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import List

import torch
import webdataset as wds

from megatron.energon import (
    Batch,
    Cooker,
    DefaultTaskEncoder,
    TextSample,
    WorkerConfig,
    basic_sample_keys,
    get_savable_loader,
    get_train_dataset,
    stateless,
)
from megatron.energon.dataset_config import MAIN_FOLDER_NAME
from megatron.energon.flavors.webdataset.structs import reraise_exception


@dataclass
class TextBatch(Batch):
    __keys__: List[str]
    txts: List[str]


def cook_text(sample: dict) -> TextSample:
    return TextSample(
        **basic_sample_keys(sample),
        text=f"<{sample['txt']}>",
    )


def cook_other(sample: dict) -> TextSample:
    d = pickle.loads(sample["pkl"])
    return TextSample(
        **basic_sample_keys(sample),
        text=f"<{sample['txt']}|{d['idx']}>",
    )


class MyTaskEncoder(DefaultTaskEncoder[TextSample, TextSample, TextBatch, TextBatch]):
    """A simple task encoder for captioning."""

    cookers = [
        Cooker(cook_text, has_subflavors={"crude_type": "txtpkl"}),
        Cooker(cook_other, has_subflavors={"crude_type": "otherpkl"}),
    ]

    def batch(self, samples: List[TextSample]) -> TextBatch:
        return TextBatch(
            __keys__=[sample.__key__ for sample in samples],
            txts=[sample.text for sample in samples],
        )
    
    def select_samples_to_pack(self, samples):
        return [[sample] for sample in samples]
    
    @stateless
    def pack_selected_samples(self, samples):
        return samples[0]


class TestDataset(unittest.TestCase):
    # Set up the test fixture
    def setUp(self):
        logging.basicConfig(stream=sys.stderr, level=logging.INFO)
        warnings.simplefilter("ignore", ResourceWarning)

        # Create a temporary directory
        self.temp_dir = tempfile.TemporaryDirectory()
        self.dataset_path = Path(self.temp_dir.name)
        # self.dataset_path = Path("./test_dataset")

        self.dataset_path.mkdir(exist_ok=True, parents=True)

        (self.dataset_path / "ds1").mkdir(exist_ok=True, parents=True)
        (self.dataset_path / "ds2").mkdir(exist_ok=True, parents=True)

        # Create a small dummy captioning dataset
        self.create_crude_text_test_dataset(self.dataset_path / "ds1", 0)
        self.create_crude_text_test_dataset(self.dataset_path / "ds2", 100)

        self.mds_path = self.dataset_path / "metadataset.yaml"
        with open(self.mds_path, "w") as f:
            f.write(
                "\n".join(
                    [
                        "__module__: megatron.energon",
                        "__class__: Metadataset",
                        "splits:",
                        "  train:",
                        "    datasets:",
                        "      - weight: 1",
                        "        path: ds1",
                        "        subflavor: ds1",
                        "        subflavors:",
                        "          source: metadataset.yaml",
                        "          number: 43",
                        "          mds: mds",
                        "          crude_type: txtpkl",
                        "        shuffle_over_epochs_multiplier: 3",
                        "      - weight: 1",
                        "        path: ds2",
                        "        subflavor: ds2",
                        "        subflavors:",
                        "          source: metadataset.yaml",
                        "          number: 44",
                        "          mds: mds",
                        "          crude_type: otherpkl",
                        "  val:",
                        "    datasets:",
                        "      - weight: 1",
                        "        path: ds1",
                        "        split_part: train",
                        "      - weight: 1",
                        "        path: ds2",
                        "        split_part: train",
                    ]
                )
            )

        print(self.dataset_path)

    def tearDown(self):
        # Remove all temporary files
        self.temp_dir.cleanup()

    @staticmethod
    def create_crude_text_test_dataset(path: Path, offset: int):
        """Creates a small dummy test dataset for testing purposes."""

        # Create num_samples unique captions
        (path / "parts").mkdir(exist_ok=True, parents=True)

        # Initialize the ShardWriter
        with wds.ShardWriter(f"{path}/parts/data-%d.tar", maxcount=10) as shard_writer:
            for idx in range(55):
                # Write individual files to shards
                shard_writer.write(
                    {
                        "__key__": f"{idx + offset:06d}",
                        "txt": f"{idx + offset}".encode(),
                        "pkl": pickle.dumps({"idx": idx + offset}),
                    },
                )
            total_shards = shard_writer.shard

        from megatron.energon.flavors import BaseWebdataset

        BaseWebdataset.prepare_dataset(
            path,
            [f"parts/data-{{0..{total_shards-1}}}.tar"],
            split_parts_ratio=[("train", 1.0)],
            shuffle_seed=None,
        )

        with open(path / MAIN_FOLDER_NAME / "dataset.yaml", "w") as f:
            f.write(
                "\n".join(
                    [
                        "__module__: megatron.energon",
                        "__class__: CrudeWebdataset",
                        "subflavors:",
                        "  dataset.yaml: true",
                        "  number: 42",
                    ]
                )
            )

    def test_metadataset(self):
        torch.manual_seed(42)
        worker_config = WorkerConfig(
            rank=0,
            world_size=1,
            num_workers=0,
        )

        # Train mode dataset
        torch.manual_seed(42)
        train_dataset = get_train_dataset(
            self.mds_path,
            worker_config=worker_config,
            batch_size=3,
            task_encoder=MyTaskEncoder(),
            shuffle_buffer_size=None,
            max_samples_per_sequence=None,
            handler=reraise_exception,
        )
        loader = get_savable_loader(
            train_dataset,
            worker_config=worker_config,
        )

        print(len(train_dataset))
        # assert len(train_dataset) == 11

        for idx, data in enumerate(loader):
            if idx >= len(train_dataset):
                break

            assert isinstance(data, TextBatch)

            print("Batch", idx)
            for txt, key in zip(data.txts, data.__keys__):
                key_int = int(key.split("/")[-1])
                if key_int < 100:
                    assert txt == f"<{key_int}>"
                else:
                    assert txt == f"<{key_int}|{key_int}>"

                print(key, txt)

    def test_loader(self):
        torch.manual_seed(42)
        worker_config = WorkerConfig(
            rank=0,
            world_size=1,
            num_workers=2,
        )

        loader = get_savable_loader(
            get_train_dataset(
                self.mds_path,
                batch_size=2,
                worker_config=worker_config,
                task_encoder=MyTaskEncoder(),
                shuffle_buffer_size=None,
                max_samples_per_sequence=None,
                packing_buffer_size=2,
            ),
            worker_config=worker_config,
            checkpoint_every_sec=0,
            checkpoint_every_min_n_samples=1,
            n_checkpoints=4,
        )
        samples = [s.__keys__ for idx, s in zip(range(100), loader)]

        print(samples)

        state = loader.save_state_rank()

        samples_after = [s.__keys__ for idx, s in zip(range(100, 200), loader)]
        print(samples_after)

        loader = get_savable_loader(
            get_train_dataset(
                self.mds_path,
                batch_size=2,
                worker_config=worker_config,
                task_encoder=MyTaskEncoder(),
                shuffle_buffer_size=None,
                max_samples_per_sequence=None,
                packing_buffer_size=2,
            ),
            worker_config=worker_config,
            checkpoint_every_sec=0,
            checkpoint_every_min_n_samples=1,
            n_checkpoints=4,
        )

        loader.restore_state_rank(state)

        samples_restored = [s.__keys__ for idx, s in zip(range(100, 200), loader)]
        print(samples_restored)

        assert all([a == b for a, b in zip(samples_after, samples_restored)])


if __name__ == "__main__":
    unittest.main()
