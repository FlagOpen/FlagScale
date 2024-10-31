# Copyright (c) 2024, NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause

"""This module defines tests for meta datasets."""

import gc
import logging
import sys
import tempfile
import time
import unittest
import warnings
from collections import Counter
from pathlib import Path

import torch
import webdataset as wds

from megatron.energon import (
    WorkerConfig,
    get_loader,
    get_savable_loader,
    get_train_dataset,
    get_val_dataset,
    load_dataset,
)
from megatron.energon.dataset_config import MAIN_FOLDER_NAME


def _norng_state(state):
    if isinstance(state, bytes):
        if len(state) > 100:
            return state[:5] + f"...<len={len(state)}>".encode()
        return state
    elif isinstance(state, str):
        if len(state) > 100:
            return state[:5] + f"...<len={len(state)}>"
        return state
    elif isinstance(state, dict):
        return {k: _norng_state(v) for k, v in state.items()}
    elif isinstance(state, (list, tuple)):
        if len(state) > 100:
            state = state[:5]
        return type(state)(_norng_state(v) for v in state)
    else:
        return state


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
        self.create_text_test_dataset(self.dataset_path / "ds1", 0)
        self.create_text_test_dataset(self.dataset_path / "ds2", 100)

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
                        "        shuffle_over_epochs_multiplier: 3",
                        "      - weight: 1",
                        "        path: ds2",
                        "        subflavor: ds2",
                        "        subflavors:",
                        "          source: metadataset.yaml",
                        "          number: 44",
                        "          mds: mds",
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
        self.nested_mds_path = self.dataset_path / "nested_metadataset.yaml"
        with open(self.nested_mds_path, "w") as f:
            f.write(
                "\n".join(
                    [
                        "splits:",
                        "  train:",
                        "    datasets:",
                        "      - weight: 4",
                        "        path: ./metadataset.yaml",
                        "        split_part: train",
                        "        subflavor: train",
                        "        subflavors:",
                        "          source: nested_metadataset.yaml",
                        "          mds: nested_train",
                        "      - path: ./metadataset.yaml",
                        "        split_part: val",
                        "        subflavors:",
                        "          source: nested_metadataset.yaml",
                        "          mds: nested_val",
                    ]
                )
            )
        print(self.dataset_path)

    def tearDown(self):
        # Remove all temporary files
        self.temp_dir.cleanup()

    @staticmethod
    def create_text_test_dataset(path: Path, offset: int):
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
                        "__class__: TextWebdataset",
                        "field_map:",
                        "  text: txt",
                        "subflavors:",
                        "  source: dataset.yaml",
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
            seed_offset=42,
        )

        # Train mode dataset
        train_dataset = get_train_dataset(
            self.mds_path,
            worker_config=worker_config,
            batch_size=10,
            shuffle_buffer_size=None,
            max_samples_per_sequence=None,
        )
        print(len(train_dataset))
        assert len(train_dataset) == 11

        train_loader1 = get_loader(train_dataset, worker_config=worker_config)

        train_order1 = [
            text for idx, data in zip(range(55 * 10), train_loader1) for text in data.text
        ]
        print(train_order1[:10])
        print(Counter(train_order1))
        assert len(Counter(train_order1)) == 110
        assert all(48 <= v <= 52 for v in Counter(train_order1).values())

        train_subflavors = [
            subflavor
            for idx, data in zip(range(55), train_loader1)
            for subflavor in data.__subflavor__
        ]
        print(train_subflavors[:10])
        print(Counter(train_subflavors))
        assert len(Counter(train_subflavors)) == 2
        assert all(250 <= v <= 300 for v in Counter(train_subflavors).values())

        # Train mode dataset
        train_dataset = get_train_dataset(
            self.mds_path,
            worker_config=worker_config,
            batch_size=10,
            shuffle_buffer_size=25,
            max_samples_per_sequence=25,
        )
        print(len(train_dataset))
        assert len(train_dataset) == 11

        train_loader1 = get_loader(train_dataset, worker_config=worker_config)

        train_order1 = [
            text for idx, data in zip(range(55 * 10), train_loader1) for text in data.text
        ]
        print(train_order1[:10])
        print(Counter(train_order1))
        assert len(Counter(train_order1)) == 110
        assert all(48 <= v <= 52 for v in Counter(train_order1).values())

        # Val mode dataset
        val_dataset = get_val_dataset(self.mds_path, worker_config=worker_config, batch_size=10)
        print(len(val_dataset))
        assert len(val_dataset) == 11

        val_loader1 = get_loader(val_dataset, worker_config=worker_config)

        val_order1 = [text for data in val_loader1 for text in data.text]
        assert len(val_order1) == 110
        print(Counter(val_order1))
        assert all(v == 1 for v in Counter(val_order1).values())

    def test_nested_metadataset(self):
        torch.manual_seed(42)
        worker_config = WorkerConfig(
            rank=0,
            world_size=1,
            num_workers=0,
        )

        dataset = load_dataset(self.nested_mds_path)

        raw_datasets = dataset.get_datasets(
            training=False, split_part="train", worker_config=worker_config
        )
        assert [weight for _raw_dataset, weight in raw_datasets] == [0.4, 0.4, 0.1, 0.1]
        assert [raw_dataset.path.name for raw_dataset, _weight in raw_datasets] == [
            "ds1",
            "ds2",
            "ds1",
            "ds2",
        ]
        assert [raw_dataset.subflavor for raw_dataset, _weight in raw_datasets] == [
            "train",
            "train",
            None,
            None,
        ]
        print([raw_dataset.subflavors for raw_dataset, _weight in raw_datasets])
        assert [raw_dataset.subflavors for raw_dataset, _weight in raw_datasets] == [
            {
                "source": "nested_metadataset.yaml",
                "dataset.yaml": True,
                "number": 43,
                "mds": "nested_train",
            },
            {
                "source": "nested_metadataset.yaml",
                "dataset.yaml": True,
                "number": 44,
                "mds": "nested_train",
            },
            {
                "source": "nested_metadataset.yaml",
                "dataset.yaml": True,
                "number": 42,
                "mds": "nested_val",
            },
            {
                "source": "nested_metadataset.yaml",
                "dataset.yaml": True,
                "number": 42,
                "mds": "nested_val",
            },
        ]

        # Train mode dataset
        train_dataset = get_train_dataset(
            self.nested_mds_path,
            worker_config=worker_config,
            batch_size=10,
            shuffle_buffer_size=None,
            max_samples_per_sequence=None,
        )
        print(len(train_dataset))
        assert len(train_dataset) == 22

        train_loader1 = get_loader(train_dataset, worker_config=worker_config)

        train_order1 = [
            text for idx, data in zip(range(55 * 10), train_loader1) for text in data.text
        ]
        print(train_order1[:10])
        print(Counter(train_order1))
        assert len(Counter(train_order1)) == 110
        assert all(48 <= v <= 53 for v in Counter(train_order1).values())

        train_subflavors = [
            subflavor
            for idx, data in zip(range(55), train_loader1)
            for subflavor in data.__subflavor__
        ]
        cnt = Counter(train_subflavors)
        print(train_subflavors[:10])
        print(cnt)
        avg = 55 * 10 / 5
        assert len(Counter(train_subflavors)) == 2
        assert avg * 4 - 40 < cnt["train"] < avg * 4 + 40
        assert avg - 10 < cnt[None] < avg + 10

        train_subflavorss = [
            tuple(subflavor.items())
            for idx, data in zip(range(55), train_loader1)
            for subflavor in data.__subflavors__
        ]
        cnt = Counter(train_subflavorss)
        print(train_subflavorss[:10])
        print(cnt)
        assert len(Counter(train_subflavorss)) == 3
        assert (
            avg * 2 - 20
            < cnt[
                (
                    ("source", "nested_metadataset.yaml"),
                    ("dataset.yaml", True),
                    ("number", 43),
                    ("mds", "nested_train"),
                )
            ]
            < avg * 2 + 20
        )
        assert (
            avg * 2 - 20
            < cnt[
                (
                    ("source", "nested_metadataset.yaml"),
                    ("dataset.yaml", True),
                    ("number", 44),
                    ("mds", "nested_train"),
                )
            ]
            < avg * 2 + 20
        )
        assert (
            avg * 1 - 20
            < cnt[
                (
                    ("source", "nested_metadataset.yaml"),
                    ("dataset.yaml", True),
                    ("number", 42),
                    ("mds", "nested_val"),
                )
            ]
            < avg * 1 + 20
        )

        # Train mode dataset
        train_dataset = get_train_dataset(
            self.mds_path,
            worker_config=worker_config,
            batch_size=10,
            shuffle_buffer_size=25,
            max_samples_per_sequence=25,
        )
        print(len(train_dataset))
        assert len(train_dataset) == 11

        train_loader1 = get_loader(train_dataset, worker_config=worker_config)

        train_order1 = [
            text for idx, data in zip(range(55 * 10), train_loader1) for text in data.text
        ]
        print(train_order1[:10])
        print(Counter(train_order1))
        assert len(Counter(train_order1)) == 110
        assert all(48 <= v <= 52 for v in Counter(train_order1).values())

        # Val mode dataset
        val_dataset = get_val_dataset(self.mds_path, worker_config=worker_config, batch_size=10)
        print(len(val_dataset))
        assert len(val_dataset) == 11

        val_loader1 = get_loader(val_dataset, worker_config=worker_config)

        val_order1 = [text for data in val_loader1 for text in data.text]
        assert len(val_order1) == 110
        print(Counter(val_order1))
        assert all(v == 1 for v in Counter(val_order1).values())

    def test_save_restore_state_train(self):
        torch.manual_seed(42)

        worker_config = WorkerConfig(
            rank=0,
            world_size=1,
            num_workers=0,
            seed_offset=42,
        )

        def new_loader():
            return get_savable_loader(
                get_train_dataset(
                    self.mds_path,
                    worker_config=worker_config,
                    batch_size=10,
                    parallel_shard_iters=2,
                    shuffle_buffer_size=None,
                    max_samples_per_sequence=None,
                    shuffle_over_epochs_multiplier=2,
                ),
                worker_config=worker_config,
            )

        # Train mode dataset
        loader = new_loader()
        state_0 = loader.save_state()
        order_0 = [data.text for idx, data in zip(range(10), loader)]
        state_1 = loader.save_state()
        # print("save state done")
        order_1 = [data.text for idx, data in zip(range(20), loader)]

        state_2 = loader.save_state()
        # print("save state done")
        # Iterated 30 samples, afterwards 50 samples. Checkpoint should be around that
        order_2 = [data.text for idx, data in zip(range(20), loader)]

        state_3 = loader.save_state()
        # print("save state done")
        # Iterated 50 samples, afterwards 53 samples. Checkpoint should be around that
        order_3 = [data.text for idx, data in zip(range(3), loader)]

        state_4 = loader.save_state()
        # print("save state done")
        # Dataset size is 55, want to save one sample before end of epoch
        # Iterated 53 samples, afterwards 54 samples. Checkpoint should be around that
        order_4 = [data.text for idx, data in zip(range(1), loader)]

        state_5 = loader.save_state()
        # print("save state done")
        # Dataset size is 55, want to save one sample before end of epoch
        # Iterated 54 samples, afterwards 55 samples. Checkpoint should be around that
        order_5 = [data.text for idx, data in zip(range(1), loader)]

        state_6 = loader.save_state()
        # print("save state done")
        # Dataset size is 55, want to save one sample before end of epoch
        # Iterated 55 samples, afterwards 75 samples. Checkpoint should be around that
        order_6 = [data.text for idx, data in zip(range(70), loader)]

        loader = new_loader()
        print("state_1:", _norng_state(state_1))
        loader.restore_state(state_1)
        order_1_rest = [data.text for idx, data in zip(range(len(order_1)), loader)]
        assert order_1 == order_1_rest

        loader = new_loader()
        loader.restore_state(state_0)
        order_0_rest = [data.text for idx, data in zip(range(len(order_0)), loader)]
        assert order_0 == order_0_rest

        loader = new_loader()
        print("state_2:", _norng_state(state_2))
        loader.restore_state(state_2)
        order_2_rest = [data.text for idx, data in zip(range(len(order_2)), loader)]
        print("order_2:", order_2)
        print("order_2_rest:", order_2_rest)
        assert order_2 == order_2_rest

        loader = new_loader()
        print("state_3:", _norng_state(state_3))
        loader.restore_state(state_3)
        order_3_rest = [data.text for idx, data in zip(range(len(order_3)), loader)]
        print("order_3:", order_3)
        print("order_3_rest:", order_3_rest)
        assert order_3 == order_3_rest

        loader = new_loader()
        print("state_4:", _norng_state(state_4))
        loader.restore_state(state_4)
        order_4_rest = [data.text for idx, data in zip(range(len(order_4)), loader)]
        print("order_4:", order_4)
        print("order_4_rest:", order_4_rest)
        assert order_4 == order_4_rest

        loader = new_loader()
        print("state_5:", _norng_state(state_5))
        loader.restore_state(state_5)
        order_5_rest = [data.text for idx, data in zip(range(len(order_5)), loader)]
        print("order_5:", order_5)
        print("order_5_rest:", order_5_rest)
        assert order_5 == order_5_rest

        loader = new_loader()
        print("state_6:", _norng_state(state_6))
        loader.restore_state(state_6)
        order_6_rest = [data.text for idx, data in zip(range(len(order_6)), loader)]
        print("order_6:", order_6)
        print("order_6_rest:", order_6_rest)
        assert order_6 == order_6_rest

        wrk_cfg = worker_config.config()
        assert wrk_cfg == {
            "rank": 0,
            "world_size": 1,
            "num_workers": 0,
            "data_parallel_group": None,
        }
        print(loader.config())
        assert loader.config() == {
            "type": "SavableDataLoader",
            "num_workers": 0,
            "persistent_workers": False,
            "pin_memory": True,
            "prefetch_factor": None,
            "dataset": {
                "type": "MapDataset",
                "dataset": {
                    "type": "BatchDataset",
                    "batch_size": 10,
                    "batcher": "megatron.energon.task_encoder.base.DefaultTaskEncoder.batch",
                    "batcher_stateless": True,
                    "drop_last": False,
                    "error_handler": "megatron.energon.wrappers._log_exception.log_exception",
                    "worker_config": wrk_cfg,
                    "dataset": {
                        "type": "MapDataset",
                        "dataset": {
                            "type": "BlendDataset",
                            "dataset_weights": [
                                (
                                    {
                                        "type": "TextWebdataset",
                                        "path": str(self.dataset_path / "ds1"),
                                        "training": True,
                                        "worker_config": wrk_cfg,
                                        "dataset": {
                                            "type": "MapDataset",
                                            "dataset": {
                                                "type": "IterMapDataset",
                                                "dataset": {
                                                    "type": "WebdatasetSampleLoaderDataset",
                                                    "shards": [
                                                        [
                                                            {
                                                                "name": "parts/data-0.tar",
                                                                "path": str(
                                                                    self.dataset_path
                                                                    / "ds1/parts/data-0.tar"
                                                                ),
                                                                "offset": 0,
                                                                "count": 10,
                                                                "byte_offset": 0,
                                                                "byte_size": 20480,
                                                            },
                                                            {
                                                                "name": "parts/data-1.tar",
                                                                "path": str(
                                                                    self.dataset_path
                                                                    / "ds1/parts/data-1.tar"
                                                                ),
                                                                "offset": 0,
                                                                "count": 10,
                                                                "byte_offset": 0,
                                                                "byte_size": 20480,
                                                            },
                                                            {
                                                                "name": "parts/data-2.tar",
                                                                "path": str(
                                                                    self.dataset_path
                                                                    / "ds1/parts/data-2.tar"
                                                                ),
                                                                "offset": 0,
                                                                "count": 10,
                                                                "byte_offset": 0,
                                                                "byte_size": 20480,
                                                            },
                                                            {
                                                                "name": "parts/data-3.tar",
                                                                "path": str(
                                                                    self.dataset_path
                                                                    / "ds1/parts/data-3.tar"
                                                                ),
                                                                "offset": 0,
                                                                "count": 10,
                                                                "byte_offset": 0,
                                                                "byte_size": 20480,
                                                            },
                                                            {
                                                                "name": "parts/data-4.tar",
                                                                "path": str(
                                                                    self.dataset_path
                                                                    / "ds1/parts/data-4.tar"
                                                                ),
                                                                "offset": 0,
                                                                "count": 10,
                                                                "byte_offset": 0,
                                                                "byte_size": 20480,
                                                            },
                                                            {
                                                                "name": "parts/data-5.tar",
                                                                "path": str(
                                                                    self.dataset_path
                                                                    / "ds1/parts/data-5.tar"
                                                                ),
                                                                "offset": 0,
                                                                "count": 5,
                                                                "byte_offset": 0,
                                                                "byte_size": 10240,
                                                            },
                                                        ]
                                                    ],
                                                    "worker_config": wrk_cfg,
                                                    "exclude": [],
                                                    "loop": True,
                                                    "shuffle_over_epochs": 6,
                                                    "parallel_shard_iters": 2,
                                                },
                                                "iter_map_fn": "webdataset.filters.<unknown>",
                                                "len_map_fn": "megatron.energon.wrappers.iter_map_dataset.IterMapDataset.<lambda>",
                                                "error_handler": "megatron.energon.flavors.base_webdataset.BaseWebdataset.error_handler",
                                            },
                                            "map_fn": "megatron.energon.flavors.base_webdataset.DefaultGenericWebdataset._load_sample",
                                        },
                                        "subflavor": "ds1",
                                        "subflavors": {
                                            "source": "metadataset.yaml",
                                            "dataset.yaml": True,
                                            "number": 43,
                                            "mds": "mds",
                                        },
                                        "sample_loader": "megatron.energon.flavors.base_webdataset.DefaultGenericWebdataset.__init__.<locals>.<lambda>",
                                        "image_decode": "torchrgb",
                                        "ignore_decoder_errors": False,
                                    },
                                    0.5,
                                ),
                                (
                                    {
                                        "type": "TextWebdataset",
                                        "path": str(self.dataset_path / "ds2"),
                                        "training": True,
                                        "worker_config": wrk_cfg,
                                        "dataset": {
                                            "type": "MapDataset",
                                            "dataset": {
                                                "type": "IterMapDataset",
                                                "dataset": {
                                                    "type": "WebdatasetSampleLoaderDataset",
                                                    "shards": [
                                                        [
                                                            {
                                                                "name": "parts/data-0.tar",
                                                                "path": str(
                                                                    self.dataset_path
                                                                    / "ds2/parts/data-0.tar"
                                                                ),
                                                                "offset": 0,
                                                                "count": 10,
                                                                "byte_offset": 0,
                                                                "byte_size": 20480,
                                                            },
                                                            {
                                                                "name": "parts/data-1.tar",
                                                                "path": str(
                                                                    self.dataset_path
                                                                    / "ds2/parts/data-1.tar"
                                                                ),
                                                                "offset": 0,
                                                                "count": 10,
                                                                "byte_offset": 0,
                                                                "byte_size": 20480,
                                                            },
                                                            {
                                                                "name": "parts/data-2.tar",
                                                                "path": str(
                                                                    self.dataset_path
                                                                    / "ds2/parts/data-2.tar"
                                                                ),
                                                                "offset": 0,
                                                                "count": 10,
                                                                "byte_offset": 0,
                                                                "byte_size": 20480,
                                                            },
                                                            {
                                                                "name": "parts/data-3.tar",
                                                                "path": str(
                                                                    self.dataset_path
                                                                    / "ds2/parts/data-3.tar"
                                                                ),
                                                                "offset": 0,
                                                                "count": 10,
                                                                "byte_offset": 0,
                                                                "byte_size": 20480,
                                                            },
                                                            {
                                                                "name": "parts/data-4.tar",
                                                                "path": str(
                                                                    self.dataset_path
                                                                    / "ds2/parts/data-4.tar"
                                                                ),
                                                                "offset": 0,
                                                                "count": 10,
                                                                "byte_offset": 0,
                                                                "byte_size": 20480,
                                                            },
                                                            {
                                                                "name": "parts/data-5.tar",
                                                                "path": str(
                                                                    self.dataset_path
                                                                    / "ds2/parts/data-5.tar"
                                                                ),
                                                                "offset": 0,
                                                                "count": 5,
                                                                "byte_offset": 0,
                                                                "byte_size": 10240,
                                                            },
                                                        ]
                                                    ],
                                                    "worker_config": wrk_cfg,
                                                    "exclude": [],
                                                    "loop": True,
                                                    "shuffle_over_epochs": 2,
                                                    "parallel_shard_iters": 2,
                                                },
                                                "iter_map_fn": "webdataset.filters.<unknown>",
                                                "len_map_fn": "megatron.energon.wrappers.iter_map_dataset.IterMapDataset.<lambda>",
                                                "error_handler": "megatron.energon.flavors.base_webdataset.BaseWebdataset.error_handler",
                                            },
                                            "map_fn": "megatron.energon.flavors.base_webdataset.DefaultGenericWebdataset._load_sample",
                                        },
                                        "subflavor": "ds2",
                                        "subflavors": {
                                            "source": "metadataset.yaml",
                                            "dataset.yaml": True,
                                            "number": 44,
                                            "mds": "mds",
                                        },
                                        "sample_loader": "megatron.energon.flavors.base_webdataset.DefaultGenericWebdataset.__init__.<locals>.<lambda>",
                                        "image_decode": "torchrgb",
                                        "ignore_decoder_errors": False,
                                    },
                                    0.5,
                                ),
                            ],
                            "worker_config": wrk_cfg,
                        },
                        "map_fn": "megatron.energon.task_encoder.base.DefaultTaskEncoder.encode_sample",
                    },
                },
                "map_fn": "megatron.energon.task_encoder.base.DefaultTaskEncoder.encode_batch",
            },
        }

    def test_save_restore_state_train_workers(self):
        torch.manual_seed(42)

        worker_config = WorkerConfig(
            rank=0,
            world_size=1,
            num_workers=1,
            seed_offset=42,
        )

        def new_loader():
            return get_savable_loader(
                get_train_dataset(
                    self.mds_path,
                    worker_config=worker_config,
                    batch_size=10,
                    parallel_shard_iters=2,
                    shuffle_buffer_size=None,
                    max_samples_per_sequence=None,
                ),
                worker_config=worker_config,
                checkpoint_every_sec=0.5,
                checkpoint_every_min_n_samples=1,
                n_checkpoints=5,
            )

        # Train mode dataset
        loader = new_loader()
        state_0 = loader.save_state()
        order_0 = [data.text for idx, data in zip(range(10), loader)]
        time.sleep(0.5)
        state_1 = loader.save_state()
        # print("save state done")
        order_1 = [data.text for idx, data in zip(range(20), loader)]

        # Ensure a checkpoint is created on next()
        time.sleep(1.5)

        state_2 = loader.save_state()
        # print("save state done")
        # Iterated 30 samples, afterwards 50 samples. Checkpoint should be around that
        order_2 = [data.text for idx, data in zip(range(20), loader)]

        state_3 = loader.save_state()
        # print("save state done")
        # Iterated 50 samples, afterwards 53 samples. Checkpoint should be around that
        order_3 = [data.text for idx, data in zip(range(3), loader)]

        # Ensure a checkpoint is created on next()
        time.sleep(1.5)

        state_4 = loader.save_state()
        # print("save state done")
        # Dataset size is 55, want to save one sample before end of epoch
        # Iterated 1 samples, afterwards 54 samples. Checkpoint should be around that
        order_4 = [data.text for idx, data in zip(range(1), loader)]

        # Ensure a checkpoint is created on next()
        time.sleep(1.5)

        state_5 = loader.save_state()
        # print("save state done")
        # Dataset size is 55, want to save one sample before end of epoch
        # Iterated 1 samples, afterwards 55 samples. Checkpoint should be around that
        order_5 = [data.text for idx, data in zip(range(1), loader)]

        # Ensure a checkpoint is created on next()
        time.sleep(1.5)

        state_6 = loader.save_state()
        # print("save state done")
        # Dataset size is 55, want to save one sample before end of epoch
        # Iterated 1 samples, afterwards 55 samples. Checkpoint should be around that
        order_6 = [data.text for idx, data in zip(range(10), loader)]

        loader = new_loader()
        print("state_1:", _norng_state(state_1))
        loader.restore_state(state_1)
        order_1_rest = [data.text for idx, data in zip(range(len(order_1)), loader)]
        print("order_1:", order_1)
        print("order_1_rest:", order_1_rest)
        assert order_1 == order_1_rest

        loader = new_loader()
        loader.restore_state(state_0)
        order_0_rest = [data.text for idx, data in zip(range(len(order_0)), loader)]
        assert order_0 == order_0_rest

        loader = new_loader()
        print("state_2:", _norng_state(state_2))
        loader.restore_state(state_2)
        order_2_rest = [data.text for idx, data in zip(range(len(order_2)), loader)]
        print("order_2:", order_2)
        print("order_2_rest:", order_2_rest)
        assert order_2 == order_2_rest

        loader = new_loader()
        print("state_3:", _norng_state(state_3))
        loader.restore_state(state_3)
        order_3_rest = [data.text for idx, data in zip(range(len(order_3)), loader)]
        print("order_3:", order_3)
        print("order_3_rest:", order_3_rest)
        assert order_3 == order_3_rest

        loader = new_loader()
        print("state_4:", _norng_state(state_4))
        loader.restore_state(state_4)
        order_4_rest = [data.text for idx, data in zip(range(len(order_4)), loader)]
        print("order_4:", order_4)
        print("order_4_rest:", order_4_rest)
        assert order_4 == order_4_rest

        loader = new_loader()
        print("state_5:", _norng_state(state_5))
        loader.restore_state(state_5)
        order_5_rest = [data.text for idx, data in zip(range(len(order_5)), loader)]
        print("order_5:", order_5)
        print("order_5_rest:", order_5_rest)
        assert order_5 == order_5_rest

        loader = new_loader()
        print("state_6:", _norng_state(state_6))
        loader.restore_state(state_6)
        order_6_rest = [data.text for idx, data in zip(range(len(order_6)), loader)]
        print("order_6:", order_6)
        print("order_6_rest:", order_6_rest)
        assert order_6 == order_6_rest

    def test_save_restore_state_train_epochize_workers(self):
        torch.manual_seed(42)
        psi = 2
        vel = 19
        sbs = 10

        worker_config = WorkerConfig(
            rank=0,
            world_size=1,
            num_workers=2,
            seed_offset=42,
        )

        # Train mode dataset
        torch.manual_seed(42)
        loader = get_savable_loader(
            get_train_dataset(
                self.mds_path,
                worker_config=worker_config,
                batch_size=1,
                parallel_shard_iters=psi,
                virtual_epoch_length=vel,
                shuffle_buffer_size=sbs,
                max_samples_per_sequence=sbs,
            ),
            worker_config=worker_config,
        )
        state_0 = loader.save_state()
        order_1 = [data.text[0] for data in loader]
        state_1 = loader.save_state()
        order_2 = [data.text[0] for data in loader]
        state_2 = loader.save_state()
        order_3 = [data.text[0] for idx, data in zip(range(17), loader)]

        torch.manual_seed(42)
        loader = get_savable_loader(
            get_train_dataset(
                self.mds_path,
                worker_config=worker_config,
                batch_size=1,
                parallel_shard_iters=psi,
                virtual_epoch_length=vel,
                shuffle_buffer_size=sbs,
                max_samples_per_sequence=sbs,
            ),
            worker_config=worker_config,
        )
        print("state_0:", _norng_state(state_0))
        loader.restore_state(state_0)
        order_5 = [data.text[0] for data in loader]
        print("order_1:", order_1)
        print("order_5:", order_5)
        assert order_1 == order_5

        torch.manual_seed(42)
        loader = get_savable_loader(
            get_train_dataset(
                self.mds_path,
                worker_config=worker_config,
                batch_size=1,
                parallel_shard_iters=psi,
                virtual_epoch_length=vel,
                shuffle_buffer_size=sbs,
                max_samples_per_sequence=sbs,
            ),
            worker_config=worker_config,
        )
        print("state_1:", _norng_state(state_1))
        loader.restore_state(state_1)
        order_6 = [data.text[0] for data in loader]
        print("order_2:", order_2)
        print("order_6:", order_6)
        assert order_2 == order_6

        torch.manual_seed(42)
        loader = get_savable_loader(
            get_train_dataset(
                self.mds_path,
                worker_config=worker_config,
                batch_size=1,
                parallel_shard_iters=psi,
                virtual_epoch_length=vel,
                shuffle_buffer_size=sbs,
                max_samples_per_sequence=sbs,
            ),
            worker_config=worker_config,
        )
        print("state_2:", _norng_state(state_2))
        loader.restore_state(state_2)
        order_7 = [data.text[0] for idx, data in zip(range(17), loader)]
        print("order_3:", order_3)
        print("order_7:", order_7)
        assert order_3 == order_7

    def test_save_restore_state_val(self):
        torch.manual_seed(42)

        worker_config = WorkerConfig(
            rank=0,
            world_size=1,
            num_workers=0,
            seed_offset=42,
        )

        # Train mode dataset
        loader = get_savable_loader(
            get_val_dataset(self.mds_path, worker_config=worker_config, batch_size=10),
            worker_config=worker_config,
        )
        state_0 = loader.save_state()
        order_1 = [data.text for idx, data in zip(range(55 * 20), loader)]
        state_1 = loader.save_state()
        # print("save state done")
        order_2 = [data.text for idx, data in zip(range(55 * 20), loader)]

        loader = get_savable_loader(
            get_val_dataset(self.mds_path, worker_config=worker_config, batch_size=10),
            worker_config=worker_config,
        )
        loader.restore_state(state_1)
        order_3 = [data.text for idx, data in zip(range(55 * 20), loader)]
        assert order_2 == order_3

        loader = get_savable_loader(
            get_val_dataset(self.mds_path, worker_config=worker_config, batch_size=10),
            worker_config=worker_config,
        )
        loader.restore_state(state_0)
        order_4 = [data.text for idx, data in zip(range(55 * 20), loader)]
        assert order_1 == order_4

    def test_blending_randomness(self):
        import random

        import numpy

        for num_workers in [0, 1, 2]:  # Especially also check the num_workers=0 case

            world_size = 4
            micro_batch_size = 1
            seed = 42

            configs = (
                WorkerConfig(rank=0, world_size=world_size, num_workers=num_workers),
                WorkerConfig(rank=1, world_size=world_size, num_workers=num_workers),
                WorkerConfig(rank=2, world_size=world_size, num_workers=num_workers),
            )

            all_ranks_subflavors = []
            for rank_config in configs:
                torch.manual_seed(seed)
                numpy.random.seed(seed)
                random.seed(seed)

                ds = get_train_dataset(
                    self.mds_path,
                    split_part="train",
                    worker_config=rank_config,
                    batch_size=micro_batch_size,
                    shuffle_buffer_size=None,
                    max_samples_per_sequence=None,
                )
                loader = get_loader(ds, worker_config=rank_config)

                subflavors = [data.__subflavor__[0] for idx, data in zip(range(25), loader)]

                all_ranks_subflavors.append(subflavors)

                print(f"Subflavors for rank {rank_config.rank}:", subflavors)

            # Assert that all ranks got different data
            for i in range(len(all_ranks_subflavors)):
                for j in range(i + 1, len(all_ranks_subflavors)):
                    assert (
                        all_ranks_subflavors[i] != all_ranks_subflavors[j]
                    ), f"Rank {i} and rank {j} got the same subflavors."

            # Delete all locals, otherwise loaders might be kept alive
            locals().clear()
            gc.collect()


if __name__ == "__main__":
    unittest.main()
