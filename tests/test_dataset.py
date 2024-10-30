# Copyright (c) 2024, NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause

"""This module defines tests for the dataset."""

import dataclasses
import io
import json
import logging
import math
import random
import sys
import tempfile
import unittest
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple, Type, Union

import numpy as np
import torch
import webdataset as wds
from click.testing import CliRunner
from PIL import Image

from megatron.energon import (
    AugmentTaskEncoder,
    Batch,
    BatchDataset,
    BlendDataset,
    CaptioningSample,
    DefaultTaskEncoder,
    MapDataset,
    MixBatchDataset,
    SavableDataLoader,
    TaskEncoder,
    WorkerConfig,
    generic_batch,
    get_loader,
    get_savable_loader,
    get_train_dataset,
    get_val_dataset,
    homogeneous_concat_mix,
)
from megatron.energon.dataset_config import MAIN_FOLDER_NAME, get_dataset_from_config
from megatron.energon.flavors import BaseWebdataset
from megatron.energon.task_encoder.base import stateless
from megatron.energon.tools.analyze_debug import command as analyze_debug_command
from megatron.energon.tools.lint import command as lint_command
from megatron.energon.tools.prepare import command as prepare_command

DATASET_SIZE = 50

no_worker_config = WorkerConfig(rank=0, world_size=1, num_workers=0)


@dataclass
class ExtendedCaptioningSample(CaptioningSample):
    batch_index: int
    sample_index: int
    rand_num: int


@dataclass
class EncodedCaptioningSample:
    __key__: str
    __restore_key__: Tuple[Union[str, int, tuple], ...]
    image: torch.Tensor
    caption: torch.Tensor


@dataclass
class CaptioningBatch(Batch):
    __key__: List[str]
    __restore_key__: Tuple[Union[str, int, tuple], ...]
    image: torch.Tensor
    caption: torch.Tensor


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

        # Create a small dummy captioning dataset
        self.samples = self.create_captioning_test_dataset(self.dataset_path, DATASET_SIZE)
        print(self.dataset_path)

    def tearDown(self):
        # Remove all temporary files
        self.temp_dir.cleanup()

    @staticmethod
    def create_captioning_test_dataset(path: Union[str, Path], num_samples: int = 50):
        """Creates a small dummy captioning dataset for testing purposes."""
        path = Path(path)

        animals = (
            "ant bee beetle bug bumblebee butterfly caterpillar cicada cricket dragonfly earwig "
            "firefly grasshopper honeybee hornet inchworm ladybug locust mantis mayfly mosquito "
            "moth sawfly silkworm termite wasp woodlouse"
        ).split()
        adjectives = (
            "adorable affable amazing amiable attractive beautiful calm charming cherubic classic "
            "classy convivial cordial cuddly curly cute debonair elegant famous fresh friendly "
            "funny gorgeous graceful gregarious grinning handsome hilarious hot interesting kind "
            "laughing lovely meek mellow merciful neat nifty notorious poetic pretty refined "
            "refreshing sexy smiling sociable spiffy stylish sweet tactful whimsical"
        ).split()

        # Set random seeds for numpy and torch
        np.random.seed(42)
        torch.manual_seed(42)

        entries = []

        assert num_samples < len(animals) * len(
            adjectives
        ), "Cannot generate more samples than unique captions."

        # Create num_samples unique captions
        captions = set()
        while len(captions) < num_samples:
            # Create random description by sampling from adjectives and animals
            adjective = np.random.choice(adjectives)
            prefix = "An" if adjective[0] in "aeiou" else "A"
            description = f"{prefix} {adjective} {np.random.choice(animals)}."
            captions.add(description)

        (path / "parts").mkdir(exist_ok=True, parents=True)

        # Initialize the ShardWriter
        with wds.ShardWriter(f"{path}/parts/data-%d.tar", maxcount=30) as shard_writer:
            for idx in range(num_samples):
                # Create a dummy image with random noise and save to disk
                img_buf = io.BytesIO()
                randimg = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
                image = Image.fromarray(randimg)
                image.save(img_buf, format="PNG")
                img_bytes = img_buf.getvalue()

                description = captions.pop()

                entries.append({"image": randimg, "caption": description})

                # Write individual files to shards
                shard_writer.write(
                    {
                        "__key__": f"{idx:06d}",
                        "png": img_bytes,
                        "txt": description.encode("utf-8"),
                        "json": json.dumps({"caption": description}),
                    },
                )
            total_shards = shard_writer.shard

        BaseWebdataset.prepare_dataset(
            path,
            [f"parts/data-{{0..{total_shards-1}}}.tar"],
            split_parts_ratio=[("train", 1.0)],
        )

        with open(path / MAIN_FOLDER_NAME / "dataset.yaml", "w") as f:
            f.write(
                "\n".join(
                    [
                        "__module__: megatron.energon",
                        "__class__: CaptioningWebdataset",
                        "field_map:",
                        "  image: png",
                        "  caption: txt",
                    ]
                )
            )

        with open(path / MAIN_FOLDER_NAME / "dataset_field.yaml", "w") as f:
            f.write(
                "\n".join(
                    [
                        "__module__: megatron.energon",
                        "__class__: CaptioningWebdataset",
                        "field_map:",
                        "  image: png",
                        "  caption: json[caption]",
                    ]
                )
            )

        with open(path / MAIN_FOLDER_NAME / "dataset_sample_loader.yaml", "w") as f:
            f.write(
                "\n".join(
                    [
                        "__module__: megatron.energon",
                        "__class__: CaptioningWebdataset",
                        "sample_loader: sample_loader.py:sample_loader",
                        "part_filter: sample_loader.py:part_filter",
                    ]
                )
            )

        with open(path / MAIN_FOLDER_NAME / "dataset_sample_loader_key.yaml", "w") as f:
            f.write(
                "\n".join(
                    [
                        "__module__: megatron.energon",
                        "__class__: CaptioningWebdataset",
                        "sample_loader: sample_loader.py:sample_loader_key",
                        "part_filter: sample_loader.py:part_filter",
                    ]
                )
            )

        with open(path / MAIN_FOLDER_NAME / "sample_loader.py", "w") as f:
            f.write(
                "\n".join(
                    [
                        "def sample_loader(raw: dict) -> dict:",
                        "    assert 'txt' not in raw",
                        "    return dict(",
                        '        image=raw["png"],',
                        '        caption="<SL>" + raw["json"]["caption"],',
                        "    )",
                        "",
                        "def sample_loader_key(raw: dict) -> dict:",
                        "    assert 'txt' not in raw",
                        "    return dict(",
                        '        __key__="<SL>" + raw["__key__"],',
                        '        image=raw["png"],',
                        '        caption="<SL>" + raw["json"]["caption"],',
                        "    )",
                        "",
                        "def part_filter(part: str) -> bool:",
                        '    return part in ["json", "png"]',
                        "",
                    ]
                )
            )

        with open(path / MAIN_FOLDER_NAME / "dataset_exclude.yaml", "w") as f:
            f.write(
                "\n".join(
                    [
                        "__module__: megatron.energon",
                        "__class__: CaptioningWebdataset",
                        "field_map:",
                        "  image: png",
                        "  caption: txt",
                        "split_config: split2.yaml",
                    ]
                )
            )

        with open(path / MAIN_FOLDER_NAME / "split2.yaml", "w") as f:
            with open(path / MAIN_FOLDER_NAME / "split.yaml", "r") as rf:
                origsplit = rf.read()
            f.write(
                origsplit
                + "\n"
                + "\n".join(
                    [
                        "exclude:",
                        "  - parts/data-0.tar",
                        "  - parts/data-1.tar/00003{5..9}",
                    ]
                )
            )

        return entries

    def test_captioning_dataset(self):
        ds = get_dataset_from_config(
            self.dataset_path,
            split_part="train",
            worker_config=no_worker_config,
            training=False,
            sample_type=CaptioningSample,
        )

        ds = MapDataset(
            ds,
            lambda x: CaptioningSample(
                __key__=x.__key__,
                __restore_key__=x.__restore_key__,
                __subflavor__=x.__subflavor__,
                __subflavors__=x.__subflavors__,
                image=x.image,
                caption=torch.tensor(np.frombuffer(x.caption.encode(), dtype=np.uint8)),
            ),
            worker_config=no_worker_config,
        )

        def get_ld(ds):
            return get_loader(ds, worker_config=no_worker_config)

        # Check len operator
        assert len(ds) == 50
        # Check if iterating returns the same
        iter1 = list(get_ld(ds))
        iter2 = list(get_ld(ds))
        assert len(iter1) == 50
        assert len(iter2) == 50
        assert all(elem1.__key__ == elem2.__key__ for elem1, elem2 in zip(iter1, iter2))

        # Check case when batch size is larger than dataset size
        batch_sizes = []
        for wrapped_sample in get_ld(
            BatchDataset(
                ds,
                batch_size=DATASET_SIZE * 2,
                batcher=generic_batch,
                worker_config=no_worker_config,
            )
        ):
            batch_sizes.append(wrapped_sample.image.shape[0])
        assert batch_sizes == [DATASET_SIZE]

        # Check returned dimensions and batch sizes if batch size is smaller than dataset size
        batch_size = 4
        assert batch_size < DATASET_SIZE

        batched_ds = BatchDataset(
            ds, batch_size=batch_size, batcher=generic_batch, worker_config=no_worker_config
        )

        cnt = 0
        expected_num_batches = math.ceil(DATASET_SIZE / batch_size)
        for idx, wrapped_sample in enumerate(get_ld(batched_ds)):
            # Check batch sizes
            if idx < expected_num_batches - 1:
                assert wrapped_sample.image.shape[0] == batch_size
                assert wrapped_sample.caption.shape[0] == batch_size
            else:
                assert wrapped_sample.image.shape[0] == DATASET_SIZE % batch_size
                assert wrapped_sample.caption.shape[0] == DATASET_SIZE % batch_size

            # Check image size
            assert tuple(wrapped_sample.image.shape[1:]) == (3, 100, 100)

            cnt += 1

            logging.info(f"  Batch {idx}:")
            logging.info(f"    {wrapped_sample.image.shape=}")
            logging.info(f"    {wrapped_sample.caption.shape=}")

        assert cnt == expected_num_batches

        # Check if actual image and caption data are correct
        loader = get_ld(
            BatchDataset(ds, batch_size=9, batcher=generic_batch, worker_config=no_worker_config),
        )
        batch_sizes = []
        dataset_samples = {sample["caption"]: sample["image"] for sample in self.samples}
        for idx, sample in enumerate(loader):
            batch_sizes.append(sample.image.shape[0])
            for bidx in range(sample.image.shape[0]):
                refimg = dataset_samples.pop(
                    sample.caption[bidx].numpy().tobytes().rstrip(b"\0").decode()
                )
                assert torch.allclose(
                    sample.image[bidx],
                    torch.permute(torch.tensor(refimg, dtype=torch.float32) / 255, (2, 0, 1)),
                )
        assert len(dataset_samples) == 0
        assert batch_sizes == [9, 9, 9, 9, 9, 5]

    def test_field_access(self):
        ds = get_dataset_from_config(
            self.dataset_path,
            dataset_config="dataset_field.yaml",
            split_part="train",
            worker_config=no_worker_config,
            training=False,
            sample_type=CaptioningSample,
        )
        captions = set(sample["caption"] for sample in self.samples)
        for sample in get_loader(ds, worker_config=no_worker_config):
            captions.remove(sample.caption)
        assert len(captions) == 0

    def test_sample_loader(self):
        ds = get_dataset_from_config(
            self.dataset_path,
            dataset_config="dataset_sample_loader.yaml",
            split_part="train",
            worker_config=no_worker_config,
            training=False,
            sample_type=CaptioningSample,
        )
        captions = set(sample["caption"] for sample in self.samples)
        for sample in get_loader(ds, worker_config=no_worker_config):
            assert sample.caption[:4] == "<SL>"
            captions.remove(sample.caption[4:])
        assert len(captions) == 0

    def test_sample_loader_key(self):
        ds = get_dataset_from_config(
            self.dataset_path,
            dataset_config="dataset_sample_loader_key.yaml",
            split_part="train",
            worker_config=no_worker_config,
            training=False,
            sample_type=CaptioningSample,
        )
        captions = set(sample["caption"] for sample in self.samples)
        keys = set(
            f"<SL>parts/data-{idx // 30:d}.tar/{idx:06d}" for idx in range(len(self.samples))
        )
        for sample in get_loader(ds, worker_config=no_worker_config):
            assert sample.caption[:4] == "<SL>"
            captions.remove(sample.caption[4:])
            keys.remove(sample.__key__)
        assert len(captions) == 0
        assert len(keys) == 0

    def test_exclusion(self):
        ds = get_dataset_from_config(
            self.dataset_path,
            dataset_config="dataset_exclude.yaml",
            split_part="train",
            worker_config=no_worker_config,
            training=False,
            sample_type=CaptioningSample,
        )

        keys = [entry.__key__ for entry in get_loader(ds, worker_config=no_worker_config)]
        assert keys == [
            f"parts/data-1.tar/{i:06d}" for i in list(range(30, 35)) + list(range(40, 50))
        ]

    def test_loader(self):
        torch.manual_seed(42)

        class TestTaskEncoder(DefaultTaskEncoder):
            def __init__(self):
                super().__init__(raw_batch_type=CaptioningBatch)

            def encode_sample(self, sample: CaptioningSample) -> EncodedCaptioningSample:
                return EncodedCaptioningSample(
                    __key__=sample.__key__,
                    __restore_key__=sample.__restore_key__,
                    image=sample.image,
                    caption=torch.frombuffer(sample.caption.encode(), dtype=torch.uint8),
                )

        loader = get_loader(
            get_train_dataset(
                self.dataset_path,
                batch_size=10,
                worker_config=no_worker_config,
                parallel_shard_iters=2,
                virtual_epoch_length=2,
                shuffle_buffer_size=None,
                max_samples_per_sequence=None,
                task_encoder=TestTaskEncoder(),
            ),
            worker_config=no_worker_config,
        )

        assert len(loader) == 2

        def hist(data):
            """Histogram function"""
            r = {}
            for k in data:
                r.setdefault(k, 0)
                r[k] += 1
            return r

        print([[batch.__key__ for batch in loader] for _ in range(100)])
        keys = [key for _ in range(100) for batch in loader for key in batch.__key__]
        # 100 iterations, 2 virtual epoch size, batch size 10
        print(len(keys), keys)
        assert len(keys) == 100 * 2 * 10
        # Data should be approximately sampled uniformly (40+-1 samples per key)
        assert all(v in (39, 40, 41) for v in hist(keys).values())
        assert len(hist(keys)) == 50

        loader2 = get_loader(
            get_val_dataset(
                self.dataset_path,
                split_part="train",
                batch_size=10,
                worker_config=no_worker_config,
                task_encoder=TestTaskEncoder(),
            ),
            worker_config=no_worker_config,
        )
        assert len(loader2) == 5
        # The order in the split is shuffled this way
        assert list(key for batch in loader2 for key in batch.__key__) == [
            f"parts/data-1.tar/{i:06d}" for i in range(30, 50)
        ] + [f"parts/data-0.tar/{i:06d}" for i in range(30)]

    def test_default_dataset(self):
        torch.manual_seed(42)

        train_loader = get_loader(
            get_train_dataset(
                self.dataset_path,
                batch_size=10,
                worker_config=no_worker_config,
                shuffle_buffer_size=None,
                max_samples_per_sequence=None,
            ),
            worker_config=no_worker_config,
        )

        val_loader = get_loader(
            get_val_dataset(
                self.dataset_path,
                split_part="train",
                batch_size=10,
                worker_config=no_worker_config,
            ),
            worker_config=no_worker_config,
        )

        n_samples = 0
        for i, sample in zip(range(100), train_loader):
            assert sample.image.shape == (10, 3, 100, 100)
            n_samples += sample.image.shape[0]
        assert n_samples == 1000
        n_samples = 0
        for sample in val_loader:
            assert sample.image.shape == (10, 3, 100, 100)
            n_samples += sample.image.shape[0]
        assert n_samples == 50

    def test_dataset_len(self):
        torch.manual_seed(42)

        worker_config = WorkerConfig(rank=0, world_size=1, num_workers=4)

        train_dataset = get_train_dataset(
            self.dataset_path,
            batch_size=11,
            worker_config=worker_config,
            virtual_epoch_length=12,
            shuffle_buffer_size=None,
            max_samples_per_sequence=None,
        )
        train_loader = get_loader(
            train_dataset,
            worker_config=worker_config,
        )

        assert len(train_dataset) == 12
        assert len(train_loader) == 12
        assert len(list(train_loader)) == 12

        val_dataset = get_val_dataset(
            self.dataset_path, split_part="train", batch_size=1, worker_config=no_worker_config
        )
        val_loader = get_loader(val_dataset, worker_config=no_worker_config)
        assert len(val_loader) == 50
        assert len(list(val_loader)) == 50

        val_dataset = get_val_dataset(
            self.dataset_path, split_part="train", batch_size=11, worker_config=worker_config
        )
        val_loader = get_loader(val_dataset, worker_config=worker_config)

        # n samples: ceil(50 / 11) // 4 * 4
        assert len(val_dataset) == 8
        assert len(val_loader) == 8
        assert len(list(val_loader)) == 8
        assert [len(entry.__key__) for entry in val_loader] == [11, 11, 11, 11, 1, 2, 1, 2]
        assert sum(len(entry.__key__) for entry in val_loader) == 50

    def test_multirank_dataset(self):
        torch.manual_seed(42)

        worker_config_r0 = WorkerConfig(rank=0, world_size=2, num_workers=2)
        worker_config_r1 = WorkerConfig(rank=1, world_size=2, num_workers=2)

        train_dataset = get_train_dataset(
            self.dataset_path,
            batch_size=11,
            worker_config=worker_config_r0,
            virtual_epoch_length=12,
            shuffle_buffer_size=None,
            max_samples_per_sequence=None,
        )
        train_loader = get_loader(
            train_dataset,
            worker_config=worker_config_r0,
        )

        assert len(train_dataset) == 12
        assert len(train_loader) == 12
        assert len(list(train_loader)) == 12

        val_dataset0 = get_val_dataset(
            self.dataset_path, split_part="train", batch_size=1, worker_config=worker_config_r0
        )
        val_loader0 = get_loader(val_dataset0, worker_config=worker_config_r0)
        print(len(val_loader0))
        assert len(val_loader0) == 25
        keys0 = set(key for entry in val_loader0 for key in entry.__key__)
        assert len(keys0) == 25

        val_dataset0b11 = get_val_dataset(
            self.dataset_path, split_part="train", batch_size=11, worker_config=worker_config_r0
        )
        val_loader0b11 = get_loader(val_dataset0b11, worker_config=worker_config_r0)

        assert len(val_dataset0b11) == 4
        assert len(val_loader0b11) == 4
        assert len(list(val_loader0b11)) == 4
        keys0b11 = set(key for entry in val_loader0b11 for key in entry.__key__)
        print([len(entry.__key__) for entry in val_loader0b11])
        assert [len(entry.__key__) for entry in val_loader0b11] == [11, 11, 1, 2]
        assert len(keys0b11) == 25

        assert keys0b11 == keys0

        val_dataset1 = get_val_dataset(
            self.dataset_path, split_part="train", batch_size=1, worker_config=worker_config_r1
        )
        val_loader1 = get_loader(val_dataset1, worker_config=worker_config_r1)
        print(len(val_loader1))
        assert len(val_loader1) == 25
        keys1 = set(key for entry in val_loader1 for key in entry.__key__)
        assert len(keys1) == 25
        print(sorted(keys1))
        print(sorted(keys0))
        assert keys1.isdisjoint(keys0)

        val_dataset1b11 = get_val_dataset(
            self.dataset_path, split_part="train", batch_size=11, worker_config=worker_config_r1
        )
        val_loader1b11 = get_loader(val_dataset1b11, worker_config=worker_config_r1)

        assert len(val_dataset1b11) == 4
        assert len(val_loader1b11) == 4
        assert len(list(val_loader1b11)) == 4
        keys1b11 = set(key for entry in val_loader1b11 for key in entry.__key__)
        print([len(entry.__key__) for entry in val_loader1b11])
        assert [len(entry.__key__) for entry in val_loader1b11] == [11, 11, 1, 2]
        assert len(keys1b11) == 25
        assert keys1b11.isdisjoint(keys0b11)

        assert keys1b11 == keys1

    def test_weight_aug(self):
        class WeightAugmentTaskEncoder(AugmentTaskEncoder):
            def __init__(self, task_encoder: TaskEncoder, weight: float, target_data_class: type):
                super().__init__(task_encoder)
                self.weight = weight
                self.target_data_class = target_data_class

            def encode_sample(self, sample):
                sample = super().encode_sample(sample)
                return self.target_data_class(**dataclasses.asdict(sample), weight=self.weight)

        torch.manual_seed(42)

        @dataclass
        class WeightedCaptioningBatch(Batch):
            __key__: List[str]
            __restore_key__: Tuple[Union[str, int], ...]
            __subflavor__: List[str]
            __subflavors__: List[Dict[str, Any]]
            image: torch.Tensor
            caption: List[str]
            weight: float

        loader = get_loader(
            get_val_dataset(
                self.dataset_path,
                split_part="train",
                batch_size=10,
                worker_config=no_worker_config,
                task_encoder=WeightAugmentTaskEncoder(
                    DefaultTaskEncoder(),
                    weight=0.8,
                    target_data_class=WeightedCaptioningBatch,
                ),
            ),
            worker_config=no_worker_config,
        )

        for data in loader:
            assert data.weight == [0.8] * 10

    def test_blending(self):
        torch.manual_seed(42)

        loader = get_loader(
            BlendDataset(
                (
                    get_train_dataset(
                        self.dataset_path,
                        batch_size=10,
                        worker_config=no_worker_config,
                        shuffle_buffer_size=None,
                        max_samples_per_sequence=None,
                    ),
                    2,
                ),
                (
                    get_train_dataset(
                        self.dataset_path,
                        batch_size=20,
                        worker_config=no_worker_config,
                        shuffle_buffer_size=None,
                        max_samples_per_sequence=None,
                    ),
                    8,
                ),
                worker_config=no_worker_config,
            ),
            worker_config=no_worker_config,
        )

        bs_hist = {10: 0, 20: 0}
        for i, sample in zip(range(1000), loader):
            bs_hist[sample.image.shape[0]] += 1
        print(bs_hist)
        assert 150 <= bs_hist[10] <= 250
        assert 750 <= bs_hist[20] <= 850

    def test_mixing_homogeneous(self):
        @dataclass
        class TestBatch(Batch):
            __key__: List[str]
            __restore_key__: Tuple[Union[str, int], ...]
            __subflavor__: List[str]
            __subflavors__: List[Dict[str, Any]]
            image: torch.Tensor
            caption: List[str]
            source: int

        class TestTaskEncoder(TaskEncoder):
            def __init__(self, source: int):
                self.source = source

            def encode_batch(self, batch):
                return TestBatch(**dataclasses.asdict(batch), source=self.source)

        loader = get_loader(
            MixBatchDataset(
                (
                    get_train_dataset(
                        self.dataset_path,
                        batch_size=1,
                        worker_config=no_worker_config,
                        task_encoder=TestTaskEncoder(source=0),
                        shuffle_buffer_size=None,
                        max_samples_per_sequence=None,
                    ),
                    2,
                ),
                (
                    get_train_dataset(
                        self.dataset_path,
                        batch_size=1,
                        worker_config=no_worker_config,
                        task_encoder=TestTaskEncoder(source=1),
                        shuffle_buffer_size=None,
                        max_samples_per_sequence=None,
                    ),
                    8,
                ),
                batch_size=10,
                batch_mix_fn=homogeneous_concat_mix,
                worker_config=no_worker_config,
            ),
            worker_config=no_worker_config,
        )

        source_hist = {0: 0, 1: 0}
        for i, sample in zip(range(1000), loader):
            assert sample.image.shape == (10, 3, 100, 100)
            for source in sample.source:
                source_hist[source] += 1
        assert 1500 <= source_hist[0] <= 2500
        assert 7500 <= source_hist[1] <= 8500

    def test_mixing_heterogeneous(self):
        @dataclass
        class TestBatch1(Batch):
            __key__: List[str]
            __restore_key__: Tuple[Union[str, int], ...]
            __subflavor__: List[str]
            __subflavors__: List[Dict[str, Any]]
            image: torch.Tensor
            caption: List[str]
            source: int

        @dataclass
        class TestBatch2(TestBatch1):
            pass

        class TestTaskEncoder(TaskEncoder):
            def __init__(self, source: int, batch_cls: Type[TestBatch1]):
                self.source = source
                self.batch_cls = batch_cls

            def encode_batch(self, batch):
                return self.batch_cls(**dataclasses.asdict(batch), source=self.source)

        loader = get_loader(
            MixBatchDataset(
                (
                    get_train_dataset(
                        self.dataset_path,
                        batch_size=1,
                        worker_config=no_worker_config,
                        task_encoder=TestTaskEncoder(source=0, batch_cls=TestBatch1),
                        shuffle_buffer_size=None,
                        max_samples_per_sequence=None,
                    ),
                    2,
                ),
                (
                    get_train_dataset(
                        self.dataset_path,
                        batch_size=1,
                        worker_config=no_worker_config,
                        task_encoder=TestTaskEncoder(source=1, batch_cls=TestBatch2),
                        shuffle_buffer_size=None,
                        max_samples_per_sequence=None,
                    ),
                    8,
                ),
                batch_size=10,
                worker_config=no_worker_config,
            ),
            worker_config=no_worker_config,
        )

        source_hist = {0: 0, 1: 0}
        for i, samples in zip(range(1000), loader):
            assert len(samples) == 10
            for sample in samples:
                assert sample.image.shape == (1, 3, 100, 100)
                source_hist[sample.source] += 1
        assert 1500 <= source_hist[0] <= 2500
        assert 7500 <= source_hist[1] <= 8500

    def test_val_limit(self):
        torch.manual_seed(42)

        loader = get_loader(
            get_val_dataset(
                self.dataset_path,
                split_part="train",
                batch_size=2,
                worker_config=no_worker_config,
                limit=3,
            ),
            worker_config=no_worker_config,
        )

        assert len(loader) == 3

        samples = [[batch.__key__ for batch in loader] for _ in range(10)]
        print(samples)
        assert all(samples[0] == one_ep_samples for one_ep_samples in samples)

        worker_config = WorkerConfig(rank=0, world_size=1, num_workers=2)

        loader = get_loader(
            get_val_dataset(
                self.dataset_path,
                split_part="train",
                batch_size=2,
                worker_config=worker_config,
                limit=3,
            ),
            worker_config=worker_config,
        )

        assert len(loader) == 3

        samples_wrk2 = [[batch.__key__ for batch in loader] for _ in range(10)]
        print(samples)
        assert all(samples_wrk2[0] == one_ep_samples for one_ep_samples in samples_wrk2)
        # TODO: This should be the same.
        # assert samples_wrk2 == samples

    def test_current_batch_index(self):
        # Tests if the get_current_batch_index works properly
        torch.manual_seed(42)

        class TestTaskEncoder(TaskEncoder):

            @stateless(restore_seeds=True)
            def encode_sample(self, sample):
                # print("si stack:", WorkerConfig._sample_index_stack)
                return ExtendedCaptioningSample.extend(
                    sample,
                    batch_index=self.current_batch_index,
                    sample_index=self.current_sample_index,
                    rand_num=random.randint(0, 1000),
                )

        # First, test simple single main-thread loader with accessing get_current_batch_index
        loader = get_loader(
            get_train_dataset(
                self.dataset_path,
                batch_size=2,
                task_encoder=TestTaskEncoder(),
                worker_config=no_worker_config,
                shuffle_buffer_size=20,
                max_samples_per_sequence=10,
            ),
            worker_config=no_worker_config,
        )

        batches = list(zip(range(20), loader))
        print("bi", [batch.batch_index for batch_idx, batch in batches])
        assert all(all(bi == batch_idx for bi in batch.batch_index) for batch_idx, batch in batches)

        print("si", [batch.sample_index for batch_idx, batch in batches])
        assert all(
            all(
                si == sample_offset + batch_idx * 2
                for sample_offset, si in enumerate(batch.sample_index)
            )
            for batch_idx, batch in batches
        )

        print("rk", [batch.__restore_key__ for batch_idx, batch in batches])
        assert loader.can_restore_sample()

        # These need to be hard coded to detect breaking changes
        # If a change is expected, update the values with the ones printed below
        ref_batch_rand_nums = [
            [661, 762],
            [206, 470],
            [130, 283],
            [508, 61],
            [625, 661],
            [296, 376],
            [632, 514],
            [715, 406],
            [555, 27],
            [760, 36],
            [607, 610],
            [825, 219],
            [564, 832],
            [876, 512],
            [632, 605],
            [357, 738],
            [40, 378],
            [609, 444],
            [610, 367],
            [367, 69],
        ]

        batch_rand_nums = []
        for batch_idx, batch in batches:
            restore_batch = loader.restore_sample(batch.__restore_key__)
            assert restore_batch.batch_index == batch.batch_index
            assert restore_batch.sample_index == batch.sample_index
            assert restore_batch.rand_num == batch.rand_num

            batch_rand_nums.append(restore_batch.rand_num)
            assert np.allclose(restore_batch.image, batch.image)

        # For constructing the test data above:
        print("batch_rand_nums: ", batch_rand_nums)
        assert batch_rand_nums == ref_batch_rand_nums

        # Now, test multi-worker loader with accessing get_current_batch_index
        worker_config_r0 = WorkerConfig(rank=0, world_size=2, num_workers=2)
        worker_config_r1 = WorkerConfig(rank=1, world_size=2, num_workers=2)

        loader = get_loader(
            get_train_dataset(
                self.dataset_path,
                batch_size=2,
                task_encoder=TestTaskEncoder(),
                worker_config=worker_config_r0,
                shuffle_buffer_size=20,
                max_samples_per_sequence=10,
            ),
            worker_config=worker_config_r0,
        )
        loader_r1 = get_loader(
            get_train_dataset(
                self.dataset_path,
                batch_size=2,
                task_encoder=TestTaskEncoder(),
                worker_config=worker_config_r1,
                shuffle_buffer_size=20,
                max_samples_per_sequence=10,
            ),
            worker_config=worker_config_r1,
        )

        batches = list(zip(range(20), loader))
        print("bir0", [batch.batch_index for batch_idx, batch in batches])
        assert all(all(bi == batch_idx for bi in batch.batch_index) for batch_idx, batch in batches)

        print("sir0", [batch.sample_index for batch_idx, batch in batches])
        assert all(all(bi == batch_idx for bi in batch.batch_index) for batch_idx, batch in batches)
        assert all(
            all(
                si == 2 * sample_offset + (batch_idx * 2 - batch_idx % 2)
                for sample_offset, si in enumerate(batch.sample_index)
            )
            for batch_idx, batch in batches
        )

        batches_r1 = list(zip(range(20), loader_r1))
        print("bir0", [batch.batch_index for batch_idx, batch in batches_r1])
        print("sir1", [batch.sample_index for batch_idx, batch in batches_r1])
        assert all(
            all(bi == batch_idx for bi in batch.batch_index) for batch_idx, batch in batches_r1
        )
        assert all(
            all(
                si == 2 * sample_offset + (batch_idx * 2 - batch_idx % 2)
                for sample_offset, si in enumerate(batch.sample_index)
            )
            for batch_idx, batch in batches_r1
        )

        # Now, test multi-worker loader with accessing get_current_batch_index and save/restore state
        loader = get_savable_loader(
            get_train_dataset(
                self.dataset_path,
                batch_size=2,
                task_encoder=TestTaskEncoder(),
                worker_config=worker_config_r0,
                shuffle_buffer_size=20,
                max_samples_per_sequence=10,
            ),
            worker_config=worker_config_r0,
        )
        loader_r1 = get_savable_loader(
            get_train_dataset(
                self.dataset_path,
                batch_size=2,
                task_encoder=TestTaskEncoder(),
                worker_config=worker_config_r1,
                shuffle_buffer_size=20,
                max_samples_per_sequence=10,
            ),
            worker_config=worker_config_r1,
        )

        batches = list(zip(range(20), loader))
        print([batch.batch_index for batch_idx, batch in batches])
        assert all(all(bi == batch_idx for bi in batch.batch_index) for batch_idx, batch in batches)
        assert all(
            all(
                si == 2 * sample_offset + (batch_idx * 2 - batch_idx % 2)
                for sample_offset, si in enumerate(batch.sample_index)
            )
            for batch_idx, batch in batches
        )

        batches_r1 = list(zip(range(20), loader_r1))
        print([batch.batch_index for batch_idx, batch in batches_r1])
        assert all(
            all(bi == batch_idx for bi in batch.batch_index) for batch_idx, batch in batches_r1
        )
        assert all(
            all(
                si == 2 * sample_offset + (batch_idx * 2 - batch_idx % 2)
                for sample_offset, si in enumerate(batch.sample_index)
            )
            for batch_idx, batch in batches_r1
        )

        # Save and restore state
        state = loader.save_state_rank()

        # Restore state and check if the batch index is restored correctly
        loader = get_savable_loader(
            get_train_dataset(
                self.dataset_path,
                batch_size=2,
                task_encoder=TestTaskEncoder(),
                worker_config=worker_config_r0,
                shuffle_buffer_size=20,
                max_samples_per_sequence=10,
            ),
            worker_config=worker_config_r0,
        )
        loader.restore_state_rank(state)

        batches = list(zip(range(20, 40), loader))
        print([batch.batch_index for batch_idx, batch in batches])
        print([batch.sample_index for batch_idx, batch in batches])
        assert all(all(bi == batch_idx for bi in batch.batch_index) for batch_idx, batch in batches)
        assert all(
            all(
                si == 2 * sample_offset + (batch_idx * 2 - batch_idx % 2)
                for sample_offset, si in enumerate(batch.sample_index)
            )
            for batch_idx, batch in batches
        )

    def test_current_batch_index_generator(self):
        # Tests if the get_current_batch_index works properly
        torch.manual_seed(42)

        class TestTaskEncoder(TaskEncoder):

            @stateless(restore_seeds=True)
            def encode_sample(self, sample):
                # print("si stack:", WorkerConfig._sample_index_stack)
                yield ExtendedCaptioningSample.extend(
                    sample,
                    batch_index=self.current_batch_index,
                    sample_index=self.current_sample_index,
                    rand_num=random.randint(0, 1000) + 0,
                )

                yield ExtendedCaptioningSample.extend(
                    sample,
                    batch_index=self.current_batch_index,
                    sample_index=self.current_sample_index,
                    rand_num=random.randint(0, 1000) + 1000,
                )

        # First, test simple single main-thread loader with accessing get_current_batch_index
        loader = get_loader(
            get_train_dataset(
                self.dataset_path,
                batch_size=3,
                task_encoder=TestTaskEncoder(),
                worker_config=no_worker_config,
                shuffle_buffer_size=20,
                max_samples_per_sequence=10,
            ),
            worker_config=no_worker_config,
        )

        batches = list(zip(range(20), loader))
        print("bi", [batch.batch_index for batch_idx, batch in batches])
        assert all(all(bi == batch_idx for bi in batch.batch_index) for batch_idx, batch in batches)

        print("si", [batch.sample_index for batch_idx, batch in batches])
        assert all(
            all(
                si == (sample_offset + batch_idx * 3) // 2
                for sample_offset, si in enumerate(batch.sample_index)
            )
            for batch_idx, batch in batches
        )

        print("rk", [batch.__restore_key__ for batch_idx, batch in batches])
        assert loader.can_restore_sample()

        # These need to be hard coded to detect breaking changes
        # If a change is expected, update the values with the ones printed below
        ref_batch_rand_nums = [
            [661, 1747, 762],
            [1171, 206, 1921],
            [470, 1705, 130],
            [1722, 283, 1990],
            [508, 1041, 61],
            [1102, 625, 1559],
            [661, 1512, 296],
            [1866, 376, 1345],
            [632, 1176, 514],
            [1652, 715, 1702],
            [406, 1552, 555],
            [1303, 27, 1520],
            [760, 1380, 36],
            [1869, 607, 1292],
            [610, 1084, 825],
            [1113, 219, 1102],
            [564, 1695, 832],
            [1612, 876, 2000],
            [512, 1308, 632],
            [1425, 605, 1931],
        ]

        batch_rand_nums = []
        for batch_idx, batch in batches:
            restore_batch = loader.restore_sample(batch.__restore_key__)
            assert restore_batch.batch_index == batch.batch_index
            assert restore_batch.sample_index == batch.sample_index
            assert restore_batch.rand_num == batch.rand_num

            batch_rand_nums.append(restore_batch.rand_num)
            assert np.allclose(restore_batch.image, batch.image)

        # For constructing the test data above:
        print("batch_rand_nums: ", batch_rand_nums)
        assert batch_rand_nums == ref_batch_rand_nums

        # Now, test multi-worker loader with accessing get_current_batch_index
        worker_config_r0 = WorkerConfig(rank=0, world_size=2, num_workers=2)
        worker_config_r1 = WorkerConfig(rank=1, world_size=2, num_workers=2)

        loader = get_loader(
            get_train_dataset(
                self.dataset_path,
                batch_size=3,
                task_encoder=TestTaskEncoder(),
                worker_config=worker_config_r0,
                shuffle_buffer_size=20,
                max_samples_per_sequence=10,
            ),
            worker_config=worker_config_r0,
        )
        loader_r1 = get_loader(
            get_train_dataset(
                self.dataset_path,
                batch_size=3,
                task_encoder=TestTaskEncoder(),
                worker_config=worker_config_r1,
                shuffle_buffer_size=20,
                max_samples_per_sequence=10,
            ),
            worker_config=worker_config_r1,
        )

        batches = list(zip(range(20), loader))
        print("bir0", [batch.batch_index for batch_idx, batch in batches])
        assert all(all(bi == batch_idx for bi in batch.batch_index) for batch_idx, batch in batches)

        print("sir0", [batch.sample_index for batch_idx, batch in batches])
        # [[0, 0, 2], [1, 1, 3], [2, 4, 4], [3, 5, 5], [6, 6, 8], [7, 7, 9], [8, 10, 10], [9, 11, 11], [12, 12, 14], [13, 13, 15], [14, 16, 16], [15, 17, 17], [18, 18, 20], [19, 19, 21], [20, 22, 22], [21, 23, 23], [24, 24, 26], [25, 25, 27], [26, 28, 28], [27, 29, 29]]
        assert all(
            all(
                si == batch_idx + (batch_idx // 4 + ((batch_idx // 2 % 2) + sample_offset) // 2) * 2
                for sample_offset, si in enumerate(batch.sample_index)
            )
            for batch_idx, batch in batches
        )

        batches_r1 = list(zip(range(20), loader_r1))
        print("bir0", [batch.batch_index for batch_idx, batch in batches_r1])
        print("sir1", [batch.sample_index for batch_idx, batch in batches_r1])
        assert all(
            all(bi == batch_idx for bi in batch.batch_index) for batch_idx, batch in batches_r1
        )
        assert all(
            all(
                si == batch_idx + (batch_idx // 4 + ((batch_idx // 2 % 2) + sample_offset) // 2) * 2
                for sample_offset, si in enumerate(batch.sample_index)
            )
            for batch_idx, batch in batches_r1
        )

        # Now, test multi-worker loader with accessing get_current_batch_index and save/restore state
        loader = get_savable_loader(
            get_train_dataset(
                self.dataset_path,
                batch_size=3,
                task_encoder=TestTaskEncoder(),
                worker_config=worker_config_r0,
                shuffle_buffer_size=20,
                max_samples_per_sequence=10,
            ),
            worker_config=worker_config_r0,
        )
        loader_r1 = get_savable_loader(
            get_train_dataset(
                self.dataset_path,
                batch_size=3,
                task_encoder=TestTaskEncoder(),
                worker_config=worker_config_r1,
                shuffle_buffer_size=20,
                max_samples_per_sequence=10,
            ),
            worker_config=worker_config_r1,
        )

        batches = list(zip(range(20), loader))
        print("bi:", [batch.batch_index for batch_idx, batch in batches])
        print("si:", [batch.sample_index for batch_idx, batch in batches])
        assert all(all(bi == batch_idx for bi in batch.batch_index) for batch_idx, batch in batches)
        assert all(
            all(
                si == batch_idx + (batch_idx // 4 + ((batch_idx // 2 % 2) + sample_offset) // 2) * 2
                for sample_offset, si in enumerate(batch.sample_index)
            )
            for batch_idx, batch in batches
        )

        batches_r1 = list(zip(range(20), loader_r1))
        print([batch.batch_index for batch_idx, batch in batches_r1])
        assert all(
            all(bi == batch_idx for bi in batch.batch_index) for batch_idx, batch in batches_r1
        )
        assert all(
            all(
                si == batch_idx + (batch_idx // 4 + ((batch_idx // 2 % 2) + sample_offset) // 2) * 2
                for sample_offset, si in enumerate(batch.sample_index)
            )
            for batch_idx, batch in batches_r1
        )

        # Save and restore state
        state = loader.save_state_rank()

        # Iter next 20 from the loader
        cmp_batches = list(zip(range(20, 40), loader))
        print("bi:", [batch.batch_index for batch_idx, batch in cmp_batches])
        print("si:", [batch.sample_index for batch_idx, batch in cmp_batches])
        print("rnd:", [batch.rand_num for batch_idx, batch in cmp_batches])
        assert all(
            all(bi == batch_idx for bi in batch.batch_index) for batch_idx, batch in cmp_batches
        )
        assert all(
            all(
                si == batch_idx + (batch_idx // 4 + ((batch_idx // 2 % 2) + sample_offset) // 2) * 2
                for sample_offset, si in enumerate(batch.sample_index)
            )
            for batch_idx, batch in cmp_batches
        )

        # Restore state and check if the batch index is restored correctly
        loader = get_savable_loader(
            get_train_dataset(
                self.dataset_path,
                batch_size=3,
                task_encoder=TestTaskEncoder(),
                worker_config=worker_config_r0,
                shuffle_buffer_size=20,
                max_samples_per_sequence=10,
            ),
            worker_config=worker_config_r0,
        )
        loader.restore_state_rank(state)

        batches = list(zip(range(20, 40), loader))
        print("bi:", [batch.batch_index for batch_idx, batch in batches])
        print("si:", [batch.sample_index for batch_idx, batch in batches])
        print("rnd:", [batch.rand_num for batch_idx, batch in batches])
        assert all(all(bi == batch_idx for bi in batch.batch_index) for batch_idx, batch in batches)
        assert all(
            all(
                si == batch_idx + (batch_idx // 4 + ((batch_idx // 2 % 2) + sample_offset) // 2) * 2
                for sample_offset, si in enumerate(batch.sample_index)
            )
            for batch_idx, batch in batches
        )
        assert all(
            all(b1s == b2s for b1s, b2s in zip(b1.rand_num, b2.rand_num))
            for (_b1idx, b1), (_b2idx, b2) in zip(batches, cmp_batches)
        )

    def test_packing(self):
        torch.manual_seed(42)

        class TestTaskEncoder(DefaultTaskEncoder):
            def __init__(self):
                super().__init__(raw_batch_type=CaptioningBatch)

            @stateless
            def encode_sample(self, sample: CaptioningSample) -> EncodedCaptioningSample:
                return EncodedCaptioningSample(
                    __key__=sample.__key__,
                    __restore_key__=sample.__restore_key__,
                    image=sample.image,
                    caption=torch.frombuffer(sample.caption.encode(), dtype=torch.uint8),
                )

            def select_samples_to_pack(
                self, samples: List[EncodedCaptioningSample]
            ) -> List[List[EncodedCaptioningSample]]:
                assert len(samples) == 21
                return [samples[:1], samples[1 : 1 + 4], samples[1 + 4 : 1 + 4 + 16]]

            @stateless
            def pack_selected_samples(
                self, samples: List[EncodedCaptioningSample]
            ) -> EncodedCaptioningSample:
                return EncodedCaptioningSample(
                    __key__=",".join([sample.__key__ for sample in samples]),
                    __restore_key__=(),
                    image=torch.stack([sample.image for sample in samples]),
                    caption=torch.cat([sample.caption for sample in samples]),
                )

        loader = get_loader(
            get_train_dataset(
                self.dataset_path,
                batch_size=2,
                packing_buffer_size=21,
                worker_config=no_worker_config,
                virtual_epoch_length=6,
                shuffle_buffer_size=None,
                max_samples_per_sequence=None,
                task_encoder=TestTaskEncoder(),
            ),
            worker_config=no_worker_config,
        )

        assert len(loader) == 6

        samples = list(loader)

        print([batch.__key__ for batch in samples])
        print([batch.__restore_key__ for batch in samples])
        print([len(batch.__key__) for batch in samples])
        print([[len(batch_key.split(",")) for batch_key in batch.__key__] for batch in samples])

        # Each batch should have 2 samples
        assert [len(batch.__key__) for batch in samples] == [
            2,
            2,
            2,
            2,
            2,
            2,
        ]

        # The packs of lengths 1, 4, 16 should be unrolled repeatedly across the batches of size 2
        assert [
            [len(batch_key.split(",")) for batch_key in batch.__key__] for batch in samples
        ] == [[1, 4], [16, 1], [4, 16], [1, 4], [16, 1], [4, 16]]

        restored_sample_1 = loader.restore_sample(samples[1].__restore_key__)
        assert restored_sample_1.__key__ == samples[1].__key__
        assert restored_sample_1.__restore_key__ == samples[1].__restore_key__

        worker_config_r0 = WorkerConfig(rank=0, world_size=2, num_workers=2)

        loader_r0 = get_savable_loader(
            get_train_dataset(
                self.dataset_path,
                batch_size=2,
                packing_buffer_size=21,
                worker_config=worker_config_r0,
                virtual_epoch_length=8,
                shuffle_buffer_size=None,
                max_samples_per_sequence=None,
                task_encoder=TestTaskEncoder(),
            ),
            worker_config=worker_config_r0,
            checkpoint_every_min_n_samples=1,
            checkpoint_every_sec=0,
        )

        samples_r0 = list(loader_r0)
        assert [
            [len(batch_key.split(",")) for batch_key in batch.__key__] for batch in samples_r0
        ] == [[1, 4], [1, 4], [16, 1], [16, 1], [4, 16], [4, 16], [1, 4], [1, 4]]

        restored_sample_1 = loader_r0.restore_sample(samples_r0[1].__restore_key__)
        assert restored_sample_1.__key__ == samples_r0[1].__key__
        assert restored_sample_1.__restore_key__ == samples_r0[1].__restore_key__

        rank_state_r0 = loader_r0.save_state_rank()
        samples_r0_cmp = list(loader_r0)
        assert [
            [len(batch_key.split(",")) for batch_key in batch.__key__] for batch in samples_r0_cmp
        ] == [[16, 1], [16, 1], [4, 16], [4, 16], [1, 4], [1, 4], [16, 1], [16, 1]]

        loader_r0 = get_savable_loader(
            get_train_dataset(
                self.dataset_path,
                batch_size=2,
                packing_buffer_size=21,
                worker_config=worker_config_r0,
                virtual_epoch_length=8,
                shuffle_buffer_size=None,
                max_samples_per_sequence=None,
                task_encoder=TestTaskEncoder(),
            ),
            worker_config=worker_config_r0,
            checkpoint_every_min_n_samples=1,
            checkpoint_every_sec=0,
        )

        loader_r0.restore_state_rank(rank_state_r0)

        samples_r0_restored = list(loader_r0)
        print("cmp", [batch.__key__ for batch in samples_r0_cmp])
        print("rst", [batch.__key__ for batch in samples_r0_restored])
        assert [
            [len(batch_key.split(",")) for batch_key in batch.__key__]
            for batch in samples_r0_restored
        ] == [[16, 1], [16, 1], [4, 16], [4, 16], [1, 4], [1, 4], [16, 1], [16, 1]]

        assert all(s0.__key__ == s1.__key__ for s0, s1 in zip(samples_r0_cmp, samples_r0_restored))

    def test_debug_dataset(self):
        torch.manual_seed(42)
        worker_config = WorkerConfig(
            rank=0,
            world_size=1,
            num_workers=2,
            worker_log_level=3,
            worker_debug_path=str(self.dataset_path) + "/worker_debug/{worker_id}.jsonl",
        )

        # Reset this to 0 to make sure the test is deterministic
        SavableDataLoader._next_id = 0

        loader = get_savable_loader(
            get_val_dataset(
                self.dataset_path,
                split_part="train",
                batch_size=5,
                worker_config=worker_config,
            ),
            worker_config=worker_config,
        )

        assert len(loader) == 10

        samples = [[batch.__key__ for batch in loader] for _ in range(2)]
        print(samples)

        debug_log_path = self.dataset_path / "worker_debug"
        assert (debug_log_path / "0.jsonl").is_file()
        assert (debug_log_path / "1.jsonl").is_file()
        assert (debug_log_path / "2.jsonl").is_file()

        collected_keys_order = [[None] * 10 for _ in range(2)]
        with (debug_log_path / "0.jsonl").open() as rf:
            for line in rf:
                line_data = json.loads(line)
                if line_data["t"] == "SavableDataLoader.yield":
                    print(line_data)
                    for i in range(len(collected_keys_order)):
                        if collected_keys_order[i][line_data["idx"]] is None:
                            collected_keys_order[i][line_data["idx"]] = line_data["keys"]
                            break
                    else:
                        assert False, "Too many entries for key"

        print(collected_keys_order)
        assert collected_keys_order == samples

        runner = CliRunner()
        result = runner.invoke(
            analyze_debug_command,
            [
                str(debug_log_path),
                "--include-modality",
                "train,val",
                "--heatmap-path",
                str(self.dataset_path / "heatmap.png"),
            ],
            catch_exceptions=False,
        )
        print(result.stdout)
        assert result.exit_code == 0, "Debug analysis failed, see output"
        assert "Analyzing 3 logs" in result.stdout
        assert "Found 50 unique sample keys, 20 steps" in result.stdout

    def test_validate_captioning_dataset(self):
        runner = CliRunner()
        result = runner.invoke(
            lint_command,
            [str(self.dataset_path), "--split-parts=train"],
            catch_exceptions=False,
        )
        assert result.exit_code == 0, "Validation failed, see output"

    def test_prepare_dataset(self):
        runner = CliRunner()
        result = runner.invoke(
            prepare_command,
            [str(self.dataset_path)],
            catch_exceptions=False,
            input="y\n1,0,0\ny\n0\nY\npng\ntxt\n",
        )
        assert result.exit_code == 0, "Prepare failed, see output"
        assert "Done" in result.stdout, "Prepare failed, see output"


if __name__ == "__main__":
    unittest.main()
