# Copyright (c) 2024, NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause

"""This module defines tests for the dataset."""

import gc
import logging
import sys
import tempfile
import unittest
import warnings
from collections import Counter
from pathlib import Path

import torch
import webdataset as wds

from megatron.energon import TextSample, WorkerConfig, get_loader, get_train_dataset
from megatron.energon.dataset_config import MAIN_FOLDER_NAME, get_dataset_from_config
from megatron.energon.loader import get_savable_loader


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

        # Create a small dummy captioning dataset
        self.create_text_test_dataset(self.dataset_path)
        print(self.dataset_path)

    def tearDown(self):
        # Remove all temporary files
        self.temp_dir.cleanup()

    @staticmethod
    def create_text_test_dataset(path: Path):
        """Creates a small dummy test dataset for testing purposes."""

        # Create num_samples unique captions
        (path / "parts").mkdir(exist_ok=True, parents=True)

        # Initialize the ShardWriter
        with wds.ShardWriter(f"{path}/parts/data-%d.tar", maxcount=100) as shard_writer:
            for idx in range(55):
                # Write individual files to shards
                shard_writer.write(
                    {
                        "__key__": f"{idx:06d}",
                        "txt": f"{idx}".encode(),
                    },
                )
                # Also create smaller shards, to verify distributions
                if idx in (1, 3, 6, 10, 20, 30, 40, 50):
                    shard_writer.next_stream()
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
                    ]
                )
            )

    def test_text_dataset(self):
        worker_config = WorkerConfig(rank=0, world_size=1, num_workers=0)

        ds = get_dataset_from_config(
            self.dataset_path,
            split_part="train",
            training=False,
            sample_type=TextSample,
            worker_config=worker_config,
        )

        # Check len operator
        assert len(ds) == 55
        # Check if iterating returns the same
        iter1 = list(get_loader(ds, worker_config=worker_config))
        iter2 = list(get_loader(ds, worker_config=worker_config))
        assert len(iter1) == 55
        assert len(iter2) == 55
        assert all(elem1.__key__ == elem2.__key__ for elem1, elem2 in zip(iter1, iter2))
        assert all(
            f"{idx}" == x.text for idx, x in enumerate(get_loader(ds, worker_config=worker_config))
        )

        del ds
        gc.collect()

    def test_epoch(self):
        torch.manual_seed(42)

        worker_config = WorkerConfig(rank=0, world_size=1, num_workers=5)

        # Without shuffle buffer, should yield everything exactly once
        ds3 = get_dataset_from_config(
            self.dataset_path,
            split_part="train",
            training=True,
            sample_type=TextSample,
            worker_config=worker_config,
        )
        loader5 = get_loader(ds3, worker_config=worker_config)
        order9 = [data.text for idx, data in zip(range(55), loader5)]
        print(order9)
        print(Counter(order9))
        assert all(v == 1 for v in Counter(order9).values())

    def test_determinism(self):
        worker_config2 = WorkerConfig(rank=0, world_size=1, num_workers=2)
        worker_config2b = WorkerConfig(rank=0, world_size=1, num_workers=2, seed_offset=43)
        worker_config4 = WorkerConfig(rank=0, world_size=1, num_workers=4)

        # This seed is used by the dataset to shuffle the data
        torch.manual_seed(42)
        ds1 = get_train_dataset(
            self.dataset_path,
            split_part="train",
            sample_type=TextSample,
            worker_config=worker_config2,
            batch_size=1,
            shuffle_buffer_size=42,
            max_samples_per_sequence=2,
        )
        ds1b = get_train_dataset(  # Same but different seed
            self.dataset_path,
            split_part="train",
            sample_type=TextSample,
            worker_config=worker_config2b,
            batch_size=1,
            shuffle_buffer_size=42,
            max_samples_per_sequence=2,
        )
        ds2 = get_train_dataset(
            self.dataset_path,
            split_part="train",
            sample_type=TextSample,
            worker_config=worker_config2,
            batch_size=1,
            shuffle_buffer_size=42,
            max_samples_per_sequence=2,
        )
        ds3 = get_train_dataset(
            self.dataset_path,
            split_part="train",
            sample_type=TextSample,
            worker_config=worker_config4,
            batch_size=1,
            shuffle_buffer_size=42,
            max_samples_per_sequence=2,
        )

        # Fork the dataset twice
        loader1 = get_loader(ds1, worker_config=worker_config2)
        loader2 = get_loader(ds1, worker_config=worker_config2)

        order4 = [data.text[0] for idx, data in zip(range(55 * 20), loader1)]
        order5 = [data.text[0] for idx, data in zip(range(55 * 20), loader1)]
        order6 = [data.text[0] for idx, data in zip(range(55 * 20), loader2)]
        print(order4)
        print(Counter(order4))
        # +-1 is possible due to the random shuffling (actually +-2 is possible)
        assert all(17 <= v <= 22 for v in Counter(order4).values())

        assert order4 != order5
        assert order4 == order6

        loader3 = get_loader(ds1b, worker_config=worker_config2b)
        order7 = [data.text[0] for idx, data in zip(range(55 * 20), loader3)]
        assert order6 != order7

        loader4 = get_loader(ds3, worker_config=worker_config4)
        order8 = [data.text[0] for idx, data in zip(range(55 * 100), loader4)]
        assert order6 != order8[: len(order6)]
        print(Counter(order8))
        assert all(90 <= v <= 110 for v in Counter(order8).values())

        # Delete all locals, otherwise loaders might be kept alive
        locals().clear()
        gc.collect()

    def test_restore_state(self):
        worker_config = WorkerConfig(rank=0, world_size=1, num_workers=0)

        count1 = 55 * 20
        count2 = 55 * 20
        sbs = 42
        # count1 = 4
        # count2 = 2
        # sbs = None
        psi = None

        # This seed is used by the dataset to shuffle the data
        torch.manual_seed(42)

        loader = get_savable_loader(
            get_train_dataset(
                self.dataset_path,
                split_part="train",
                sample_type=TextSample,
                worker_config=worker_config,
                batch_size=1,
                shuffle_buffer_size=sbs,
                max_samples_per_sequence=2,
                parallel_shard_iters=psi,
            ),
            worker_config=worker_config,
        )

        # print("save state")
        state_0 = loader.save_state()
        # print("save state done")
        order_1 = [data.text[0] for idx, data in zip(range(count1), loader)]
        assert len(order_1) == count1
        # print("save state")
        state_1 = loader.save_state()
        # print("save state done")
        order_2 = [data.text[0] for idx, data in zip(range(count2), loader)]
        assert len(order_2) == count2

        print("state0", state_0)
        print("state1", state_1)

        torch.manual_seed(42)
        loader = get_savable_loader(
            get_train_dataset(
                self.dataset_path,
                split_part="train",
                sample_type=TextSample,
                worker_config=worker_config,
                batch_size=1,
                shuffle_buffer_size=sbs,
                max_samples_per_sequence=2,
                parallel_shard_iters=psi,
            ),
            worker_config=worker_config,
        )
        loader.restore_state(state_0)
        order_45 = [data.text[0] for idx, data in zip(range(count1 + count2), loader)]
        order_4 = order_45[:count1]
        order_5 = order_45[count1:]
        # print("order1", order_1)
        # print("order2", order_2)
        # print("order4", order_4)
        assert order_1 == order_4
        # print("order5", order_5)
        assert order_2 == order_5

        torch.manual_seed(42)
        loader = get_savable_loader(
            get_train_dataset(
                self.dataset_path,
                split_part="train",
                sample_type=TextSample,
                worker_config=worker_config,
                batch_size=1,
                shuffle_buffer_size=sbs,
                max_samples_per_sequence=2,
                parallel_shard_iters=psi,
            ),
            worker_config=worker_config,
        )
        # print("restore state")
        loader.restore_state(state_1)
        # print("restore state done")
        order_3 = [data.text[0] for idx, data in zip(range(count2), loader)]
        # print("order1", order_1)
        # print("order2", order_2[:100])
        # print("order3", order_3[:100])
        assert order_2 == order_3

    def test_restore_state_workers(self):
        worker_config = WorkerConfig(rank=0, world_size=1, num_workers=2)

        psi = 2
        sbs = 42
        n1 = 18
        n2 = 109
        n3 = 28
        ces = 0

        # This seed is used by the dataset to shuffle the data
        torch.manual_seed(42)
        ds = get_train_dataset(
            self.dataset_path,
            split_part="train",
            sample_type=TextSample,
            worker_config=worker_config,
            batch_size=1,
            shuffle_buffer_size=sbs,
            max_samples_per_sequence=2,
            parallel_shard_iters=psi,
        )
        loader = get_savable_loader(ds, worker_config=worker_config, checkpoint_every_sec=ces)

        # print("save state")
        state_0 = loader.save_state()
        it1 = iter(loader)
        # print("save state done")
        order_1 = [data.text[0] for idx, data in zip(range(n1), it1)]
        # print("save state")
        # time.sleep(0.5)
        state_1 = loader.save_state()
        # print("save state done")
        order_2 = [data.text[0] for idx, data in zip(range(n2), it1)]
        state_2 = loader.save_state()
        order_3 = [data.text[0] for idx, data in zip(range(n3), it1)]

        print("order_1", order_1)
        print("order_2", order_2)
        print("order_3", order_3)

        # print("state0", state_0)
        print("state1", state_1)
        print("state2", state_2)

        # Restoring the state of a new dataset should also yield the same
        torch.manual_seed(42)
        ds = get_train_dataset(
            self.dataset_path,
            split_part="train",
            sample_type=TextSample,
            worker_config=worker_config,
            batch_size=1,
            shuffle_buffer_size=sbs,
            max_samples_per_sequence=2,
            parallel_shard_iters=psi,
        )
        loader = get_savable_loader(ds, worker_config=worker_config)
        loader.restore_state(state_0)
        order_6 = [data.text[0] for idx, data in zip(range(n1), loader)]
        print("order1", order_1)
        print("order6", order_6)
        assert order_6 == order_1

        # Restoring the state of a new dataset should also yield the same
        torch.manual_seed(42)
        ds = get_train_dataset(
            self.dataset_path,
            split_part="train",
            sample_type=TextSample,
            worker_config=worker_config,
            batch_size=1,
            shuffle_buffer_size=sbs,
            max_samples_per_sequence=2,
            parallel_shard_iters=psi,
        )
        loader = get_savable_loader(ds, worker_config=worker_config)
        loader.restore_state(state_1)
        order_7 = [data.text[0] for idx, data in zip(range(n2), loader)]
        print("order2", order_2[:100])
        print("order7", order_7[:100])
        assert order_7 == order_2

        # Restoring the state of a new dataset should also yield the same
        torch.manual_seed(42)
        ds = get_train_dataset(
            self.dataset_path,
            split_part="train",
            sample_type=TextSample,
            worker_config=worker_config,
            batch_size=1,
            max_samples_per_sequence=2,
            shuffle_buffer_size=sbs,
            parallel_shard_iters=psi,
        )
        loader = get_savable_loader(ds, worker_config=worker_config)
        loader.restore_state(state_2)
        order_8 = [data.text[0] for idx, data in zip(range(n3), loader)]
        print("order3", order_3)
        print("order8", order_8)
        assert order_8 == order_3

    def test_invariance_global_samples(self):
        # We'd like to ensure that the user can keep the same global batches
        # (deterministic pseudo random order) when changing the number of ranks (world size).

        # This can be achieved by obeying a few constraints:
        # - Global batch size must stay the same across runs
        # - Global batch size must be a multiple of (micro-batch size * world_size * num_workers)
        #   - Global batch size = micro-batch size * world_size * num_workers * gradient_accum_steps
        # - world_size * num_workers must stay the same across runs
        # Set the same torch.manual_seed(...) on each rank before constructing the dataset and the data loader

        scenarios = [
            dict(
                configs=(WorkerConfig(rank=0, world_size=1, num_workers=4),),
                micro_batch_size=2,
                global_batch_size=8,
            ),
            dict(
                configs=(
                    WorkerConfig(rank=0, world_size=2, num_workers=2),
                    WorkerConfig(rank=1, world_size=2, num_workers=2),
                ),
                micro_batch_size=2,
                global_batch_size=8,
            ),
            dict(
                configs=(
                    WorkerConfig(rank=0, world_size=4, num_workers=1),
                    WorkerConfig(rank=1, world_size=4, num_workers=1),
                    WorkerConfig(rank=2, world_size=4, num_workers=1),
                    WorkerConfig(rank=3, world_size=4, num_workers=1),
                ),
                micro_batch_size=2,
                global_batch_size=8,
            ),
            dict(
                configs=(
                    WorkerConfig(rank=0, world_size=2, num_workers=2),
                    WorkerConfig(rank=1, world_size=2, num_workers=2),
                ),
                micro_batch_size=1,  # Micro-batch 1, more accum
                global_batch_size=8,
            ),
        ]

        # Constraints to user:

        global_batches_per_scenario = []
        for scenario in scenarios:
            assert (
                scenario["global_batch_size"] % scenario["micro_batch_size"] == 0
            ), "Global batch size must be a multiple of the micro-batch size."

            world_size = len(scenario["configs"])
            gradient_accum_steps = scenario["global_batch_size"] // (
                scenario["micro_batch_size"] * world_size
            )

            batches_per_rank = []

            for rank_config in scenario["configs"]:
                torch.manual_seed(42)
                ds = get_train_dataset(
                    self.dataset_path,
                    split_part="train",
                    sample_type=TextSample,
                    worker_config=rank_config,
                    batch_size=scenario["micro_batch_size"],
                    shuffle_buffer_size=42,
                    max_samples_per_sequence=2,
                )
                loader = get_loader(ds, worker_config=rank_config)

                micro_batches = [
                    data.text
                    for idx, data in zip(
                        range(55 * 8 // (world_size * scenario["micro_batch_size"])), loader
                    )
                ]
                batches_per_rank.append(micro_batches)

            # Compose global batches
            global_batches_cur_rank = []
            batch_index = 0
            while batch_index < len(batches_per_rank[0]):
                global_batch = []
                for _ in range(gradient_accum_steps):
                    for rank_batches in batches_per_rank:
                        global_batch.extend(rank_batches[batch_index])
                    batch_index += 1
                    if batch_index >= len(batches_per_rank[0]):
                        # last global batch may be smaller
                        break
                global_batches_cur_rank.append(sorted(global_batch))

            global_batches_per_scenario.append(global_batches_cur_rank)

        # Check that the global batches are the same

        # Assert that all scenarios produced the same number of global batches
        assert all(
            len(global_batches) == len(global_batches_per_scenario[0])
            for global_batches in global_batches_per_scenario
        ), "Number of global batches per scenario does not match."

        for global_batches in global_batches_per_scenario:
            print("= Global batches per scenario")
            for global_batch in global_batches:
                print("  Global batch: ", global_batch)

        # Assert that all global batches are the same
        for i in range(len(global_batches_per_scenario[0])):
            for scenerio_idx, global_batches in enumerate(global_batches_per_scenario):
                assert (
                    global_batches[i] == global_batches_per_scenario[0][i]
                ), f"Global batch {i} of scenario {scenerio_idx} does not match."

        # Delete all locals, otherwise loaders might be kept alive
        locals().clear()
        gc.collect()


if __name__ == "__main__":
    unittest.main()
