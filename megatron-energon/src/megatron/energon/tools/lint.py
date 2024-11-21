# Copyright (c) 2024, NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause

import logging
import traceback
from typing import Optional

import click
import tqdm
import yaml

from megatron.energon import WorkerConfig, get_loader, get_val_dataset
from megatron.energon.epathlib import EPath
from megatron.energon.flavors import BaseWebdataset
from megatron.energon.flavors.webdataset.itar import TarIndexReader


def check_index(dataset_path: EPath):
    if not dataset_path.is_dir():
        click.echo(f"{dataset_path} is not a directory, therefore the index will not be checked")
        return

    ok = True

    # Get info file
    info_file = dataset_path / ".nv-meta/.info.yaml"
    info = yaml.safe_load(info_file.read_text())

    click.echo("Checking the index files...")
    shards = info["shard_counts"]
    for shard_file, length in shards.items():
        with TarIndexReader(dataset_path / shard_file) as itar:
            l = len(itar)
            if l - 1 != length:
                ok = False
                print(
                    f"Error in shard {shard_file}: Shard length in Info file {length} != {l - 1} (length in index)"
                )

    return ok


@click.command(name="lint")
@click.argument(
    "path",
    type=click.Path(path_type=EPath),
)
@click.option(
    "--split-parts", default="train,val,test", help="The splits to verify", show_default=True
)
@click.option(
    "--dataset-config", default="dataset.yaml", help="Dataset config file name", show_default=True
)
@click.option(
    "--split-config", default="split.yaml", help="Split config file name", show_default=True
)
@click.option(
    "--parallel", default=1, help="Number of parallel workers", show_default=True, type=int
)
def command(path: EPath, split_parts: str, dataset_config: str, split_config: str, parallel: int):
    """Check energon dataset for errors.

    The PATH should point to the folder with the dataset.
    The dataset must comply with the energon dataset format. See README.md for more details."""

    path = path.absolute()

    # Check the tar file index
    if not check_index(path):
        raise click.ClickException("Validation failed with errors, see logs for details.")

    # Check the dataset
    failed = False

    ignore_list = []

    def handler(exc: Exception, key: Optional[str] = None) -> None:
        nonlocal failed
        failed = True
        logging.exception(str(exc))
        if key is not None:
            ignore_list.append(key)

    kwargs = {}
    if dataset_config != "dataset.yaml":
        kwargs["dataset_config"] = dataset_config
    if split_config != "split.yaml":
        kwargs["split_config"] = split_config

    worker_config = WorkerConfig(rank=0, world_size=1, num_workers=parallel)

    for split_part in split_parts.split(","):
        try:
            dataset = get_val_dataset(
                EPath(path),
                split_part=split_part,
                worker_config=worker_config,
                batch_size=1,
                handler=handler,
                **kwargs,
            )
        except BaseWebdataset.EmptyDatasetError:
            click.echo(f"Skipping empty split part {split_part}")
            continue

        try:
            for _ in tqdm.tqdm(get_loader(dataset, worker_config=worker_config)):
                pass
        except InterruptedError:
            raise
        except BaseException:
            traceback.print_exc()
            raise click.ClickException("Validation failed with errors, see logs for details.")

    if failed:
        click.echo(
            "The following shards/samples failed (maybe set as dataset.yaml:ignore_list):", err=True
        )
        for item in ignore_list:
            click.echo(f"- {item}", err=True)
        raise click.ClickException("Validation failed with errors, see logs for details.")


if __name__ == "__main__":
    command()
