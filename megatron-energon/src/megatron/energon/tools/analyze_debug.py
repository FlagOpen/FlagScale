# Copyright (c) 2024, NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause

import json
import time
import traceback
from concurrent.futures.process import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import (
    Container,
    Dict,
    Generator,
    Iterable,
    List,
    Literal,
    Optional,
    Tuple,
    TypedDict,
    Union,
)

import click
import numpy as np
from PIL import Image
from tqdm import tqdm

cpal = np.array(
    [
        [int(x) for x in line.split(" ")]
        for line in """255 255 255
1 0 103
213 255 0
255 0 86
158 0 142
14 76 161
255 229 2
0 95 57
0 255 0
149 0 58
255 147 126
164 36 0
0 21 68
145 208 203
98 14 0
107 104 130
0 0 255
0 125 181
106 130 108
0 174 126
194 140 159
190 153 112
0 143 156
95 173 78
255 0 0
255 0 246
255 2 157
104 61 59
255 116 163
150 138 232
152 255 82
167 87 64
1 255 254
255 238 232
254 137 0
189 198 255
1 208 255
187 136 0
117 68 177
165 255 210
255 166 254
119 77 0
122 71 130
38 52 0
0 71 84
67 0 44
181 0 255
255 177 103
255 219 102
144 251 146
126 45 210
189 211 147
229 111 254
222 255 116
0 255 120
0 155 255
0 100 1
0 118 255
133 169 0
0 185 23
120 130 49
0 255 198
255 110 65
232 94 190""".split(
            "\n"
        )
    ],
    dtype=np.int32,
)


class YieldBatchLogLine(TypedDict):
    # Json example:
    # {
    #   "t": "yield_batch",
    #   "r": 1,
    #   "w": 1,
    #   "m": "train",
    #   "idx": 1,
    #   "keys": ["parts/data-train-000051.tar/528866", ...],
    # }
    t: Literal["yield_batch"]
    r: int
    w: int
    m: Literal["train", "val"]
    idx: int
    keys: List[str]


class SampleLoaderYieldLogLine(TypedDict):
    # Json example:
    # {
    #   "t": "WebdatasetSampleLoaderDataset._shards_iter.yield",
    #   "r": 1,
    #   "w": 1,
    #   "key": "parts/data-train-000051.tar/528866",
    #   "shard": "parts/data-train-000051.tar",
    #   "count": 633,
    #   "epoch": 0,
    #   "epoch_count": 633
    # }
    t: Literal["WebdatasetSampleLoaderDataset._shards_iter.yield"]
    r: int
    w: int
    key: str
    shard: str
    count: int
    epoch: int
    epoch_count: int


class AutosizingHeatmapWriter:
    """Writes a heatmap, automatically resizing it if necessary."""

    def __init__(self, heatmap_samples: int, heatmap_steps: int, colorize: bool = True):
        self.heatmap = np.zeros((heatmap_samples, heatmap_steps, 3), dtype=np.int32)
        self.heatmap_sample_factor = 1
        self.heatmap_step_factor = 1

        self.heatmap_sample_max = -1
        self.heatmap_step_max = -1

        self.colors_size = cpal.shape[0] if colorize else 1

    def add(self, sample_id: int, step: int, src: int) -> None:
        """
        Add a point to the heatmap (i.e. increase count at that position).

        Args:
            sample_id: The sample id (y-axis)
            step: The step (x-axis)
        """
        # Resize heatmap?
        while self.heatmap.shape[0] * self.heatmap_sample_factor <= sample_id:
            self.heatmap[: self.heatmap.shape[0] // 2] = self.heatmap[::2] + self.heatmap[1::2]
            self.heatmap[self.heatmap.shape[0] // 2 :] = 0
            self.heatmap_sample_factor *= 2
            self.heatmap_sample_max = 0
        while self.heatmap.shape[1] * self.heatmap_step_factor <= step:
            self.heatmap[:, : self.heatmap.shape[1] // 2] = (
                self.heatmap[:, ::2] + self.heatmap[:, 1::2]
            )
            self.heatmap[:, self.heatmap.shape[1] // 2 :] = 0
            self.heatmap_step_factor *= 2
            self.heatmap_step_max = 0
        # Save point
        step //= self.heatmap_step_factor
        sample_id //= self.heatmap_sample_factor
        self.heatmap[sample_id, step] += cpal[src % self.colors_size]
        self.heatmap_step_max = max(self.heatmap_step_max, step)
        self.heatmap_sample_max = max(self.heatmap_sample_max, sample_id)

    def save(self, path: Union[Path, str], gain: float):
        """
        Save the heatmap to the given path.

        Args:
            path: The path to save the heatmap to.
            gain: The gain (=multiplication factor) for the heatmap.

        Returns:
            The maximum sample id and step id that were used in the heatmap.
        """
        heatmap = self.heatmap[: self.heatmap_sample_max + 1, : self.heatmap_step_max + 1]

        heatmap = heatmap.astype(np.float32)
        heatmap = np.clip(heatmap * gain / heatmap.max((0, 1)) * 255, 0, 255).astype(np.uint8)

        Image.fromarray(heatmap).save(path)
        return (
            self.heatmap_sample_max * self.heatmap_sample_factor,
            self.heatmap_step_max * self.heatmap_step_factor,
        )


@click.command(name="analyze-debug")
@click.argument(
    "log_paths",
    nargs=-1,
    type=click.Path(exists=True, file_okay=True, dir_okay=True, path_type=Path),
)
@click.option(
    "--heatmap-path",
    type=click.Path(exists=False, writable=True, dir_okay=False, path_type=Path),
    default=Path("heatmap.png"),
)
@click.option(
    "--heatmap-steps",
    type=int,
    default=1000,
    help="Size of the heatmap in step direction. All steps will be downscaled to this size.",
)
@click.option(
    "--heatmap-samples",
    type=int,
    default=1000,
    help="Size of the heatmap in sample direction. All samples will be downscaled to this size.",
)
@click.option(
    "--heatmap-gain",
    type=float,
    default=10,
    help="Gain (=multiplication factor) for the heatmap",
)
@click.option(
    "--force-loading-order",
    is_flag=True,
    default=False,
    help="If true, force using the dataloader loading order instead of batch data",
)
@click.option(
    "--include-modality",
    type=str,
    default="train",
    help="Choose which modality/modalities (train,val) to include. Comma separate for multiple.",
)
@click.option(
    "--skip",
    type=int,
    default=0,
    help="If >0, skip this many steps at the beginning of log file parsing.",
)
@click.option(
    "--no-colors",
    is_flag=True,
    default=False,
    help="If set, disable colorizing ranks.",
)
def command(
    log_paths: List[Path],
    heatmap_path: Path,
    heatmap_steps: int,
    heatmap_samples: int,
    heatmap_gain: float,
    force_loading_order: bool,
    include_modality: str,
    skip: int,
    no_colors: bool,
):
    """Check energon dataset for errors.

    The LOG_PATH should point to the folder with the debug log, or to a single log file."""

    if len(log_paths) == 0:
        raise click.ClickException("No log paths specified")
    log_files = []
    for log_path in log_paths:
        if log_path.is_dir():
            log_files.extend(sorted(log_path.glob("*.jsonl")))
        elif log_path.is_file():
            log_files.append(log_path)
        else:
            raise click.ClickException(f"Invalid log path: {log_path}")

    if len(log_files) == 0:
        raise click.ClickException("No log files found")

    heatmap = AutosizingHeatmapWriter(heatmap_samples, heatmap_steps, colorize=not no_colors)

    print(f"Analyzing {len(log_files)} logs...")

    modalities = [m.strip() for m in include_modality.split(",")]

    key_index = {}
    count = 0
    if not force_loading_order:
        loaders = [LoaderLogIter(log_file, start_idx=skip) for log_file in log_files]
        loaders_by_id: Dict[int, Tuple[LoaderInfo, List[LoaderLogIter]]] = {}
        with ProcessPoolExecutor(max_workers=16) as executor:
            for loader, loader_info in tqdm(
                executor.map(_proc_map_loader, loaders), total=len(loaders)
            ):
                for loader_id, loader_info in loader_info.items():
                    if loader_id in loaders_by_id:
                        existing_loader_info, existing_loaders = loaders_by_id[loader_id]
                        assert (
                            existing_loader_info.modality == loader_info.modality
                            and existing_loader_info.path == loader_info.path
                        ), f"Found multiple loaders for {loader_id}: {existing_loader_info.modality, existing_loader_info.path} and {loader_info.modality, loader_info.path}"
                        existing_loader_info.global_count = max(
                            existing_loader_info.global_count, loader_info.global_count
                        )
                        existing_loaders.append(loader)
                    else:
                        loaders_by_id[loader_id] = (loader_info, [loader])
        print("Available loaders:")
        selected_loader_id = None
        must_select = False
        for loader_id, (loader_info, _iters) in loaders_by_id.items():
            print(
                f"  {loader_id}: {loader_info.modality} {loader_info.path} {loader_info.global_count} steps"
            )
            if loader_info.modality in modalities:
                if selected_loader_id is None:
                    selected_loader_id = loader_id
                else:
                    # Have multiple loaders
                    must_select = True
        if must_select:
            while True:
                loader_id_str = input("Choose loader id: ")
                try:
                    selected_loader_id = int(loader_id_str)
                except ValueError:
                    print(f"Invalid loader id {loader_id_str} 1")
                    continue
                if selected_loader_id in loaders_by_id:
                    break
                print(f"Invalid loader id {selected_loader_id}")
        assert selected_loader_id is not None
        selected_loader_info, selected_loader_readers = loaders_by_id[selected_loader_id]
        print(
            f"Reading for loader {selected_loader_id}: {selected_loader_info.modality} {selected_loader_info.path}"
        )
        log_iters = [
            (idx, loader.log_entries(loader_ids={selected_loader_id}))
            for idx, loader in enumerate(selected_loader_readers)
        ]
        with tqdm(total=selected_loader_info.global_count) as pbar:
            while len(log_iters) > 0:
                cur_count = 0
                # Iterate over all iterators for this count and put into heatmap
                for src_idx, log_iter in tuple(log_iters):
                    # Iterate until None (=next count) is encountered
                    while True:
                        try:
                            log_keys = next(log_iter)
                        except StopIteration:
                            log_iters.remove((src_idx, log_iter))
                            break
                        except OSError:
                            traceback.print_exc()
                            log_iters.remove((src_idx, log_iter))
                            break
                        else:
                            if log_keys is None:
                                break
                            for log_key in log_keys:
                                key_id = key_index.setdefault(log_key, len(key_index))
                                heatmap.add(key_id, count, src_idx)
                                cur_count += 1
                if cur_count == 0:
                    print(f"No data for step {count}")
                count += 1
                pbar.update(1)

    if len(key_index) == 0:
        if force_loading_order:
            print("Forcing to use sample loader logs")
        else:
            print("No batch information in logs, trying sample loader logs...")
        if modalities != {"train", "val"}:
            print("  Data includes all modalities (train and val)")
        print(
            "  Shuffle buffer and batching will not be considered, only the loading order from disk"
        )
        log_iters = [
            _iter_sl_log_line_keys(_iter_sl_log_samples(log_file), start_idx=skip)
            for log_file in log_files
        ]
        key_index = {}
        count = 0
        start = time.time()
        while len(log_iters) > 0:
            cur_count = 0
            # Iterate over all iterators for this count and put into heatmap
            for log_iter in tuple(log_iters):
                # Iterate until None (=next count) is encountered
                while True:
                    try:
                        log_key = next(log_iter)
                    except StopIteration:
                        log_iters.remove(log_iter)
                        break
                    except OSError:
                        traceback.print_exc()
                        log_iters.remove(log_iter)
                        break
                    else:
                        if log_key is None:
                            break
                        key_id = key_index.setdefault(log_key, len(key_index))
                        heatmap.add(key_id, count)
                        cur_count += 1
            if cur_count == 0:
                print(f"No data for step {count}")
            if time.time() - start > 10:
                print(f"  Step {count}")
                start = time.time()
            count += 1

    if count == 0:
        raise click.ClickException("No data found in logs")

    print(f"Found {len(key_index)} unique sample keys, {count} steps")

    # print(f"Heatmap factors: {heatmap_sample_factor} samples, {heatmap_step_factor} steps")
    # print(f"Heatmap max: {heatmap_sample_max} samples, {heatmap_step_max} steps")
    n_samples, n_steps = heatmap.save(heatmap_path, heatmap_gain)
    print(f"Wrote heatmap to {heatmap_path}")
    print("Heatmap axes:")
    print(f"  x-axis: {n_steps} worker steps")
    print(f"  y-axis: {n_samples} samples")


class LoaderInitLogLine(TypedDict):
    t: Literal["SavableLoader.__init__", "BasicDataLoader.__init__"]
    r: int
    w: None
    id: int
    config: dict


class LoaderIterLogLine(TypedDict):
    t: Literal["SavableDataLoader.iter", "BasicDataLoader.iter"]
    r: int
    w: None
    id: int
    iter_id: int


class LoaderYieldLogLine(TypedDict):
    t: Literal["SavableDataLoader.yield", "BasicDataLoader.yield"]
    r: int
    w: None
    id: int
    iter_id: int
    worker_id: int
    worker_idx: int
    idx: int
    iter_idx: int
    global_idx: int
    keys: Optional[List[str]]


class LoaderStopLogLine(TypedDict):
    t: Literal["SavableDataLoader.StopIteration", "BasicDataLoader.StopIteration"]
    r: int
    w: None
    id: int
    iter_id: int


LoaderLines = Union[
    LoaderInitLogLine,
    LoaderIterLogLine,
    LoaderYieldLogLine,
    LoaderStopLogLine,
]

LOADER_LOG_LINE_TYPES_T = (
    "SavableLoader.__init__",
    "BasicDataLoader.__init__",
    "SavableDataLoader.iter",
    "BasicDataLoader.iter",
    "SavableDataLoader.yield",
    "BasicDataLoader.yield",
    "SavableDataLoader.StopIteration",
    "BasicDataLoader.StopIteration",
)


@dataclass
class LoaderInfo:
    id: int
    modality: str
    path: str
    global_count: int


class LoaderLogIter:
    def __init__(self, path: Path, start_idx: int = 0):
        self._path = path
        self._start_idx = start_idx

    def _iter_log_lines(self, which: Iterable[str]) -> Generator[LoaderLines, None, None]:
        try:
            with self._path.open("r") as rf:
                for line in rf:
                    if any(f'"t": "{t}"' in line for t in which):
                        try:
                            yield json.loads(line.strip())
                        except json.JSONDecodeError:
                            print("Cannot decode line", repr(line))
        except IOError as e:
            print(f"Ignoring IOError: {e} for {self._path}")

    @staticmethod
    def _find_config_modality(config: dict) -> Literal["train", "val"]:
        assert isinstance(config, dict)
        if "training" in config:
            return "train" if config["training"] else "val"
        elif "dataset" in config:
            return LoaderLogIter._find_config_modality(config["dataset"])
        elif "dataset_weights" in config:
            return LoaderLogIter._find_config_modality(config["dataset_weights"][0][0])
        elif "datasets" in config:
            return LoaderLogIter._find_config_modality(config["datasets"][0])
        assert False, f"Unrecognized config {config}"

    @staticmethod
    def _find_config_path(config: dict) -> str:
        assert isinstance(config, dict)
        if "path" in config:
            return config["path"]
        elif "dataset" in config:
            return LoaderLogIter._find_config_path(config["dataset"])
        elif "dataset_weights" in config:
            return LoaderLogIter._find_config_path(config["dataset_weights"][0][0])
        elif "datasets" in config:
            return LoaderLogIter._find_config_path(config["datasets"][0])
        assert False, f"Unrecognized config {config}"

    def loaders(self) -> Dict[int, LoaderInfo]:
        loaders = {}
        for log_line in self._iter_log_lines(
            (
                "SavableLoader.__init__",
                "BasicDataLoader.__init__",
                "SavableDataLoader.yield",
                "BasicDataLoader.yield",
            )
        ):
            if log_line["t"] in ("SavableLoader.__init__", "BasicDataLoader.__init__"):
                loaders[log_line["id"]] = LoaderInfo(
                    id=log_line["id"],
                    modality=self._find_config_modality(log_line["config"]),
                    path=self._find_config_path(log_line["config"]),
                    global_count=0,
                )
            elif log_line["t"] in ("SavableDataLoader.yield", "BasicDataLoader.yield"):
                loaders[log_line["id"]].global_count = log_line["global_idx"]
        return loaders

    def log_entries(self, loader_ids: Container[int]) -> Generator[Optional[List[str]], None, None]:
        idx = self._start_idx
        for log_line in self._iter_log_lines(("SavableDataLoader.yield", "BasicDataLoader.yield")):
            if (
                log_line["t"] in ("SavableDataLoader.yield", "BasicDataLoader.yield")
                and log_line["id"] in loader_ids
            ):
                assert (
                    log_line["global_idx"] >= idx
                ), f"Found entry {log_line} with wrong idx <{idx}"
                while log_line["global_idx"] != idx:
                    yield None
                    idx += 1
                if "keys" in log_line:
                    yield log_line["keys"]

    def __repr__(self) -> str:
        return f"log({str(self._path)})"


def _proc_map_loader(loader: LoaderLogIter) -> Tuple[LoaderLogIter, Dict[int, LoaderInfo]]:
    return (loader, loader.loaders())


def _iter_sl_log_line_keys(
    log_lines: Iterable[SampleLoaderYieldLogLine],
    start_idx: int = 0,
) -> Generator[Optional[str], None, None]:
    count = start_idx
    for log_line in log_lines:
        if log_line["count"] < start_idx:
            continue
        assert log_line["count"] >= count
        while log_line["count"] != count:
            yield None
            count += 1
        yield log_line["key"]


def _iter_sl_log_samples(path: Path) -> Generator[SampleLoaderYieldLogLine, None, None]:
    with path.open("r") as rf:
        for line in rf:
            if '"t": "WebdatasetSampleLoaderDataset._shards_iter.yield"' in line:
                try:
                    yield json.loads(line.strip())
                except json.JSONDecodeError:
                    print("Cannot decode line", repr(line))


if __name__ == "__main__":
    command()
