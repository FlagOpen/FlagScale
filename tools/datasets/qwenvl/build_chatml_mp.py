import json
import os
import pickle

from argparse import ArgumentParser
from multiprocessing import Pool, cpu_count

import cv2
import webdataset as wds
import yaml

from tqdm import tqdm
from webdataset.writer import add_handlers, default_handlers, imageencoder

from megatron.energon.epathlib import EPath
from megatron.energon.flavors import BaseWebdatasetFactory


def process_entry_mp(args):
    entry, idx, dataset_dir, image_key, video_key, sort_function = args
    image_datas = []
    video_datas = []
    second_per_grid_ts = []

    image_paths = entry.get(image_key, [])
    if isinstance(image_paths, str):
        image_paths = [image_paths]

    for image in image_paths:
        img = cv2.imread(os.path.join(dataset_dir, image), cv2.IMREAD_UNCHANGED)
        if img is not None:
            image_datas.append(img)

    for video in entry.get(video_key, []):
        video_noext, _ = os.path.splitext(video)
        frame_folder = os.path.join(dataset_dir, video_noext)
        if os.path.exists(frame_folder + ".json"):
            with open(frame_folder + ".json", "r") as f:
                fps = float(json.load(f)["fps"])
        else:
            fps = 2.0

        frames = []
        for frame in sort_function(os.listdir(frame_folder)):
            img = cv2.imread(os.path.join(frame_folder, frame), cv2.IMREAD_UNCHANGED)
            if img is not None:
                frames.append(img)

        if len(frames) % 2 == 1:
            frames = frames[:-1]
        video_datas.append(frames)
        second_per_grid_ts.append(1 / fps)

    return {
        "__key__": entry.get("id", str(idx)),
        "jpgs": image_datas,
        "videos": video_datas,
        "json": json.dumps(
            {"conversations": entry["conversations"], "second_per_grid_ts": second_per_grid_ts}
        ).encode("utf-8"),
    }


def convert(
    dataset_dir,
    output_dir,
    json_name,
    sort_function=sorted,
    max_count=10000,
    image_key="images",
    video_key="videos",
    vision_dir=None,
):
    if vision_dir is None:
        vision_dir = dataset_dir
    json_file = os.path.join(dataset_dir, json_name)
    output = os.path.join(output_dir, "wds")
    os.makedirs(output, exist_ok=True)

    # Load dataset
    try:
        with open(json_file, "r") as f:
            data = json.load(f)
    except:
        with open(json_file, "r") as f:
            data = [json.loads(l) for l in f.readlines()]

    print(f"Loaded {len(data)} entries")

    # Setup WebDataset handlers
    add_handlers(
        default_handlers, "jpgs", lambda data: pickle.dumps([imageencoder(d, "jpg") for d in data])
    )
    add_handlers(
        default_handlers,
        "videos",
        lambda data: pickle.dumps([[imageencoder(d, "jpg") for d in video] for video in data]),
    )

    # Prepare multiprocessing arguments
    num_workers = min(cpu_count(), 32)
    task_args = [
        (entry, idx, vision_dir, image_key, video_key, sort_function)
        for idx, entry in enumerate(data)
    ]

    with Pool(num_workers) as pool, wds.ShardWriter(
        os.path.join(output, "pretrain-%d.tar"), maxcount=max_count
    ) as shard_writer:
        for sample in tqdm(pool.imap(process_entry_mp, task_args), total=len(task_args)):
            shard_writer.write(sample)

    print(f"Dataset successfully converted to wds")
    return output


def generate_configs(path: EPath, split, shuffle_tars=True, num_workers=32):
    all_tars = list(path.glob("**/*.tar")) + list(path.glob("**/*.tgz"))
    all_tars = [str(p.relative_to(path)) for p in sorted(all_tars)]
    split_parts_ratio = [("train", split[0]), ("val", split[1]), ("test", split[2])]

    BaseWebdatasetFactory.prepare_dataset(
        path,
        all_tars,
        split_parts_ratio=split_parts_ratio,
        split_parts_patterns=None,
        tar_index_only=False,
        shuffle_seed=42 if shuffle_tars else None,
        workers=num_workers,
    )

    metadata = {
        "__class__": "ChatMLWebdataset",
        "__module__": "tools.datasets.qwenvl.data.energon.chatml",
        "field_map": {"imgs": "jpgs", "videos": "videos", "conversation": "json"},
    }
    os.makedirs(os.path.join(path.url, ".nv-meta"), exist_ok=True)
    with open(os.path.join(path.url, ".nv-meta", "dataset.yaml"), "w") as f:
        yaml.safe_dump(metadata, f)


if __name__ == "__main__":
    argparser = ArgumentParser()
    argparser.add_argument("--dataset-root", required=True, type=str)
    argparser.add_argument("--output-root", required=True, type=str)
    argparser.add_argument("--vision-root", default=None, type=str)
    argparser.add_argument("--json", default="dataset.json", type=str)
    argparser.add_argument("--images-key", default="images", type=str)
    argparser.add_argument("--videos-key", default="videos", type=str)
    argparser.add_argument("--max-samples-per-tar", default=10000, type=int)
    argparser.add_argument("--train-split", default=9, type=float)
    argparser.add_argument("--val-split", default=1, type=float)
    argparser.add_argument("--test-split", default=0, type=float)
    argparser.add_argument("--shuffle-tars", action="store_true")

    args = argparser.parse_args()

    output_dir = convert(
        args.dataset_root,
        args.output_root,
        args.json,
        max_count=args.max_samples_per_tar,
        image_key=args.images_key,
        video_key=args.videos_key,
        vision_dir=args.vision_root
    )

    print(f"Generating Configurations")
    generate_configs(EPath(output_dir), [args.train_split, args.val_split, args.test_split], shuffle_tars=args.shuffle_tars)
    print(f"Configurations Generated")
