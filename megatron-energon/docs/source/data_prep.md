<!--- Copyright (c) 2024, NVIDIA CORPORATION.
SPDX-License-Identifier: BSD-3-Clause -->

# Data Preparation

The aim of data preparation is to convert your data to a format that the energon loader can understand and iterate.
The outcome will be a webdataset with some extra information stored in a folder called `.nv-meta`. See [](data_on_disk) for details about this format and how to adapt the meta information to your needs.

For data preparation, we provide a few helper functions to get you started quickly. There are a few cases to consider:
- [Data Preparation](#data-preparation)
  - [Webdataset format](#webdataset-format)
    - [Compatible format](#compatible-format)
      - [Example 1: No presplit shards, captioning webdataset](#example-1-no-presplit-shards-captioning-webdataset)
      - [Example 2: Presplit shards by prefix](#example-2-presplit-shards-by-prefix)
      - [Example 3: Presplit shards by folder](#example-3-presplit-shards-by-folder)
    - [Special format](#special-format)
  - [Convert to webdataset](#convert-to-webdataset)
    - [Webdataset Format](#webdataset-format-1)
    - [Build using Python](#build-using-python)

(wds-format)=
## Webdataset format

If you already have a dataset in webdataset format, you're lucky: It should work out-of-the-box.

(wds-format-compatible)=
### Compatible format

Example for a compatible format (e.g. captioning dataset):
```text
shard_000.tar
├── samples/sample_0000.jpg
├── samples/sample_0000.txt
├── samples/sample_0000.json
├── samples/sample_0001.jpg
├── samples/sample_0001.txt
├── samples/sample_0001.json
└── ...
```
With the default webdataset loading semantic, the images (in this case the `jpg` part), text (`txt`) and json
are loaded automatically if specified in the `field_map`. The dataset preparation wizard will ask you for the mapping
of those fields.

The shards may be pre-split or not split beforehand. Exemplary structures and [dataset preparation commands](cli#energon_data_prepare):

#### Example 1: No presplit shards, captioning webdataset
```text
shards
├── shard_0000.tar
├── shard_0001.tar
└── ...
```
Commandline:
```
> energon prepare ./
# Exemplary answers to interactive questions:
Ratio: 8,1,1
Dataset class: CaptioningWebdataset
Field map: Yes
  image: jpg
  caption: txt  # if txt contains the caption
# or
  caption: json[caption]  # if .json contains {"caption": "My nice image"}
```

#### Example 2: Presplit shards by prefix
```text
shards
├── train_shard_0000.tar
├── train_shard_0001.tar
├── ...
├── val_shard_0000.tar
├── val_shard_0001.tar
└── ...

```
Commandline:
```
> energon prepare --split-parts 'train:shards/train_.*' --split-parts 'val:shards/val_.*' ./
```

#### Example 3: Presplit shards by folder
```text
shards
├── train
│   ├── shard_00001.tar
│   ├── shard_00001.tar
│   └── ...
├── val
│   ├── shard_00001.tar
│   ├── shard_00001.tar
│   └── ...
└── ...
```
Commandline:
```
> energon prepare --split-parts 'train:shards/train/.*' --split-parts 'val:shards/val/.*' ./
```

(wds-format-special)=
### Special format
Sometimes, your data will not be easily represented as a `field_map` explained above. 
For example, your data may contain

* structured data like nested boxes for each sample
* custom binary formats
* xml / html / pickle etc.

In those cases you have two options:

1. Creating a custom `sample_loader.py` in the `.nv-meta` folder
    * This will typically do the job and is preferred if you only have to do some small conversions.
2. Using a `CrudeWebdataset`
    * For more intricate conversions, you can use a CrudeWebdataset that will pass your samples in a raw form into your TaskEncoder where you can then convert them based on the subflavor for example. For more details see [](crude-data).

Even for these specific wds formats, you would start preparing your data using the [dataset preparation command](cli#energon_data_prepare), but you will need to define a custom sample loader or select `CrudeWebdataset` in the dataprep wizard. 

Example for a special format (e.g. ocr dataset) for which we will use a custom `sample_loader.py`:

```text
parts
├── segs-000000.tar
│   ├── 636f6d706f6e656e747362656e6566693030616e6472(0025).jp2
│   ├── 636f6d706f6e656e747362656e6566693030616e6472(0025).lines.png
│   ├── 636f6d706f6e656e747362656e6566693030616e6472(0025).mp
│   ├── 636f6d706f6e656e747362656e6566693030616e6472(0025).words.png
│   ├── 636f6d706f6e656e747362656e6566693030616e6472(0075).jp2
│   ├── 636f6d706f6e656e747362656e6566693030616e6472(0075).lines.png
│   ├── 636f6d706f6e656e747362656e6566693030616e6472(0075).mp
│   └── ...
└── ...
```
`.mp` (`msgpack` content) files are automatically decoded, containing:
```json
{
  "identifier": "componentsbenefi00andr",
  "pageno": 25,
  "size": {"w": 2286, "h": 3179},
  "lines": [
    {"l": 341, "t": 569, "b": 609, "r": 1974, "text": "CHAPTER 4  ADVANCED TRAFFIC CONTROL SYSTEMS IN INDIANA"},
    {"l": 401, "t": 770, "b": 815, "r": 2065, "text": "A variety of traffic control systems currently exist"},
    //...
  ],
  "words": [
    {"l": 341, "t": 577, "b": 609, "r": 544, "text": "CHAPTER"},
    {"l": 583, "t": 578, "b": 607, "r": 604, "text": "4"},
    //...
  ],
  "chars": [
    {"t": 579, "b": 609, "l": 341, "r": 363, "text": "C"},
    {"t": 579, "b": 609, "l": 370, "r": 395, "text": "H"},
    //...
  ],
}
```

`sample_loader.py`:
```python
import torch


def sample_loader(raw: dict) -> dict:
    return dict(
        __key__=raw["__key__"],
        image=raw["jp2"],
        text="\n".join(line["text"] for line in raw["mp"]["lines"]),
        lines_boxes=torch.tensor(
            [
                (line["l"], line["t"], line["r"] - line["l"], line["b"] - line["t"])
                for line in raw["mp"]["lines"]
            ],
            dtype=torch.int64,
        ),
        lines_text=[line["text"] for line in raw["mp"]["lines"]],
        words_boxes=torch.tensor(
            [
                (line["l"], line["t"], line["r"] - line["l"], line["b"] - line["t"])
                for line in raw["mp"]["words"]
            ],
            dtype=torch.int64,
        ),
        words_text=[line["text"] for line in raw["mp"]["words"]],
        chars_boxes=torch.tensor(
            [
                (line["l"], line["t"], line["r"] - line["l"], line["b"] - line["t"])
                for line in raw["mp"]["chars"]
            ],
            dtype=torch.int64,
        ),
        chars_text=[line["text"] for line in raw["mp"]["chars"]],
    )


def part_filter(part: str) -> bool:
    return part in ("jp2", "mp")
```

For more information please also read [](custom-sample-loader).

(convert-to-wds)=
## Convert to webdataset

### Webdataset Format
```
shards
├── shard_0000.tar
│   ├── sample_0000.jpg
│   ├── sample_0000.txt
│   ├── sample_0000.detail.json
│   ├── sample_0001.jpg
│   ├── sample_0001.txt
│   └── sample_0001.detail.json
├── shard_0001.tar
│   ├── sample_0002.jpg
│   ├── sample_0002.txt
│   ├── sample_0002.detail.json
│   ├── sample_0003.jpg
│   ├── sample_0003.txt
│   └── sample_0003.detail.json
└── ...
```
The order of samples in the tar file is important. Samples with the same base name (~before the first dot of the filename) must follow each other.
The base name is used to group the samples, i.e. in the example `sample_0000` is the first group name, with the part types `jpg`, `txt`, `detail.json`.
If the default `webdataset` decoder is used, files are automatically parsed by extensions (e.g. ending on `.json` will automatically use `json.loads`, `.png` will load the image).
Each sample is yielded as a `dict`. Here that would be:
```py
{
    '__key__': 'sample_0000',
    'jpg': torch.Tensor(...),
    'txt': '...',
    'detail.json': {'key': 'value', 'key2': 'value2', ...},
}
{
    '__key__': 'sample_0001',
    'jpg': torch.Tensor(...),
    'txt': '...',
    'detail.json': {'key': 'value', 'key2': 'value2', ...},
}
...
```

### Build using Python
```py
import webdataset as wds


if __name__ == '__main__':
    # Wherever your dataset comes from
    my_dataset = ...
  
    with wds.ShardWriter("parts/data-%d.tar", maxcount=10000) as shard_writer:
        for key, data in my_dataset:
            sample = {
                "__key__": key,
                "png": data['image'],
            }
            shard_writer.write(sample)
```


