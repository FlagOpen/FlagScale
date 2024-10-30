<!--- Copyright (c) 2024, NVIDIA CORPORATION.
SPDX-License-Identifier: BSD-3-Clause -->

(custom-sample-loader)=
# Custom Sample Loader

As discussed in [](data_on_disk), you can specify how samples are loaded from disk and mapped to
the sample dataclass in memory.

Instead of using a `field_map` in your `dataset.yaml`, you can also create custom python code for sample loading.

Here's an example for your updated `dataset.yaml`:
```yaml
__class__: OCRWebdataset
__module__: megatron.energon
sample_loader: sample_loader.py:sample_loader
part_filter: sample_loader.py:part_filter
```

In addition, you need to create a python file inside the `.nv-meta` folder. In this case it's
called `sample_loader.py`.
That file needs to contain the two methods referenced above:

```python
import json
import torch

def sample_loader(raw: dict) -> dict:
    data = json.loads(raw["bbox.json"])

    return dict(
        __key__=raw["__key__"],
        image=raw["jpg"],
        text=raw["gt.txt"],
        lines_boxes=torch.tensor([box["bbox"] for box in data], dtype=torch.int64),
        lines_text=[box["text"] for box in data],
    )

def part_filter(part: str) -> bool:
    return part in ("bbox.json", "gt.txt", "jpg")
```

Your `sample_loader` method must accept a dictionary as argument and return a dictionary. It directly operates on 
the webdataset samples and the resulting dictionary keys should map to the corresponding sample class members.
In this case an {py:class}`OCRSample <megatron.energon.OCRSample>`.

With the optional `part_filter` method, you can prevent some webdataset fields from being loaded.
Given a field name, the method should return True if the field is to be kept.

(crude-data)=
# Crude Data and How to Cook It 👨‍🍳

Sometimes, you cannot do all the conversion work to obtain a {py:class}`Sample <megatron.energon.Sample>` right in a small `sample_loader.py` next to the dataset.
Maybe your data

- is complex and contains custom binary formats and pickle data
- or needs project-specific pre-processing that you don't want to place next to the dataset

In this case, you can feed the data in its "crude" or "raw" form into your code and do the conversion there. 

Let's try it. If you already have an existing energon-compliant data set, you may want to create a copy of your `dataset.yaml` specifically for this use case and keep the original:

Original `dataset.yaml`:
```
__class__: OCRWebdataset
__module__: megatron.energon
sample_loader: sample_loader.py:sample_loader
part_filter: sample_loader.py:part_filter
```

New `crude.yaml` next to it:
```
__class__: CrudeWebdataset
__module__: megatron.energon
subflavors:
  example_crude_selector: foobar
```

We recommend that you assign subflavors to differentiate the data.
This helps, if you decide to mix different crude datasets or crude with non-crude data using a [Metadataset](metadatasets.md).
You can then use the subflavors to assign how each sample shall be processed.

Let's see how we set up our [TaskEncoder](task_encoders.md) to cook the crude samples and turn them into real samples like {py:class}`TextSample <megatron.energon.TextSample>`.

A crude sample is essentially a dictionary ({py:class}`CrudeSample <megatron.energon.CrudeSample>` inherits from `dict`) and the loader will look for a {py:class}`Cooker <megatron.energon.task_encoder.Cooker>` that can handle and convert your crude samples.
You can provide cookers as follows:

```python
from megatron.energon.task_encoder.cooking import Cooker, basic_sample_keys

# ...

def cook_text(sample: dict) -> TextSample:
    return TextSample(
        **basic_sample_keys(sample),
        text=f">{sample['txt'].decode()}<",
    )


class MyTaskEncoder(DefaultTaskEncoder[TextSample, TextSample, TextRawBatch, TextBatch]):
    cookers = [
        Cooker(cook_text, has_subflavors={"example_crude_selector": "foobar"}),
        Cooker(...)  # other cookers for other crude data if needed
    ]

    # ...
```

In the example above, the cooker acts on all crude samples that have a subflavor `example_crude_selector` set to `foobar`.
The cooker will convert the dictionary to a {py:class}`TextSample <megatron.energon.TextSample>` by decoding the raw bytes and decorating the text with some nice angle brackets.
Probably you noticed the {py:meth}`basic_sample_keys <megatron.energon.task_encoder.cooking.basic_sample_keys>` helper that we inserted. All it does is to forward the key, restore key and flavors from the dict to the real sample. You will always need to forward these, or your dataset will not be restorable.

In a real use-case you will want to do a lot more here and we recommend keeping the cook methods in separate files and importing them where you define your TaskEncoder.

### Other Filters for Cookers

You can filter using the `subflavors` as above, you can also filter using the deprecated single `subflavor` like this:

```python
Cooker(cook_text, is_subflavor="helloworld")
```

or if you need custom filtering, you can provide a method to filter the sample:

```python
Cooker(cook_text, condition=lambda sample: return sample['myprop'] == 'yes_thats_it')
```

If you use multiple filters, they must all be satisfied for the sample to match.
