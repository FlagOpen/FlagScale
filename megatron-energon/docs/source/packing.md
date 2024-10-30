<!--- Copyright (c) 2024, NVIDIA CORPORATION.
SPDX-License-Identifier: BSD-3-Clause -->

# Packing

Packing (sometimes also called sequence packing), enables you to selectively compress multiple
input samples into a single sample, for example depending on their length.

This technique is commonly used with large language models, if the input samples have very different
lengths leading to lots of padding and hence wasted compute.

This section explains how you can pack samples together and utilize the full context length.

## How to pack samples on the fly

To use packing, you need to implement the TaskEncoder methods {py:meth}`select_samples_to_pack <megatron.energon.TaskEncoder.select_samples_to_pack>`
and {py:meth}`pack_selected_samples <megatron.energon.TaskEncoder.pack_selected_samples>`.
Furthermore, you need to initialize the loader with the `packing_buffer_size` argument set to a non-zero number.

The `select_samples_to_pack` method will receive a list of samples (size according to the selected `packing_buffer_size`),
and should partition those samples into groups that shall be packed together. Hence the function returns
a list of lists of samples.

For each group, the second method `pack_selected_samples` will be called. You need to implement how a group of
samples will be mapped to a single sample. In terms of LLMs for example, this method might concatenate the input tokens.


```{warning}
You can set the `__restore_key__` of the packed sample to an empty tuple, since energon will set the correct
restore key afterwards, based on the samples that went in.
```

```{warning}
To handle attention masks and tokenized inputs, you will want to operate on a different sample type.
The `pack_selected_samples` method may return a different sample type that is expected as the input for the `batch` method.
```

It is important, to mark custom functions like `encode_sample` and `pack_selected_samples` as `@stateless` to allow saving
samples for packing. If augmentations happen, it should be marked with
`@stateless(restore_seeds=True)`, to deterministically set the seeds based on the `TaskEncoder.current_sample_index`.
You have to make sure the methods are actually stateless, meaning that they will produce the same output when invoked
with the same input and random states.

Example for padding for a large language model extending the example from the [Task Encoders](task_encoders) section:

```python
class PackingCaptioningTaskEncoder(CaptioningTaskEncoder):
    """This class extends the CaptioningTaskEncoder and adds select_samples_to_pack and pack_selected_samples for packing samples
    efficiently on-the-fly.
    Set the `packing_buffer_size` of the get_(train|val)_dataset to an accordingly large number to get a
    properly sized input sample buffer with good diversity.
    """

    @stateless(restore_seeds=True)
    def encode_sample(self, ...):
        # Added `stateless` decorator to allow saving samples for packing. Will set the seed
        # deterministically based on the self.current_sample_index.
        ...

    def select_samples_to_pack(self, samples: List[CaptioningSample]) -> List[List[CaptioningSample]]:
        # Do something intelligent here, e.g. sort by caption length and concat where possible.
        # This could be better, but it's just an example.
        samples.sort(key=lambda x: len(x.caption))
        groups = []
        while len(samples) > 0:
            batch = []
            caption_len = 0
            while len(samples) > 0 and caption_len + len(samples[0].caption) < self.max_length:
                sample = samples.pop(0)
                batch.append(sample)
                caption_len += len(sample.caption)
            groups.append(batch)
        return groups
    
    @stateless
    def pack_selected_samples(self, samples: List[CaptioningSample]) -> CaptioningSample:
        # Construct a new CaptioningSample by concatenating the captions
        ...

```
