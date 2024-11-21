<!--- Copyright (c) 2024, NVIDIA CORPORATION.
SPDX-License-Identifier: BSD-3-Clause -->

# Dataset Concepts

## Dataset Design Guidelines

* Datasets should be sharded (i.e. a few samples per "shard", like <1% of the dataset each, optimally more lots more shards than total number of workers used in training)
  * This allows for parallel loading of shards, split over the workers
  * With webdataset, the shards can be streamed (i.e. no random access, but iterate over the shards very fast)
  * The datasets are split across all ranks and workers to
* TorchNet IndexedDatasets are also supported, they will be partially streamed, similar to shards
  * Virtual shards are created during training, i.e. an offset index and number of samples is resampled every time, then this portion is streamed for performance
* All datasets are {py:class}`torch.data.IterableDataset`s, i.e. they do not support random access via `__getitem__` (i.e. no `dataset[index]`, but you may only iterate over it).
  * No concept of "epochs", datasets are infinitely looped for training, concatenated for validation / testing
  * This gives more freedom in how the data is loaded
  * Enables streaming of shards (requirement for high-performance loading)
  * Enables blending of datasets (i.e. mix different datasets together with a weighted random sampler)
  * For validation / testing, shards are strided across workers, thus some workers may have less/no data.

## Statistical Analysis of Dataset Loading
* As each webdataset dataloader worker gets a all shards to work on, this is statistically fine.
  * After iterating over the dataset once, the shards are reshuffled, thus each "worker epoch", every sample will be seen once (i.e. it is exactly balanced once every (*total number of workers* x *total number of samples*) samples have been iterated.
  * Because webdatasets shards are typically iterated linearly, lots of shuffling randomness potential is not available. We thus slice the shards into smaller parts (configured by `max_samples_per_sequence`), such that the shuffling is more fine-grained. Applying this, effectively, the size of shards does not matter too much any more at the performance cost of more seeking.
* The {py:class}`BlendDataset <megatron.energon.BlendDataset>` will always yield full batches of one underlying dataset loader,
  but across different nodes/ranks(=GPUs), different batches will be yielded according to the blend weights.
  * Typically, gradients are accumulated across ranks. Thus, the distribution should approximately match the given weights for a high total number of ranks (like at least 8 ranks).
  * If this behaviour is not desired, but mixing should happen within batches, the {py:class}`MixBatchDataset <megatron.energon.MixBatchDataset>` can be used instead.
* The {py:class}`GroupBatchDataset <megatron.energon.GroupBatchDataset>` will only yield as soon as a full batch of one group was collected.
  This could potentially lead to corner cases, such as that rare groups are filled very slowly (or even only with a single example).
  * Still, statistically, this should be fine over lots of samples, even if there is one unbalanced group, as it will eventually yield nevertheless.

## Types

Following will show the type hierarchy of python classes.

(flavors_details)=
### Dataset Types / Flavors

These are the available dataset types for the `dataset.yaml`.

Type hierarchy:
* ({py:class}`torch.data.IterableDataset`: All datasets implement the torch {py:class}`IterableDataset <torch.data.IterableDataset>` interface)
  * ({py:class}`BaseCoreDataset <megatron.energon.BaseCoreDataset>`: Base class for all datasets.)
    * ({py:class}`BaseWebdataset <megatron.energon.BaseWebdataset>`: For more customizable webdataset based datasets.)
      * {py:class}`DefaultGenericWebdataset <megatron.energon.DefaultGenericWebdataset>`: Webdataset based dataset consisting of sharded .tar files.
        * {py:class}`DefaultImageWebdataset <megatron.energon.DefaultImageWebdataset>`: On top of the {py:class}`DefaultGenericWebdataset <megatron.energon.DefaultGenericWebdataset>`, loads all images.
          * {py:class}`CaptioningWebdataset <megatron.energon.CaptioningWebdataset>`: Yields {py:class}`CaptioningSample <megatron.energon.CaptioningSample>` from webdataset format
          * {py:class}`ImageWebdataset <megatron.energon.ImageWebdataset>`: Yields {py:class}`ImageSample <megatron.energon.ImageSample>` from webdataset format
          * {py:class}`OCRWebdataset <megatron.energon.OCRWebdataset>`: Yields {py:class}`OCRSample <megatron.energon.OCRSample>` from webdataset format
          * {py:class}`VQAWebdataset <megatron.energon.VQAWebdataset>`: Yields {py:class}`VQASample <megatron.energon.VQASample>` from webdataset format
    * {py:class}`BaseIndexedDataset <megatron.energon.BaseIndexedDataset>`: TorchNet IndexedDataset/MMapIndexedDataset dataset, consisting of .bin/.idx file(s).
      * {py:class}`CaptioningIndexedDataset <megatron.energon.CaptioningIndexedDataset>`: Yields {py:class}`CaptioningSample <megatron.energon.CaptioningSample>` from TorchNet IndexedDataset format
      * {py:class}`ImageIndexedDataset <megatron.energon.ImageIndexedDataset>`: Yields {py:class}`ImageSample <megatron.energon.ImageSample>` from TorchNet IndexedDataset format
      * {py:class}`TextIndexedDataset <megatron.energon.TextIndexedDataset>`: Yields {py:class}`TextSample <megatron.energon.TextSample>` from TorchNet IndexedDataset format

From the above, you will want to use the innermost (non-abstract) classes for your `dataset.yaml`.
Hence, if you have a captioning dataset stored in the `.idx/.bin` format that megatron uses, you need `CaptioningIndexedDataset`.
For an ocr dataset stored as a webdataset, you will use `OCRWebdataset`.

### Sample Types

These are the available sample types and their attributes yielded by the datasets above.

Type hierarchy:

* {py:class}`Sample <megatron.energon.Sample>`: Base class
  * Attributes:
    * {py:attr}`__key__: str <megatron.energon.Sample.__key__>`: Unique identifier of the sample within the dataset. Useful for backtracking the source of a single sample.
  * {py:class}`CaptioningSample <megatron.energon.CaptioningSample>`: Represents a sample for captioning
    * Attributes:
      * {py:attr}`__key__: str <megatron.energon.Sample.__key__>` (inherited)
      * {py:attr}`image: torch.Tensor <megatron.energon.CaptioningSample.image>`: The input image tensor
      * {py:attr}`caption: str <megatron.energon.CaptioningSample.caption>`: The target caption string
  * {py:class}`ImageSample <megatron.energon.ImageSample>`: Represents a sample which only contains an image (e.g. for reconstruction)
    * Attributes:
      * {py:attr}`__key__: str <megatron.energon.Sample.__key__>` (inherited)
      * {py:attr}`image: torch.Tensor <megatron.energon.CaptioningSample.image>`: The image tensor
  * {py:class}`OCRSample <megatron.energon.OCRSample>`: Represents a sample which only contains ocr image and text
    * Attributes:
      * {py:attr}`__key__: str <megatron.energon.Sample.__key__>` (inherited)
      * {py:attr}`image: str <megatron.energon.OCRSample.image>`: The input image
      * {py:attr}`text: str <megatron.energon.OCRSample.text>`: The text string for the whole image
      * {py:attr}`lines_boxes: Optional[torch.Tensor] <megatron.energon.OCRSample.lines_boxes>`: The bounding boxes of the text lines in the image
      * {py:attr}`lines_text: Optional[torch.Tensor] <megatron.energon.OCRSample.lines_text>`: The text content of the text lines in the image
      * {py:attr}`words_boxes: Optional[torch.Tensor] <megatron.energon.OCRSample.words_boxes>`: The bounding boxes of the text words in the image
      * {py:attr}`words_text: Optional[torch.Tensor] <megatron.energon.OCRSample.words_text>`: The text content of the text words in the image
      * {py:attr}`chars_boxes: Optional[torch.Tensor] <megatron.energon.OCRSample.chars_boxes>`: The bounding boxes of the text characters in the image
      * {py:attr}`chars_text: Optional[torch.Tensor] <megatron.energon.OCRSample.chars_text>`: The text content of the text characters in the image
  * {py:class}`TextSample <megatron.energon.TextSample>`: Represents a sample which only contains a text string (e.g. for text generation)
    * Attributes:
      * {py:attr}`__key__: str <megatron.energon.Sample.__key__>` (inherited)
      * {py:attr}`text: str <megatron.energon.TextSample.text>`: The text string
  * {py:class}`VQASample <megatron.energon.VQASample>`: Represents a sample which contains an image, a question/context and an answer
    * Attributes:
      * {py:attr}`__key__: str <megatron.energon.Sample.__key__>` (inherited)
      * {py:attr}`image: torch.Tensor <megatron.energon.VQASample.image>`: The input image tensor
      * {py:attr}`context: str <megatron.energon.VQASample.context>`: The target caption string
      * {py:attr}`answer: str <megatron.energon.VQASample.answer>`: The target caption string

```{note}
Images are always of shape `(C, H, W)`    
```
