# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""CPU regression tests for sample boundaries in Qwen3.5 multimodal packing."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.transformer.multi_token_prediction import roll_tensor
from tools.datasets.qwenvl.data import dataset_helpers as data

from flagscale.models.megatron.qwen35.rope import get_rope_index
from flagscale.train.megatron import train_qwen35


def sample(key, tokens, image=False, video=False):
    return data.ImageTaskSample(
        __key__=key,
        __subflavors__={},
        imgs=np.ones((4, 12), dtype=np.float32) if image else [],
        videos=np.full((4, 12), 2, dtype=np.float32) if video else [],
        image_thw_grids=np.array([[1, 2, 2]]) if image else np.empty((0, 3), dtype=np.int64),
        video_thw_grids=np.array([[1, 2, 2]]) if video else np.empty((0, 3), dtype=np.int64),
        image_input_mask=np.array(tokens) == 91 if image else None,
        video_input_mask=np.array(tokens) == 92 if video else None,
        second_per_grid_ts=np.array([0.5]) if video else np.empty(0),
        text=np.array(tokens, dtype=np.int64),
        target=np.array([*tokens[1:], data.IGNORE_IDX], dtype=np.int64),
    )


@pytest.fixture
def encoder(monkeypatch):
    encoder = data.TaskEncoder.__new__(data.TaskEncoder)
    encoder.args = SimpleNamespace(enable_variable_seq_lengths=True)
    encoder.seq_len = 16
    encoder.tp_size, encoder.cp_size, encoder.sequence_parallel = 4, 1, True
    encoder.temporal_patch_size, encoder.patch_size = 1, 2
    encoder.tokenizer = SimpleNamespace(pad_token_id=0)
    monkeypatch.setattr(data, "FIRST_MAX_PADDING_FLAG", False)
    monkeypatch.setattr(data, "MAX_IMG_THRESHOLD", 5000)
    return encoder


def test_next_fit_preserves_order_and_flushes_tail(encoder, caplog):
    encoder.seq_len = 10
    samples = [sample(str(i), list(range(1, n + 1))) for i, n in enumerate([6, 5, 4, 11, 3])]
    groups = encoder.select_samples_to_pack(samples)
    assert [[s.__key__ for s in group] for group in groups] == [["0"], ["1", "2"], ["4"]]
    assert "Skipping packing sample 3" in caplog.text
    assert encoder.select_samples_to_pack([]) == []
    assert encoder.select_samples_to_pack([sample("empty", [])]) == []


def test_exact_capacity(encoder):
    encoder.seq_len = 6
    groups = encoder.select_samples_to_pack([sample("a", [1, 2]), sample("b", [3, 4, 5, 6])])
    assert len(groups) == 1
    assert len(encoder.pack_selected_samples(groups[0]).text) == 6


def test_visual_alignment_and_shifted_labels(encoder):
    samples = [
        sample("text", [11, 12]),
        sample("image", [90, 91, 13], image=True),
        sample("video", [90, 92, 14], video=True),
    ]
    originals = [s.target.copy() for s in samples]
    packed = encoder.pack_selected_samples(samples)
    assert packed.__subflavors__["packed_seq_lengths"] == [2, 3, 3]
    np.testing.assert_array_equal(packed.target, np.concatenate(originals))
    # First labels of later samples are valid next-token targets, not boundary losses.
    assert packed.target[2] == 91 and packed.target[5] == 92
    np.testing.assert_array_equal(
        packed.image_input_mask, [False, False, False, True, False, False, False, False]
    )
    np.testing.assert_array_equal(
        packed.video_input_mask, [False, False, False, False, False, False, True, False]
    )
    assert packed.imgs.shape == packed.videos.shape == (4, 12)
    np.testing.assert_array_equal(packed.second_per_grid_ts, [0.5])
    for s, original in zip(samples, originals):
        np.testing.assert_array_equal(s.target, original)
    batch = encoder.encode_batch(encoder.batch([packed]))
    assert batch["cu_seqlens"].tolist() == [0, 2, 5, 8]
    assert batch["image_thw_grids"].prod(-1).sum() == batch["imgs"].shape[0]
    assert batch["video_thw_grids"].prod(-1).sum() == batch["videos"].shape[0]


@pytest.mark.parametrize("capacity_padding", [False, True])
def test_padding_stays_in_last_segment_with_ignored_labels(encoder, monkeypatch, capacity_padding):
    monkeypatch.setattr(data, "FIRST_MAX_PADDING_FLAG", capacity_padding)
    packed = encoder.pack_selected_samples([sample("a", [1, 2]), sample("b", [3, 4, 5])])
    batch = encoder.encode_batch(encoder.batch([packed]))
    end = 16 if capacity_padding else 8
    assert batch["cu_seqlens"].dtype == torch.int32
    assert batch["cu_seqlens"].tolist() == [0, 2, end]
    assert torch.all(batch["target"][0, 5:] == data.IGNORE_IDX)
    assert torch.all(batch["text"][0, 5:] == 0)


def test_unpacked_batch_has_no_packing_metadata(encoder):
    batch = encoder.encode_batch(encoder.batch([sample("a", [1, 2, 3])]))
    assert "cu_seqlens" not in batch


def test_packed_batch_rejects_truncation_and_multiple_rows(encoder):
    packed = encoder.pack_selected_samples([sample("a", [1, 2, 3]), sample("b", [4, 5])])
    with pytest.raises(ValueError, match="micro_batch_size"):
        encoder.batch([packed, packed])
    encoder.sequence_parallel = False
    encoder.seq_len = 4
    with pytest.raises(ValueError, match="truncated"):
        encoder.batch([packed])


def rope(tokens, images=None, videos=None, mask=None, cu=None):
    return get_rope_index(
        2,
        91,
        92,
        90,
        input_ids=tokens,
        image_grid_thw=images,
        video_grid_thw=videos,
        attention_mask=mask,
        cu_seqlens=cu,
    )[0]


def test_packed_mrope_matches_individual_text_image_and_video_samples():
    # Each video frame has its own vision-start marker; two video grids must be expanded once.
    segments = [
        torch.tensor([[11, 12]]),
        torch.tensor([[90, 91, 13]]),
        torch.tensor([[90, 92, 90, 92, 14]]),
        torch.tensor([[90, 92, 15, 0]]),
    ]
    image = torch.tensor([[1, 2, 2]])
    videos = torch.tensor([[2, 2, 2], [1, 2, 2]])
    expected = torch.cat(
        [
            rope(segments[0], mask=segments[0] != 0),
            rope(segments[1], images=image, mask=segments[1] != 0),
            rope(segments[2], videos=videos[:1], mask=segments[2] != 0),
            rope(segments[3], videos=videos[1:], mask=segments[3] != 0),
        ],
        dim=-1,
    )
    tokens = torch.cat(segments, dim=-1)
    result = rope(
        tokens,
        images=image,
        videos=videos,
        mask=tokens != 0,
        cu=torch.tensor([0, 2, 5, 10, 14], dtype=torch.int32),
    )
    torch.testing.assert_close(result, expected)
    assert result[:, 0, [0, 2, 5, 10]].eq(0).all()


@pytest.mark.parametrize("cu", [[1, 4], [0, 0, 4], [0, 5], [0, 3, 2, 4]])
def test_mrope_rejects_invalid_boundaries(cu):
    with pytest.raises(ValueError, match="cu_seqlens"):
        rope(torch.tensor([[1, 2, 3, 4]]), cu=torch.tensor(cu))


def test_mtp_shifting_respects_packed_boundaries():
    cu = torch.tensor([0, 3, 5], dtype=torch.int32)
    params = PackedSeqParams(qkv_format="thd", cu_seqlens_q=cu, cu_seqlens_kv=cu)
    shifted, _ = roll_tensor(torch.tensor([[10, 11, 12, 20, 21]]), packed_seq_params=params)
    assert shifted.tolist() == [[11, 12, 0, 21, 0]]
    mask, _ = roll_tensor(torch.tensor([[1.0, 1.0, 0.0, 1.0, 0.0]]), packed_seq_params=params)
    assert mask.tolist() == [[1.0, 0.0, 0.0, 0.0, 0.0]]


def args(**overrides):
    values = dict(
        packing_buffer_size=8,
        enable_vision=True,
        micro_batch_size=1,
        context_parallel_size=1,
        pipeline_model_parallel_size=1,
        max_padding_length=16,
        seq_length=16,
    )
    return SimpleNamespace(**(values | overrides))


@pytest.mark.parametrize(
    "override",
    [
        dict(packing_buffer_size=-1),
        dict(micro_batch_size=2),
        dict(context_parallel_size=2),
        dict(pipeline_model_parallel_size=2),
        dict(max_padding_length=8),
        dict(enable_vision=False),
    ],
)
def test_unsupported_packing_config_is_rejected(override):
    with pytest.raises(ValueError):
        train_qwen35.validate_packing_args(args(**override))


def test_disabled_packing_preserves_existing_layouts():
    train_qwen35.validate_packing_args(
        args(
            packing_buffer_size=0,
            micro_batch_size=4,
            context_parallel_size=2,
            pipeline_model_parallel_size=2,
        )
    )


@pytest.mark.parametrize("packed", [False, True])
def test_get_batch_uses_batch_metadata_for_unpacked_validation(encoder, monkeypatch, packed):
    samples = [sample("a", [1, 2]), sample("b", [3, 4])]
    value = encoder.pack_selected_samples(samples) if packed else samples[0]
    batch = encoder.encode_batch(encoder.batch([value]))
    monkeypatch.setattr(train_qwen35, "get_args", lambda: args())
    monkeypatch.setattr(train_qwen35, "get_tokenizer", lambda: encoder.tokenizer)
    monkeypatch.setattr(train_qwen35, "get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(
        train_qwen35,
        "broadcast_data",
        lambda keys, data, dtype: {k: data[k].to(dtype) for k in keys},
    )
    monkeypatch.setattr(
        train_qwen35,
        "cur_platform",
        SimpleNamespace(range_push=lambda _: None, range_pop=lambda: None),
    )
    seen = []

    def get_positions(**kwargs):
        seen.append(kwargs["cu_seqlens"])
        return rope(
            kwargs["input_ids"], mask=kwargs["attention_mask"], cu=kwargs["cu_seqlens"]
        ), None

    output = train_qwen35.get_batch(
        iter([batch]), model=SimpleNamespace(get_rope_index=get_positions)
    )
    assert (output[-1] is not None) == packed
    assert (seen[0] is not None) == packed
    assert output[4].shape == (3, 1, batch["text"].shape[-1])
