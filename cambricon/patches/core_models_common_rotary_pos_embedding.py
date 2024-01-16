import os
import torch
import torch_mlu
import megatron
from megatron.core.models.common.rotary_pos_embedding import _rotate_half

def apply_rotary_pos_emb(t, freqs):
    """
    input tensor t is of shape [seq_length, ..., dim]
    rotary positional embeding tensor freqs is of shape [seq_length, ..., dim]
    check https://kexue.fm/archives/8265 for detailed formulas
    """
    t_shape = t.shape
    freqs_shape = freqs.shape
    t = t.reshape(t_shape[0], t_shape[1] * t_shape[2], t_shape[3])
    freqs = freqs.reshape(freqs_shape[0], freqs_shape[1] * freqs_shape[2], freqs_shape[3])

    rot_dim = freqs.shape[-1]
    # ideally t_pass is empty so rotary pos embedding is applied to all tensor t
    t, t_pass = t[..., :rot_dim], t[..., rot_dim:]

    # first part is cosine component
    # second part is sine component, need to change signs with _rotate_half method
    inference_flag = (os.getenv("AQUILA_INFERENCE_ROTARY_EMBEDDING", "false").lower() == "true")
    t = (t * freqs.cos()) + (_rotate_half(t) * freqs.sin())

    t = t.reshape(t_shape[0], t_shape[1], t_shape[2], t_shape[3])
    t_pass = t_pass.reshape(t_shape[0], t_shape[1], t_shape[2], t_pass.shape[-1])

    return torch.cat((t, t_pass), dim=-1)


megatron.core.models.common.rotary_pos_embedding.apply_rotary_pos_emb = apply_rotary_pos_emb
