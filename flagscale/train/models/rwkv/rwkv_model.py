# model_rwkv.py
# Minimal Megatron-LM compatible wrapper around RWKV-LM-V7 (src/model.py)
# - Reuses your Block (which internally uses RWKV_Tmix_x070 / RWKV_CMix_x070, token_shift, fused kernels, etc.)
# - If labels are provided: returns per-token loss [B, T] (Megatron will aggregate with loss_mask)
# - If labels are None: returns logits [B, T, V]
from typing import Dict, Literal, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch import Tensor
from megatron.core.inference.contexts import BaseInferenceContext
from megatron.core.packed_seq_params import PackedSeqParams
from rwkvfla.modules.token_shift import token_shift
from rwkvfla.ops.rwkv7.fused_addcmul import fused_addcmul_rwkv7
from rwkvfla.ops.rwkv7.fused_k_update import fused_k_rwkv7
# Import your original Block from RWKV-LM-V7
# Adjust this import path if your project layout differs

try:
    import torch._dynamo as _dynamo
except Exception:
    class _Dummy:
        def disable(self, f): return f
        def is_compiling(self): return False
    _dynamo = _Dummy()

_raw_token_shift = token_shift

@_dynamo.disable
def _safe_token_shift(*args, **kwargs):
    out = _raw_token_shift(*args, **kwargs)
    if isinstance(out, tuple):
        main, cache_out = out
        return main
    return out

token_shift = _safe_token_shift

def __nop(ob):
    return ob

ROCm_flag = torch.version.hip is not None
CompileFunction = __nop
if os.environ["RWKV_COMPILE_ON"] == "1":
    CompileFunction = torch.compile

#################################################################
# CUDA Kernel
#################################################################


HEAD_SIZE = int(os.environ["RWKV_HEAD_SIZE"])

if "x070" in os.environ["RWKV_MY_TESTING"]:
    CHUNK_LEN = 16

    if ROCm_flag is True:
        flags = [
            f"-D_C_={HEAD_SIZE}",
            f"-D_CHUNK_LEN_={CHUNK_LEN}",
            "-xhip",
            "-fopenmp",
            "-ffast-math",
            "-O3",
            "-munsafe-fp-atomics",
        ]
        load(
            name="wind_backstepping_hip",
            sources=["cuda/wkv7_hip.hip", "cuda/wkv7_op.hip"],
            is_python_module=False,
            verbose=True,
            extra_cuda_cflags=flags,
        )
    else:
        flags = [
            "-res-usage",
            f"-D_C_={HEAD_SIZE}",
            f"-D_CHUNK_LEN_={CHUNK_LEN}",
            "--use_fast_math",
            "-O3",
            "-Xptxas -O3",
            "--extra-device-vectorization",
        ]
        load(
            name="wind_backstepping",
            sources=["cuda/wkv7_cuda.cu", "cuda/wkv7_op.cpp"],
            is_python_module=False,
            verbose=True,
            extra_cuda_cflags=flags,
        )

    class WindBackstepping(torch.autograd.Function):
        @staticmethod
        def forward(ctx, w, q, k, v, z, b):
            B, T, H, C = w.shape
            if not isinstance(T, int) or T <= 0:
                raise ValueError(f"Invalid sequence length T: {T}. Expected a positive integer.")
            if T % CHUNK_LEN != 0:
                raise ValueError(f"T must be a multiple of CHUNK_LEN ({CHUNK_LEN}), got T={T}.")
            assert all(i.dtype == torch.bfloat16 for i in [w, q, k, v, z, b])
            assert all(i.is_contiguous() for i in [w, q, k, v, z, b])
            y = torch.empty_like(v)
            s = torch.empty(
                B, H, T // CHUNK_LEN, C, C, dtype=torch.float32, device=w.device
            )
            sa = torch.empty(B, T, H, C, dtype=torch.float32, device=w.device)
            torch.ops.wind_backstepping.forward(w, q, k, v, z, b, y, s, sa)
            ctx.save_for_backward(w, q, k, v, z, b, s, sa)
            return y

        @staticmethod
        def backward(ctx, dy):
            assert all(i.dtype == torch.bfloat16 for i in [dy])
            assert all(i.is_contiguous() for i in [dy])
            w, q, k, v, z, b, s, sa = ctx.saved_tensors
            grad_placeholders = {
                "dw": torch.empty_like(w),
                "dq": torch.empty_like(q),
                "dk": torch.empty_like(k),
                "dv": torch.empty_like(v),
                "dz": torch.empty_like(z),
                "db": torch.empty_like(b),
            }
            torch.ops.wind_backstepping.backward(
                w, q, k, v, z, b, dy, s, sa,
                grad_placeholders["dw"],
                grad_placeholders["dq"],
                grad_placeholders["dk"],
                grad_placeholders["dv"],
                grad_placeholders["dz"],
                grad_placeholders["db"],
            )
            return grad_placeholders["dw"], grad_placeholders["dq"], grad_placeholders["dk"], grad_placeholders["dv"], grad_placeholders["dz"], grad_placeholders["db"]

    def RUN_CUDA_RWKV7g(q, w, k, v, a, b):
        B, T, HC = q.shape
        q, w, k, v, a, b = [i.view(B, T, HC // 64, 64)
                            for i in [q, w, k, v, a, b]]
        return WindBackstepping.apply(w, q, k, v, a, b).view(B, T, HC)


#################################################################

class RWKV_Tmix_x070(nn.Module):
    @torch.no_grad()
    def __init__(self, args, layer_id):
        super().__init__()
        self.args = args
        self.layer_id = layer_id
        self.my_testing = args.my_testing

        self.head_size = args.head_size
        self.n_head = args.dim_att // self.head_size
        assert args.dim_att % self.n_head == 0
        H = self.n_head
        N = self.head_size
        C = args.n_embd

        ratio_0_to_1 = layer_id / (args.n_layer - 1)  # 0 to 1
        ratio_1_to_almost0 = 1.0 - (layer_id / args.n_layer)  # 1 to ~0
        ddd = torch.ones(1, 1, C)
        for i in range(C):
            ddd[0, 0, i] = i / C

        self.x_r = nn.Parameter(1.0 - torch.pow(ddd, 0.2 * ratio_1_to_almost0))
        self.x_w = nn.Parameter(1.0 - torch.pow(ddd, 0.9 * ratio_1_to_almost0))
        self.x_k = nn.Parameter(1.0 - torch.pow(ddd, 0.7 * ratio_1_to_almost0))
        self.x_v = nn.Parameter(1.0 - torch.pow(ddd, 0.7 * ratio_1_to_almost0))
        self.x_a = nn.Parameter(1.0 - torch.pow(ddd, 0.9 * ratio_1_to_almost0))
        self.x_g = nn.Parameter(1.0 - torch.pow(ddd, 0.2 * ratio_1_to_almost0))

        def ortho_init(x, scale):
            shape = x.shape
            if len(shape) == 2:
                gain = (
                    math.sqrt(shape[0] / shape[1]
                              ) if shape[0] > shape[1] else 1
                )
                nn.init.orthogonal_(x, gain=gain * scale)
            elif len(shape) == 3:
                gain = (
                    math.sqrt(shape[1] / shape[2]
                              ) if shape[1] > shape[2] else 1
                )
                for i in range(shape[0]):
                    nn.init.orthogonal_(x[i], gain=gain * scale)
            else:
                raise ValueError(f"Unexpected shape: {shape}")
            return x

        www = torch.zeros(C)
        zigzag = torch.zeros(C)
        linear = torch.zeros(C)
        for n in range(C):
            linear[n] = n / (C - 1) - 0.5
            zigzag[n] = ((n % N) - ((N - 1) / 2)) / ((N - 1) / 2)
            zigzag[n] = zigzag[n] * abs(zigzag[n])
            www[n] = -6 + 6 * (n / (C - 1)) ** (1 + 1 * ratio_0_to_1**0.3)

        # Increase lora dimension for headdim>64
        factor = self.head_size / 64
        D_DECAY_LORA = max(
            32, int(round((2.5 * (C**0.5)) * factor / 32) * 32))  # suggestion
        self.w1 = nn.Parameter(torch.zeros(C, D_DECAY_LORA))
        self.w2 = nn.Parameter(ortho_init(torch.zeros(D_DECAY_LORA, C), 0.1))
        # !!! 0.5 comes from F.softplus !!!
        self.w0 = nn.Parameter(www.reshape(1, 1, C) + 0.5 + zigzag * 2.5)

        D_AAA_LORA = max(
            32, int(round((2.5 * (C**0.5)) * factor / 32) * 32))  # suggestion
        self.a1 = nn.Parameter(torch.zeros(C, D_AAA_LORA))
        self.a2 = nn.Parameter(ortho_init(torch.zeros(D_AAA_LORA, C), 0.1))
        self.a0 = nn.Parameter(
            torch.zeros(1, 1, C) - 0.19 + zigzag * 0.3 + linear * 0.4
        )

        D_MV_LORA = max(
            32, int(round((1.7 * (C**0.5)) * factor / 32) * 32))  # suggestion
        self.v1 = nn.Parameter(torch.zeros(C, D_MV_LORA))
        self.v2 = nn.Parameter(ortho_init(torch.zeros(D_MV_LORA, C), 0.1))
        self.v0 = nn.Parameter(torch.zeros(1, 1, C) + 0.73 - linear * 0.4)

        # Note: for some data, you can reduce D_GATE_LORA or even remove this gate
        D_GATE_LORA = max(
            32, int(round((5 * (C**0.5)) / 32) * 32))  # suggestion
        self.g1 = nn.Parameter(torch.zeros(C, D_GATE_LORA))
        self.g2 = nn.Parameter(ortho_init(torch.zeros(D_GATE_LORA, C), 0.1))

        self.k_k = nn.Parameter(torch.zeros(1, 1, C) + 0.71 - linear * 0.1)
        self.k_a = nn.Parameter(torch.zeros(1, 1, C) + 1.02)
        self.r_k = nn.Parameter(torch.zeros(H, N) - 0.04)

        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))
        self.receptance = nn.Linear(C, C, bias=False)
        self.key = nn.Linear(C, C, bias=False)
        self.value = nn.Linear(C, C, bias=False)
        self.output = nn.Linear(C, C, bias=False)
        # !!! notice eps value !!!
        self.ln_x = nn.GroupNorm(H, C, eps=64e-5)

        self.receptance.weight.data.uniform_(-0.5 / (C**0.5), 0.5 / (C**0.5))
        self.key.weight.data.uniform_(-0.05 / (C**0.5), 0.05 / (C**0.5))
        self.value.weight.data.uniform_(-0.5 / (C**0.5), 0.5 / (C**0.5))
        self.output.weight.data.zero_()
        del www, zigzag, linear, ddd

    @CompileFunction
    def forward(self, x, v_first):
        B, T, C = x.size()
        xx = token_shift(x)
        xr, xw, xk, xv, xa, xg = fused_addcmul_rwkv7(x, xx, self.x_r,
                                                     self.x_w, self.x_k, self.x_v,
                                                     self.x_a, self.x_g)

        r = self.receptance(xr)
        # soft-clamp to (-inf, -0.5)
        w = -F.softplus(-(self.w0 + torch.tanh(xw @ self.w1) @ self.w2)) - 0.5
        k = self.key(xk)
        v = self.value(xv)
        if self.layer_id == 0:
            v_first = v  # store the v of the first layer
        else:
            v = torch.lerp(
                v, v_first, torch.sigmoid(self.v0 + (xv @ self.v1) @ self.v2)
            )  # add value residual
        # a is "in-context learning rate"
        a = torch.sigmoid(self.a0 + (xa @ self.a1) @ self.a2)
        g = torch.sigmoid(xg @ self.g1) @ self.g2

        kk = F.normalize((k * self.k_k).view(B, T, self.n_head, -1),
                         dim=-1, p=2.0).view(B, T, C)
        k = fused_k_rwkv7(k, a, self.k_a)

        x = RUN_CUDA_RWKV7g(r, w, k, v, -kk, kk * a)
        x = self.ln_x(x.view(B * T, C)).view(B, T, C)

        x = x + (
            (r.view(B, T, self.n_head, -1) * k.view(B, T, self.n_head, -1) * self.r_k).sum(
                dim=-1, keepdim=True
            )
            * v.view(B, T, self.n_head, -1)
        ).view(B, T, C)
        x = self.output(x * g)
        return x, v_first

#################################################################


class RWKV_CMix_x070(nn.Module):
    @torch.no_grad()
    def __init__(self, args, layer_id):
        super().__init__()
        self.args = args
        self.layer_id = layer_id
        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))

        ratio_1_to_almost0 = 1.0 - (layer_id / args.n_layer)  # 1 to ~0
        ddd = torch.ones(1, 1, args.n_embd)
        for i in range(args.n_embd):
            ddd[0, 0, i] = i / args.n_embd
        self.x_k = nn.Parameter(1.0 - torch.pow(ddd, ratio_1_to_almost0**4))

        self.key = nn.Linear(args.n_embd, args.n_embd * 4, bias=False)
        self.value = nn.Linear(args.n_embd * 4, args.n_embd, bias=False)

        self.key.weight.data.uniform_(
            -0.5 / (args.n_embd**0.5), 0.5 / (args.n_embd**0.5)
        )
        self.value.weight.data.zero_()

    @CompileFunction
    def forward(self, x):
        xx = token_shift(x)
        k = torch.addcmul(x, xx, self.x_k)
        k = torch.relu(self.key(k)) ** 2
        return self.value(k)

#################################################################

class Block(nn.Module):
    def __init__(self, args, layer_id):
        super().__init__()
        self.args = args
        self.layer_id = layer_id

        self.ln1 = nn.LayerNorm(args.n_embd)
        self.ln2 = nn.LayerNorm(args.n_embd)

        if self.layer_id == 0:
            self.ln0 = nn.LayerNorm(args.n_embd)

        self.att = RWKV_Tmix_x070(args, layer_id)
        self.ffn = RWKV_CMix_x070(args, layer_id)

    @CompileFunction
    def forward(self, x, v_first):
        if self.layer_id == 0:
            x = self.ln0(x)

        x_attn, v_first = self.att(self.ln1(x), v_first)
        x = x + x_attn

        x = x + self.ffn(self.ln2(x))
        return x, v_first

#################################################################
# The RWKV Model with blocks
#################################################################

class RWKVModel(nn.Module):
    """
    Forward contract:
      - forward(tokens) -> logits [B, T, V]
      - forward(tokens, labels, loss_mask) -> per-token loss [B, T]
        (No aggregation here; Megatron's loss_func will multiply by loss_mask and reduce.)

    Notes:
      - This version does not enable tensor/pipeline model parallelism (TP/PP). Start with TP=1/PP=1.
      - pre_process/post_process are kept for future PP use; ignored in TP=1/PP=1.
      - If you later enable TP, replace nn.Embedding/nn.Linear with Megatron's parallel layers.
    """

    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
        n_layer: int,
        *,
        pre_process: bool = True,
        post_process: bool = True,
        parallel_output: bool = True,   # keep for future TP; no effect with TP=1
        use_grad_checkpoint: bool = False,  # PyTorch checkpoint at block granularity
        dtype: torch.dtype | None = None,   # e.g., torch.bfloat16 to match --bf16
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.n_layer = n_layer
        self.pre_process = pre_process
        self.post_process = post_process
        self.parallel_output = parallel_output
        self.use_grad_checkpoint = use_grad_checkpoint

        # Embedding / Blocks / Head — align with your original RWKV stack
        self.emb = nn.Embedding(vocab_size, hidden_size)
        self.blocks = nn.ModuleList([Block(_ArgsShim(hidden_size, n_layer), i)
                                     for i in range(n_layer)])
        self.ln_out = nn.LayerNorm(hidden_size)
        self.head = nn.Linear(hidden_size, vocab_size, bias=False)

        if dtype is not None:
            # Optional: move module parameters to a specific dtype (e.g., bf16)
            self.to(dtype=dtype)

    def forward(
        self,
        input_ids: Tensor,
        labels: Tensor = None,
        loss_mask: Optional[Tensor] = None,
    ):
        """
        Returns:
          - If labels is None: logits [B, T, V]
          - Else: per-token loss [B, T] (no reduction), optionally multiplied by loss_mask
        """
        # 1) Token embedding
        x = self.emb(input_ids)  # [B, T, H]

        # 2) Run N RWKV blocks; maintain v_first across blocks as in original implementation
        v_first = torch.empty_like(x)
        if self.use_grad_checkpoint:
            # PyTorch gradient checkpointing to reduce activation memory
            for blk in self.blocks:
                x, v_first = torch.utils.checkpoint.checkpoint(
                    blk, x, v_first, use_reentrant=False
                )
        else:
            for blk in self.blocks:
                x, v_first = blk(x, v_first)

        # 3) Output projection to logits
        x = self.ln_out(x)
        logits = self.head(x)  # [B, T, V]

        # 4) Training branch: return per-token cross entropy for Megatron to aggregate
        if labels is not None:
            # Use transpose for CE: [B, V, T] vs labels [B, T]
            per_tok = F.cross_entropy(
                logits.transpose(1, 2), labels,
                reduction="none"
            )  # [B, T]

            # Do NOT reduce here. Optionally pre-mask per-token loss for clarity.
            if loss_mask is not None:
                per_tok = per_tok * loss_mask  # still [B, T]

            return per_tok

        # Inference / eval branch: return logits
        return logits

class _ArgsShim:
    """
    Minimal adapter to provide attributes that Block(...) expects from `args`.
    Extend this class if Block accesses additional fields in your repo.
    """
    def __init__(self, n_embd: int, n_layer: int):
        self.n_embd = n_embd
        self.n_layer = n_layer

        # Common defaults for RWKV-7
        self.dim_att = n_embd
        # Typical RWKV-7 FFN size: 4x n_embd, padded to a multiple of 32
        self.dim_ffn = int((n_embd * 4) // 32 * 32)

        # Gradient checkpoint flag expected by some branches; we control it externally
        self.grad_cp = 0

        # Version switch used in your repo (e.g., "x070"); set to match RWKV-7 defaults
        self.my_testing = "x070"
        # Add/adjust fields here if your Block reads more attributes from args
