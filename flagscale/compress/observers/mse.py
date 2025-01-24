from typing import Any, Iterable, Optional, Tuple, Union
import torch
from torch import FloatTensor, IntTensor, Tensor
from compressed_tensors.quantization.utils import calculate_qparams
from compressed_tensors.quantization.quant_args import QuantizationArgs
from compressed_tensors.quantization.lifecycle import fake_quantize
from .base import Observer
from compressed_tensors.registry.registry import _REGISTRY, _ALIAS_REGISTRY

_REGISTRY[Observer]["moving_mse"] = _REGISTRY[Observer]["mse"]
_REGISTRY[Observer].pop("mse")
_ALIAS_REGISTRY[Observer]["moving_mse"] = "moving_mse"
_ALIAS_REGISTRY[Observer].pop("mse")

print(_REGISTRY[Observer])
@Observer.register("mse")
class MSEObserver(Observer):
    def __init__(
        self,
        quantization_args: QuantizationArgs,
        grid: float = 100.0,
        maxshrink: float = 0.80,
        norm: float = 2.4,
    ):
        super().__init__(quantization_args=quantization_args)
        self.grid = grid
        self.maxshrink = maxshrink
        self.norm = norm

    def calculate_mse_min_max(
        self,
        observed: Tensor,
        reduce_dims: Optional[Tuple[int]] = None,
    ):
        if not reduce_dims:
            absolute_min_val, absolute_max_val = torch.aminmax(observed)
        else:
            absolute_min_val = torch.amin(observed, dim=reduce_dims, keepdims=True)
            absolute_max_val = torch.amax(observed, dim=reduce_dims, keepdims=True)

        best = torch.full_like(
            absolute_min_val, torch.finfo(absolute_min_val.dtype).max
        )
        min_val = torch.ones_like(absolute_min_val)
        max_val = torch.zeros_like(absolute_max_val)
        for i in range(int(self.maxshrink * self.grid)):
            p = 1 - i / self.grid
            shrinked_min_val = p * absolute_min_val
            shrinked_max_val = p * absolute_max_val

            candidate_scales, candidate_zero_points = calculate_qparams(
                shrinked_min_val, shrinked_max_val, self.quantization_args
            )
            q = fake_quantize(
                observed,
                candidate_scales,
                candidate_zero_points,
                self.quantization_args,
            )

            q -= observed
            q.abs_()
            q.pow_(self.norm)
            if not reduce_dims:
                err = torch.sum(q)
            else:
                err = torch.sum(q, reduce_dims, keepdims=True)

            tmp = err < best
            if torch.any(tmp):
                best[tmp] = err[tmp]
                min_val[tmp] = shrinked_min_val[tmp]
                max_val[tmp] = shrinked_max_val[tmp]
        return min_val, max_val

    def calculate_qparams(
        self,
        observed: Tensor,
        reduce_dims: Optional[Tuple[int]] = None,
    ) -> Tuple[FloatTensor, IntTensor]:
        min_val, max_val = self.calculate_mse_min_max(observed, reduce_dims)

        return calculate_qparams(
            min_val, max_val, self.quantization_args
        )
        
    def get_qparams_along_dim(
        self,
        observed,
        dim: Union[int, Iterable[int]],
        tensor_id: Optional[Any] = None,
    ):
        if isinstance(dim, int):
            dim = [dim]
        dim = set(dim)

        reduce_dims = tuple(idx for idx in range(observed.ndim) if idx not in dim)
        return self.calculate_qparams(
            observed, reduce_dims=reduce_dims
        )



