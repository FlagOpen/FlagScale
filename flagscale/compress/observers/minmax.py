from typing import Any, Iterable, Optional, Tuple, Union
import torch
from torch import FloatTensor, IntTensor, Tensor
from compressed_tensors.quantization.utils import calculate_qparams
from compressed_tensors.quantization.quant_args import QuantizationArgs
from .base import Observer
from compressed_tensors.registry.registry import _REGISTRY, _ALIAS_REGISTRY

_REGISTRY[Observer]["moving_minmax"] = _REGISTRY[Observer]["minmax"]
_REGISTRY[Observer].pop("minmax")
_ALIAS_REGISTRY[Observer]["moving_minmax"] = "moving_minmax"
_ALIAS_REGISTRY[Observer].pop("minmax")

@Observer.register("minmax")
class MinMaxObserver(Observer):
    def __init__(
        self, quantization_args: QuantizationArgs
    ):
        super().__init__(quantization_args=quantization_args)

    def calculate_qparams(
        self,
        observed: Tensor,
        reduce_dims: Optional[Tuple[int]] = None,
    ) -> Tuple[FloatTensor, IntTensor]:
        if not reduce_dims:
            min_val, max_val = torch.aminmax(observed)
        else:
            min_val = torch.amin(observed, dim=reduce_dims, keepdims=True)
            max_val = torch.amax(observed, dim=reduce_dims, keepdims=True)

        return calculate_qparams(
            min_val, max_val, self.quantization_args
        )
        
    def get_qparams_along_dim(
        self,
        observed,
        dim: Union[int, Iterable[int]],
    ):
        if isinstance(dim, int):
            dim = [dim]
        dim = set(dim)

        reduce_dims = tuple(idx for idx in range(observed.ndim) if idx not in dim)
        return self.calculate_qparams(
            observed, reduce_dims=reduce_dims
        )
 