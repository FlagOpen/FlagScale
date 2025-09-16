import torch
import torch.nn as nn
from typing import Any, Callable, List, Optional, Tuple, Type
from dataclasses import dataclass, field

from flagscale.train.peft.peft import PEFT, AdapterWrapper
from flagscale.train.peft.utils import (
    get_adapter_attributes_from_linear,
    is_expert_linear,
    ParallelLinearAdapter,
    match_module,
)


class LoRALinear(AdapterWrapper):
    """An adapter wrapper that adds the output of the adapter to the output of the wrapped module.

    This class is designed to be used with LoRA (Low-Rank Adaptation) and similar techniques
    where the adapter's output is added to the main module's output. It extends the AdapterWrapper
    class to provide a specific implementation of the forward method.
    """

    def forward(
        self,
        x: torch.Tensor,
        *args,
        **kwargs,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        # pylint: disable=C0115,C0116
        linear_output, bias, layernorm_output = self.base_linear_forward(x, *args, **kwargs)
        adapter_output = self.adapter(layernorm_output.contiguous())
        adapter_output = adapter_output.reshape(linear_output.shape)
        return linear_output + adapter_output, bias



class LoRA(PEFT):
    """
    Implements the LoRA (Low-Rank Adaptation) module for parameter-efficient fine-tuning.

    LoRA uses a low-rank projection to adapt the weights of a pre-trained model to a new downstream task.
    This class facilitates the application of LoRA to specific modules within the model architecture.
    """
    def __init__(
        self,
        target_modules: list[str] = ['linear_qkv', 'linear_proj', 'linear_fc1', 'linear_fc2'],
        dim: int = 32,
        alpha: int = 64,
        dropout: float = 0.0,
        dropout_position: str = 'pre',
        in_init_method: str = 'xavier',
        out_init_method: str = 'zero',
    ):
        self.target_modules = target_modules
        self.dim = dim
        self.alpha = alpha
        self.dropout = dropout
        self.dropout_position = dropout_position
        self.in_init_method = in_init_method
        self.out_init_method = out_init_method

    def transform(self, m: nn.Module, name=None, prefix=None):
        """
        Applies LoRA to a specific module within the model architecture.

        Args:
            m (nn.Module): The module to apply LoRA to.
            name (str, optional): Name of the module (if applicable). Defaults to None.

        Returns:
            nn.Module: The modified module with LoRA applied, or the original module if not a target.
        """
        if (ans := match_module(m, name, prefix, self.target_modules)) is not None:
            (match, full_name) = ans

            input_is_parallel, in_features, out_features, disable_sp_comm, base_linear_is_parallel = (
                get_adapter_attributes_from_linear(m)
            )

            adapter = ParallelLinearAdapter(
                in_features=in_features,
                out_features=out_features,
                dim=self.dim,
                model_parallel_config=getattr(m, "config", None),
                gather_output=False,
                input_is_parallel=input_is_parallel,
                is_expert=is_expert_linear(full_name),
                in_init_method=self.in_init_method,
                out_init_method=self.out_init_method,
                dropout=self.dropout,
                alpha=self.alpha,
                dropout_position=self.dropout_position,
                disable_sequence_parallel_comm=disable_sp_comm,
            )

            lora_linear = LoRALinear(m, adapter)
            return lora_linear
            
        return m
