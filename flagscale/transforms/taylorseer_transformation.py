# Modified from https://github.com/vipshop/cache-dit/blob/v0.3.0/src/cache_dit/cache_factory/cache_contexts/taylorseer.py

import math

from typing import Any, List

import torch
import torch.nn as nn

from flagscale.inference.runtime_context import current_ctx
from flagscale.transforms.hook import ModelHook, ModuleHookRegistry
from flagscale.transforms.state_store import BaseState, StateStore
from flagscale.transforms.transformation import Transformation

# block level taylorseer

# module pattern name or class type

# https://docs.nvidia.com/nemo/automodel/latest/apidocs/nemo_automodel/nemo_automodel.components._peft.module_matcher.html
# learn how to use module matcher


"""
1. given wildcard pattern, find all the modules in the model via module matcher
2. for each module, apply the transformation
3. reset scope in the context for new pipe call

"""


class TaylorSeerState(BaseState):
    def __init__(self, order: int, warmup_steps: int, skip_interval_steps: int):
        # Order of the Taylor series
        self.order: int = order
        # Number of full computation steps before approximating the output for the first time
        self.warmup_steps: int = warmup_steps
        # Maximum number of steps to approximate the output before doing full computation again
        self.skip_interval_steps: int = skip_interval_steps

        # Taylor-series derivatives of the previous inference step
        self.previous_derivatives: List[torch.Tensor] = [None] * self.order
        # Taylor-series derivatives of the current inference step
        self.current_derivatives: List[torch.Tensor] = [None] * self.order
        # Last timestep where the model actually did inference
        self.previous_forward_step: int = -1

        assert self.order > 0 and self.warmup_steps > 0 and self.skip_interval_steps > 0

    def approximate_derivative(self, output: torch.Tensor, step: int):
        """Approximate the derivative of the taylor series"""

        distance = step - self.previous_forward_step

        current: List[torch.Tensor] = [None] * self.order
        current[0] = output
        for i in range(self.order - 1):
            if (
                self.previous_derivatives[i] is not None
                # The official implementation from https://github.com/Shenyi-Z/TaylorSeer/blob/main/TaylorSeer-Wan2.1/wan/taylorseer/taylorseer_utils/__init__.py seems to be buggy,
                # The derivatives are only computed after the warmup steps, so the condition is always true.
                # Here we follow the implementation from cache-dit
                and step > 1
            ):
                current[i + 1] = (current[i] - self.previous_derivatives[i]) / distance
            else:
                break

        return current

    def update(self, output: torch.Tensor) -> None:
        """Update the derivatives with the new output

        Args:
            output (torch.Tensor): The module output of the current timestep
        """

        ctx = current_ctx()
        step: int = ctx.timestep_index

        self.previous_derivatives = self.current_derivatives
        self.current_derivatives = self.approximate_derivative(output, step)
        self.previous_forward_step = step

    def approximate_output(self) -> torch.Tensor:
        """Approximate the moduleoutput by using the taylor series

        Returns:
            (torch.Tensor): The approximate module output
        """

        ctx = current_ctx()
        step: int = ctx.timestep_index

        elapsed = step - self.previous_forward_step
        output = 0
        for i, derivative in enumerate(self.current_derivatives):
            if derivative is not None:
                output += (1 / math.factorial(i)) * derivative * (elapsed**i)
            else:
                break
        return output

    # TODO(yupu): This should be the hook's responsibility
    def needs_exact_forward(self) -> bool:
        """Check if an exact forward is needed at the current step

        Returns:
            True if an exact forward is needed at the current step, False otherwise.
        """

        ctx = current_ctx()
        step: int = ctx.timestep_index
        if step == -1:
            raise ValueError("Timestep index is not set")

        return (
            step < self.warmup_steps
            or (step - self.warmup_steps + 1) % self.skip_interval_steps == 0
        )

    def reset(self, *args, **kwargs):
        self.previous_derivatives = [None] * self.order
        self.current_derivatives = [None] * self.order
        self.previous_forward_step = -1


# TODO(yupu): We may need multiple stores for different outputs
class TaylorSeerHook(ModelHook):
    def __init__(self, state_store: StateStore):
        super().__init__()
        self.state_store = state_store
        self.register_stateful(state_store)
        self.is_tuple = False

    def approximate_derivative(self, output: Any):
        # TODO(yupu): Support tuple output, do we need to know the index of the output?
        if isinstance(output, tuple):
            assert len(output) == 1, "Tuple with multiple elements are not supported"
            output_tensor = output[0]
            state = self.state_store.get_or_create_state()
            state.update(output_tensor)
            self.is_tuple = True
            # raise NotImplementedError("List and tuple are not supported")
        elif isinstance(output, torch.Tensor):
            state = self.state_store.get_or_create_state()
            state.update(output)
        else:
            raise NotImplementedError("Unsupported output type")

    def custom_forward(self, module: nn.Module, *args, **kwargs) -> Any:
        state = self.state_store.get_or_create_state()

        if state.needs_exact_forward():
            print("needs exact forward")
            output = self.fn_ref.original_forward(*args, **kwargs)
            self.approximate_derivative(output)
            return output
        if self.is_tuple:
            print("approximate output tuple")
            return (state.approximate_output(),)
        else:
            print("approximate output tensor")
            return state.approximate_output()
        # return state.approximate_output()


class TaylorSeerTransformation(Transformation):
    # TODO(yupu): Define patterns
    # TODO(yupu): module output index????
    def __init__(self, order: int, warmup_steps: int, skip_interval_steps: int):
        super().__init__()

        self.state_store = StateStore(
            TaylorSeerState,
            init_kwargs={
                "order": order,
                "warmup_steps": warmup_steps,
                "skip_interval_steps": skip_interval_steps,
            },
        )

    def supports(self, module: nn.Module) -> bool:
        # inspect model output type?
        pass

    def apply(self, module: nn.Module) -> bool:
        reg = ModuleHookRegistry.get_or_create_registry(module)
        hook = TaylorSeerHook(self.state_store)
        reg.register_hook(hook, "taylorseer")

        return True
