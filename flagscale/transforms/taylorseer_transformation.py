# Modified from
# https://github.com/vipshop/cache-dit/blob/v0.3.0/src/cache_dit/cache_factory/cache_contexts/taylorseer.py

import math

from typing import Any, Iterable, List, Optional, Tuple, Union

import torch
import torch.nn as nn

from omegaconf import DictConfig

from flagscale.inference.runtime_context import current_ctx
from flagscale.transforms.hook import ModelHook, ModuleHookRegistry
from flagscale.transforms.state_store import BaseState, StateStore
from flagscale.transforms.transformation import Selector, Transformation, build_selector


class TaylorSeerState(BaseState):
    """State for TaylorSeer approximation.

    Holds the most recent exact-forward output and divided-difference terms
    for a Taylor expansion. Can operate with either index-based spacing
    (diffusion step indices) or real scheduler timestep deltas.
    """

    def __init__(
        self,
        order: int,
        warmup_steps: int,
        skip_interval_steps: int,
        use_timestep_delta: bool = False,
    ) -> None:
        # Order of the Taylor series
        self.order: int = order
        # Number of full computation steps before approximating the output for the first time
        self.warmup_steps: int = warmup_steps
        # Maximum number of steps to approximate the output before doing full computation again
        self.skip_interval_steps: int = skip_interval_steps
        # Whether to use real scheduler timestep deltas instead of index deltas
        self.use_timestep_delta: bool = use_timestep_delta

        # Taylor-series derivatives of the previous inference step
        self.previous_derivatives: List[torch.Tensor] = [None] * self.order
        # Taylor-series derivatives of the current inference step
        self.current_derivatives: List[torch.Tensor] = [None] * self.order
        # Last timestep where the model actually did inference
        self.previous_forward_step: int = -1
        # Last actual scheduler timestep value (float) when we last did exact forward.
        # Negative value indicates "no previous exact forward" (sentinel).
        self.previous_forward_time: float = -1.0

        self._is_tuple: bool = False

        assert self.order > 0 and self.warmup_steps > 0 and self.skip_interval_steps > 0

    def approximate_derivative(
        self, output: torch.Tensor, step: int, timestep: float
    ) -> List[torch.Tensor]:
        """Update divided differences for the Taylor series.

        Args:
            output: The module output at the current step.
            step: Diffusion step index for the current update.
            timestep: Raw scheduler timestep value (may be a scalar tensor or
                float); used only when use_timestep_delta is enabled.

        Returns:
            Updated derivative list (0..order-1), where index 0 is the latest
            output and higher orders are divided differences.
        """

        if self.use_timestep_delta:
            if self.previous_forward_time < 0 or timestep is None:
                distance = None
            else:
                distance = timestep - self.previous_forward_time
        else:
            if self.previous_forward_step < 0:
                distance = None
            else:
                distance = step - self.previous_forward_step

        current: List[torch.Tensor] = [None] * self.order
        current[0] = output
        for i in range(self.order - 1):
            if self.previous_derivatives[i] is not None and distance not in (None, 0):
                current[i + 1] = (current[i] - self.previous_derivatives[i]) / distance
            else:
                break

        return current

    def update(self, output: Union[torch.Tensor, Tuple[torch.Tensor, ...]]) -> None:
        """Ingest an exact-forward output and refresh series terms.

        Args:
            output: Tensor (or single-tensor tuple) produced by the module at
                the current diffusion step.
        """
        if isinstance(output, tuple):
            assert len(output) == 1, "Tuple with multiple elements are not supported"
            output = output[0]
            self._is_tuple = True

        ctx = current_ctx()
        timestep_index: int = ctx.timestep_index
        timestep: float = ctx.timestep

        self.previous_derivatives = self.current_derivatives
        self.current_derivatives = self.approximate_derivative(output, timestep_index, timestep)
        self.previous_forward_step = timestep_index
        self.previous_forward_time = timestep

    def approximate_output(self) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
        """Approximate the module output using the Taylor series.

        Returns:
            A tensor approximating the next output; if the last exact forward
            observed a single-tensor tuple, returns a tuple with one tensor to
            match the original signature.
        """

        ctx = current_ctx()
        timestep_index: int = ctx.timestep_index
        timestep: float = ctx.timestep

        if self.use_timestep_delta:
            if self.previous_forward_time < 0 or timestep is None:
                elapsed = 0
            else:
                elapsed = timestep - self.previous_forward_time
        else:
            if self.previous_forward_step < 0:
                elapsed = 0
            else:
                elapsed = timestep_index - self.previous_forward_step
        output = 0
        for i, derivative in enumerate(self.current_derivatives):
            if derivative is not None:
                output += (1 / math.factorial(i)) * derivative * (elapsed**i)
            else:
                break
        return (output,) if self._is_tuple else output

    def needs_exact_forward(self) -> bool:
        """Decide whether to run an exact forward at the current step.

        Returns:
            True if in warmup or at the configured skip cadence; False if the
            next output can be approximated from the series.
        """

        ctx = current_ctx()
        timestep_index: int = ctx.timestep_index
        if timestep_index == -1:
            raise ValueError("Timestep index is not set")

        return (
            timestep_index < self.warmup_steps
            or (timestep_index - self.warmup_steps + 1) % self.skip_interval_steps == 0
        )

    def reset(self, *args, **kwargs):
        """Reset the state of the TaylorSeer."""
        self.previous_derivatives = [None] * self.order
        self.current_derivatives = [None] * self.order
        self.previous_forward_step = -1


class TaylorSeerHook(ModelHook):
    """Hook that swaps the module's forward for TaylorSeer approximation."""

    def __init__(self, state_store: StateStore):
        """Initialize the hook.

        Args:
            state_store: Per-module store used to retrieve the TaylorSeer
                state under the current scope (e.g., uncond/cond).
        """

        super().__init__()
        self.state_store = state_store
        self.register_stateful(state_store)

    def custom_forward(self, _module: nn.Module, *args, **kwargs) -> Any:
        """TaylorSeer-aware forward.

        Runs the original forward when needed to refresh the series terms;
        otherwise returns the extrapolated output.
        """

        state = self.state_store.get_or_create_state()

        if state.needs_exact_forward():
            output = self.fn_ref.original_forward(*args, **kwargs)
            state.update(output)
            return output

        return state.approximate_output()


class TaylorSeerTransformation(Transformation):
    """Apply [TaylorSeer](https://github.com/Shenyi-Z/TaylorSeer) to selected submodules.

    Approximates outputs between exact forwards via a Taylor series in either
    diffusion-step index or raw scheduler timestep domain. Currently supports
    modules that return a tensor or a single-tensor tuple.
    """

    def __init__(
        self,
        order: int,
        warmup_steps: int,
        skip_interval_steps: int,
        targets: Optional[DictConfig] = None,
        use_timestep_delta: bool = False,
    ):
        """Initialize the transformation.

        Args:
            order: Taylor series order (>=1). order=1 is zero-th term only
                (hold); order=2 includes first derivative (linear), etc.
            warmup_steps: Number of initial exact forwards before starting to
                approximate.
            skip_interval_steps: Maximum number of successive approximations
                between exact forwards (after warmup).
            targets: Target selector config, e.g., by_name patterns or by_type.
            use_timestep_delta: If True, use raw scheduler deltas instead of
                index deltas for the Taylor domain.
        """

        super().__init__()

        self._order = order
        self._warmup_steps = warmup_steps
        self._skip_interval_steps = skip_interval_steps

        self._selector: Selector = build_selector(targets)
        self._use_timestep_delta: bool = use_timestep_delta

    def targets(self, scope: nn.Module) -> Iterable[Tuple[str, nn.Module]]:
        """Enumerate target modules for this transformation.

        Args:
            scope: Module tree to scan for matches.

        Returns:
            Iterator of (qualified_name, module) pairs.
        """

        return self._selector(scope)

    def apply(self, module: nn.Module) -> bool:
        """Attach a TaylorSeer hook with a fresh per-module state store.

        Args:
            module: Submodule to modify.

        Returns:
            True if successfully attached.
        """

        reg = ModuleHookRegistry.get_or_create_registry(module)

        state_store = StateStore(
            TaylorSeerState,
            init_kwargs={
                "order": self._order,
                "warmup_steps": self._warmup_steps,
                "skip_interval_steps": self._skip_interval_steps,
                "use_timestep_delta": self._use_timestep_delta,
            },
        )

        hook = TaylorSeerHook(state_store)
        reg.register_hook(hook, "taylorseer")

        return True
