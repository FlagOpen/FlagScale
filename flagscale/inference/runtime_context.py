import contextlib
import contextvars
import itertools

from typing import Callable, ContextManager, List, Optional, Union

import torch

# The current context in use
_current_ctx: contextvars.ContextVar["RuntimeContext | None"] = contextvars.ContextVar(
    "flagscale_runtime_ctx", default=None
)


class RuntimeContext:
    """Context shared across a model's forward pass.
    `Transformation`s could write/read information from this context.
    """

    def __init__(self, state_scopes: Optional[List[str]] = None, num_timesteps: int = -1):
        if state_scopes is not None and len(state_scopes) > 0:
            # A provider of state scope names.
            # If set, the scope names would be used by `Transformation`s to access different streams of the `StateStore`.
            # The scope names will be cycled through in order indefinitely.
            self.state_scope_provider: Optional[Callable[[], str | None]] = itertools.cycle(
                state_scopes
            ).__next__
        else:
            self.state_scope_provider: Optional[Callable[[], str | None]] = None

        # ==========================================
        #           DIFFUSION MODEL SETTINGS
        # ==========================================
        # Number of diffusion timesteps
        self.num_timesteps: int = num_timesteps
        # The current timestep retrieved from the root module's input
        self.timestep: Optional[Union[torch.Tensor, int, float]] = None
        # The current timestep index. Range from 0 to num_timesteps - 1
        self.timestep_index: int = -1

    @contextlib.contextmanager
    def session(self) -> ContextManager["RuntimeContext"]:
        """Activate this context for the current call stack (process-local).
        Use once around the model call in each worker/rank.
        """

        token = _current_ctx.set(self)
        try:
            yield self
        finally:
            _current_ctx.reset(token)

    @classmethod
    def current(cls) -> Optional["RuntimeContext"]:
        """Get the current active context.

        Returns:
            Optional[RuntimeContext]: The current active context, or None if no context is active.
        """

        return _current_ctx.get()

    @property
    def state_scope(self) -> Optional[str]:
        """Get the current state context name.

        Returns:
            Optional[str]: The current state context name, or None if no provider is set.
        """

        p = self.state_scope_provider
        return p() if callable(p) else None

    def update_timestep(self, t: Union[torch.Tensor, int, float]) -> None:
        """Update the current timestep only when a new timestep arrives.

        Args:
            t (Union[torch.Tensor, int, float]): The new timestep.
        """

        if self.timestep != t:
            self.timestep = t
            self.timestep_index += 1
            print(f"update_timestep: {self.timestep_index}")

        assert self.timestep_index < self.num_timesteps, "`timestep_index` is out of range"


def current_ctx() -> Optional[RuntimeContext]:
    """Optional module-level alias for convenience."""

    return RuntimeContext.current()
