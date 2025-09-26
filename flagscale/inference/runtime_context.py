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

    def __init__(self, state_scopes: Optional[List[str]] = None):
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
        # The current timestep retrieved from the root module's input
        self.timestep: float = -1
        # The current timestep index. Range from 0 to num_inference_steps - 1
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

        scalar_t: Optional[float] = None
        if isinstance(t, torch.Tensor):
            if t.ndim == 0:
                scalar_t = float(t.item())
            else:
                scalar_t = float(t.flatten()[0].item())
        elif isinstance(t, (int, float)):
            scalar_t = float(t)
        else:
            raise ValueError(f"Unexpected timestep type: {type(t)}")

        # In case there are multiple model forward calls in the same timestep
        if self.timestep != scalar_t:
            self.timestep = scalar_t
            self.timestep_index += 1


def current_ctx() -> Optional[RuntimeContext]:
    """Optional module-level alias for convenience."""

    return RuntimeContext.current()
