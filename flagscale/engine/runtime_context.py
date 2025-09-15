import contextlib
import contextvars
import uuid

from typing import Any, Callable, ContextManager, Dict, Optional

# The current context in use
_current_ctx: contextvars.ContextVar["RuntimeContext | None"] = contextvars.ContextVar(
    "flagscale_runtime_ctx", default=None
)


class RuntimeContext:
    """Context shared across a model's forward pass.
    `Transform`s could write/read information from this context.
    """

    def __init__(self):
        # TODO(yupu): Do we need this?
        self.run_id: str = uuid.uuid4().hex
        # Extra information that could be set/get by `Transform`s.
        # TODO(yupu): Actually use this and see if it fits. Print the content for debugging when errors happen.
        self.extras: Dict[str, Any] = {}
        # A provider of state context names. The context names could be used by `Transform`s to access different part of the state stores.
        self.state_ctx_provider: Optional[Callable[[], str | None]] = None

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

    @classmethod
    def is_active(cls) -> bool:
        """Check if an active context is available.

        Returns:
            bool: True if an active context is available, False otherwise.
        """

        return cls.current() is not None

    @property
    def state_ctx(self) -> Optional[str]:
        """Get the current state context name.

        Returns:
            Optional[str]: The current state context name, or None if no provider is set.
        """

        p = self.state_ctx_provider
        return p() if callable(p) else None


def current_ctx() -> Optional[RuntimeContext]:
    """Optional module-level alias for convenience."""

    return RuntimeContext.current()
