import contextlib
import contextvars
import uuid

from typing import Callable, ContextManager, Optional

# The current context in use
_current_ctx: contextvars.ContextVar["RuntimeContext | None"] = contextvars.ContextVar(
    "flagscale_runtime_ctx", default=None
)


class RuntimeContext:
    """Context shared across a model's forward pass.
    `Transformation`s could write/read information from this context.
    """

    def __init__(self):
        # TODO(yupu): Do we need this?
        self.run_id: str = uuid.uuid4().hex
        # A provider of state context names. The context names could be used by `Transformation`s to access different parts of the state stores.
        self.state_scope_provider: Optional[Callable[[], str | None]] = None

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


def current_ctx() -> Optional[RuntimeContext]:
    """Optional module-level alias for convenience."""

    return RuntimeContext.current()
