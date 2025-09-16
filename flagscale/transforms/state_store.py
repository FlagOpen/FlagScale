# Modified from https://github.com/huggingface/diffusers/blob/4a7556eaecc9872dea50ce161301edfa6392693c/src/diffusers/hooks/hooks.py

from typing import Dict, Generic, Optional, Type, TypeVar

S = TypeVar("S")


class StateStore(Generic[S]):
    """A state store that manages states for different contexts."""

    def __init__(self, state_cls: Type[S], init_args=None, init_kwargs=None) -> None:
        self._state_cls: Type[S] = state_cls
        # TODO(yupu): Fixed args/kwargs?
        self._init_args = init_args if init_args is not None else ()
        self._init_kwargs = init_kwargs if init_kwargs is not None else {}
        # TODO(yupu): Do we need to distinguish between global and module-specific states?
        # Mapping: state context name -> state.
        # State can be shared across modules or by a single module.
        self._state_by_context: Dict[str, S] = {}
        # The state context name that is currently in use.
        self._active_scope: Optional[str] = None

    def get_or_create_state(self) -> S:
        """Get or create a state for the current context.

        Returns:
            The state for the current context.

        Raises:
            ValueError: If no state context is currently in use.
        """
        if self._active_scope is None:
            raise ValueError(
                "No state context is currently in use. Please set a state context first."
            )
        if self._active_scope not in self._state_by_context:
            self._state_by_context[self._active_scope] = self._state_cls(
                *self._init_args, **self._init_kwargs
            )
        return self._state_by_context[self._active_scope]

    def set_scope(self, name: str):
        """Set the current state context.

        Args:
            name: The name of the state context to set.
        """
        self._active_scope = name

    # TODO(yupu): Reset?
