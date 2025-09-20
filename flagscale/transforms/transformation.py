from abc import ABC, abstractmethod
from fnmatch import fnmatch
from typing import Iterable, Protocol, Tuple

from omegaconf import DictConfig
from torch import nn


class Selector(Protocol):
    """Protocol for target selectors.

    A selector inspects a given module scope and yields (qualified_name, submodule)
    pairs that a transformation should be applied to.
    """

    def __call__(self, scope: nn.Module) -> Iterable[Tuple[str, nn.Module]]: ...


class SelectSelf:
    """Selector that yields the provided scope itself."""

    def __call__(self, scope: nn.Module) -> Iterable[Tuple[str, nn.Module]]:
        yield "", scope


class ByType:
    """Select submodules by nn.Module types.

    Args:
        *types: One or more nn.Module classes (e.g., nn.Linear, nn.LayerNorm).
    """

    def __init__(self, *types: type):
        self._types: Tuple[type, ...] = tuple(types)

    def __call__(self, scope: nn.Module) -> Iterable[Tuple[str, nn.Module]]:
        for name, m in scope.named_modules():
            if name and isinstance(m, self._types):
                yield name, m


class ByName:
    """Select submodules by qualified name patterns (fnmatch).

    Args:
        *patterns: One or more fnmatch-style patterns (e.g., "blocks.*.mlp.*").
    """

    def __init__(self, *patterns: str):
        self._patterns: Tuple[str, ...] = tuple(patterns)

    def __call__(self, scope: nn.Module) -> Iterable[Tuple[str, nn.Module]]:
        for name, m in scope.named_modules():
            if name and any(fnmatch(name, p) for p in self._patterns):
                print(f"yield name: {name}")
                yield name, m


class Or:
    """Union of multiple selectors; deduplicates by module identity."""

    def __init__(self, *selectors: Selector):
        self._selectors: Tuple[Selector, ...] = tuple(selectors)

    def __call__(self, scope: nn.Module) -> Iterable[Tuple[str, nn.Module]]:
        seen: set[int] = set()
        for sel in self._selectors:
            for name, m in sel(scope):
                mid = id(m)
                if mid in seen:
                    continue
                seen.add(mid)
                yield name, m


def _resolve_types(type_names: Iterable[str]) -> Tuple[type, ...]:
    """Resolve string type names to torch.nn classes.

    Accepts names like "Linear", "LayerNorm", "torch.nn.Linear".
    """
    resolved: list[type] = []
    for n in type_names:
        leaf = n.split(".")[-1]
        cls = getattr(nn, leaf, None)
        if cls is None or not isinstance(cls, type):
            raise ValueError(f"Unknown nn.Module type: {n}")
        resolved.append(cls)
    return tuple(resolved)


def build_selector(config: DictConfig | None) -> Selector:
    """Build a selector from a simple dict config.

    Supported keys (preferred):
      - by_type: list[str] of nn.Module class names (e.g., ["torch.nn.Linear"]).
      - by_name: list[str] of fnmatch patterns over qualified names
        (e.g., ["unet.down_blocks.*.attentions.*"]).

    If both are provided, returns Or(ByType(...), ByName(...)).
    If none provided, defaults to SelectSelf().
    """
    if not config:
        return SelectSelf()

    sels: list[Selector] = []
    by_type = config.get("by_type") if isinstance(config, DictConfig) else None
    by_name = config.get("by_name") if isinstance(config, DictConfig) else None

    if by_type:
        sels.append(ByType(*_resolve_types(by_type)))
    if by_name:
        sels.append(ByName(*by_name))

    if not sels:
        return SelectSelf()
    if len(sels) == 1:
        return sels[0]
    return Or(*sels)


class Transformation(ABC):
    """
    Base transform class.

    A transform is a class that can be applied to a model.
    It can be used to modify the model in some way.

    For example, a transform can be used to apply a pre-processing step to the model,
    or to apply a post-processing step to the model.
    """

    def preflight(self) -> bool:
        """
        Check if a hardware/python package requirement is met.

        Returns:
            True if the hardware/python package requirement is met, False otherwise.
        """
        return True

    def targets(self, scope: nn.Module) -> Iterable[Tuple[str, nn.Module]]:
        """
        Enumerate target modules for this transformation.

        Args:
            scope: The module scope provided by the engine.

        Returns:
            Iterable of (qualified_name, module). Default: yields the scope itself.
        """
        selector: Selector = SelectSelf()
        return selector(scope)

    @abstractmethod
    def apply(self, model: nn.Module) -> bool:
        """
        Apply the transform to the model.

        Args:
            model: The model to apply the transform to.

        Returns:
            True if the transform is applied successfully, False otherwise.
        """
        raise NotImplementedError
