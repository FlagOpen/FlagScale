from abc import ABC, abstractmethod
from typing import Callable, Dict, Iterable

from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from torch import nn

from flagscale.runner.utils import logger


class BaseAdapter(ABC):
    """
    Base class for all model adapters.

    An `Adapter` is aimed to provide an stable interface for the `Tranform`s to be applied to the model.
    """

    def __init__(self, model_or_pipeline: nn.Module | DiffusionPipeline) -> None:
        """
        Args:
            model_or_pipeline: The model or pipeline to be adapted.
        """
        self._model = model_or_pipeline
        # A dict of capabilities, which will be accessed during `Transform`s.
        self._caps: Dict[str, Callable] = {}

    @abstractmethod
    def backbone(self) -> nn.Module | Iterable[nn.Module]:
        """Get the diffusion model backbone. It could be a UNet (such as `UNet2DConditionModel` in
         `StableDiffusionPipeline`), a ModuleList of DiT blocks, etc.

        It should be the main model we'll be applying the `Transform`s to.

        Returns:
            nn.Module | Iterable[nn.Module]: The backbone modules.
        """
        ...

    def register_capability(self, name: str, capability: Callable) -> None:
        """Register a callable capability under the given name.

        Args:
            name: The name of the capability.
            capability: The callable capability.
        """
        if name in self._caps:
            logger.warning(f"Capability {name} is already registered.")
            return
        self._caps[name] = capability

    def get_capability(self, name: str) -> Callable | None:
        """
        Get the capability under the given name.

        Args:
            name: The name of the capability.

        Returns:
            Callable | None: The capability callable or None if not found.
        """
        return self._caps.get(name)

    def has_capability(self, name: str) -> bool:
        """
        Check if the capability is registered.

        Args:
            name: The name of the capability.

        Returns:
            bool: True if the capability is registered, False otherwise.
        """
        return name in self._caps

    # TODO(yupu): Each type may have its own saving method (image, video, protein sequence)
    def save(self) -> bool:
        return True
