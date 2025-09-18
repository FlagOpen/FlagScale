from abc import ABC, abstractmethod

from torch import nn


class Transformation(ABC):
    """
    Base transform class.

    A transform is a class that can be applied to a model.
    It can be used to modify the model in some way.

    For example, a transform can be used to apply a pre-processing step to the model,
    or to apply a post-processing step to the model.
    """

    def supports(self, model: nn.Module) -> bool:
        """
        Check if the transform supports the model.

        Args:
            model: The model to check if the transform supports.

        Returns:
            True if the transform supports the model, False otherwise.
        """
        return True

    def preflight(self) -> bool:
        """
        Check if a hardware/python package requirement is met.

        Returns:
            True if the hardware/python package requirement is met, False otherwise.
        """
        return True

    @abstractmethod
    def apply(self, model: nn.Module) -> bool:
        """
        Apply the transform to the model.

        Args:
            model: The model to apply the transform to.

        Returns:
            True if the transform is applied successfully, False otherwise.
        """
        ...
