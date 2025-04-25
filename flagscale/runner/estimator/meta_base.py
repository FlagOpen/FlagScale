from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from flagscale.runner.estimator.meta_registry import get_registry


class MetaModule:
    """
    Base class for model components that track computational resources.

    This class provides automatic tracking of FLOPs, parameters, and activations
    through the ModelStatsRegistry system with hierarchical level information.
    Derived classes can either provide their own implementations of compute_* methods,
    or rely on the default behavior.
    """

    _path = None
    _counter = 0

    def __init__(self, shard_specs=None, model_id="default"):
        """
        Initialize a MetaModule with sharding specifications.

        Parameters:
        -----------
        shard_specs : list
            Specifications for tensor sharding
        model_id : str, optional
            Identifier for the model, used to retrieve the correct registry
        name : str, optional
            Name of this module component (defaults to class name + counter)
        """
        self.shard_specs = shard_specs
        self.model_id = model_id
        self.registry = get_registry(model_id)
        self.shared_params = False

    def add_flops(self, *args, **kwargs):
        """
        Add FLOPs to the registry with level information.

        Parameters:
        -----------
        *args
            Positional arguments for FLOP calculation
        level : bool, optional
            Whether to include level information in the tag
        **kwargs
            Keyword arguments for FLOP calculation

        Returns:
        --------
        int
            Number of FLOPs added
        """
        return 0

    def add_params(self, *args, **kwargs):
        """
        Add parameters to the registry with level information.

        Parameters:
        -----------
        *args
            Positional arguments for parameter calculation
        level : bool, optional
            Whether to include level information in the tag
        **kwargs
            Keyword arguments for parameter calculation

        Returns:
        --------
        int
            Number of parameters added
        """
        return 0

    def add_acts(self, *args, **kwargs):
        """
        Add activation elements to the registry with level information.

        Parameters:
        -----------
        *args
            Positional arguments for activation calculation
        level : bool, optional
            Whether to include level information in the tag
        **kwargs
            Keyword arguments for activation calculation

        Returns:
        --------
        int
            Number of activation elements added
        """
        return 0

    def get_flops(self):
        """
        Get previously computed FLOPs from module's registry.

        This method returns the tracked FLOPs from the module's registry rather
        than computing them. To compute and add FLOPs to the registry, use add_flops().

        Returns:
        --------
        int or float
            Number of floating-point operations previously recorded in registry
        """
        return self.registry.total_flops

    def get_params(self):
        """
        Get previously computed parameter count from module's registry.

        This method returns the tracked parameter count from the module's registry
        rather than computing them. To compute and add parameters to the registry,
        use add_params().

        Returns:
        --------
        int
            Number of parameters previously recorded in registry
        """
        return self.registry.total_params

    def get_acts(self):
        """
        Get previously computed activation memory from module's registry.

        This method returns the tracked activation memory from the module's registry
        rather than computing them. To compute and add activations to the registry,
        use add_acts().

        Returns:
        --------
        int
            Number of activation elements previously recorded in registry
        """
        return self.registry.total_acts

    def share_params(self):
        """
        Share parameters with another module.

        This method allows sharing parameters between two modules, which is useful
        for weight tying in some models. The derived class should implement this method
        to handle the sharing logic.

        Returns:
        --------
        MetaModule
            Reference to the current module for method chaining
        """
        self.shared_params = True

    def update_registry(self, *args, **kwargs):
        """
        Update registry with computed metrics.

        This method calculates metrics using the add_* methods and adds them
        to the registry with hierarchy information.

        Parameters:
        -----------
        *args, **kwargs
            Arguments passed to the add_* methods

        Returns:
        --------
        tuple
            (flops, params, acts) added to the registry
        """
        # Compute metrics with level information
        flops = self.add_flops(*args, **kwargs)
        params = 0
        if not self.shared_params:
            params = self.add_params(*args, **kwargs)
        acts = self.add_acts(*args, **kwargs)

        # Add to registry with appropriate tags
        self.registry.add_flops(flops, path=MetaModule._path)
        self.registry.add_params(params, path=MetaModule._path)
        self.registry.add_acts(acts, path=MetaModule._path)

        return (flops, params, acts)

    def __call__(self, *args, **kwargs):
        """
        Execute the module's operation with automatic resource tracking.

        This method:
        1. Validates model_id consistency across input tensors
        2. Updates the registry with computed metrics
        3. Calls the forward method that derived classes should implement
        4. Ensures output tensors have consistent model_ids

        Parameters:
        -----------
        *args, **kwargs
            Arguments to pass to the module implementation

        Returns:
        --------
        object
            Result of the module's operation

        Raises:
        -------
        ValueError
            If input tensors have inconsistent model_ids that cannot be resolved
        """
        import warnings

        from flagscale.runner.estimator.meta_tensor import MetaTensor

        # Track model_ids and their sources for better error reporting
        model_id_sources = {}

        # Helper function to process a potential MetaTensor argument
        def process_tensor(tensor, source_desc):
            if isinstance(tensor, MetaTensor):
                if tensor.model_id != "default":
                    if tensor.model_id not in model_id_sources:
                        model_id_sources[tensor.model_id] = []
                    model_id_sources[tensor.model_id].append(source_desc)
                    return tensor.model_id
            return None

        # Process positional arguments
        for i, arg in enumerate(args):
            if isinstance(arg, MetaTensor):
                process_tensor(arg, f"args[{i}]")
            elif isinstance(arg, list):
                # Handle lists of MetaTensor objects
                for j, item in enumerate(arg):
                    process_tensor(item, f"args[{i}][{j}]")

        # Process keyword arguments
        for key, value in kwargs.items():
            if isinstance(value, MetaTensor):
                process_tensor(value, f"kwargs['{key}']")
            elif isinstance(value, list):
                # Handle lists of MetaTensor objects
                for j, item in enumerate(value):
                    process_tensor(item, f"kwargs['{key}'][{j}]")

        # Determine which model_id to use
        target_model_id = self.model_id

        # If this module has default model_id but inputs have specific ones
        if target_model_id == "default" and model_id_sources:
            # If all inputs have the same non-default model_id, use that
            if len(model_id_sources) == 1:
                target_model_id = next(iter(model_id_sources.keys()))
                # Update module's model_id and registry
                self.model_id = target_model_id
                self.registry = get_registry(target_model_id)
            else:
                # If inputs have different model_ids, we have a conflict
                model_id_details = []
                for mid, sources in model_id_sources.items():
                    model_id_details.append(f"  - '{mid}' from {', '.join(sources)}")

                raise ValueError(
                    f"Module {self.__class__.__name__} received inputs with inconsistent model_ids:\n"
                    f"{chr(10).join(model_id_details)}\n"
                    f"Either use consistent model_ids or explicitly set the module's model_id."
                )

        # If module has non-default model_id but inputs have different ones
        elif (
            target_model_id != "default"
            and model_id_sources
            and target_model_id not in model_id_sources
        ):
            model_id_details = []
            for mid, sources in model_id_sources.items():
                model_id_details.append(f"  - '{mid}' from {', '.join(sources)}")

            raise ValueError(
                f"Module {self.__class__.__name__} has model_id '{target_model_id}', "
                f"but received inputs with different model_ids:\n"
                f"{chr(10).join(model_id_details)}\n"
                f"Model IDs must be consistent for proper metric tracking."
            )

        # Save parent path before modifying
        parent_path = MetaModule._path

        # Update global counter
        MetaModule._counter += 1

        name = f"{self.__class__.__name__}_{MetaModule._counter}"

        # Update path using the module's name (includes class name and instance ID)
        if parent_path is None:
            MetaModule._path = name
        else:
            MetaModule._path = f"{parent_path}/{name}"

        # Update registry with metrics
        self.update_registry(*args, **kwargs)

        # Call the implementation
        output = self.forward(*args, **kwargs)

        # Helper function to check and propagate model_id to output tensors
        def check_and_propagate_model_id(result, path="output"):
            if isinstance(result, MetaTensor):
                if result.model_id != "default" and result.model_id != target_model_id:
                    warnings.warn(
                        f"Module {self.__class__.__name__} (model_id='{target_model_id}') produced output at {path} "
                        f"with different model_id='{result.model_id}'. This may lead to incorrect metric tracking.",
                        UserWarning,
                    )
                # Always set to the correct model_id
                result.model_id = target_model_id
            elif isinstance(result, (list, tuple)):
                for i, item in enumerate(result):
                    check_and_propagate_model_id(item, f"{path}[{i}]")

        # Apply model_id check and propagation to outputs
        check_and_propagate_model_id(output)

        # Restore parent path when exiting
        MetaModule._path = parent_path

        return output

    def forward(self, *args, **kwargs):
        """
        Implement the actual module operation.

        Derived classes should override this method to provide their functionality.

        Parameters:
        -----------
        *args, **kwargs
            Arguments to the module

        Returns:
        --------
        object
            Result of the operation

        Raises:
        -------
        NotImplementedError
            If the derived class doesn't implement this method
        """
        raise NotImplementedError(f"Class {self.__class__.__name__} must implement forward method")


@dataclass
class ModelConfig:
    """
    Base configuration class for model architectures.

    This class serves as a base for all model configurations, providing
    common parameters and type hints for model initialization.
    Model-specific configurations should inherit from this class.
    """

    # Model identification
    model_id: str = "default"

    # Training configuration
    batch_size: int = 1
    seq_length: int = 512

    # Parallelism configuration
    data_parallel_size: int = 1
    tensor_parallel_size: int = 1
    pipeline_parallel_size: int = 1
    expert_parallel_size: int = 1
    pipeline_rank: int = 0

    # Other common parameters
    dtype: str = "float16"

    # Additional model-specific configurations
    kwargs: Dict[str, Any] = field(default_factory=dict)
