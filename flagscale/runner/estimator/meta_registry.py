from dataclasses import dataclass, field
from typing import Any, Dict, Optional


class ModelStatsRegistry:
    """Registry for tracking model statistics with detailed operation logging and hierarchy."""

    def __init__(self, model_id=None):
        """
        Initialize a model statistics registry.

        Parameters:
        -----------
        model_id : str, optional
            Identifier for the model (defaults to "default")
        """
        self.model_id = model_id or "default"
        self.reset()

    def reset(self):
        """Reset all statistics and logs."""
        self.total_flops = 0
        self.total_params = 0
        self.total_acts = 0

        self.flops_logs = []
        self.params_logs = []
        self.acts_logs = []

        self.flops_counter = 0
        self.params_counter = 0
        self.acts_counter = 0

        self.flops_by_module = {}
        self.params_by_module = {}
        self.acts_by_module = {}

        self.module_ids = {}

    def add_flops(self, value, path=None):
        """
        Add FLOPs to the registry with path information.

        Parameters:
        -----------
        value : int or float
            Number of FLOPs to add
        path : str, optional
            Full path identifier for the module
        """
        if value < 0:
            raise ValueError(f"Cannot add negative FLOPs: {value}")

        self.total_flops += value

        # Increment counter for this operation type
        self.flops_counter += 1

        # Generate a path if none provided
        if path is None:
            path = f"unnamed_flop_{self.flops_counter}"

        # Calculate level from path (number of / characters)
        level = path.count("/")

        # Extract module ID from path for sorting
        # Format is typically: ModuleName_ID/SubmoduleName_ID/...
        try:
            module_name = path.split("/")[-1]
            module_id = int(module_name.split("_")[-1]) if "_" in module_name else 0
            self.module_ids[path] = module_id
        except (ValueError, IndexError):
            self.module_ids[path] = 999999  # Fallback ID for malformed paths

        # Update module-specific flops
        if path not in self.flops_by_module:
            self.flops_by_module[path] = 0
        else:
            raise ValueError(f"Path already exists: {path}")
        self.flops_by_module[path] += value

        # Log the operation with path, level and accumulated value
        self.flops_logs.append((value, path, level, self.total_flops))

    def add_params(self, value, path=None):
        """
        Add parameters to the registry with path information.

        Parameters:
        -----------
        value : int or float
            Number of parameters to add
        path : str, optional
            Full path identifier for the module
        """
        if value < 0:
            raise ValueError(f"Cannot add negative parameters: {value}")

        self.total_params += value

        # Increment counter for this operation type
        self.params_counter += 1

        # Generate a path if none provided
        if path is None:
            path = f"unnamed_param_{self.params_counter}"

        # Calculate level from path (number of / characters)
        level = path.count("/")

        # Extract module ID from path for sorting
        # Format is typically: ModuleName_ID/SubmoduleName_ID/...
        try:
            module_name = path.split("/")[-1]
            module_id = int(module_name.split("_")[-1]) if "_" in module_name else 0
            self.module_ids[path] = module_id
        except (ValueError, IndexError):
            self.module_ids[path] = 999999  # Fallback ID for malformed paths

        # Update module-specific parameters
        if path not in self.params_by_module:
            self.params_by_module[path] = 0
        self.params_by_module[path] += value

        # Log the operation with path, level and accumulated value
        self.params_logs.append((value, path, level, self.total_params))

    def add_acts(self, value, path=None):
        """
        Add activation elements to the registry with path information.

        Parameters:
        -----------
        value : int or float
            Number of activation elements to add
        path : str, optional
            Full path identifier for the module
        """
        if value < 0:
            raise ValueError(f"Cannot add negative activations: {value}")

        self.total_acts += value

        # Increment counter for this operation type
        self.acts_counter += 1

        # Generate a path if none provided
        if path is None:
            path = f"unnamed_act_{self.acts_counter}"

        # Calculate level from path (number of / characters)
        level = path.count("/")

        # Extract module ID from path for sorting
        # Format is typically: ModuleName_ID/SubmoduleName_ID/...
        try:
            module_name = path.split("/")[-1]
            module_id = int(module_name.split("_")[-1]) if "_" in module_name else 0
            self.module_ids[path] = module_id
        except (ValueError, IndexError):
            self.module_ids[path] = 999999  # Fallback ID for malformed paths

        # Update module-specific activations
        if path not in self.acts_by_module:
            self.acts_by_module[path] = 0
        self.acts_by_module[path] += value

        # Log the operation with path, level and accumulated value
        self.acts_logs.append((value, path, level, self.total_acts))

    def print_logs(self, metric_type=None, include_summary=False):
        """
        Print logs in hierarchical format while preserving module creation order within each level.

        Parameters:
        -----------
        metric_type : str or list, optional
            Type(s) of metrics to print. Can be "flops", "params", "acts",
            or a list containing any of these. If None, prints all metrics.
        include_summary : bool, optional
            Whether to include summary information at the end (defaults to False)
        """
        # Parse and validate metric_type
        all_metric_types = ["flops", "params", "acts"]

        if metric_type is None:
            metric_types = all_metric_types
        elif isinstance(metric_type, str):
            if metric_type not in all_metric_types:
                raise ValueError(
                    f"Invalid metric_type: {metric_type}. Must be one of {all_metric_types}"
                )
            metric_types = [metric_type]
        elif isinstance(metric_type, (list, tuple)):
            for m in metric_type:
                if m not in all_metric_types:
                    raise ValueError(f"Invalid metric_type: {m}. Must be one of {all_metric_types}")
            metric_types = metric_type
        else:
            raise ValueError(
                f"metric_type must be None, str, list, or tuple, got {type(metric_type)}"
            )

        # Print header
        metric_list = ", ".join(metric_types).upper()
        print(f"\n===== {metric_list} Statistics for '{self.model_id}' =====")

        # Collect all unique module paths with their levels and module IDs
        module_info = {}

        for logs, metric in [
            (self.flops_logs, "flops"),
            (self.params_logs, "params"),
            (self.acts_logs, "acts"),
        ]:
            if metric not in metric_types:
                continue

            for value, path, level, _ in logs:
                if path not in module_info:
                    # Extract module name (last part of path) for display
                    name = path.split("/")[-1] if "/" in path else path
                    module_info[path] = {
                        "level": level,
                        "name": name,
                        "path": path,
                        "parent_path": ("/".join(path.split("/")[:-1]) if "/" in path else ""),
                        "module_id": self.module_ids.get(path, 999999),
                    }

        # Organize modules into a hierarchy while preserving module ID order
        hierarchy = {}

        # First, identify all root modules (level 0 or no parent in module_info)
        root_modules = []
        for path, info in module_info.items():
            if info["level"] == 0 or not info["parent_path"]:
                root_modules.append(path)

        # Sort root modules by module ID
        root_modules.sort(key=lambda p: module_info[p]["module_id"])

        # Build hierarchy dictionary
        for root in root_modules:
            hierarchy[root] = self._build_module_hierarchy(root, module_info)

        # Calculate column widths
        max_name_len = max(
            [len(info["name"]) + info["level"] * 2 for info in module_info.values()], default=30
        )
        max_name_len = max(max_name_len, 30)  # Minimum width

        # Print column headers
        header = f"{'Module':<{max_name_len + 5}}"
        for metric in metric_types:
            header += f"{metric.upper():>15} {metric.upper()+'-TOTAL':>15}"
        print(f"\n{header}")
        print("-" * (max_name_len + 5 + len(metric_types) * 30))

        # Calculate accumulated metrics
        accumulated = self._calculate_accumulated_metrics()

        # Print hierarchy
        for root in root_modules:
            self._print_module_hierarchy(
                root, hierarchy[root], module_info, metric_types, max_name_len + 5, accumulated
            )

        # Print summary if requested
        if include_summary:
            print("\n===== Summary =====")
            for metric in metric_types:
                total = getattr(self, f"total_{metric}")
                print(f"Total {metric.upper()}: {total:,}")

    def _calculate_accumulated_metrics(self):
        """
        Calculate accumulated metrics for each module by summing its children's metrics.

        This is used to display total resource usage per module including its submodules.

        Returns:
        --------
        dict
            Dictionary with accumulated metrics for each metric type
        """
        accumulated = {"flops": {}, "params": {}, "acts": {}}

        # Get all module paths from all metric dictionaries
        all_paths = set()
        all_paths.update(self.flops_by_module.keys())
        all_paths.update(self.params_by_module.keys())
        all_paths.update(self.acts_by_module.keys())

        # Define a function to get all children of a path
        def get_children(path):
            return {p for p in all_paths if p.startswith(path + "/") or p == path}

        # Calculate accumulated metrics for each path
        for path in all_paths:
            children = get_children(path)

            # Sum up metrics from all children for each metric type
            for metric, module_dict in [
                ("flops", self.flops_by_module),
                ("params", self.params_by_module),
                ("acts", self.acts_by_module),
            ]:
                accumulated[metric][path] = sum(module_dict.get(child, 0) for child in children)

        return accumulated

    def _build_module_hierarchy(self, parent_path, module_info):
        """
        Recursively build a hierarchy of modules while preserving module ID order.

        Parameters:
        -----------
        parent_path : str
            Path of the parent module
        module_info : dict
            Dictionary of module information

        Returns:
        --------
        dict
            Hierarchical structure of modules
        """
        # Find all children of this parent
        children = []
        for path, info in module_info.items():
            if info["parent_path"] == parent_path:
                children.append(path)

        # Sort children by module ID
        children.sort(key=lambda p: module_info[p]["module_id"])

        # Build hierarchy recursively
        result = {}
        for child in children:
            result[child] = self._build_module_hierarchy(child, module_info)

        return result

    def _print_module_hierarchy(
        self, path, children, module_info, metric_types, name_width, accumulated, indent=0
    ):
        """
        Recursively print a module hierarchy with metrics.

        Parameters:
        -----------
        path : str
            Path of the current module
        children : dict
            Dictionary of child modules
        module_info : dict
            Dictionary of module information
        metric_types : list
            List of metric types to display
        name_width : int
            Width for the name column
        accumulated : dict
            Dictionary of accumulated metrics
        indent : int, optional
            Current indentation level
        """
        # Skip if module doesn't exist in module_info (should never happen)
        if path not in module_info:
            return

        info = module_info[path]

        # Create indentation
        indent_str = "  " * indent

        # Display name with indentation
        display_name = f"{indent_str}{info['name']}"

        # Prepare line with metrics
        line = f"{display_name:<{name_width}}"

        # Add metrics with their accumulated values
        has_metrics = False
        for metric in metric_types:
            module_dict = getattr(self, f"{metric}_by_module")
            direct_value = module_dict.get(path, 0)
            accum_value = accumulated[metric].get(path, 0)

            if direct_value > 0 or accum_value > 0:
                has_metrics = True

            # Format for display
            direct_str = f"{direct_value:,}" if direct_value > 0 else "-"
            accum_str = f"{accum_value:,}" if accum_value > 0 else "-"

            # Add to output line
            line += f"{direct_str:>15} {accum_str:>15}"

        # Print line if module has any metrics
        if has_metrics:
            print(line)

        # Recursively print children sorted by module_id
        for child_path in sorted(children.keys(), key=lambda p: module_info[p]["module_id"]):
            self._print_module_hierarchy(
                child_path,
                children[child_path],
                module_info,
                metric_types,
                name_width,
                accumulated,
                indent + 1,
            )


# Registry to store stats for different models
_model_registries = {}


def register_model(model_id="default"):
    """
    Register a model with a specific configuration.

    Parameters:
    -----------
    model_id : str
        Unique identifier for the model

    Returns:
    --------
    ModelStatsRegistry
        The newly created registry

    Raises:
    -------
    ValueError
        If model_id already exists in registry
    """
    global _model_registries
    if model_id in _model_registries:
        if model_id == "default":
            return _model_registries[model_id]
        else:
            raise ValueError(f"Model ID {model_id} already exists in registry")
    _model_registries[model_id] = ModelStatsRegistry(model_id)
    return _model_registries[model_id]


def get_registry(model_id="default"):
    """
    Get or create a registry for the specified model.

    If the specified model_id doesn't exist but "default" does exist,
    returns the default registry.

    Parameters:
    -----------
    model_id : str, optional
        Identifier for the model (defaults to "default")

    Returns:
    --------
    ModelStatsRegistry
        Registry for the specified model or default registry
    """
    global _model_registries

    # Remove debug print statement

    # Ensure default registry exists
    if "default" not in _model_registries:
        _model_registries["default"] = ModelStatsRegistry("default")

    # Return requested registry if it exists, otherwise default
    if model_id not in _model_registries:
        raise ValueError(f"Model ID {model_id} not found in registry")

    return _model_registries[model_id]


def reset_registry(model_id="default"):
    """
    Reset the registry for the specified model.

    Parameters:
    -----------
    model_id : str, optional
        Identifier for the model (defaults to "default")

    Raises:
    -------
    ValueError
        If model_id not found in registry
    """
    global _model_registries

    # Ensure default registry exists
    if "default" not in _model_registries:
        _model_registries["default"] = ModelStatsRegistry("default")

    # Reset requested registry
    if model_id not in _model_registries:
        raise ValueError(f"Model ID {model_id} not found in registry")
    _model_registries[model_id].reset()


def remove_registry(model_id="default"):
    """
    Remove the registry for the specified model.

    Parameters:
    -----------
    model_id : str, optional
        Identifier for the model (defaults to "default")

    Raises:
    -------
    ValueError
        If model_id not found in registry
    """
    global _model_registries

    # Ensure default registry exists
    if "default" not in _model_registries:
        _model_registries["default"] = ModelStatsRegistry("default")

    # Remove requested registry
    if model_id not in _model_registries:
        raise ValueError(f"Model ID {model_id} not found in registry")
    del _model_registries[model_id]
