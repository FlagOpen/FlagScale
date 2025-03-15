from typing import Any, Dict

from flagscale.runner.estimator.meta_base import (
    ModelConfig,
    get_registry,
    register_model,
)


def compute_memory(config: ModelConfig, params: int, acts: int) -> int:
    """
    Compute memory requirements for the model.

    Parameters:
    -----------
    config : ModelConfig
        Model configuration
    params : int
        Number of parameters
    acts : int
        Number of activations

    Returns:
    --------
    int
        Memory requirement in bytes
    """
    # Determine parameter bytes based on dtype
    dtype_bytes = {
        "fp32": 4,
        "fp16": 2,
        "bf16": 2,
        "float32": 4,
        "float16": 2,
        "bfloat16": 2,
    }

    # Use bf16 as default if dtype is not specified or not recognized
    param_dtype = getattr(config, "dtype", "bf16")
    param_bytes = dtype_bytes.get(param_dtype, 2)

    # Check if distributed optimizer is enabled
    use_distributed_optimizer = getattr(config, "use_distributed_optimizer", False)
    data_parallel_size = getattr(config, "data_parallel_size", 1)

    # Calculate parameter memory based on optimizer type and data type
    if use_distributed_optimizer:
        # Distributed optimizer memory formulas based on README.md
        if param_dtype in ["fp16", "float16"]:
            # fp16 param, fp16 grads: 4 + 16/d bytes per parameter
            param_memory = params * (4 + 16 / data_parallel_size)
        elif param_dtype in ["bf16", "bfloat16"]:
            # bf16 param, fp32 grads: 6 + 12/d bytes per parameter
            param_memory = params * (6 + 12 / data_parallel_size)
        elif param_dtype in ["fp32", "float32"]:
            # fp32 param, fp32 grads: 8 + 8/d bytes per parameter
            param_memory = params * (8 + 8 / data_parallel_size)
        else:
            # Default to bf16 formula if unknown dtype
            param_memory = params * (6 + 12 / data_parallel_size)
    else:
        # Non-distributed optimizer memory formulas based on README.md
        if param_dtype in ["fp16", "float16"]:
            # fp16 param, fp16 grads: 20 bytes per parameter
            param_memory = params * 20
        elif param_dtype in ["bf16", "bfloat16"]:
            # bf16 param, fp32 grads: 18 bytes per parameter
            param_memory = params * 18
        elif param_dtype in ["fp32", "float32"]:
            # fp32 param, fp32 grads: 16 bytes per parameter
            param_memory = params * 16
        else:
            # Default to bf16 formula if unknown dtype
            param_memory = params * 18

    # Determine activation bytes based on model dtype
    act_bytes = dtype_bytes.get(param_dtype, 2)

    # Dynamic memory for activations
    act_memory = acts * act_bytes

    return param_memory, act_memory


def print_banner(text: str) -> None:
    """
    Print a banner with the specified text.

    Parameters:
    -----------
    text : str
        Text to display in banner
    """
    width = 60
    print("\n" + "=" * width)
    print(f"{text.center(width)}")
    print("=" * width)


def print_results(results: Dict[str, Any], show_details: bool = False) -> None:
    """
    Print resource estimation results in a formatted table.

    Parameters:
    -----------
    results : dict
        Dictionary with resource estimates
    show_details : bool, optional
        Whether to show detailed breakdown
    """
    model_id = results["model_id"]
    params = results["model_size"]
    flops = results["flops"]
    params_memory = results["params_memory"]
    act_memory = results["activation_memory"]
    total_memory = results["total_memory"]

    # Convert to GB for display
    params_memory_gb = params_memory / (1024**3)
    act_memory_gb = act_memory / (1024**3)
    total_memory_gb = total_memory / (1024**3)

    # Format model name for display
    display_name = model_id.upper()

    # Print main banner
    print_banner(f"{display_name} MODEL ESTIMATION")

    # Model size information
    print("\nMODEL SIZE:")
    print(f"Parameters:        {params/1e9:.3f} B")

    # Compute information
    print("\nCOMPUTATION:")
    print(f"FLOPs:            {flops/1e9:.3f} B")

    # Memory usage
    print("\nMEMORY USAGE:")
    if params_memory > 0:  # Only show breakdown if we have it
        print(f"Parameter Memory: {params_memory_gb:.2f} GB")
        print(f"Activation Memory: {act_memory_gb:.2f} GB")
    print(f"Total Memory:     {total_memory_gb:.2f} GB")

    if show_details:
        # Get registry with detailed metrics
        registry = get_registry(model_id)
        print("\nDETAILED BREAKDOWN:")
        registry.print_logs()
