#!/usr/bin/env python3

#!/usr/bin/env python3

import argparse
import json
import math
import os
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from flagscale.runner.estimator.meta_base import ModelConfig
from flagscale.runner.estimator.meta_gpt import GPTModel
from flagscale.runner.estimator.meta_registry import get_registry, register_model
from flagscale.runner.estimator.meta_tensor import MetaTensor
from flagscale.runner.estimator.utils import compute_memory, print_results

# Expanded model selection
MODEL_SIZES = {
    "gpt-345m": {
        "hidden_size": 1024,
        "num_layers": 24,
        "num_attention_heads": 16,
        "vocab_size": 50257,
        "max_position_embeddings": 1024,
    },
    "gpt-1.3b": {
        "hidden_size": 2048,
        "num_layers": 24,
        "num_attention_heads": 16,
        "vocab_size": 50257,
        "max_position_embeddings": 1024,
    },
    "gpt-6.7b": {
        "hidden_size": 4096,
        "num_layers": 32,
        "num_attention_heads": 32,
        "vocab_size": 50257,
        "max_position_embeddings": 2048,
    },
    "gpt-13b": {
        "hidden_size": 5120,
        "num_layers": 40,
        "num_attention_heads": 40,
        "vocab_size": 50257,
        "max_position_embeddings": 2048,
    },
    "gpt-30b": {
        "hidden_size": 7168,
        "num_layers": 48,
        "num_attention_heads": 56,
        "vocab_size": 50257,
        "max_position_embeddings": 2048,
    },
    "gpt-66b": {
        "hidden_size": 9216,
        "num_layers": 64,
        "num_attention_heads": 72,
        "vocab_size": 50257,
        "max_position_embeddings": 2048,
    },
    "gpt-175b": {
        "hidden_size": 12288,
        "num_layers": 96,
        "num_attention_heads": 96,
        "vocab_size": 50257,
        "max_position_embeddings": 2048,
    },
}


@dataclass
class GPTConfig(ModelConfig):
    """Configuration class for GPT model estimation."""

    # Core architecture parameters specific to GPT
    hidden_size: int = 768
    num_layers: int = 12
    num_attention_heads: int = 12
    vocab_size: int = 50257
    max_position_embeddings: int = 1024

    # Optional parameters with defaults
    ffn_hidden_size: Optional[int] = None
    head_dim: Optional[int] = None

    # Training parameters
    batch_size: int = 1
    seq_length: int = 1024
    dtype: str = "bf16"
    use_distributed_optimizer: bool = False

    # Parallelism parameters
    tensor_parallel_size: int = 1
    pipeline_parallel_size: int = 1
    pipeline_rank: Optional[int] = None
    expert_parallel_size: int = 1
    data_parallel_size: int = 1

    # Model behavior parameters
    activation_func: str = "gelu"
    layernorm_epsilon: float = 1e-5
    embedding_dropout: float = 0.1
    attention_dropout: float = 0.1
    hidden_dropout: float = 0.1
    norm_type: str = "layernorm"
    use_rotary_position_embeddings: bool = False
    use_post_attention_layernorm: bool = True
    add_linear_bias: bool = False
    untie_embeddings_and_output_weights: bool = False
    kv_channels: Optional[int] = None

    # Attention parameters
    attention_softmax_in_fp32: bool = True
    apply_query_key_layer_scaling: bool = True

    def __post_init__(self):
        """
        Initialize derived values and handle parameter relationships.

        This method calculates default values for optional parameters
        and ensures consistency between related parameters.
        """
        # Set default for ffn_hidden_size if not provided
        if self.ffn_hidden_size is None:
            self.ffn_hidden_size = 4 * self.hidden_size

        # Set default for head_dim if not provided
        if self.head_dim is None:
            self.head_dim = self.hidden_size // self.num_attention_heads

        # Set default for kv_channels if not provided
        if self.kv_channels is None:
            self.kv_channels = self.hidden_size // self.num_attention_heads

        # If pipeline_rank is None but pipeline_parallel_size > 1, set it to 0
        if self.pipeline_parallel_size > 1 and self.pipeline_rank is None:
            self.pipeline_rank = 0


def get_model_config(
    model_id: str,
    batch_size: int,
    seq_length: int,
    tensor_parallel_size: int = 1,
    pipeline_parallel_size: int = 1,
    pipeline_rank: int = 0,
    expert_parallel_size: int = 1,
    data_parallel_size: int = 1,
    dtype: str = "bf16",
    use_distributed_optimizer: bool = False,
) -> GPTConfig:
    """
    Get configuration for a predefined model size.

    Parameters:
    -----------
    model_id : str
        Model identifier (e.g., 'gpt-345m', 'gpt-175b')
    batch_size : int
        Batch size for estimation
    seq_length : int
        Sequence length for estimation
    tensor_parallel_size : int
        Tensor parallel size for distributed training
    pipeline_parallel_size : int
        Pipeline parallel size for distributed training
    pipeline_rank : Optional[int]
        Pipeline rank for current stage (None for full model simulation)
    expert_parallel_size : int
        Expert parallel size for MoE models
    data_parallel_size : int
        Data parallel size for distributed training
    dtype : str
        Data type for model parameters ("fp32", "bf16", "fp16")
    use_distributed_optimizer : bool
        Whether to use distributed optimizer

    Returns:
    --------
    GPTConfig
        Configuration object for the model

    Raises:
    -------
    ValueError
        If model_id is not recognized
    """
    if model_id not in MODEL_SIZES:
        valid_models = ", ".join(MODEL_SIZES.keys())
        raise ValueError(f"Unknown model: {model_id}. Valid models: {valid_models}")

    model_info = MODEL_SIZES[model_id]
    hidden_size = model_info["hidden_size"]

    # Calculate FFN hidden size (typically 4x hidden size)
    ffn_hidden_size = 4 * hidden_size

    return GPTConfig(
        hidden_size=hidden_size,
        num_layers=model_info["num_layers"],
        num_attention_heads=model_info["num_attention_heads"],
        vocab_size=model_info["vocab_size"],
        max_position_embeddings=model_info["max_position_embeddings"],
        batch_size=batch_size,
        seq_length=seq_length,
        ffn_hidden_size=ffn_hidden_size,
        tensor_parallel_size=tensor_parallel_size,
        pipeline_parallel_size=pipeline_parallel_size,
        pipeline_rank=pipeline_rank,
        expert_parallel_size=expert_parallel_size,
        data_parallel_size=data_parallel_size,
        dtype=dtype,
        use_distributed_optimizer=use_distributed_optimizer,
        activation_func="gelu",
        norm_type="layernorm",
    )


def estimate(config: GPTConfig, model_id: str = "gpt_model") -> Dict[str, Any]:
    """
    Estimate computational resources for a GPT model with given configuration.

    Parameters:
    -----------
    config : GPTConfig
        Configuration for the GPT model
    model_id : str, optional
        Identifier for the model registry

    Returns:
    --------
    dict
        Dictionary with resource estimates
    """
    # Register model in the registry
    register_model(model_id)

    # Create GPT model
    model = GPTModel(config, model_id=model_id)

    # Create input shapes
    batch_size = config.batch_size
    seq_length = config.seq_length

    # Create input tensors for the model
    input_ids = MetaTensor(
        shape=[batch_size, seq_length], shard_spec=[config.data_parallel_size, 1]
    )

    # Create attention mask for causal attention
    attention_mask = MetaTensor(
        shape=[batch_size, 1, seq_length, seq_length],
        shard_spec=[config.data_parallel_size, 1, 1, 1],
    )

    # Forward pass to compute metrics
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)

    # Get registry with accumulated metrics
    registry = get_registry(model_id)

    # Extract model information
    params = registry.total_params
    flops = registry.total_flops
    acts = registry.total_acts

    # Compute memory estimates
    params_memory, activation_memory = compute_memory(config, params, acts)

    # Scale metrics based on parallelism
    effective_params = params
    if config.pipeline_parallel_size > 1 and config.pipeline_rank is not None:
        # Only count parameters for this pipeline stage
        effective_params = params * config.pipeline_parallel_size

    return {
        "model_id": model_id,
        "model_size": params,
        "effective_model_size": effective_params,
        "flops": flops,
        "activations": acts,
        "params_memory": params_memory,
        "activation_memory": activation_memory,
        "total_memory": params_memory + activation_memory,
        "parallelism": {
            "tensor": config.tensor_parallel_size,
            "pipeline": config.pipeline_parallel_size,
            "data": config.data_parallel_size,
            "expert": config.expert_parallel_size,
        },
    }


def main():
    parser = argparse.ArgumentParser(
        description="Estimate computational resources for GPT models"
    )

    # Main options
    parser.add_argument(
        "--model",
        choices=list(MODEL_SIZES.keys()),
        default="gpt-345m",
        help="Predefined model size to estimate",
    )
    parser.add_argument("--bs", type=int, default=1, help="Batch size for estimation")
    parser.add_argument(
        "--seq", type=int, default=1024, help="Sequence length for estimation"
    )

    # Parallelism options
    parser.add_argument("--tp", type=int, default=1, help="Tensor parallel size")
    parser.add_argument("--dp", type=int, default=1, help="Data parallel size")
    parser.add_argument("--pp", type=int, default=1, help="Pipeline parallel size")
    parser.add_argument("--pp-rank", type=int, default=0, help="Pipeline parallel rank")
    parser.add_argument("--ep", type=int, default=1, help="Expert parallel size")

    # Model precision options
    parser.add_argument(
        "--dtype",
        choices=["fp32", "bf16", "fp16"],
        default="bf16",
        help="Data type for model parameters",
    )
    parser.add_argument(
        "--dist-opt", action="store_true", help="Use distributed optimizer"
    )

    # Output options
    parser.add_argument(
        "--details",
        action="store_true",
        help="Show detailed breakdown of resource usage",
    )

    # Custom model parameters
    parser.add_argument(
        "--hidden-size", type=int, default=None, help="Override hidden size"
    )
    parser.add_argument(
        "--num-layers", type=int, default=None, help="Override number of layers"
    )
    parser.add_argument(
        "--num-heads", type=int, default=None, help="Override number of attention heads"
    )

    args = parser.parse_args()

    # Get model configuration
    config = get_model_config(
        args.model,
        args.bs,
        args.seq,
        tensor_parallel_size=args.tp,
        pipeline_parallel_size=args.pp,
        pipeline_rank=args.pp_rank,
        expert_parallel_size=args.ep,
        data_parallel_size=args.dp,
        dtype=args.dtype,
        use_distributed_optimizer=args.dist_opt,
    )

    # Override with any custom parameters
    if args.hidden_size is not None:
        config.hidden_size = args.hidden_size
        if config.ffn_hidden_size is None:
            config.ffn_hidden_size = 4 * args.hidden_size
        if config.head_dim is None:
            config.head_dim = args.hidden_size // config.num_attention_heads

    if args.num_layers is not None:
        config.num_layers = args.num_layers

    if args.num_heads is not None:
        config.num_attention_heads = args.num_heads
        if config.head_dim is None:
            config.head_dim = config.hidden_size // args.num_heads

    # Print config
    print("\nModel Configuration:")
    for key, value in vars(config).items():
        print(f"  {key}: {value}")

    # Estimate resources
    results = estimate(config, model_id=args.model)

    # Print results
    print_results(results, args.details)


if __name__ == "__main__":
    main()
