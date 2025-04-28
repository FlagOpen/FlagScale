"""Functional wrappers for meta tensor operations.

This module provides functional interfaces for common tensor operations,
wrapping the MetaModule implementations for easier use.
"""

from typing import List, Optional, Tuple, Union

from flagscale.runner.estimator.meta_modules import (
    GELU,
    Baddbmm,
    Bmm,
    CrossEntropy,
    Dropout,
    Matmul,
    SiLU,
    Softmax,
    SwiGLU,
)
from flagscale.runner.estimator.meta_tensor import MetaTensor


def baddbmm(
    input: MetaTensor,
    batch1: MetaTensor,
    batch2: MetaTensor,
    *,
    beta: float = 1.0,
    alpha: float = 1.0,
) -> MetaTensor:
    """
    Batched matrix multiplication with beta and alpha scaling.

    Performs: out = beta * input + alpha * (batch1 @ batch2)

    Parameters:
    -----------
    input : MetaTensor
        Input tensor [..., m, k]
    batch1 : MetaTensor
        First batch [..., m, n]
    batch2 : MetaTensor
        Second batch [..., n, k]
    beta : float, optional
        Scaling factor for input
    alpha : float, optional
        Scaling factor for matrix product
    shard_specs : list, optional
        Specification for tensor sharding
    model_id : str, optional
        Identifier for the model

    Returns:
    --------
    MetaTensor
        Output tensor after operation [..., m, k]
    """
    module = Baddbmm(shard_specs=None, model_id=input.model_id)
    return module(input, batch1, batch2, beta=beta, alpha=alpha)


def bmm(batch1: MetaTensor, batch2: MetaTensor) -> MetaTensor:
    """
    Batched matrix multiplication operation.

    Performs: out = batch1 @ batch2

    Parameters:
    -----------
    batch1 : MetaTensor
        First batch [..., m, n]
    batch2 : MetaTensor
        Second batch [..., n, k]
    shard_specs : list, optional
        Specification for tensor sharding
    model_id : str, optional
        Identifier for the model

    Returns:
    --------
    MetaTensor
        Output tensor after operation [..., m, k]
    """
    module = Bmm(shard_specs=None, model_id=batch1.model_id)
    return module(batch1, batch2)


def matmul(input1: MetaTensor, input2: MetaTensor) -> MetaTensor:
    """
    Matrix multiplication between input1 and input2.

    Handles various cases:
    - Vector-vector: Dot product
    - Matrix-vector: Matrix-vector multiplication
    - Matrix-matrix: Standard matrix multiplication
    - Batched versions of the above

    Parameters:
    -----------
    input1 : MetaTensor
        First input tensor
    input2 : MetaTensor
        Second input tensor
    shard_specs : list, optional
        Specification for tensor sharding
    model_id : str, optional
        Identifier for the model

    Returns:
    --------
    MetaTensor
        Output tensor after matrix multiplication
    """
    module = Matmul(shard_specs=None, model_id=input.model_id)
    return module(input1, input2)


def softmax(input: MetaTensor, dim: int = -1) -> MetaTensor:
    """
    Softmax activation function.

    Applies softmax normalization across a specified dimension.

    Parameters:
    -----------
    input : MetaTensor
        Input tensor
    dim : int, optional
        Dimension along which softmax is computed
    shard_specs : list, optional
        Specification for tensor sharding
    model_id : str, optional
        Identifier for the model

    Returns:
    --------
    MetaTensor
        Output tensor after softmax (same shape as input)
    """
    module = Softmax(dim=dim, shard_specs=None, model_id=input.model_id)
    return module(input)


def dropout(input: MetaTensor, p: float = 0.5, training: bool = True) -> MetaTensor:
    """
    Dropout regularization function.

    Randomly zeros elements of the input tensor with probability p during training.
    During evaluation, the input is returned unchanged.

    Parameters:
    -----------
    input : MetaTensor
        Input tensor
    p : float, optional
        Probability of zeroing elements
    training : bool, optional
        Whether in training mode (apply dropout) or not
    shard_specs : list, optional
        Specification for tensor sharding
    model_id : str, optional
        Identifier for the model

    Returns:
    --------
    MetaTensor
        Output tensor after dropout (same shape as input)
    """
    if not training:
        return input.clone()

    module = Dropout(p=p, shard_specs=None, model_id=input.model_id)
    return module(input)


def gelu(input: MetaTensor, approximate: str = "none") -> MetaTensor:
    """
    Gaussian Error Linear Unit (GELU) activation function.

    Implements the GELU activation function:
    GELU(x) = x * Φ(x) where Φ is the cumulative distribution function of the standard normal distribution.

    Parameters:
    -----------
    input : MetaTensor
        Input tensor
    approximate : str, optional
        Approximation method: "none", "tanh", or "sigmoid"
    shard_specs : list, optional
        Specification for tensor sharding
    model_id : str, optional
        Identifier for the model

    Returns:
    --------
    MetaTensor
        Output tensor with GELU activation applied (same shape as input)
    """
    module = GELU(approximate=approximate, shard_specs=None, model_id=input.model_id)
    return module(input)


def silu(input: MetaTensor) -> MetaTensor:
    """
    SiLU (Sigmoid Linear Unit) activation function.

    Implements the SiLU function:
    SiLU(x) = x * sigmoid(x)

    Parameters:
    -----------
    input : MetaTensor
        Input tensor
    shard_specs : list, optional
        Specification for tensor sharding
    model_id : str, optional
        Identifier for the model

    Returns:
    --------
    MetaTensor
        Output tensor with SiLU activation applied (same shape as input)
    """
    module = SiLU(shard_specs=None, model_id=input.model_id)
    return module(input)


def swiglu(input: MetaTensor) -> MetaTensor:
    """
    SwiGLU activation function.

    Implements the SwiGLU activation function:
    SwiGLU(x) = x * sigmoid(x)

    Parameters:
    -----------
    input : MetaTensor
        Input tensor
    shard_specs : list, optional
        Specification for tensor sharding
    model_id : str, optional
        Identifier for the model

    Returns:
    --------
    MetaTensor
        Output tensor with SwiGLU activation applied (same shape as input)
    """
    module = SwiGLU(shard_specs=None, model_id=input.model_id)
    return module(input)


def cross_entropy(input: MetaTensor, target: MetaTensor) -> MetaTensor:
    """
    Cross-entropy loss function.

    Computes the cross-entropy loss between input and target tensors.

    Parameters:
    -----------
    input : MetaTensor
        Input tensor
    target : MetaTensor
        Target tensor
    shard_specs : list, optional
        Specification for tensor sharding
    model_id : str, optional
        Identifier for the model

    Returns:
    --------
    MetaTensor
        Output tensor with cross-entropy loss (scalar)
    """
    module = CrossEntropy(shard_specs=None, model_id=input.model_id)
    return module(input, target)
