from flagscale.runner.estimator.meta_base import MetaModule, MetaTensor, ShardedDim


class Linear(MetaModule):
    """
    Linear transformation (fully connected layer) with optional bias.

    Performs y = xA^T + b where A is the weight matrix and b is the bias vector.
    """

    def __init__(
        self,
        in_features,
        out_features,
        bias=True,
        shard_specs=[[1, 1]],
        model_id="default",
    ):
        super().__init__(shard_specs, model_id)

        # Use integer division for sharding to avoid floating point imprecision
        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias

    def add_flops(self, input: MetaTensor):
        """
        Compute FLOPs for linear transformation.

        FLOPs include:
        - Matrix multiplication: 2 * batch_dims * in_features * out_features (2 for multiply-add)
        - Bias addition: batch_dims * out_features (if bias=True)

        Parameters:
        -----------
        input : MetaTensor
            Input tensor with shape [..., in_features]

        Returns:
        --------
        int
            Number of FLOPs for the linear operation
        """
        # Calculate batch dimensions (all except the last feature dimension)
        batch_size = 1
        for i in range(len(input) - 1):
            batch_size *= input[i].sharded_dim()

        # Apply tensor parallelism sharding to feature dimensions
        sharded_in_features = self.in_features // self.shard_specs[0][0]
        sharded_out_features = self.out_features // self.shard_specs[0][1]

        # Matrix multiplication: 2 * batch_size * in_features * out_features
        # The factor 2 accounts for multiply-add operations
        flops = 2 * batch_size * sharded_in_features * sharded_out_features

        # Bias addition: batch_size * out_features (if bias=True)
        if self.bias:
            flops += batch_size * sharded_out_features

        return flops

    def add_params(self, input: MetaTensor):
        """
        Compute number of parameters for linear transformation.

        Parameters include:
        - Weight matrix: in_features * out_features
        - Bias vector: out_features (if bias=True)

        Parameters:
        -----------
        input : MetaTensor
            Input tensor (unused but required for API consistency)

        Returns:
        --------
        int
            Number of parameters for the linear layer
        """
        # Apply tensor parallelism sharding to parameter count
        sharded_in_features = self.in_features // self.shard_specs[0][0]
        sharded_out_features = self.out_features // self.shard_specs[0][1]

        # Weight parameters: in_features * out_features
        params = sharded_in_features * sharded_out_features

        # Bias parameters: out_features (if bias=True)
        if self.bias:
            params += sharded_out_features

        return params

    def add_acts(self, input: MetaTensor):
        """
        Compute activation memory elements for linear transformation.

        For linear layers, we need to store for backward pass:
        1. Input tensor (needed for weight gradient calculation)

        We don't count the output tensor as it will be provided as gradient
        by the following layer during backpropagation.

        Parameters:
        -----------
        input : MetaTensor
            Input tensor (typically [batch_size, in_features]
            or [batch_size, seq_len, in_features])

        Returns:
        --------
        int
            Number of elements in tensors needed for backward computation
        """
        # Count input tensor elements (needed for backward)
        input_elements = input.total_elements(apply_sharding=True)

        # We don't count output tensor as it comes from the gradient computation
        # in the backward pass from the subsequent layer

        # Total activations needed for backward computation
        return input_elements

    def forward(self, input: MetaTensor):
        """
        Process input shape and return output shape.
        Updates registry with computed metrics.

        Parameters:
        -----------
        input : MetaTensor
            Input tensor

        Returns:
        --------
        MetaTensor
            Output tensor after linear transformation
        """
        output = input[:-1] + ShardedDim(self.out_features, self.shard_specs[0][1])
        return output


class Embedding(MetaModule):
    """
    Embedding layer that maps indices to dense vectors.
    """

    def __init__(
        self,
        num_embeddings,
        embedding_dim,
        shard_specs=[[1, 1]],
        model_id="default",
    ):
        super().__init__(shard_specs, model_id)

        # Store raw values for later calculations
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim

        # Pad num_embeddings to ensure it's divisible by the sharding factor
        sharding_factor = self.shard_specs[0][0]
        if num_embeddings % sharding_factor != 0:
            self.num_embeddings = (
                (num_embeddings + sharding_factor - 1) // sharding_factor
            ) * sharding_factor

    def add_flops(self, input: MetaTensor):
        """
        Compute FLOPs for embedding lookup.

        While the embedding operation is primarily a lookup, we count
        the operations needed to gather the vectors.

        Parameters:
        -----------
        input : MetaTensor, optional
            Shape of input tensor [batch_size, seq_len]

        Returns:
        --------
        int
            Number of FLOPs (0 for embeddings as it's primarily memory-bound)
        """
        # Embedding lookup is considered memory-bound, not compute-bound
        return 0

    def add_params(self, input: MetaTensor):
        """
        Compute number of parameters for embedding layer.

        Parameters:
        -----------
        input : MetaTensor, optional
            Input tensor (unused, kept for API consistency)

        Returns:
        --------
        int
            Number of parameters in embedding table
        """
        # Each embedding vector has embedding_dim parameters
        # Apply sharding to parameter count
        sharded_num_embeddings = self.num_embeddings // self.shard_specs[0][0]
        sharded_embedding_dim = self.embedding_dim // self.shard_specs[0][1]
        params = sharded_num_embeddings * sharded_embedding_dim
        return params

    def add_acts(self, input: MetaTensor):
        """
        Compute activation memory elements for embedding layer.

        For embedding layers, we count:
        1. The output tensor (forward activation)
        2. The input tensor indices (needed for both forward and backward)

        Parameters:
        -----------
        input : MetaTensor
            Input tensor with indices to look up

        Returns:
        --------
        int
            Number of elements in the activation tensors
        """
        return input.total_elements(apply_sharding=True)

    def forward(self, input: MetaTensor):
        """
        Process input and return output shape.
        Updates registry with computed metrics.

        Parameters:
        -----------
        input : MetaTensor
            Input tensor containing indices to look up

        Returns:
        --------
        MetaTensor
            Output tensor after embedding lookup
        """
        output = input + [ShardedDim(self.embedding_dim, self.shard_specs[0][1])]
        return output


class RotaryEmbedding(MetaModule):
    """
    Rotary position embeddings implementation.

    Implements rotary position embeddings (RoPE) as described in the paper:
    "RoFormer: Enhanced Transformer with Rotary Position Embedding"
    https://arxiv.org/abs/2104.09864
    """

    def __init__(
        self,
        dim,
        max_seq_len=512,
        base=10000,
        shard_specs=None,
        model_id="default",
    ):
        super().__init__(shard_specs, model_id)
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base

    def add_flops(self, input: MetaTensor, positions=None):
        """
        Compute FLOPs for applying rotary embeddings.

        Rotary embeddings involve complex number operations (rotations) for each element.

        Parameters:
        -----------
        input : MetaTensor
            Input tensor [batch_size, seq_len, dim]

        Returns:
        --------
        int
            Number of FLOPs for rotary embedding application
        """
        batch_size = input[0].dim
        seq_len = input[1].dim
        # dim is divided by 2 because rotary embedding is applied to pairs of dimensions
        dim = min(self.dim, input[2].dim)

        # Each element requires 4 multiply-adds for the rotation (complex number multiplication)
        # 4 ops per element: 2 multiplications + 2 additions for the rotation matrix application
        flops = 4 * batch_size * seq_len * dim

        return flops

    def add_params(self, input: MetaTensor, positions=None):
        """
        Compute number of parameters for rotary embeddings.

        Rotary embeddings have no learnable parameters.

        Returns:
        --------
        int
            Number of parameters (0 for rotary embeddings)
        """
        return 0

    def add_acts(self, input: MetaTensor, positions=None):
        """
        Compute activation memory for rotary embeddings.

        For rotary embeddings, we count:
        1. The input tensor (needed for backward pass)
        2. The sin/cos position tables (needed for both forward and backward)

        For rotary embeddings, the computation can often be done in-place,
        so we don't need to separately count the output tensor.

        Parameters:
        -----------
        input : MetaTensor
            Input tensor [batch_size, seq_len, dim]

        Returns:
        --------
        int
            Number of activation elements
        """
        # We need to store the input tensor for backward pass
        input_elements = input.total_elements(apply_sharding=True)

        # We also need to store the sin and cos tables for the positions
        # These are needed for both forward and backward pass
        seq_len = min(input[1].dim, self.max_seq_len)
        dim = (
            min(self.dim, input[2].dim) // 2
        )  # Only need half dim for sin & half for cos
        pos_table_elements = 2 * seq_len * dim  # 2x for both sin and cos tables

        # We don't count output tensor as it can reuse the input tensor's memory
        # In rotary embeddings, the result is often computed in-place

        return input_elements + pos_table_elements

    def forward(self, input: MetaTensor, positions=None):
        """
        Apply rotary embeddings to the input tensor.

        Parameters:
        -----------
        input : MetaTensor
            Input tensor [batch_size, seq_len, dim]
        positions : MetaTensor, optional
            Optional positions tensor, if None, uses sequential positions

        Returns:
        --------
        MetaTensor
            Output tensor with rotary embeddings applied (same shape as input)
        """
        # Output has same shape as input
        output = input.copy()

        return output


class Baddbmm(MetaModule):
    """
    Batched matrix multiplication with beta and alpha scaling.

    Performs: out = beta * input + alpha * (batch1 @ batch2)
    """

    def __init__(
        self,
        shard_specs=None,
        model_id="default",
    ):
        super().__init__(shard_specs, model_id)

    def add_flops(
        self,
        input: MetaTensor,
        batch1: MetaTensor,
        batch2: MetaTensor,
        *,
        beta=1,
        alpha=1,
    ):
        """
        Compute FLOPs for batched matrix multiplication with beta and alpha scaling.

        For the operation: out = beta * input + alpha * (batch1 @ batch2)

        Works with tensors of any dimensionality as long as the last two dimensions
        are compatible for matrix multiplication.

        FLOPs include:
        - Matrix multiplication: 2 * batch_dims * m * n * k (2 for multiply-add)
        - Beta scaling: batch_dims * m * k (if beta != 0 and beta != 1)
        - Alpha scaling: batch_dims * m * k (if alpha != 1)
        - Addition: batch_dims * m * k (if beta != 0)

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

        Returns:
        --------
        float
            Total FLOPs required for the operation
        """
        # Validate tensor shapes - must have at least 2D for matrix multiplication
        if len(batch1) < 2 or len(batch2) < 2:
            raise ValueError("batch1 and batch2 must be at least 2D tensors")

        if input is None or len(input) < 2:
            raise ValueError("input must be at least a 2D tensor")

        # Check that batch1 and batch2 have compatible dimensions for matrix multiplication
        if batch1[-1].dim != batch2[-2].dim:
            raise ValueError(
                f"Matrix dimensions mismatch for multiplication: "
                f"batch1[...{batch1[-1].dim}] and batch2[{batch2[-2].dim}...]"
            )

        # Check input shape is compatible with the result of batch1 @ batch2
        expected_output_shape = batch1[:-1] + [batch2[-1]]
        if (
            len(input) != len(expected_output_shape)
            or input[-2].dim != batch1[-2].dim
            or input[-1].dim != batch2[-1].dim
        ):
            raise ValueError(
                f"Input shape doesn't match expected shape for the operation. "
                f"Expected output has last dimensions [{batch1[-2].dim}, {batch2[-1].dim}], "
                f"but input has [{input[-2].dim}, {input[-1].dim}]"
            )

        # Calculate batch size (all dimensions except the last two)
        batch_size = 1
        for i in range(len(batch1) - 2):
            # Check that batch dimensions match
            if batch1[i].dim != batch2[i].dim or batch1[i].dim != input[i].dim:
                raise ValueError(
                    f"Batch dimension {i} mismatch: {batch1[i].dim}, {batch2[i].dim}, {input[i].dim}"
                )
            batch_size *= batch1[i].sharded_dim()

        # Get the matrix dimensions with proper sharding consideration
        m = batch1[-2].sharded_dim()  # rows of batch1
        n = batch1[-1].sharded_dim()  # columns of batch1 (inner dimension)
        k = batch2[-1].sharded_dim()  # columns of batch2

        # Matrix multiplication flops (multiply-add)
        # Each output element requires n multiply-adds
        mm_flops = 2 * batch_size * m * n * k

        # Beta scaling flops (only if beta != 0 and beta != 1)
        beta_flops = batch_size * m * k if beta != 0 and beta != 1 else 0

        # Alpha scaling flops (only if alpha != 1)
        alpha_flops = batch_size * m * k if alpha != 1 else 0

        # Addition flops (only if beta != 0)
        addition_flops = batch_size * m * k if beta != 0 else 0

        # Calculate total FLOPs
        total_flops = mm_flops + beta_flops + alpha_flops + addition_flops

        return total_flops

    def add_params(
        self,
        input: MetaTensor,
        batch1: MetaTensor,
        batch2: MetaTensor,
        *,
        beta=1,
        alpha=1,
    ):
        """
        Compute number of parameters for Baddbmm operation.

        Baddbmm is just an operation, not a layer with parameters.
        """
        return 0

    def add_acts(
        self,
        input: MetaTensor,
        batch1: MetaTensor,
        batch2: MetaTensor,
        *,
        beta=1,
        alpha=1,
    ):
        """
        Compute activation memory elements for Baddbmm operation.

        For backward computation, we need to store:
        1. input tensor (for gradient calculation)
        2. batch1 (needed for gradient w.r.t. batch2)
        3. batch2 (needed for gradient w.r.t. batch1)

        We need to store all three input tensors for the backward pass, as each is
        required for calculating gradients with respect to the others.

        Parameters:
        -----------
        input : MetaTensor
            Input tensor [batch_size, m, k]
        batch1 : MetaTensor
            First batch [batch_size, m, n]
        batch2 : MetaTensor
            Second batch [batch_size, n, k]

        Returns:
        --------
        int
            Number of activation elements needed for backward computation
        """
        # Count input elements (same shape as output)
        input_elements = input.total_elements(apply_sharding=True)

        # Count batch1 elements (needed for gradient w.r.t. batch2)
        batch1_elements = batch1.total_elements(apply_sharding=True)

        # Count batch2 elements (needed for gradient w.r.t. batch1)
        batch2_elements = batch2.total_elements(apply_sharding=True)

        # Total activations needed for backward
        return input_elements + batch1_elements + batch2_elements

    def forward(
        self,
        input: MetaTensor,
        batch1: MetaTensor,
        batch2: MetaTensor,
        *,
        beta=1,
        alpha=1,
    ):
        """
        Process inputs and return output shape.
        Updates registry with computed metrics.

        Performs: out = beta * input + alpha * (batch1 @ batch2)

        Supports tensors of any dimensionality as long as the last two dimensions
        are compatible for matrix multiplication.

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

        Returns:
        --------
        MetaTensor
            Output tensor after operation [..., m, k]
        """
        # Validate tensor shapes - must have at least 2D for matrix multiplication
        if len(batch1) < 2 or len(batch2) < 2:
            raise ValueError("batch1 and batch2 must be at least 2D tensors")

        if input is None or len(input) < 2:
            raise ValueError("input must be at least a 2D tensor")

        # Check that batch1 and batch2 have compatible dimensions for matrix multiplication
        if batch1[-1].dim != batch2[-2].dim:
            raise ValueError(
                f"Matrix dimensions mismatch for multiplication: "
                f"batch1[...{batch1[-1].dim}] and batch2[{batch2[-2].dim}...]"
            )

        # Check input shape is compatible with the result of batch1 @ batch2
        if input[-2].dim != batch1[-2].dim or input[-1].dim != batch2[-1].dim:
            raise ValueError(
                f"Input shape doesn't match expected shape for the operation. "
                f"Expected output has last dimensions [{batch1[-2].dim}, {batch2[-1].dim}], "
                f"but input has [{input[-2].dim}, {input[-1].dim}]"
            )

        # Check that batch dimensions match (all except the last two)
        if len(batch1) != len(batch2) or len(batch1) != len(input):
            raise ValueError(
                f"Tensor dimension counts must match. Got batch1: {len(batch1)}, "
                f"batch2: {len(batch2)}, input: {len(input)}"
            )

        for i in range(len(batch1) - 2):
            if batch1[i].dim != batch2[i].dim or batch1[i].dim != input[i].dim:
                raise ValueError(
                    f"Batch dimension {i} mismatch: "
                    f"batch1[{i}]={batch1[i].dim}, batch2[{i}]={batch2[i].dim}, input[{i}]={input[i].dim}"
                )

        # Use MetaTensor operations to create the output shape
        # batch1[:-1] gets all batch dimensions plus the 'm' dimension
        # batch2[-1] gets the 'k' dimension
        output = batch1[:-1] + [batch2[-1]]
        return output


class Bmm(MetaModule):
    """
    Batched matrix multiplication operation.

    Performs: out = batch1 @ batch2
    """

    def __init__(
        self,
        shard_specs=None,
        model_id="default",
    ):
        super().__init__(shard_specs, model_id)

    def add_flops(self, batch1: MetaTensor, batch2: MetaTensor):
        """
        Compute FLOPs for batched matrix multiplication.

        For the operation: out = batch1 @ batch2

        Supports tensors of any dimensionality as long as the last two dimensions
        are compatible for matrix multiplication.

        FLOPs include:
        - Matrix multiplication: 2 * batch_dims * m * n * k (2 for multiply-add)

        Parameters:
        -----------
        batch1 : MetaTensor
            First batch [..., m, n]
        batch2 : MetaTensor
            Second batch [..., n, k]

        Returns:
        --------
        float
            Total FLOPs required for the operation
        """
        # Validate tensor shapes - must have at least 2D for matrix multiplication
        if len(batch1) < 2 or len(batch2) < 2:
            raise ValueError("batch1 and batch2 must be at least 2D tensors")

        # Check that batch1 and batch2 have compatible dimensions for matrix multiplication
        if batch1[-1].dim != batch2[-2].dim:
            raise ValueError(
                f"Matrix dimensions mismatch for multiplication: "
                f"batch1[...{batch1[-1].dim}] and batch2[{batch2[-2].dim}...]"
            )

        # Check that batch dimensions match (all except the last two)
        if len(batch1) != len(batch2):
            raise ValueError(
                f"Tensor dimension counts must match. Got batch1: {len(batch1)}, "
                f"batch2: {len(batch2)}"
            )

        # For dimensions except the last two, check that they match
        for i in range(len(batch1) - 2):
            if batch1[i].dim != batch2[i].dim:
                raise ValueError(
                    f"Batch dimension {i} mismatch: "
                    f"batch1[{i}]={batch1[i].dim}, batch2[{i}]={batch2[i].dim}"
                )

        # Calculate batch size (all dimensions except the last two)
        batch_size = 1
        for i in range(len(batch1) - 2):
            batch_size *= batch1[i].sharded_dim()

        # Get the matrix dimensions with proper sharding consideration
        m = batch1[-2].sharded_dim()  # rows of batch1
        n = batch1[-1].sharded_dim()  # columns of batch1 (inner dimension)
        k = batch2[-1].sharded_dim()  # columns of batch2

        # Matrix multiplication flops (multiply-add)
        # Each output element requires n multiply-adds
        mm_flops = 2 * batch_size * m * n * k

        return mm_flops

    def add_params(self, batch1: MetaTensor, batch2: MetaTensor):
        """
        Compute number of parameters for Bmm operation.

        Bmm is just an operation, not a layer with parameters.
        """
        return 0

    def add_acts(self, batch1: MetaTensor, batch2: MetaTensor):
        """
        Compute activation memory elements for Bmm operation.

        For backward computation, we need to store:
        1. batch1 (needed for gradient w.r.t. batch2)
        2. batch2 (needed for gradient w.r.t. batch1)

        Parameters:
        -----------
        batch1 : MetaTensor
            First batch [batch_size, m, n]
        batch2 : MetaTensor
            Second batch [batch_size, n, k]

        Returns:
        --------
        int
            Number of activation elements needed for backward computation
        """
        if batch1 is None or batch2 is None:
            return 0

        # Count batch1 elements (needed for backward)
        batch1_elements = batch1.total_elements(apply_sharding=True)

        # Count batch2 elements (needed for backward)
        batch2_elements = batch2.total_elements(apply_sharding=True)

        # Total activations needed for backward
        return batch1_elements + batch2_elements

    def forward(
        self,
        batch1: MetaTensor,
        batch2: MetaTensor,
        *,
        beta=1,
        alpha=1,
    ):
        """
        Process inputs and return output shape.
        Updates registry with computed metrics.

        Performs: out = beta * input + alpha * (batch1 @ batch2)

        Supports tensors of any dimensionality as long as the last two dimensions
        are compatible for matrix multiplication.

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

        Returns:
        --------
        MetaTensor
            Output tensor after operation [..., m, k]
        """

        # Validate tensor shapes - must have at least 2D for matrix multiplication
        if len(batch1) < 2 or len(batch2) < 2:
            raise ValueError("batch1 and batch2 must be at least 2D tensors")

        # Check that batch1 and batch2 have compatible dimensions for matrix multiplication
        if batch1[-1].dim != batch2[-2].dim:
            raise ValueError(
                f"Matrix dimensions mismatch for multiplication: "
                f"batch1[...{batch1[-1].dim}] and batch2[{batch2[-2].dim}...]"
            )

        # Check that batch dimensions match (all except the last two)
        if len(batch1) != len(batch2):
            raise ValueError(
                f"Tensor dimension counts must match. Got batch1: {len(batch1)}, "
                f"batch2: {len(batch2)}"
            )

        for i in range(len(batch1) - 2):
            if batch1[i].dim != batch2[i].dim:
                raise ValueError(
                    f"Batch dimension {i} mismatch: "
                    f"batch1[{i}]={batch1[i].dim}, batch2[{i}]={batch2[i].dim}"
                )

        # Use MetaTensor operations to create the output shape
        # batch1[:-1] gets all batch dimensions plus the 'm' dimension
        # batch2[-1] gets the 'k' dimension
        output = batch1[:-1] + [batch2[-1]]

        return output


class Softmax(MetaModule):
    """
    Softmax activation function.

    Applies softmax normalization across a specified dimension.
    """

    def __init__(
        self,
        dim=-1,  # Default to last dimension
        shard_specs=None,
        model_id="default",
    ):
        super().__init__(shard_specs, model_id)
        self.dim = dim

    def add_flops(self, input: MetaTensor):
        """
        Compute FLOPs for softmax operation.

        Softmax is primarily a memory-bound operation, but we could count exponentials
        and divisions if desired.

        Parameters:
        -----------
        input : MetaTensor
            Input tensor

        Returns:
        --------
        int
            Number of FLOPs (0 for softmax as it's primarily memory-bound)
        """
        # We consider softmax primarily memory-bound for this estimator
        return 0

    def add_params(self, input: MetaTensor):
        """
        Compute number of parameters for softmax operation.

        Softmax has no learnable parameters.

        Returns:
        --------
        int
            Number of parameters (0 for softmax)
        """
        return 0

    def add_acts(self, input: MetaTensor):
        """
        Compute activation memory elements for softmax operation.

        Parameters:
        -----------
        input : MetaTensor
            Input tensor

        Returns:
        --------
        int
            Number of elements in input tensor
        """
        return input.total_elements(apply_sharding=True)

    def forward(self, input: MetaTensor):
        """
        Process input and return output shape.
        Updates registry with computed metrics.

        Parameters:
        -----------
        input : MetaTensor
            Input tensor

        Returns:
        --------
        MetaTensor
            Output tensor after softmax (same shape as input)
        """
        output = input.copy()
        return output


class Dropout(MetaModule):
    """
    Dropout regularization layer.

    Randomly zeros elements of the input tensor with probability p.
    """

    def __init__(
        self,
        p=0.5,
        shard_specs=None,
        model_id="default",
    ):
        super().__init__(shard_specs, model_id)
        self.p = p

    def add_flops(self, input: MetaTensor):
        """
        Compute FLOPs for dropout operation.

        Dropout is primarily a memory-bound operation.

        Parameters:
        -----------
        input : MetaTensor
            Input tensor

        Returns:
        --------
        int
            Number of FLOPs (0 for dropout as it's primarily memory-bound)
        """
        # We consider dropout primarily memory-bound for this estimator
        return 0

    def add_params(self, input: MetaTensor):
        """
        Compute number of parameters for dropout operation.

        Dropout has no learnable parameters.

        Returns:
        --------
        int
            Number of parameters (0 for dropout)
        """
        return 0

    def add_acts(self, input: MetaTensor):
        """
        Compute activation memory elements for dropout operation.

        For backward computation, we need to store:
        1. Dropout mask (binary tensor with same shape as input)
        2. Output tensor (not needed if we keep the mask)

        Parameters:
        -----------
        input : MetaTensor
            Input tensor

        Returns:
        --------
        int
            Number of activation elements needed for backward computation
        """
        return input.total_elements(apply_sharding=True)

    def forward(self, input: MetaTensor):
        """
        Process input and return output shape.
        Updates registry with computed metrics.

        Parameters:
        -----------
        input : MetaTensor
            Input tensor

        Returns:
        --------
        MetaTensor
            Output tensor after dropout (same shape as input)
        """
        output = input.copy()
        return output


class GELU(MetaModule):
    """
    Gaussian Error Linear Unit (GELU) activation function.

    Implements the GELU activation function as described in:
    "Gaussian Error Linear Units (GELUs)" (Hendrycks & Gimpel, 2016)
    https://arxiv.org/abs/1606.08415
    """

    def __init__(self, approximate="none", shard_specs=None, model_id="default"):
        super().__init__(shard_specs, model_id)
        self.approximate = approximate  # Options: "none", "tanh", "sigmoid"

    def add_flops(self, input: MetaTensor):
        """
        Compute FLOPs for GELU activation.

        GELU involves several elementary operations per element:
        - For standard GELU: ~8-10 operations per element (erf calculation)
        - For tanh approximation: ~7 operations per element

        Parameters:
        -----------
        input : MetaTensor
            Input tensor

        Returns:
        --------
        int
            Number of FLOPs for GELU activation
        """
        # Count total number of elements after applying sharding
        num_elements = input.total_elements(apply_sharding=True)

        # Estimate operations based on GELU variant
        if self.approximate == "tanh":
            # tanh approximation: x * 0.5 * (1.0 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
            # ~7 ops per element: multiply, add, cube, multiply, add, tanh, multiply
            ops_per_element = 7
        elif self.approximate == "sigmoid":
            # sigmoid approximation: x * sigmoid(1.702 * x)
            # ~3 ops per element: multiply, sigmoid, multiply
            ops_per_element = 3
        else:
            # standard GELU: x * 0.5 * (1.0 + erf(x / sqrt(2)))
            # ~10 ops per element for erf calculation
            ops_per_element = 10

        return num_elements * ops_per_element

    def add_params(self, input: MetaTensor):
        """
        Compute number of parameters for GELU activation.

        GELU has no learnable parameters.

        Returns:
        --------
        int
            Number of parameters (0 for GELU)
        """
        return 0

    def add_acts(self, input: MetaTensor):
        """
        Compute activation memory for GELU activation.

        For GELU, we need to store the input tensor and the output tensor.

        Parameters:
        -----------
        input : MetaTensor
            Input tensor

        Returns:
        --------
        int
            Number of activation elements
        """
        return input.total_elements(apply_sharding=True)

    def forward(self, input: MetaTensor):
        """
        Apply GELU activation to the input tensor.

        Parameters:
        -----------
        input : MetaTensor
            Input tensor

        Returns:
        --------
        MetaTensor
            Output tensor with GELU activation applied (same shape as input)
        """
        # Output has the same shape and sharding as input
        output = input.copy()
        return output


class Swish(MetaModule):
    """
    Swish activation function.

    Implements the Swish activation function as described in:
    "Searching for Activation Functions" (Ramachandran et al., 2017)
    https://arxiv.org/abs/1710.05941

    Swish(x) = x * sigmoid(x)
    """

    def __init__(self, shard_specs=None, model_id="default"):
        super().__init__(shard_specs, model_id)

    def add_flops(self, input: MetaTensor):
        """
        Compute FLOPs for Swish activation.

        Swish involves:
        - 1 sigmoid operation
        - 1 multiplication
        Per element

        Parameters:
        -----------
        input : MetaTensor
            Input tensor

        Returns:
        --------
        int
            Number of FLOPs for Swish activation
        """
        # Count total number of elements after applying sharding
        num_elements = input.total_elements()

        # Estimate operations: sigmoid + multiply
        ops_per_element = 2

        return num_elements * ops_per_element

    def add_params(self, input: MetaTensor):
        """
        Compute number of parameters for Swish activation.

        Swish has no learnable parameters.

        Returns:
        --------
        int
            Number of parameters (0 for Swish)
        """
        return 0

    def add_acts(self, input: MetaTensor):
        """
        Compute activation memory for Swish activation.

        For Swish, we need to store the input tensor and the output tensor.

        Parameters:
        -----------
        input : MetaTensor
            Input tensor

        Returns:
        --------
        int
            Number of activation elements
        """
        return input.total_elements()

    def forward(self, input: MetaTensor):
        """
        Apply Swish activation to the input tensor.

        Parameters:
        -----------
        input : MetaTensor
            Input tensor

        Returns:
        --------
        MetaTensor
            Output tensor with Swish activation applied (same shape as input)
        """
        # Output has the same shape and sharding as input
        output = input.copy()

        return output


class SwiGLU(MetaModule):
    """
    Swish-Gated Linear Unit (SwiGLU) activation.

    Implements SwiGLU as described in:
    "GLU Variants Improve Transformer" (Noam Shazeer, 2020)
    https://arxiv.org/abs/2002.05202

    SwiGLU(x, W, V, b, c) = Swish(xW + b) ⊗ (xV + c)

    Where Swish(x) = x * sigmoid(x)
    """

    def __init__(self, shard_specs=[[1, 1]], model_id="default"):
        super().__init__(shard_specs, model_id)

    def add_flops(self, gate: MetaTensor, value: MetaTensor):
        """
        Compute FLOPs for SwiGLU activation.

        SwiGLU involves:
        1. Sigmoid operation on gate tensor
        2. Multiplication of gate with sigmoid(gate)
        3. Element-wise multiplication between swish(gate) and value tensors

        Parameters:
        -----------
        gate : MetaTensor
            Gate tensor input
        value : MetaTensor
            Value tensor input

        Returns:
        --------
        int
            Number of FLOPs for SwiGLU activation
        """
        # Count total number of elements after applying sharding
        gate_elements = gate.total_elements(apply_sharding=True)

        # Compute sigmoid(gate) - one sigmoid operation per element
        # Sigmoid typically requires ~4 ops per element (exp, add, div)
        sigmoid_flops = gate_elements * 4

        # Compute gate * sigmoid(gate) - one multiplication per element
        swish_mult_flops = gate_elements

        # Compute swish(gate) * value - one multiplication per element
        swiglu_mult_flops = gate_elements

        # Total FLOPs
        total_flops = sigmoid_flops + swish_mult_flops + swiglu_mult_flops

        return total_flops

    def add_params(self, gate: MetaTensor, value: MetaTensor):
        """
        Compute number of parameters for SwiGLU activation.

        SwiGLU itself has no learnable parameters.

        Returns:
        --------
        int
            Number of parameters (0 for SwiGLU)
        """
        return 0

    def add_acts(self, gate: MetaTensor, value: MetaTensor):
        """
        Compute activation memory for SwiGLU activation.

        For backward computation of SwiGLU, we need:
        1. Gate tensor (for Swish gradient calculation)
        2. Value tensor (for gradient calculation)
        3. Sigmoid(gate) tensor (for gradient calculation)

        We don't need to store the final output separately as it doesn't
        factor into the gradient calculation.

        Parameters:
        -----------
        gate : MetaTensor
            Gate tensor input
        value : MetaTensor
            Value tensor input

        Returns:
        --------
        int
            Number of activation elements needed for backward computation
        """
        # Gate tensor elements (needed for backward)
        gate_elements = gate.total_elements(apply_sharding=True)

        # Value tensor elements (needed for backward)
        value_elements = value.total_elements(apply_sharding=True)

        # Sigmoid(gate) tensor (needed for backward)
        sigmoid_elements = gate_elements

        # Total activations needed for backward
        return gate_elements + value_elements + sigmoid_elements

    def forward(self, gate: MetaTensor, value: MetaTensor):
        """
        Apply SwiGLU activation: Swish(gate) * value.

        Where Swish(x) = x * sigmoid(x)

        Parameters:
        -----------
        gate : MetaTensor
            Gate tensor input
        value : MetaTensor
            Value tensor input

        Returns:
        --------
        MetaTensor
            Output tensor with SwiGLU activation applied
        """
        output = value.copy()

        return output


class LayerNorm(MetaModule):
    """
    Layer normalization module.

    Applies Layer Normalization over a mini-batch of inputs as described in
    "Layer Normalization" (Ba et al., 2016): https://arxiv.org/abs/1607.06450

    The normalization is applied across the last dimension.
    """

    def __init__(
        self,
        normalized_shape,
        eps=1e-5,
        elementwise_affine=True,
        bias=True,
        shard_specs=[[1, 1]],
        model_id="default",
    ):
        """
        Initialize a layer normalization module.

        Parameters:
        -----------
        normalized_shape : int or list
            Size of the normalized dimension(s)
        eps : float, optional
            Small constant for numerical stability
        elementwise_affine : bool, optional
            Whether to use learnable affine parameters
        bias : bool, optional
            Whether to use bias when elementwise_affine=True
        shard_specs : list, optional
            Specification for tensor sharding
        model_id : str, optional
            Identifier for the model
        """
        super().__init__(shard_specs, model_id)

        # Store configuration
        if isinstance(normalized_shape, (list, tuple)):
            # Use the last dimension if normalized_shape is a sequence
            self.hidden_size = normalized_shape[-1]
        else:
            # If it's a single integer, use it directly
            self.hidden_size = normalized_shape

        self.eps = eps
        self.elementwise_affine = elementwise_affine
        self.bias = bias if elementwise_affine else False

        assert (
            self.shard_specs[0][0] == self.shard_specs[0][1]
        ), "LayerNorm requires equal sharding for hidden size"

    def add_flops(self, input: MetaTensor):
        """
        Compute FLOPs for layer normalization.

        LayerNorm operations include:
        1. Mean calculation: sum over normalized dimension + division
        2. Variance calculation: subtract mean, square differences, sum, divide by n
        3. Normalization: add epsilon, sqrt, division by std
        4. Scaling: multiply by gamma and add beta (if elementwise_affine=True)

        Parameters:
        -----------
        input : MetaTensor
            Input tensor with shape [..., hidden_size]

        Returns:
        --------
        float
            Number of FLOPs for layer normalization
        """
        # Get total number of elements in the input, respecting sharding
        total_elements = input.total_elements(apply_sharding=True)

        # Account for potential tensor parallelism in the hidden dimension
        sharded_hidden_size = self.hidden_size // self.shard_specs[0][0]

        # Number of normalization groups - each group is normalized independently
        # For LayerNorm, we normalize over the last dimension (hidden_size)
        # So we have (total_elements / hidden_size) independent normalization groups
        batch_elements = total_elements // sharded_hidden_size

        # ------- STEP 1: Mean calculation -------
        # For each group, we:
        # - Sum hidden_size elements: (hidden_size - 1) addition ops
        # - Divide by hidden_size: 1 division op
        # Total: batch_elements * hidden_size + batch_elements ops
        mean_flops = batch_elements * sharded_hidden_size + batch_elements

        # ------- STEP 2: Variance calculation -------
        # For each element:
        # - Subtract mean: total_elements subtractions
        # - Square differences: total_elements multiplications
        # For each group:
        # - Sum squared differences: (hidden_size - 1) additions
        # - Divide by hidden_size: 1 division
        var_flops = (
            total_elements * 2 + batch_elements * sharded_hidden_size + batch_elements
        )

        # ------- STEP 3: Normalization -------
        # For each group:
        # - Add epsilon: batch_elements additions
        # - Calculate sqrt: batch_elements sqrt operations
        # For each element:
        # - Divide by std: total_elements divisions
        norm_flops = batch_elements * 2 + total_elements

        # ------- STEP 4: Scaling and bias (only if elementwise_affine=True) -------
        if self.elementwise_affine:
            # Scale by gamma: total_elements multiplications
            # Add bias (if used): total_elements additions
            scale_flops = total_elements + (total_elements if self.bias else 0)
        else:
            scale_flops = 0

        # ------- Total FLOPs -------
        # Sum of all steps
        total_flops = mean_flops + var_flops + norm_flops + scale_flops

        return total_flops

    def add_params(self, input: MetaTensor):
        """
        Compute number of parameters for layer normalization.

        Parameters include:
        - Weight (gamma): hidden_size (if elementwise_affine=True)
        - Bias (beta): hidden_size (if elementwise_affine=True and bias=True)

        Returns:
        --------
        int
            Number of parameters for layer normalization
        """
        if not self.elementwise_affine:
            return 0

        sharded_hidden_size = self.hidden_size // self.shard_specs[0][0]

        # Weight (gamma) parameters
        params = sharded_hidden_size

        # Bias (beta) parameters (if bias=True)
        if self.bias:
            params += sharded_hidden_size

        return params

    def add_acts(self, input: MetaTensor):
        """
        Compute activation memory for layer normalization.

        For backward computation of LayerNorm, we need:
        1. Input tensor (for gradient calculation)
        2. Mean tensor (one value per normalization group)
        3. Inverse std tensor (one value per normalization group)

        Parameters:
        -----------
        input : MetaTensor
            Input tensor with shape [..., hidden_size]

        Returns:
        --------
        int
            Number of activation elements needed for backward computation
        """
        # Input tensor elements (needed for backward)
        input_elements = input.total_elements(apply_sharding=True)

        # Mean and inverse std elements (one each per group)
        # Number of groups = total_elements / hidden_size
        total_elements = input.total_elements(apply_sharding=True)
        sharded_hidden_size = self.hidden_size // self.shard_specs[0][0]
        groups = total_elements // sharded_hidden_size
        stats_elements = groups * 2  # mean and inverse std

        # Total activations needed for backward
        return input_elements + stats_elements

    def forward(self, input: MetaTensor):
        """
        Apply layer normalization to the input tensor.

        Parameters:
        -----------
        input : MetaTensor
            Input tensor with shape [..., hidden_size]

        Returns:
        --------
        MetaTensor
            Output tensor after layer normalization (same shape as input)
        """
        # Validate input shape
        if len(input) < 1:
            raise ValueError("Input tensor must have at least one dimension")

        # Check that the last dimension matches hidden_size
        if input[-1].dim != self.hidden_size:
            raise ValueError(
                f"Last dimension of input ({input[-1].dim}) must match hidden_size ({self.hidden_size})"
            )

        # Output has the same shape and sharding as input
        output = input.copy()

        return output


class RMSNorm(MetaModule):
    """
    Root Mean Square Layer Normalization.

    Applies RMS normalization over a mini-batch of inputs as described in
    "Root Mean Square Layer Normalization" (Zhang & Sennrich, 2019): https://arxiv.org/abs/1910.07467

    RMSNorm simplifies LayerNorm by removing the mean centering step,
    which can improve training efficiency with comparable performance.
    """

    def __init__(
        self,
        normalized_shape,
        eps=None,
        elementwise_affine=True,
        shard_specs=[[1, 1]],
        model_id="default",
    ):
        """
        Initialize an RMS normalization module.

        Parameters:
        -----------
        normalized_shape : int or list
            Size of the normalized dimension(s)
        eps : float, optional
            Small constant for numerical stability
        elementwise_affine : bool, optional
            Whether to use learnable affine parameters
        shard_specs : list, optional
            Specification for tensor sharding
        model_id : str, optional
            Identifier for the model
        """
        super().__init__(shard_specs, model_id)

        # Store configuration
        if isinstance(normalized_shape, (list, tuple)):
            # Use the last dimension if normalized_shape is a sequence
            self.hidden_size = normalized_shape[-1]
        else:
            # If it's a single integer, use it directly
            self.hidden_size = normalized_shape

        # Set default eps value if not provided
        self.eps = eps if eps is not None else 1e-8
        self.elementwise_affine = elementwise_affine

        assert (
            self.shard_specs[0][0] == self.shard_specs[0][1]
        ), "RMSNorm requires equal sharding for hidden size"

    def add_flops(self, input: MetaTensor):
        """
        Compute FLOPs for RMS normalization.

        RMSNorm operations include:
        1. RMS calculation: square each element, sum, divide by n, sqrt
        2. Normalization: x / rms
        3. Scaling: gamma * normalized (no bias typically in RMSNorm)

        Parameters:
        -----------
        input : MetaTensor
            Input tensor with shape [..., hidden_size]

        Returns:
        --------
        float
            Number of FLOPs for RMS normalization
        """
        # Get total number of elements in the input
        total_elements = input.total_elements(apply_sharding=True)

        # Account for potential tensor parallelism in the hidden dimension
        sharded_hidden_size = self.hidden_size // self.shard_specs[0][0]

        # Number of normalization groups - each group is normalized independently
        # For RMSNorm, we normalize over the last dimension (hidden_size)
        # So we have (total_elements / hidden_size) independent normalization groups
        batch_elements = total_elements // sharded_hidden_size

        # ------- STEP 1: RMS calculation -------
        # For each element:
        # - Square each element: 1 op per element (total_elements ops)
        # For each group:
        # - Sum squared values: (hidden_size-1) ops per group (batch_elements * sharded_hidden_size ops)
        # - Divide by n: 1 op per group (batch_elements ops)
        # - Square root: 1 op per group (batch_elements ops)
        rms_flops = (
            total_elements + batch_elements * sharded_hidden_size + batch_elements * 2
        )

        # ------- STEP 2: Normalization -------
        # For each element:
        # - Divide by rms: 1 op per element (total_elements ops)
        norm_flops = total_elements

        # ------- STEP 3: Scaling -------
        # Apply scale factor (gamma): 1 op per element (if elementwise_affine=True)
        scale_flops = total_elements if self.elementwise_affine else 0

        # Total FLOPs
        total_flops = rms_flops + norm_flops + scale_flops

        return total_flops

    def add_params(self, input: MetaTensor):
        """
        Compute number of parameters for RMS normalization.

        Parameters include:
        - Weight (gamma): hidden_size (if elementwise_affine=True)

        Returns:
        --------
        int
            Number of parameters for RMS normalization
        """
        if not self.elementwise_affine:
            return 0

        sharded_hidden_size = self.hidden_size // self.shard_specs[0][0]

        # Weight (gamma) parameters
        params = sharded_hidden_size

        return params

    def add_acts(self, input: MetaTensor):
        """
        Compute activation memory for RMS normalization.

        For backward computation of RMSNorm, we need:
        1. Input tensor (for gradient calculation)
        2. Inverse RMS tensor (one value per normalization group)

        We don't store the output separately as it can be computed from
        the above values when needed.

        Parameters:
        -----------
        input : MetaTensor
            Input tensor with shape [..., hidden_size]

        Returns:
        --------
        int
            Number of activation elements needed for backward computation
        """
        if input is None:
            return 0

        # Input tensor elements (needed for backward)
        input_elements = input.total_elements(apply_sharding=True)

        # Inverse RMS elements (one per group)
        # Number of groups = total_elements / hidden_size
        total_elements = input.total_elements(apply_sharding=True)
        sharded_hidden_size = self.hidden_size // self.shard_specs[0][0]
        groups = total_elements // sharded_hidden_size
        inv_rms_elements = groups

        # Total activations needed for backward
        return input_elements + inv_rms_elements

    def forward(self, input: MetaTensor):
        """
        Apply RMS normalization to the input tensor.

        Parameters:
        -----------
        input : MetaTensor
            Input tensor with shape [..., hidden_size]

        Returns:
        --------
        MetaTensor
            Output tensor after RMS normalization (same shape as input)
        """
        # Validate input shape
        if len(input) < 1:
            raise ValueError("Input tensor must have at least one dimension")

        # Check that the last dimension matches hidden_size
        if input[-1].dim != self.hidden_size:
            raise ValueError(
                f"Last dimension of input ({input[-1].dim}) must match hidden_size ({self.hidden_size})"
            )

        # Output has the same shape and sharding as input
        output = input.copy()

        return output


class CrossEntropy(MetaModule):
    """
    Cross Entropy loss function.

    Computes the cross entropy loss between input logits and target.
    Typically used as the final loss function in classification tasks.
    """

    def __init__(
        self,
        weight=None,
        ignore_index=-100,
        reduction="mean",
        label_smoothing=0.0,
        shard_specs=None,
        model_id="default",
    ):
        """
        Initialize a cross entropy loss module.

        Parameters:
        -----------
        weight : torch.Tensor, optional
            Manual rescaling weight for each class. If given, has to be a tensor
            of size C (number of classes)
        ignore_index : int, optional
            Specifies a target value that is ignored during loss computation
        reduction : str, optional
            Specifies the reduction to apply to the output: 'none' | 'mean' | 'sum'
        label_smoothing : float, optional
            Specifies the amount of label smoothing to apply
        shard_specs : list, optional
            Specification for tensor sharding
        model_id : str, optional
            Identifier for the model
        """
        super().__init__(shard_specs, model_id)

        # Store configuration
        self.weight = weight
        self.ignore_index = ignore_index
        self.label_smoothing = label_smoothing
        self.reduction = reduction

    def add_flops(self, logits: MetaTensor, targets: MetaTensor):
        """
        Compute FLOPs for cross entropy loss calculation.

        Cross entropy operations include:
        1. Softmax calculation for each prediction
        2. Log of softmax outputs
        3. Gathering target probabilities
        4. Reduction (sum or mean)
        5. Label smoothing (if applied)
        6. Class weighting (if weight is provided)

        Parameters:
        -----------
        logits : MetaTensor
            Predicted values [batch_size, num_classes, ...] or [batch_size, seq_len, num_classes]
        targets : MetaTensor
            Target values [batch_size, ...] or [batch_size, seq_len]

        Returns:
        --------
        float
            Number of FLOPs for cross entropy calculation
        """
        if logits is None or targets is None:
            return 0

        # Extract shapes and dimensions
        if len(logits) < 2:
            raise ValueError("Logits must have at least 2 dimensions")

        # Get total elements and vocabulary size
        total_elements = logits.total_elements(apply_sharding=True)

        # For language modeling, logits are typically [batch_size, seq_len, vocab_size]
        if len(logits) == 3:
            batch_size = logits[0].sharded_dim()
            seq_len = logits[1].sharded_dim()
            vocab_size = logits[2].sharded_dim()
            predictions = batch_size * seq_len
        # For classification, logits are typically [batch_size, num_classes]
        elif len(logits) == 2:
            batch_size = logits[0].sharded_dim()
            vocab_size = logits[1].sharded_dim()
            predictions = batch_size
        else:
            # For other cases, just count the total predictions
            vocab_size = logits[-1].sharded_dim()
            predictions = total_elements // vocab_size

        # Calculate FLOPs for each operation:

        # 1. Softmax calculation per prediction
        # - Exp for each logit: 1 op per element
        # - Sum of exps: vocab_size ops per prediction
        # - Division for each probability: 1 op per element
        softmax_flops = total_elements + predictions * vocab_size + total_elements

        # 2. Log of softmax outputs: 1 op per element
        log_flops = total_elements

        # 3. Gathering target probabilities: 1 op per prediction
        gather_flops = predictions

        # 4. Apply class weights (if provided)
        weight_flops = 0
        if self.weight is not None:
            weight_flops = predictions

        # 5. Label smoothing (if used)
        label_smoothing_flops = 0
        if self.label_smoothing > 0:
            # Additional operations for smoothed targets:
            # - Create uniform distribution: vocab_size ops per prediction
            # - Scale targets: 1 op per prediction
            # - Scale uniform distribution: 1 op per prediction
            # - Mix distributions: 1 op per prediction
            label_smoothing_flops = predictions * (vocab_size + 3)

        # 6. Reduction (sum or mean)
        # - Sum: predictions ops
        # - Mean: predictions + 1 ops
        reduction_flops = predictions
        if self.reduction == "mean":
            reduction_flops += 1

        # Total FLOPs
        total_flops = (
            softmax_flops
            + log_flops
            + gather_flops
            + weight_flops
            + label_smoothing_flops
            + reduction_flops
        )

        return total_flops

    def add_params(self, logits: MetaTensor, targets: MetaTensor):
        """
        Compute number of parameters for cross entropy loss.

        Cross entropy has no learnable parameters.

        Returns:
        --------
        int
            Number of parameters (0 for cross entropy)
        """
        return 0

    def add_acts(self, logits: MetaTensor, targets: MetaTensor):
        """
        Compute activation memory for cross entropy loss.

        For backward computation of cross entropy, we need:
        1. Softmax probabilities (for gradient calculation)
        2. Target indices (for loss calculation)
        3. Weight tensor (if provided)

        Parameters:
        -----------
        logits : MetaTensor
            Predicted values [batch_size, num_classes, ...] or [batch_size, seq_len, num_classes]
        targets : MetaTensor
            Target values [batch_size, ...] or [batch_size, seq_len]

        Returns:
        --------
        int
            Number of activation elements needed for backward computation
        """
        if logits is None or targets is None:
            return 0

        # We need to store the softmax probabilities (same shape as logits)
        probs_elements = logits.total_elements(apply_sharding=True)

        # We also need to store the target indices
        targets_elements = targets.total_elements(apply_sharding=True)

        # If weight is provided, we need to store it
        weight_elements = 0
        if self.weight is not None:
            # Weight has size equal to number of classes (vocab_size)
            weight_elements = logits[-1].sharded_dim()

        # Total activations needed for backward
        return probs_elements + targets_elements + weight_elements

    def forward(self, logits: MetaTensor, targets: MetaTensor):
        """
        Compute cross entropy loss between logits and targets.

        Parameters:
        -----------
        logits : MetaTensor
            Predicted values [batch_size, num_classes, ...] or [batch_size, seq_len, num_classes]
        targets : MetaTensor
            Target values [batch_size, ...] or [batch_size, seq_len]

        Returns:
        --------
        MetaTensor
            Output loss value [1] or [batch_size, ...]
        """
        # Validate input shapes
        if len(logits) < 2:
            raise ValueError("Logits must have at least 2 dimensions")

        if targets is None:
            raise ValueError("Targets cannot be None")

        # For 'none' reduction, output has shape matching targets (except for class dimension)
        # For 'mean' or 'sum' reduction, output is a scalar
        if self.reduction in ["mean", "sum"]:
            output = MetaTensor(shape=[1], shard_spec=[1])
        else:
            # For 'none' reduction, copy the target shape except last dim for sequence tasks
            # or remove the class dimension for classification tasks
            if len(logits) == len(targets) + 1:
                # Classification case: logits [B, C], targets [B]
                output = targets.copy()
            else:
                # Sequence case: logits [B, S, C], targets [B, S]
                # Keep same shape as targets
                output = targets.copy()

        return output
