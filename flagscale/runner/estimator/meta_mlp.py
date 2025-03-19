from flagscale.runner.estimator.meta_base import MetaModule
from flagscale.runner.estimator.meta_modules import GELU, Linear, SwiGLU
from flagscale.runner.estimator.meta_registry import get_registry, register_model
from flagscale.runner.estimator.meta_tensor import MetaTensor


class MLP(MetaModule):
    """
    Multi-Layer Perceptron (MLP) module.

    Standard MLP block used in transformer architectures with:
    1. First linear projection that expands dimensions
    2. GELU activation function
    3. Dropout layer
    4. Second linear projection that reduces dimensions back
    5. Final dropout layer
    """

    def __init__(
        self,
        config,
        model_id="default",
    ):
        super().__init__(None, model_id)
        self.config = config

        # Create sub-modules with appropriate sharding
        self.fc1 = Linear(
            in_features=self.config.hidden_size,
            out_features=self.config.ffn_hidden_size,
            bias=self.config.add_linear_bias,
            shard_specs=[[1, self.config.tensor_parallel_size]],
            model_id=model_id,
        )
        self.gelu = GELU(
            approximate=self.config.activation_func,
            shard_specs=None,
            model_id=model_id,
        )
        self.fc2 = Linear(
            in_features=self.config.ffn_hidden_size,
            out_features=self.config.hidden_size,
            bias=self.config.add_linear_bias,
            shard_specs=[[self.config.tensor_parallel_size, 1]],
            model_id=model_id,
        )

    def forward(self, input: MetaTensor):
        """
        Process input through the MLP block.
        Updates registry with computed metrics.

        Parameters:
        -----------
        input : MetaTensor
            Input tensor [batch_size, seq_len, hidden_size]

        Returns:
        --------
        MetaTensor
            Output tensor after MLP processing [batch_size, seq_len, hidden_size]
        """
        x = self.fc1(input)
        x = self.gelu(x)
        x = self.fc2(x)
        # keep the first dimension as it is since it is dp applied and unshard the rest
        x = x.unshard(start=1)
        return x


class SwiGLUMLP(MetaModule):
    """
    Multi-Layer Perceptron (MLP) module with SwiGLU activation.

    SwiGLU-based MLP block used in modern transformer architectures:
    1. Split projection into gate and value pathways
    2. Apply SwiGLU activation (Swish(gate) * value)
    3. Projection back to hidden size
    4. Optional dropout
    """

    def __init__(
        self,
        config,
        model_id="default",
    ):
        super().__init__(None, model_id)
        self.config = config

        # Calculate intermediate size for SwiGLU if not explicitly provided
        # For SwiGLU, we typically use 2/3 expansion factor per tensor (4/3 combined)
        # to keep parameter count similar to standard 4x MLP
        self.intermediate_size = getattr(
            self.config, "ffn_hidden_size_swiglu", (4 * self.config.hidden_size) // 3
        )

        # Create sub-modules with appropriate sharding
        self.gate_proj = Linear(
            in_features=self.config.hidden_size,
            out_features=self.intermediate_size,
            bias=self.config.add_linear_bias,
            shard_specs=[[1, self.config.tensor_parallel_size]],
            model_id=model_id,
        )

        self.value_proj = Linear(
            in_features=self.config.hidden_size,
            out_features=self.intermediate_size,
            bias=self.config.add_linear_bias,
            shard_specs=[[1, self.config.tensor_parallel_size]],
            model_id=model_id,
        )

        self.swiglu = SwiGLU(
            shard_specs=None,
            model_id=model_id,
        )

        self.out_proj = Linear(
            in_features=self.intermediate_size,
            out_features=self.config.hidden_size,
            bias=self.config.add_linear_bias,
            shard_specs=[[self.config.tensor_parallel_size, 1]],
            model_id=model_id,
        )

    def forward(self, input: MetaTensor):
        """
        Process input through the SwiGLU MLP block.
        Updates registry with computed metrics.

        Parameters:
        -----------
        input : MetaTensor
            Input tensor [batch_size, seq_len, hidden_size]

        Returns:
        --------
        MetaTensor
            Output tensor after SwiGLU MLP processing [batch_size, seq_len, hidden_size]
        """
        # Forward pass
        gate = self.gate_proj(input)
        value = self.value_proj(input)
        swiglu_out = self.swiglu(gate, value)
        output = self.out_proj(swiglu_out)

        # keep the first dimension as it is since it is dp applied and unshard the rest
        output = output.unshard(start=1)
        return output
