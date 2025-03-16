from flagscale.runner.estimator.meta_attention import SelfAttention
from flagscale.runner.estimator.meta_base import MetaModule, MetaTensor, get_registry
from flagscale.runner.estimator.meta_mlp import MLP, SwiGLUMLP
from flagscale.runner.estimator.meta_modules import LayerNorm, RMSNorm


class TransformerLayer(MetaModule):
    """
    A single transformer layer based on MetaTensor for resource estimation.

    This implementation follows the structure of modern transformer architectures,
    supporting both pre-normalization (GPT-style) and post-normalization variants:

    Pre-normalization (default):
    1. Input → LayerNorm → Attention → Residual → LayerNorm → MLP → Residual → Output

    Post-normalization:
    1. Input → Attention → Residual → LayerNorm → MLP → Residual → LayerNorm → Output
    """

    def __init__(
        self,
        config,
        layer_number=0,
        model_id="default",
    ):
        """
        Initialize a transformer layer.

        Parameters:
        -----------
        config : object
            Configuration object containing transformer parameters
        layer_number : int, optional
            Layer number in the transformer stack
        model_id : str, optional
            Identifier for the model
        pre_normalization : bool, optional
            Whether to use pre-normalization (GPT-style) or post-normalization
        """
        # Extract tensor parallel settings
        super().__init__(None, model_id)

        # Store configuration and layer number
        self.config = config
        self.layer_number = layer_number
        self.pre_normalization = getattr(config, "pre_normalization", True)

        # Extract model architecture parameters
        self.hidden_size = config.hidden_size
        self.ffn_hidden_size = getattr(config, "ffn_hidden_size", 4 * self.hidden_size)
        self.num_attention_heads = config.num_attention_heads
        self.layernorm_epsilon = getattr(config, "layernorm_epsilon", 1e-5)

        # Determine normalization type
        self.norm_type = getattr(config, "norm_type", "layernorm").lower()
        if self.norm_type == "rmsnorm":
            NormClass = RMSNorm
            norm_bias = False
        else:
            NormClass = LayerNorm
            norm_bias = True

        # Extract dropout parameters
        self.hidden_dropout = getattr(config, "hidden_dropout", 0.1)

        # Determine activation type
        activation_type = getattr(config, "activation_func", "gelu")

        # Create normalization layers
        self.attention_norm = NormClass(
            normalized_shape=self.hidden_size,
            eps=self.layernorm_epsilon,
            shard_specs=[[1, 1]],
            model_id=model_id,
        )

        self.mlp_norm = NormClass(
            normalized_shape=self.hidden_size,
            eps=self.layernorm_epsilon,
            shard_specs=[[1, 1]],
            model_id=model_id,
        )

        # Self Attention
        self.self_attention = SelfAttention(
            config=config,
            model_id=model_id,
        )

        # MLP
        if activation_type.lower() == "swiglu":
            self.mlp = SwiGLUMLP(
                config=config,
                model_id=model_id,
            )
        else:
            self.mlp = MLP(
                config=config,
                model_id=model_id,
            )

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        context=None,
        context_mask=None,
        rotary_pos_emb=None,
        position_ids=None,
        inference_params=None,
    ):
        """
        Implement the forward pass through transformer layer.

        Parameters:
        -----------
        hidden_states : MetaTensor
            Input tensor [batch_size, seq_len, hidden_size]
        attention_mask : MetaTensor, optional
            Attention mask tensor for self-attention
        context : MetaTensor, optional
            Context tensor for cross-attention
        context_mask : MetaTensor, optional
            Mask tensor for cross-attention
        rotary_pos_emb : tuple or None, optional
            Rotary position embeddings
        position_ids : MetaTensor, optional
            Position IDs for rotary embeddings
        inference_params : dict, optional
            Parameters for inference-time optimizations

        Returns:
        --------
        MetaTensor
            Output hidden states after transformer layer processing
        """
        if self.pre_normalization:
            # GPT-style pre-normalization architecture
            # First residual branch: Attention
            residual = hidden_states
            hidden_states = self.attention_norm(hidden_states)

            attention_output = self.self_attention(
                input_tensor=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
            )

            hidden_states = residual + attention_output

            # Second residual branch: MLP
            residual = hidden_states
            hidden_states = self.mlp_norm(hidden_states)

            mlp_output = self.mlp(hidden_states)

            output = residual + mlp_output

        else:
            # Traditional post-normalization architecture
            # First branch: Attention then normalize
            residual = hidden_states

            attention_output = self.self_attention(
                input_tensor=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
            )

            hidden_states = residual + attention_output
            hidden_states = self.attention_norm(hidden_states)

            # Second branch: MLP then normalize
            residual = hidden_states

            mlp_output = self.mlp(hidden_states)

            output = residual + mlp_output
            output = self.mlp_norm(output)

        return output
