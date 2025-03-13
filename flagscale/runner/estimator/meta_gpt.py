from flagscale.runner.estimator.meta_base import MetaModule, MetaTensor, ShardedDim
from flagscale.runner.estimator.meta_modules import Embedding, Linear, LayerNorm, RMSNorm, Dropout
from flagscale.runner.estimator.meta_transformer_layer import TransformerLayer


class GPTModel(MetaModule):
    """
    GPT model architecture built on MetaTensor for resource estimation.
    
    This implementation follows the architecture of Megatron-LM's GPT model:
    1. Token embeddings + position embeddings
    2. Multiple transformer layers
    3. Final layer normalization
    4. Optional language modeling head (prediction of next token)
    
    This MetaTensor-based implementation enables accurate estimation of
    computational resources required for training and inference.
    """
    def __init__(
        self,
        config,
        model_id="default"
    ):
        """
        Initialize a GPT model with the provided configuration.
        
        Parameters:
        -----------
        config : object
            Configuration object containing model parameters
        model_id : str, optional
            Identifier for the model
        """
        # Extract tensor parallel settings
        super().__init__(None, model_id)
        
        # Store configuration
        self.config = config
        
        # Extract architecture parameters
        self.hidden_size = config.hidden_size
        self.num_layers = config.num_layers
        self.vocab_size = config.vocab_size
        self.max_seq_length = getattr(config, 'max_position_embeddings', 2048)
        self.layernorm_epsilon = getattr(config, 'layernorm_epsilon', 1e-5)
        self.activation_func = getattr(config, 'activation_func', 'gelu')
        
        # Determine normalization type
        self.norm_type = getattr(config, 'norm_type', 'layernorm').lower()
        if self.norm_type == 'rmsnorm':
            NormClass = RMSNorm
            norm_bias = False
        else:
            NormClass = LayerNorm
            norm_bias = True 
        
        # Embedding dropout rate
        self.embedding_dropout_rate = getattr(config, 'embedding_dropout', 0.1)
        
        # Create embedding components
        self.word_embeddings = Embedding(
            num_embeddings=self.vocab_size,
            embedding_dim=self.hidden_size,
            shard_specs=[[self.config.tensor_parallel_size, 1]],
            model_id=model_id,
        )
        
        # Create position embeddings if not using rotary embeddings
        self.use_rotary_position_embeddings = getattr(config, 'use_rotary_position_embeddings', False)
        if not self.use_rotary_position_embeddings:
            self.position_embeddings = Embedding(
                num_embeddings=self.max_seq_length,
                embedding_dim=self.hidden_size,
                shard_specs=[[1, 1]],
                model_id=model_id,
            )
        else:
            self.position_embeddings = None
        
        # Embedding dropout
        self.embedding_dropout = Dropout(
            p=self.embedding_dropout_rate,
            shard_specs=None,
            model_id=model_id,
        )
        
        # Create transformer layers
        self.layers = []
        for i in range(self.num_layers):
            self.layers.append(
                TransformerLayer(
                    config=config,
                    layer_number=i,
                    model_id=model_id,
                )
            )
        
        # Final layer norm
        self.final_norm = NormClass(
            normalized_shape=self.hidden_size,
            eps=self.layernorm_epsilon,
            bias=norm_bias,
            shard_specs=[[1, 1]],
            model_id=model_id,
        )
        

        self.output_layer = Linear(
            in_features=self.hidden_size,
            out_features=self.word_embeddings.num_embeddings,
            bias=False,  # GPT typically doesn't use bias in output layer
            shard_specs=[[1, self.config.tensor_parallel_size]],
            model_id=model_id,
        )
        if not config.untie_embeddings_and_output_weights:
            # Output layer (Language Model Head)
            self.output_layer.share_params()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        token_type_ids=None,
        lm_labels=None,
        inference_params=None,
    ):
        """
        Process input through the GPT model.
        Updates registry with computed metrics.
        
        Parameters:
        -----------
        input_ids : MetaTensor
            Input tokens with shape [batch_size, seq_len]
        attention_mask : MetaTensor, optional
            Attention mask with shape [batch_size, 1, 1, seq_len] or [batch_size, 1, seq_len, seq_len]
        position_ids : MetaTensor, optional
            Position indices with shape [batch_size, seq_len]
        token_type_ids : MetaTensor, optional
            Token type IDs with shape [batch_size, seq_len]
        lm_labels : MetaTensor, optional
            Language modeling labels with shape [batch_size, seq_len]
        inference_params : dict, optional
            Parameters for inference-time optimizations
            
        Returns:
        --------
        MetaTensor or tuple
            Output tensor(s) after model processing
        """
        # Get batch size and sequence length
        batch_size = input_ids[0].dim
        seq_length = input_ids[1].dim
        
        # Get pipeline parallel size from config
        pp_size = getattr(self.config, 'pipeline_parallel_size', 1)

        # Determine which pipeline stage this is
        # Default to full model (all stages) if pipeline_rank not provided
        pipeline_rank = getattr(self.config, 'pipeline_rank', 0)

        # Create position IDs if not provided
        if position_ids is None:
            position_ids = MetaTensor(
                shape=[batch_size, seq_length],
                shard_spec=[input_ids[0].shard, input_ids[1].shard]
            )
        
        # Only do embedding layer in first pipeline stage (or if not using pipeline parallel)
        if pipeline_rank == 0:
            # 1. Get token embeddings
            hidden_states = self.word_embeddings(input_ids)
            
            # 2. Add position embeddings if not using rotary
            if not self.use_rotary_position_embeddings and self.position_embeddings is not None:
                position_embeddings = self.position_embeddings(position_ids)
                hidden_states = hidden_states + position_embeddings
            
            # 3. Apply embedding dropout
            hidden_states = self.embedding_dropout(hidden_states)
        else:
            # For non-first pipeline stages, expect hidden_states as input
            # Simulate the shape that would come from previous stage
            hidden_states = MetaTensor(
                shape=[batch_size, seq_length, self.hidden_size],
                shard_spec=[input_ids[0].shard, input_ids[1].shard, 1]
            )
        
        # Calculate layers for this pipeline stage
        if pp_size > 1:
            assert self.num_layers % pp_size == 0, "Number of layers must be divisible by pipeline parallel size"
            layers_per_stage = self.num_layers // pp_size
            start_layer = pipeline_rank * layers_per_stage
            end_layer = min(start_layer + layers_per_stage, self.num_layers)
            layers_range = range(start_layer, end_layer)
        else:
            # Use all layers if not using pipeline parallel or if simulating all stages
            layers_range = range(len(self.layers))
        
        # 4. Process through transformer layers for this stage
        for i in layers_range:
            hidden_states = self.layers[i](
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids if self.use_rotary_position_embeddings else None,
                inference_params=inference_params
            )
        
        # Only do final norm and output layer in last pipeline stage (or if not using pipeline parallel)
        if pp_size == 1 or pipeline_rank == pp_size - 1:
            # 5. Apply final normalization
            hidden_states = self.final_norm(hidden_states)
            
            # 6. Apply language modeling head (output layer)
            logits = self.output_layer(hidden_states)
            return logits
        else:
            # For non-final stages, just return hidden states
            return hidden_states
    