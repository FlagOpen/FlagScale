from flagscale.runner.estimator.meta_base import  MetaTensor, MetaModule, get_registry, register_model
from flagscale.runner.estimator.meta_modules import Linear, Dropout, Baddbmm, Softmax, Bmm, RotaryEmbedding, LayerNorm, RMSNorm


class CoreAttention(MetaModule):
    """
    Core attention mechanism.
    
    Performs the central attention computation:
    1. Compute attention scores: query @ key.transpose(-2, -1)
    2. Apply softmax to get attention probabilities
    3. Apply dropout to attention probabilities
    4. Compute context: attention_probs @ value
    
    Does not include the initial projection to Q/K/V or the final output projection.
    """
    def __init__(
        self,
        dropout_prob=0.1,
        shard_specs=None,
        model_id="default",
    ):
        super().__init__(shard_specs, model_id)
        
        self.baddbmm = Baddbmm(shard_specs, model_id)
        self.softmax = Softmax(dim=-1, shard_specs=shard_specs, model_id=model_id)
        self.attention_dropout = Dropout(p=dropout_prob, shard_specs=shard_specs, model_id=model_id)
        self.bmm = Bmm(shard_specs, model_id)
    
    def forward(self, query: MetaTensor, key: MetaTensor, value: MetaTensor, attention_mask=None):
        """
        Process inputs and return core attention output with batch-first format.
        
        Parameters:
        -----------
        query : MetaTensor
            Query tensor [b, sq, np, hn] - (batch_size, seq_len_q, num_heads, head_size)
        key : MetaTensor
            Key tensor [b, sk, ng, hn] - (batch_size, seq_len_k, num_query_groups, head_size)
        value : MetaTensor
            Value tensor [b, sk, ng, hn] - (batch_size, seq_len_k, num_query_groups, head_size)
        attention_mask : MetaTensor, optional
            Attention mask tensor [b, 1, sq, sk] - (batch_size, 1, seq_len_q, seq_len_k)
            
        Returns:
        --------
        MetaTensor
            Context tensor [b, sq, hp] - (batch_size, seq_len_q, hidden_size_per_partition)
        """
        # Extract dimensions with batch as first dimension
        batch_size = query[0].dim
        seq_len_q = query[1].dim
        num_heads = query[2].dim
        head_size = query[3].dim
        
        seq_len_k = key[1].dim
        num_key_value_groups = key[2].dim
        
        # ===================================
        # Handle Group Query Attention (GQA)
        # ===================================
        
        # Calculate heads per group for GQA
        heads_per_group = 1
        if num_heads > num_key_value_groups:
            heads_per_group = num_heads // num_key_value_groups
            
            # Simple expansion of key and value for GQA
            expanded_key = MetaTensor(
                shape=[batch_size, seq_len_k, num_heads, head_size],
                shard_spec=[key[0].shard, key[1].shard, key[2].shard, key[3].shard]
            )
            
            expanded_value = MetaTensor(
                shape=[batch_size, seq_len_k, num_heads, head_size],
                shard_spec=[value[0].shard, value[1].shard, value[2].shard, value[3].shard]
            )
            
            key = expanded_key
            value = expanded_value
        
        # ===================================
        # Raw attention scores using baddbmm
        # ===================================
        
        # Direct creation of tensors for baddbmm operation
        
        # Create input buffer with proper shape [b*np, sq, sk]
        matmul_input_buffer = MetaTensor(
            shape=[batch_size * num_heads, seq_len_q, seq_len_k],
            shard_spec=[
                query[0].shard * query[2].shard,  # Combined batch and head sharding
                query[1].shard,  # seq_len_q sharding
                key[1].shard     # seq_len_k sharding
            ]
        )
        
        # Create query tensor for baddbmm [b*np, sq, hn]
        baddbmm_query = MetaTensor(
            shape=[batch_size * num_heads, seq_len_q, head_size],
            shard_spec=[
                query[0].shard * query[2].shard,  # Combined batch and head
                query[1].shard,  # seq_len_q
                query[3].shard   # head_size
            ]
        )
        
        # Create key tensor for baddbmm [b*np, hn, sk]
        baddbmm_key = MetaTensor(
            shape=[batch_size * num_heads, head_size, seq_len_k],
            shard_spec=[
                key[0].shard * key[2].shard,  # Combined batch and head
                key[3].shard,   # head_size
                key[1].shard    # seq_len_k
            ]
        )
        
        # Compute raw attention scores
        # [b*np, sq, sk] = baddbmm([b*np, sq, sk], [b*np, sq, hn], [b*np, hn, sk])
        matmul_result = self.baddbmm(
            matmul_input_buffer,
            baddbmm_query,
            baddbmm_key,
            beta=0.0,
            alpha=self.softmax_scale
        )
        
        # ===========================
        # Attention probs and dropout
        # ===========================
        
        # Create attention scores with correct shape for softmax
        attention_scores = MetaTensor(
            shape=[batch_size, num_heads, seq_len_q, seq_len_k],
            shard_spec=[
                query[0].shard,  # batch_size
                query[2].shard,  # num_heads
                query[1].shard,  # seq_len_q
                key[1].shard     # seq_len_k
            ]
        )
        
        # Apply attention mask (no actual computation in MetaTensor)
        if attention_mask is not None:
            pass
        
        # Apply softmax
        attention_probs = self.softmax(attention_scores)
        
        # Apply dropout
        attention_probs = self.attention_dropout(attention_probs)
        
        # =========================
        # Context layer using bmm
        # =========================
        
        # Create attention_probs tensor for bmm [b*np, sq, sk]
        bmm_attention_probs = MetaTensor(
            shape=[batch_size * num_heads, seq_len_q, seq_len_k],
            shard_spec=[
                attention_probs[0].shard * attention_probs[1].shard,
                attention_probs[2].shard,
                attention_probs[3].shard
            ]
        )
        
        # Create value tensor for bmm [b*np, sk, hn]
        bmm_value = MetaTensor(
            shape=[batch_size * num_heads, seq_len_k, head_size],
            shard_spec=[
                value[0].shard * value[2].shard,
                value[1].shard,
                value[3].shard
            ]
        )
        
        # Compute context [b*np, sq, hn]
        context = self.bmm(bmm_attention_probs, bmm_value)
        
        # Create final output with batch-first format
        # [b, sq, hp] where hp = np * hn
        hidden_size_per_partition = num_heads * head_size
        context_final = MetaTensor(
            shape=[batch_size, seq_len_q, hidden_size_per_partition],
            shard_spec=[
                query[0].shard,  # batch_size
                query[1].shard,  # seq_len_q
                query[2].shard   # Sharding applies to head dimension
            ]
        )
        
        return context_final


class SelfAttention(MetaModule):
    """
    Self-attention layer similar to Megatron-LM implementation but using MetaTensor.
    
    Self-attention layer takes input with size [b, sq, h]
    and returns output of the same size.
    
    This implementation shares the same structure as Megatron's SelfAttention,
    adapted to work with MetaTensor for resource estimation.
    """
    
    def __init__(
        self,
        config,
        model_id="default"
    ):
        """
        Initialize a self-attention module with configuration parameters.
        
        Parameters:
        -----------
        config : object
            Configuration object containing attention parameters
        model_id : str, optional
            Identifier for the model
        """
        # Initialize parent Attention class with the config
        super().__init__(None, model_id)

        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads

        # Extract optional parameters with defaults
        self.num_query_groups = getattr(config, 'num_query_groups', self.num_attention_heads)
        self.kv_channels = getattr(config, 'kv_channels', self.hidden_size // self.num_attention_heads)
        self.attention_head_size = self.hidden_size // self.num_attention_heads
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.add_qkv_bias = getattr(config, 'add_qkv_bias', False)
        self.add_linear_bias= getattr(config, 'add_linear_bias', True)
        
        # Per attention head and per partition values
        self.hidden_size_per_attention_head = self.kv_channels
        self.query_projection_size = self.kv_channels * self.num_attention_heads
        self.kv_projection_size = self.kv_channels * self.num_query_groups
        
        # Extract QK layernorm parameters
        self.qk_layernorm = getattr(config, 'qk_layernorm', False)
        self.qk_layernorm_dim = getattr(config, 'qk_layernorm_dim', 0)
        self.layernorm_epsilon = getattr(config, 'layernorm_epsilon', 1e-5)
        
        # Extract norm type (LayerNorm or RMSNorm)
        self.norm_type = getattr(config, 'norm_type', 'layernorm').lower()
        # Extract dropout parameters
        attention_dropout_prob = getattr(config, 'attention_dropout_prob', 0.1)
        output_dropout_prob = getattr(config, 'output_dropout_prob', 0.1)
        
        # Softmax scaling - use 1/sqrt(hidden_size_per_attention_head) if not provided
        self.softmax_scale = getattr(config, 'softmax_scale', None)
        if self.softmax_scale is None:
            self.softmax_scale = 1.0 / (self.hidden_size_per_attention_head ** 0.5)
        
        # Rotary embedding configuration
        self.rotary_embedding = getattr(config, 'use_rotary_position_embeddings', False)
        if self.rotary_embedding:
            rotary_embedding_dim = getattr(config, 'rotary_embedding_dim', 0)
            if rotary_embedding_dim <= 0:
                self.rotary_embedding_dim = self.attention_head_size
            else:
                self.rotary_embedding_dim = min(rotary_embedding_dim, self.attention_head_size)
            
            rotary_embedding_base = getattr(config, 'rotary_embedding_base', 10000)
            rotary_embedding_max_seq_len = getattr(config, 'rotary_embedding_max_seq_len', 2048)
            
            self.rope = RotaryEmbedding(
                dim=self.rotary_embedding_dim,
                max_seq_len=rotary_embedding_max_seq_len,
                base=rotary_embedding_base,
                shard_specs=None,
                model_id=model_id,
            )
        
        # Create projection layers
        # For SelfAttention, create the combined QKV projection
        self.query_key_value = Linear(
            in_features=self.hidden_size,
            out_features=self.query_projection_size + 2 * self.kv_projection_size,
            bias=self.add_linear_bias or self.add_qkv_bias,
            shard_specs=[[config.tensor_parallel_size, 1]],
            model_id=model_id,
        )
        
        # Add Q and K LayerNorms if specified
        if self.qk_layernorm:
            # Choose normalization type based on config
            if self.norm_type == 'rmsnorm':
                NormClass = RMSNorm
                # RMSNorm typically doesn't use bias
                norm_bias = False
            else:
                NormClass = LayerNorm
                # LayerNorm typically uses bias
                norm_bias = True
                
            if self.qk_layernorm_dim <= 0:
                # Apply normalization per head
                self.q_layernorm = NormClass(
                    self.hidden_size_per_attention_head,
                    eps=self.layernorm_epsilon,
                    bias=norm_bias,
                    shard_specs=[[1, 1]],
                    model_id=model_id,
                )
                self.k_layernorm = NormClass(
                    self.hidden_size_per_attention_head,
                    eps=self.layernorm_epsilon,
                    bias=norm_bias,
                    shard_specs=[[1, 1]],
                    model_id=model_id,
                )
            else:
                # Apply normalization to the entire projection
                self.q_layernorm = NormClass(
                    self.query_projection_size,
                    eps=self.layernorm_epsilon,
                    bias=norm_bias,
                    shard_specs=[[1, 1]],
                    model_id=model_id,
                )
                self.k_layernorm = NormClass(
                    self.kv_projection_size,
                    eps=self.layernorm_epsilon,
                    bias=norm_bias,
                    shard_specs=[[1, 1]],
                    model_id=model_id,
                )
        else:
            self.q_layernorm = None
            self.k_layernorm = None
        
        # Create core attention module
        self.core_attention = CoreAttention(
            dropout_prob=attention_dropout_prob,
            shard_specs=None,
            model_id=model_id,
        )
        self.core_attention.softmax_scale = self.softmax_scale
        
        # Output projection layer
        self.output_proj = Linear(
            in_features=self.all_head_size,
            out_features=self.hidden_size,
            bias=self.add_linear_bias,
            shard_specs=[[1, config.tensor_parallel_size]],
            model_id=model_id,
        )
        
        # Output dropout
        self.output_dropout = Dropout(
            p=output_dropout_prob,
            shard_specs=None,
            model_id=model_id,
        )

    def split_qkv_tensor(self, mixed_qkv, batch_size, seq_len):
        """
        Split the combined QKV tensor into separate Q, K, V tensors
        with special handling for grouped query attention.
        
        Parameters:
        -----------
        mixed_qkv : MetaTensor
            Combined QKV projection [batch_size, seq_len, total_proj_size]
        batch_size : int
            Batch size
        seq_len : int
            Sequence length
            
        Returns:
        --------
        tuple(MetaTensor, MetaTensor, MetaTensor)
            Query, key, and value tensors with appropriate shapes
        """
        # Calculate projection sizes for split
        query_size = self.query_projection_size
        kv_size = self.kv_projection_size
        
        # Create query tensor
        query = MetaTensor(
            shape=[batch_size, seq_len, query_size],
            shard_spec=[
                mixed_qkv[0].shard,  # batch_size sharding
                mixed_qkv[1].shard,  # seq_len sharding
                mixed_qkv[2].shard   # hidden_size sharding
            ]
        )
        
        # Create key tensor
        key = MetaTensor(
            shape=[batch_size, seq_len, kv_size],
            shard_spec=[
                mixed_qkv[0].shard,  # batch_size sharding
                mixed_qkv[1].shard,  # seq_len sharding
                mixed_qkv[2].shard   # hidden_size sharding
            ]
        )
        
        # Create value tensor
        value = MetaTensor(
            shape=[batch_size, seq_len, kv_size],
            shard_spec=[
                mixed_qkv[0].shard,  # batch_size sharding
                mixed_qkv[1].shard,  # seq_len sharding
                mixed_qkv[2].shard   # hidden_size sharding
            ]
        )
        
        return query, key, value
    
    def reshape_for_attention(self, tensor, batch_size, seq_len, num_heads, head_size):
        """
        Reshape a tensor for multi-head attention computation.
        
        Parameters:
        -----------
        tensor : MetaTensor
            Input tensor [batch_size, seq_len, proj_size]
        batch_size : int
            Batch size
        seq_len : int
            Sequence length
        num_heads : int
            Number of attention heads
        head_size : int
            Size of each attention head
            
        Returns:
        --------
        MetaTensor
            Reshaped tensor [batch_size, seq_len, num_heads, head_size]
        """
        return MetaTensor(
            shape=[batch_size, seq_len, num_heads, head_size],
            shard_spec=[
                tensor[0].shard,  # batch_size sharding
                1, 
                tensor[1].shard,
                tensor[2].shard,
            ]
        )
    
    def apply_qk_layernorm(self, query, key):
        """
        Apply layer normalization to query and key tensors if enabled.
        
        Parameters:
        -----------
        query : MetaTensor
            Query tensor [batch_size, seq_len, num_heads, head_size]
        key : MetaTensor
            Key tensor [batch_size, seq_len, num_groups, head_size]
            
        Returns:
        --------
        tuple(MetaTensor, MetaTensor)
            Normalized query and key tensors
        """
        if not self.qk_layernorm:
            return query, key
        
        if self.qk_layernorm_dim <= 0:
            # Apply layernorm directly to the 4D tensors
            query = self.q_layernorm(query)
            key = self.k_layernorm(key)
        else:
            # Reshape to apply layernorm to the entire projection
            query_shape = list(query.shape)
            key_shape = list(key.shape)
            
            # [b, sq, np, hn] -> [b, sq, 1, np*hn]
            query_reshaped = MetaTensor(
                shape=[query[0].dim, query[1].dim, 1, query[2].dim * query[3].dim],
                shard_spec=[query[0].shard, query[1].shard, 1, query[2].shard]
            )
            
            # [b, sk, ng, hn] -> [b, sk, 1, ng*hn]
            key_reshaped = MetaTensor(
                shape=[key[0].dim, key[1].dim, 1, key[2].dim * key[3].dim],
                shard_spec=[key[0].shard, key[1].shard, 1, key[2].shard]
            )
            
            # Apply layernorm
            query_reshaped = self.q_layernorm(query_reshaped)
            key_reshaped = self.k_layernorm(key_reshaped)
            
            # Reshape back to original shape
            query = MetaTensor(
                shape=query_shape,
                shard_spec=[query[0].shard, query[1].shard, query[2].shard, query[3].shard]
            )
            
            key = MetaTensor(
                shape=key_shape,
                shard_spec=[key[0].shard, key[1].shard, key[2].shard, key[3].shard]
            )
        
        return query, key
    
    def forward(self, input_tensor: MetaTensor, attention_mask=None, position_ids=None):
        """
        Process input and return self-attention output.
        Updates registry with computed metrics.
        
        Parameters:
        -----------
        input_tensor : MetaTensor
            Input tensor [batch_size, seq_len, hidden_size]
        attention_mask : MetaTensor, optional
            Attention mask tensor [batch_size, 1, 1, seq_len] or [batch_size, 1, seq_len, seq_len]
        position_ids : MetaTensor, optional
            Position IDs for rotary embeddings [batch_size, seq_len]
            
        Returns:
        --------
        MetaTensor
            Output tensor after self-attention [batch_size, seq_len, hidden_size]
        """
        batch_size = input_tensor[0].dim
        seq_len = input_tensor[1].dim
        
        # 1. Project input to combined query, key, value
        mixed_qkv = self.query_key_value(input_tensor)
        
        # 2. Split into separate query, key, value projections
        query, key, value = self.split_qkv_tensor(mixed_qkv, batch_size, seq_len)
        
        # 3. Reshape for multi-head attention
        # [batch_size, seq_len, proj_size] -> [batch_size, seq_len, num_heads, head_size]
        query_4d = self.reshape_for_attention(
            query, batch_size, seq_len, self.num_attention_heads, self.attention_head_size
        )
        key_4d = self.reshape_for_attention(
            key, batch_size, seq_len, self.num_query_groups, self.attention_head_size
        )
        value_4d = self.reshape_for_attention(
            value, batch_size, seq_len, self.num_query_groups, self.attention_head_size
        )
        
        # 3a. Apply layernorm to query and key if specified
        if self.qk_layernorm:
            query_4d, key_4d = self.apply_qk_layernorm(query_4d, key_4d)
        
        # 4. Apply rotary embeddings if enabled
        if self.rotary_embedding:
            # Apply rotary embeddings to query and key
            query_4d = self.rope(query_4d, positions=position_ids)
            key_4d = self.rope(key_4d, positions=position_ids)
        
        # 5. Perform core attention computation
        context = self.core_attention(query_4d, key_4d, value_4d, attention_mask)
        
        # 6. Apply output projection and dropout
        output = self.output_proj(context)
        # keep the first dimension as it is since it is dp applied and unshard the rest 
        output = output.unshard(start=1)
        output = self.output_dropout(output)
        
        return output
