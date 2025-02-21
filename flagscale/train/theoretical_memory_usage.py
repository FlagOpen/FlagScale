"""
Computes theoretical memory footprint for model training referring to megatron.
Reference: https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/training/theoretical_memory_usage.py
Activation memory is optimized with adding block recompute formula.
"""

import math

NUM_BYTES_IN_MEGABYTE = 1024 * 1024


def compute_weight_and_optimizer_memory(args, verbose=False):
    """Use megatron directly."""
    # NOTE: This function is the same as megatron
    # Attention projection size.
    query_projection_size = args.kv_channels * args.num_attention_heads
    query_projection_to_hidden_size_ratio = query_projection_size / args.hidden_size
    # Group Query Attention.
    if not args.group_query_attention:
        args.num_query_groups = args.num_attention_heads
    # MoE.
    num_experts = 1 if args.num_experts is None else args.num_experts
    gated_linear_multiplier = 3 / 2 if args.swiglu else 1
    num_parameters_in_transformer_layers = (
        2
        * args.num_layers
        * args.hidden_size
        * args.hidden_size
        * (
            # Attention.
            (
                (1 + (args.num_query_groups / args.num_attention_heads))
                * query_projection_to_hidden_size_ratio
            )
            # MLP.
            + (
                (args.ffn_hidden_size / args.hidden_size)
                * num_experts
                * gated_linear_multiplier
            )
            # Transformer layernorms.
            + (2 / args.hidden_size)
            # Final layernorm.
            + (1 / (args.num_layers * args.hidden_size))
        )
    )
    embedding_size = args.hidden_size * args.padded_vocab_size
    if args.untie_embeddings_and_output_weights:
        num_parameters_in_embedding_layers = 2 * embedding_size
    else:
        num_parameters_in_embedding_layers = embedding_size
    num_total_parameters = (
        num_parameters_in_transformer_layers + num_parameters_in_embedding_layers
    )
    if verbose:
        print(
            f"Number of parameters in transformer layers in billions: "
            f"{num_parameters_in_transformer_layers / 10**9: .2f}"
        )
        print(
            f"Number of parameters in embedding layers in billions: "
            f"{num_parameters_in_embedding_layers / 10**9:.2f}"
        )
        print(
            f"Total number of parameters in billions: {num_total_parameters / 10**9:.2f}"
        )

    # Most loaded model shard has (1/pp_size transformer layers + 1 embedding layer) / tp_size.
    num_parameters_on_most_loaded_model_shard = (
        (num_parameters_in_transformer_layers / args.pipeline_model_parallel_size)
        + embedding_size
    ) / args.tensor_model_parallel_size
    if (
        args.untie_embeddings_and_output_weights
        and args.pipeline_model_parallel_size == 1
    ):
        num_parameters_on_most_loaded_model_shard += (
            embedding_size / args.tensor_model_parallel_size
        )
    if verbose:
        print(
            f"Number of parameters in most loaded shard in billions: "
            f"{num_parameters_on_most_loaded_model_shard / 10**9:.4f}"
        )

    if args.pipeline_model_parallel_size > 1:
        # Other shards just have (1/pp_size transformer layers) / tp_size.
        num_parameters_on_other_model_shards = num_parameters_in_transformer_layers / (
            args.pipeline_model_parallel_size * args.tensor_model_parallel_size
        )
        if verbose:
            print(
                f"Number of parameters in other shards in billions: "
                f"{num_parameters_on_other_model_shards / 10**9:.4f}"
            )

    num_bytes_per_parameter = (
        18 if not args.use_distributed_optimizer else 6 + (12 / args.data_parallel_size)
    )
    weight_and_optimizer_memory = (
        num_parameters_on_most_loaded_model_shard * num_bytes_per_parameter
    )

    return weight_and_optimizer_memory


def compute_activation_memory(args, num_microbatches, verbose=False):
    # Using formula in Table 2 of https://arxiv.org/pdf/2205.05198.pdf.
    # We are trying to compute the maximum activation footprint, so all calculations in this
    # function are for the first pipeline stage.

    # Megatron TODO: This function needs to take into account query_projection_size potentially being
    # different from hidden_size.
    # But in FlagScale, we implement it.

    # Memory footprint from transformer layer (self-attention and MLP).
    gated_linear_multiplier = 3 / 2 if args.swiglu else 1

    # In FlagScale, we provide the most detailed memory information, including each sub layer of transformer layer.
    # Attention:
    # 1. Q, K, V matrix multiplies
    QKV_activation_memory = (
        2 * args.seq_length * args.micro_batch_size * args.hidden_size
    )
    # 2. QKT matrix multiply
    QKT_activation_memory = (
        2
        * args.micro_batch_size
        * args.num_attention_heads
        * args.seq_length
        * args.kv_channels
        + 2
        * args.micro_batch_size
        * args.num_query_groups
        * args.kv_channels
        * args.seq_length
    )
    # 3. Softmax
    softmax_activation_memory = (
        2
        * args.micro_batch_size
        * args.num_attention_heads
        * args.seq_length
        * args.seq_length
    )
    # 4. Softmax Dropout
    softmax_dropout_activation_memory = (
        args.micro_batch_size
        * args.num_attention_heads
        * args.seq_length
        * args.seq_length
    )
    # 5. Attention over V
    attention_over_V_activation_memory = (
        2
        * args.micro_batch_size
        * args.num_attention_heads
        * args.seq_length
        * args.seq_length
        + 2
        * args.micro_batch_size
        * args.num_query_groups
        * args.kv_channels
        * args.seq_length
    )
    # 6. Linear
    linear_activation_memory = (
        2
        * args.micro_batch_size
        * args.num_attention_heads
        * args.seq_length
        * args.kv_channels
    )
    # 7. linear dropout
    linear_dropout_activation_memory = (
        args.seq_length * args.micro_batch_size * args.hidden_size
    )
    # Split into two parts
    attention_parallel_by_tp_activation_memory = (
        QKT_activation_memory
        + softmax_activation_memory
        + softmax_dropout_activation_memory
        + attention_over_V_activation_memory
        + linear_activation_memory
    )
    attnetion_not_parallel_by_tp_activation_memory = (
        QKV_activation_memory + linear_dropout_activation_memory
    )

    # FFN
    # In FFN, we split memory into two parts, one that can paralleled by tensor parallellism and the other that can't
    ffn_parallel_by_tp_activation_memory = (
        4 * args.seq_length * args.micro_batch_size * gated_linear_multiplier
    )
    ffn_not_parallel_by_tp_activation_memory = (
        3 * args.seq_length * args.micro_batch_size * args.hidden_size
    )

    # LayerNorm
    layernorm_activation_memory = (
        4 * args.seq_length * args.micro_batch_size * args.hidden_size
    )
    # Memory can be paralleled of tensor parallelism
    parallel_by_tp_activation_memory = (
        attention_parallel_by_tp_activation_memory
        + ffn_parallel_by_tp_activation_memory
    )
    not_parallel_by_tp_activation_memory = (
        attnetion_not_parallel_by_tp_activation_memory
        + ffn_not_parallel_by_tp_activation_memory
        + layernorm_activation_memory
    )
    # Meory of bass
    bass_activation_memory = (
        5
        * args.micro_batch_size
        * args.num_attention_heads
        * args.seq_length
        * args.seq_length
    )

    # Now add activation memory required for input embeddings, last LayerNorm and output layer.
    # Input to embedding
    embedding_activation_memory = 8 * args.seq_length * args.micro_batch_size
    dropout_activation_memory = (
        args.seq_length * args.micro_batch_size * args.hidden_size
    )
    # Last LayerNorm and inputs to output layer and CE loss.
    output_layer_and_loss_activation_memory = (
        args.seq_length
        * args.micro_batch_size
        * args.hidden_size
        * 4
        * (1 + (args.padded_vocab_size / args.hidden_size))
    )

    # Multiply by interleaved PP memory factor.
    interleaved_schedule_memory_penalty = 1
    in_flight_microbatches = num_microbatches
    if args.virtual_pipeline_model_parallel_size is not None:
        interleaved_schedule_memory_penalty = 1 + (
            (args.pipeline_model_parallel_size - 1)
            / (
                args.pipeline_model_parallel_size
                * args.virtual_pipeline_model_parallel_size
            )
        )
        in_flight_microbatches = math.ceil(
            interleaved_schedule_memory_penalty * args.pipeline_model_parallel_size
        )

    # If using non-interleaved schedule, number of microbatches in pipeline can be less than pp_size,
    # so discount accordingly.
    if (
        args.virtual_pipeline_model_parallel_size is None
        and args.pipeline_model_parallel_size > 1
    ):
        if num_microbatches is not None:
            in_flight_microbatches = min(
                num_microbatches, args.pipeline_model_parallel_size
            )
        else:
            in_flight_microbatches = args.pipeline_model_parallel_size

    activation_memory = None
    if args.recompute_granularity == "selective" or args.use_flash_attn:
        parallel_by_tp_activation_memory -= bass_activation_memory

    if args.sequence_parallel:
        perlayer_activation = (
            parallel_by_tp_activation_memory + not_parallel_by_tp_activation_memory
        ) / args.tensor_model_parallel_size
    else:
        perlayer_activation = (
            parallel_by_tp_activation_memory / args.tensor_model_parallel_size
            + not_parallel_by_tp_activation_memory
        )
    if args.pipeline_model_parallel_size > 1:
        activation_memory = (
            perlayer_activation
            * args.num_layers
            / args.pipeline_model_parallel_size
            * in_flight_microbatches
            + embedding_activation_memory
            + dropout_activation_memory
        )
    else:
        activation_memory = (
            perlayer_activation * args.num_layers
            + embedding_activation_memory
            + dropout_activation_memory
            + output_layer_and_loss_activation_memory
        )

    if args.recompute_method == "uniform" and args.recompute_granularity == "full":
        if args.pipeline_model_parallel_size > 1:
            activation_memory = (
                2 * args.seq_length * args.micro_batch_size * args.hidden_size
                + embedding_activation_memory
                + dropout_activation_memory
            )
        else:
            activation_memory = (
                2 * args.seq_length * args.micro_batch_size * args.hidden_size
                + embedding_activation_memory
                + dropout_activation_memory
                + output_layer_and_loss_activation_memory
            )

    elif args.recompute_method == "block" and args.recompute_granularity == "full":
        recompute_layers = args.recompute_num_layers
        if args.pipeline_model_parallel_size > 1:
            activation_memory = (
                2 * args.seq_length * args.micro_batch_size * args.hidden_size
                + perlayer_activation
                * (
                    args.num_layers / args.pipeline_model_parallel_size
                    - recompute_layers
                )
                * in_flight_microbatches
                + embedding_activation_memory
                + dropout_activation_memory
            )
        else:
            activation_memory = (
                2 * args.seq_length * args.micro_batch_size * args.hidden_size
                + perlayer_activation * (args.num_layers - recompute_layers)
                + embedding_activation_memory
                + dropout_activation_memory
                + output_layer_and_loss_activation_memory
            )
    assert activation_memory is not None
    return activation_memory


def report_theoretical_memory(args, num_microbatches=None, verbose=False):
    weight_and_optimizer_memory = (
        compute_weight_and_optimizer_memory(args, verbose=verbose)
        / NUM_BYTES_IN_MEGABYTE
    )

    activation_memory = (
        compute_activation_memory(
            args, num_microbatches=num_microbatches, verbose=verbose
        )
        / NUM_BYTES_IN_MEGABYTE
    )
    total_memory = weight_and_optimizer_memory + activation_memory
    return int(total_memory)
