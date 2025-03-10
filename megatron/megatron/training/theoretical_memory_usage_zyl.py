# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

"""Computes theoretical memory footprint for model training."""

import os
import math

NUM_BYTES_IN_MEGABYTE = 1024 * 1024


def compute_activated_weight_number(args, verbose=False):
    if args.num_experts is None:
        return

    # Part 1: Attention ======================================================================
    if args.multi_latent_attention is False:
        # Group Query Attention.
        if not args.group_query_attention:
            args.num_query_groups = args.num_attention_heads
        # Attention projection size.
        query_projection_size = args.kv_channels * args.num_attention_heads
        kv_projection_size = args.kv_channels * args.num_query_groups

        # QKV proj
        attn_params = args.hidden_size * (query_projection_size + 2 * kv_projection_size)
        # Out proj
        attn_params += query_projection_size * args.hidden_size

        if args.qk_layernorm:
            if not args.qk_layernorm_hidden_dim:
                attn_params += 2 * query_projection_size // args.num_attention_heads
            else:
                attn_params += query_projection_size
                attn_params += kv_projection_size
    else:
        q_head_dim = args.qk_head_dim + args.qk_pos_emb_head_dim
        if args.q_lora_rank is None:
            attn_params = args.hidden_size * args.num_attention_heads * q_head_dim
        else:
            attn_params = (
                args.hidden_size * args.q_lora_rank
                + args.q_lora_rank * args.num_attention_heads * q_head_dim
            )

        attn_params += (
            + args.hidden_size * (args.kv_lora_rank + args.qk_pos_emb_head_dim)
            + args.kv_lora_rank * args.num_attention_heads * (args.qk_head_dim + args.v_head_dim)
        )

        if args.qk_layernorm and args.q_lora_rank is None:
            attn_params += args.kv_lora_rank
        elif args.qk_layernorm:
            attn_params += args.kv_lora_rank + args.q_lora_rank

        # out proj
        query_projection_size = args.v_head_dim * args.num_attention_heads
        attn_params += (
            query_projection_size * args.hidden_size
        )

    print(
        f"Number of activated attn parameters in a transformer layer in billions: "
        f"{attn_params / 10**9: .2f}"
    )

    # Part 2: MLP or MoE =====================================================================
    shared_expert_ffn_hidden_size = (
        0
        if args.moe_shared_expert_intermediate_size is None
        else args.moe_shared_expert_intermediate_size
    )
    num_experts_routed_to = args.moe_router_topk
    num_experts = 1 if args.num_experts is None else args.num_experts
    gated_linear_multiplier = 3 / 2 if args.swiglu else 1
    num_mtp_predictor = 0 if not args.num_mtp_predictor else args.num_mtp_predictor
    if args.num_experts is None:
        moe_num_layers = 0
        dense_num_layers = args.num_layers
    else:
        if isinstance(args.moe_layer_freq, int):
            moe_num_layers = args.num_layers
            dense_num_layers = 0
        else:
            moe_num_layers = sum(args.moe_layer_freq)
            dense_num_layers = args.num_layers - moe_num_layers

    dense_mlp_params = 0
    if dense_num_layers > 0:
        dense_mlp_params = (
            2
            * args.hidden_size
            * args.hidden_size
            * (
                (args.ffn_hidden_size / args.hidden_size)
                * gated_linear_multiplier
            )
            # norm
            + 2 * args.hidden_size
        )

    sparse_mlp_params = 0
    if moe_num_layers > 0:
        sparse_mlp_params = (
            2
            * args.hidden_size
            * args.hidden_size
            * (
                (args.moe_ffn_hidden_size / args.hidden_size) * num_experts_routed_to * gated_linear_multiplier
                + ((shared_expert_ffn_hidden_size / args.hidden_size) * gated_linear_multiplier)
            )
            # gate
            + args.hidden_size * num_experts
            # norm
            + 2 * args.hidden_size
        )

    print(
        f"Number of activated dense mlp parameters in a transformer layer in billions: "
        f"{dense_mlp_params / 10**9: .2f}"
    )
    print(
        f"Number of activated sparse mlp parameters in a transformer layer in billions: "
        f"{sparse_mlp_params / 10**9: .2f}"
    )

    # Part3: MTP ============================================================================
    mtp_params = 0
    if num_mtp_predictor > 0:
        mtp_params = (
            2 * args.hidden_size # tow norms
            + 2 * args.hidden_size * args.hidden_size # one linear
            + attn_params
            + sparse_mlp_params
            + args.hidden_size # final norm
        )
        print(
            f"Number of activated mtp parameters in a transformer layer in billions: "
            f"{mtp_params / 10**9: .2f}"
        )

    # PART4: TOTAL ===========================================================================
    num_parameters_in_transformer_layers = (
        dense_num_layers * (attn_params + dense_mlp_params)
        + moe_num_layers * (attn_params + sparse_mlp_params)
        + args.hidden_size # final norm
        + num_mtp_predictor * mtp_params
    )
    embedding_size = args.hidden_size * args.padded_vocab_size
    if args.untie_embeddings_and_output_weights:
        num_parameters_in_embedding_layers = 2 * embedding_size
    else:
        num_parameters_in_embedding_layers = embedding_size

    num_total_parameters = num_parameters_in_transformer_layers + num_parameters_in_embedding_layers
    if verbose:
        print(f"{dense_num_layers=}, {moe_num_layers=}, {num_mtp_predictor=}")
        print(
            f"Number of activated parameters in transformer layers in billions: "
            f"{num_parameters_in_transformer_layers / 10**9: .2f}"
        )
        print(
            f"Number of activated parameters in embedding layers in billions: "
            f"{num_parameters_in_embedding_layers / 10**9:.2f}"
        )
        print(f"Total number of activated parameters in billions: {num_total_parameters / 10**9:.2f}")


def compute_weight_and_optimizer_memory(args, verbose=False):
    # Part 1: Attention =======================================================================
    if args.multi_latent_attention is False:
        # Group Query Attention.
        if not args.group_query_attention:
            args.num_query_groups = args.num_attention_heads
        # Attention projection size.
        query_projection_size = args.kv_channels * args.num_attention_heads
        kv_projection_size = args.kv_channels * args.num_query_groups

        # QKV proj
        attn_params = args.hidden_size * (query_projection_size + 2 * kv_projection_size)
        # Out proj
        attn_params += query_projection_size * args.hidden_size

        if args.qk_layernorm:
            if not args.qk_layernorm_hidden_dim:
                attn_params += 2 * query_projection_size // args.num_attention_heads
            else:
                attn_params += query_projection_size
                attn_params += kv_projection_size
    else:
        q_head_dim = args.qk_head_dim + args.qk_pos_emb_head_dim
        if args.q_lora_rank is None:
            attn_params = args.hidden_size * args.num_attention_heads * q_head_dim
        else:
            attn_params = (
                args.hidden_size * args.q_lora_rank
                + args.q_lora_rank * args.num_attention_heads * q_head_dim
            )

        attn_params += (
            + args.hidden_size * (args.kv_lora_rank + args.qk_pos_emb_head_dim)
            + args.kv_lora_rank * args.num_attention_heads * (args.qk_head_dim + args.v_head_dim)
        )

        if args.qk_layernorm and args.q_lora_rank is None:
            attn_params += args.kv_lora_rank
        elif args.qk_layernorm:
            attn_params += args.kv_lora_rank + args.q_lora_rank

        # out proj
        query_projection_size = args.v_head_dim * args.num_attention_heads
        attn_params += (
            query_projection_size * args.hidden_size
        )

    print(
        f"Number of attn parameters in a transformer layer in billions: "
        f"{attn_params / 10**9: .2f}"
    )

    # Part 2: MLP or MoE ====================================================================
    shared_expert_ffn_hidden_size = (
        0
        if args.moe_shared_expert_intermediate_size is None
        else args.moe_shared_expert_intermediate_size
    )
    num_experts = 1 if args.num_experts is None else args.num_experts
    gated_linear_multiplier = 3 / 2 if args.swiglu else 1
    num_mtp_predictor = 0 if not args.num_mtp_predictor else args.num_mtp_predictor
    if args.num_experts is None:
        moe_num_layers = 0
        dense_num_layers = args.num_layers
    else:
        if isinstance(args.moe_layer_freq, int):
            moe_num_layers = args.num_layers
            dense_num_layers = 0
        else:
            moe_num_layers = sum(args.moe_layer_freq)
            dense_num_layers = args.num_layers - moe_num_layers

    dense_mlp_params = 0
    if dense_num_layers > 0:
        dense_mlp_params = (
            2
            * args.hidden_size
            * args.hidden_size
            * (
                (args.ffn_hidden_size / args.hidden_size)
                * gated_linear_multiplier
            )
            # mlp norm
            + 2 * args.hidden_size
        )

    sparse_mlp_params = 0
    if moe_num_layers > 0:
        sparse_mlp_params = (
            2
            * args.hidden_size
            * args.hidden_size
            * (
                (args.moe_ffn_hidden_size / args.hidden_size) * num_experts * gated_linear_multiplier
                + ((shared_expert_ffn_hidden_size / args.hidden_size) * gated_linear_multiplier)
            )
            # gate
            + args.hidden_size * num_experts
            # mlp norm
            + 2 * args.hidden_size
        )

    print(
        f"Number of dense mlp parameters in a transformer layer in billions: "
        f"{dense_mlp_params / 10**9: .2f}"
    )
    print(
        f"Number of sparse mlp parameters in a transformer layer in billions: "
        f"{sparse_mlp_params / 10**9: .2f}"
    )

    # Part3: MTP =============================================================================
    mtp_params = 0
    if num_mtp_predictor > 0:
        mtp_params = (
            2 * args.hidden_size # tow norms
            + 2 * args.hidden_size * args.hidden_size # one linear
            + attn_params + sparse_mlp_params
            + args.hidden_size # final norm
        )
        print(
            f"Number of mtp parameters in a transformer layer in billions: "
            f"{mtp_params / 10**9: .2f}"
        )

    # PART4: TOTAL =============================================================================
    num_parameters_in_transformer_layers = (
        dense_num_layers * (attn_params + dense_mlp_params)
        + moe_num_layers * (attn_params + sparse_mlp_params)
        + args.hidden_size # final norm
        + num_mtp_predictor * mtp_params
    )
    embedding_size = args.hidden_size * args.padded_vocab_size
    if args.untie_embeddings_and_output_weights:
        num_parameters_in_embedding_layers = 2 * embedding_size
    else:
        num_parameters_in_embedding_layers = embedding_size

    num_total_parameters = num_parameters_in_transformer_layers + num_parameters_in_embedding_layers
    if verbose:
        print(f"{dense_num_layers=}, {moe_num_layers=}, {num_mtp_predictor=}")
        print(
            f"Number of parameters in transformer layers in billions: "
            f"{num_parameters_in_transformer_layers / 10**9: .2f}"
        )
        print(
            f"Number of parameters in embedding layers in billions: "
            f"{num_parameters_in_embedding_layers / 10**9:.2f}"
        )
        print(f"Total number of parameters in billions: {num_total_parameters / 10**9:.2f}")

    # Most loaded model shard has (1/pp_size transformer layers + 1 embedding layer) / tp_size.
    num_parameters_on_most_loaded_model_shard = (
        (num_parameters_in_transformer_layers / args.pipeline_model_parallel_size) + embedding_size
    ) / args.tensor_model_parallel_size
    if args.untie_embeddings_and_output_weights and args.pipeline_model_parallel_size == 1:
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

    # TODO: This function needs to take into account query_projection_size potentially being
    # different from hidden_size.

    # Memory footprint from transformer layer (self-attention and MLP).
    activation_memory = (args.seq_length * args.micro_batch_size * args.hidden_size) * (
        18 + (4 * (args.ffn_hidden_size / args.hidden_size))
    )
    if verbose:
        print(
            f"Activation memory footprint per transformer layer: "
            f"{activation_memory / NUM_BYTES_IN_MEGABYTE / args.tensor_model_parallel_size:.1f} MB"
        )
    activation_memory *= args.num_layers

    # Now add activation memory required for input embeddings, last LayerNorm and output layer.

    # Input to embedding (pp_size microbatches in flight).
    activation_memory += (
        8 * args.seq_length * args.micro_batch_size * args.pipeline_model_parallel_size
    )
    # Dropout in embedding layer (pp_size microbatches in flight).
    activation_memory += (
        args.seq_length
        * args.micro_batch_size
        * args.hidden_size
        * args.pipeline_model_parallel_size
    )

    # Multiply by interleaved PP memory factor.
    if args.virtual_pipeline_model_parallel_size is not None:
        interleaved_schedule_memory_penalty = 1 + (
            (args.pipeline_model_parallel_size - 1)
            / (args.pipeline_model_parallel_size * args.virtual_pipeline_model_parallel_size)
        )
        in_flight_microbatches = math.ceil(
            interleaved_schedule_memory_penalty * args.pipeline_model_parallel_size
        )
        if verbose:
            print(
                f"Memory penalty from interleaved schedule: {interleaved_schedule_memory_penalty:.2f}"
            )
            print(f"Number of in-flight microbatches: {in_flight_microbatches}")
        activation_memory *= interleaved_schedule_memory_penalty

    # If using non-interleaved schedule, number of microbatches in pipeline can be less than pp_size,
    # so discount accordingly.
    if args.virtual_pipeline_model_parallel_size is None and args.pipeline_model_parallel_size > 1:
        if num_microbatches is not None:
            activation_memory *= min(1, num_microbatches / args.pipeline_model_parallel_size)
            in_flight_microbatches = min(num_microbatches, args.pipeline_model_parallel_size)
        else:
            in_flight_microbatches = args.pipeline_model_parallel_size
        if verbose:
            print(f"Number of in-flight microbatches: {in_flight_microbatches}")

    if args.pipeline_model_parallel_size == 1:
        # Inputs to output layer and CE loss.
        activation_memory += (
            args.seq_length
            * args.micro_batch_size
            * args.hidden_size
            * 4
            * (1 + (args.padded_vocab_size / args.hidden_size))
        )

    # Activation memory is partitioned by TP size due to tensor and sequence model parallelism.
    return activation_memory / args.tensor_model_parallel_size


def num_floating_point_operations_fwd(args, batch_size):
    # Part 1: Attention ======================================================================
    if args.multi_latent_attention is False:
        # Group Query Attention.
        if not args.group_query_attention:
            args.num_query_groups = args.num_attention_heads
        # Attention projection size.
        query_projection_size = args.kv_channels * args.num_attention_heads
        kv_projection_size = args.kv_channels * args.num_query_groups

        # QKV proj
        num_flops_attn = (
            2
            * batch_size
            * args.seq_length
            * args.hidden_size
            * (query_projection_size + 2 * kv_projection_size)
        )

        # (QK^T)V
        num_flops_attn += (
            2 * batch_size * args.seq_length**2 * query_projection_size
            + 2 * batch_size * args.seq_length**2 * kv_projection_size
        ) // 2

        # Out proj
        num_flops_attn += (
            2
            * batch_size
            * args.seq_length
            * query_projection_size
            * args.hidden_size
        )
    else:
        # QKV proj
        q_head_dim = args.qk_head_dim + args.qk_pos_emb_head_dim
        if args.q_lora_rank is None:
            # linear_q_proj
            num_flops_attn = (
                2
                * batch_size
                * args.seq_length
                * args.hidden_size
                * args.num_attention_heads
                * q_head_dim
            )
        else:
            # linear_q_down_proj + linear_q_up_proj
            num_flops_attn = (
                2
                * batch_size
                * args.seq_length
                * (
                    args.hidden_size * args.q_lora_rank
                    + args.q_lora_rank * args.num_attention_heads * q_head_dim
                )
            )

        # linear_kv_down_proj + linear_kv_up_proj
        num_flops_attn += (
            2
            * batch_size
            * args.seq_length
            * (
                args.hidden_size * (args.kv_lora_rank + args.qk_pos_emb_head_dim)
                + args.kv_lora_rank * args.num_attention_heads * (args.qk_head_dim + args.v_head_dim)
            )
        )

        # (QK^T)V
        num_flops_attn += (
            2 * batch_size * args.seq_length**2 * args.num_attention_heads * q_head_dim
            + 2 * batch_size * args.seq_length**2 * args.num_attention_heads * args.v_head_dim
        ) // 2

        # linear_proj
        query_projection_size = args.v_head_dim * args.num_attention_heads
        num_flops_attn += (
            2
            * batch_size
            * args.seq_length
            * query_projection_size
            * args.hidden_size
        )

    # Part 2: MLP or MoE =====================================================================
    num_experts_routed_to = args.moe_router_topk
    num_experts = 1 if args.num_experts is None else args.num_experts
    gated_linear_multiplier = 3 / 2 if args.swiglu else 1
    shared_expert_ffn_hidden_size = (
        0
        if args.moe_shared_expert_intermediate_size is None
        else args.moe_shared_expert_intermediate_size
    )
    if args.num_experts is None:
        moe_num_layers = 0
        dense_num_layers = args.num_layers
    else:
        if isinstance(args.moe_layer_freq, int):
            moe_num_layers = args.num_layers
            dense_num_layers = 0
        else:
            moe_num_layers = sum(args.moe_layer_freq)
            dense_num_layers = args.num_layers - moe_num_layers

    num_flops_dense_mlp = 0
    if dense_num_layers > 0:
        num_flops_dense_mlp = (
            4 # mlp(two linear)
            * batch_size
            * args.seq_length
            * args.hidden_size
            * args.ffn_hidden_size
            * gated_linear_multiplier
        )

    num_flops_sparse_mlp = 0
    if moe_num_layers > 0:
        num_flops_sparse_mlp = (
            4 # experts(two linear)
            * batch_size
            * args.seq_length
            * args.hidden_size
            * args.moe_ffn_hidden_size
            * gated_linear_multiplier
            * num_experts_routed_to
            + 4 # shared experts (two linear)
            * batch_size
            * args.seq_length
            * args.hidden_size
            * shared_expert_ffn_hidden_size
            * gated_linear_multiplier
            + 2 # gate (one linear)
            * batch_size
            * args.seq_length
            * args.hidden_size
            * num_experts
        )

    num_flops_logits = (
        2
        * batch_size
        * args.seq_length
        * args.padded_vocab_size
        * args.hidden_size
    )

    # Part3: MTP =============================================================================
    num_flops_mtp = 0
    if args.num_mtp_predictor:
        num_flops_mtp = (
            2
            * batch_size
            * args.seq_length
            * args.hidden_size
            * args.hidden_size * 2
            + num_flops_attn
            + num_flops_sparse_mlp
            + num_flops_logits
        )

    return (
        dense_num_layers * (num_flops_attn + num_flops_dense_mlp)
        + moe_num_layers * (num_flops_attn + num_flops_sparse_mlp)
        + num_flops_logits
        + args.num_mtp_predictor * num_flops_mtp
    )


def num_floating_point_operations(args, batch_size):
    num_flops_fwd = num_floating_point_operations_fwd(args, batch_size)
    num_flops_bwd = num_flops_fwd * 2
    _NVTE_FLASH_ATTN = int(os.getenv("NVTE_FLASH_ATTN", "1"))
    if _NVTE_FLASH_ATTN:
        # recompute QK^T
        if args.multi_latent_attention is False:
            # Attention projection size.
            query_projection_size = args.kv_channels * args.num_attention_heads
            num_flops_bwd += (
                2
                * args.num_layers
                * args.batch_size
                * args.seq_length ** 2
                * query_projection_size
            ) // 2
        else:
            q_head_dim = args.qk_head_dim + args.qk_pos_emb_head_dim
            num_flops_bwd += (
                2
                * args.num_layers
                * args.batch_size
                * args.seq_length**2
                * args.num_attention_heads
                * q_head_dim
            ) // 2

    return num_flops_fwd + num_flops_bwd


def report_theoretical_memory(args, num_microbatches=None, verbose=False):
    print("="*30)
    weight_and_optimizer_memory = (
        compute_weight_and_optimizer_memory(args, verbose=verbose) / NUM_BYTES_IN_MEGABYTE
    )

    print("="*30)
    compute_activated_weight_number(args, verbose=verbose)

    batch_size = 2048
    world_size = 64
    elapsed_time_per_iteration = 29.6622 # seconds

    throughput = num_floating_point_operations(args, batch_size) / (
            elapsed_time_per_iteration * 10**12 * world_size)
    print(f"{throughput=}")

    exit(0)

    # Formulae here assume sequence parallelism and selective activation recomputation.
    if not args.sequence_parallel or args.recompute_granularity != 'selective':
        print(
            f"Theoretical memory footprints: weight and optimizer={weight_and_optimizer_memory:.2f} MB"
        )
        return

    activation_memory = (
        compute_activation_memory(args, num_microbatches=num_microbatches, verbose=verbose)
        / NUM_BYTES_IN_MEGABYTE
    )
    total_memory = weight_and_optimizer_memory + activation_memory

    print(
        f"Theoretical memory footprints: weight and optimizer={weight_and_optimizer_memory:.2f} MB, "
        f"activation={activation_memory:.2f} MB, total={total_memory:.2f} MB\n"
    )
