import os
import sys
from types import SimpleNamespace


def divisible(x, y):
    if x % y == 0:
        return True
    return False


def beside(keys, strategy, history):
    """Compare strategy with history strategies Whether same besides given keys"""
    from flagscale.runner.auto_tuner.search.searcher import __BUILT_IN_STRATEGY_DIMS__

    retrieval = []
    for task in history:
        is_same = True
        for dim in task:
            if dim not in __BUILT_IN_STRATEGY_DIMS__:
                continue
            if dim in keys:
                continue
            if strategy[dim] != task[dim]:
                is_same = False
                break
        if is_same:
            retrieval.append(task)
    return retrieval


def sort_by_memory(strategy):
    """Sort strategy by memory."""
    return (
        -strategy["use_recompute"],
        -strategy["tensor_model_parallel_size"],
        (
            -strategy["sequence_parallel"]
            if strategy["sequence_parallel"] is not None
            else -float("inf")
        ),
        strategy["micro_batch_size"],
        -strategy["pipeline_model_parallel_size"],
        strategy["data_parallel_size"],
        (
            -strategy["use_distributed_optimizer"]
            if strategy["use_distributed_optimizer"] is not None
            else -float("inf")
        ),
    )


def sort_by_memory_model(strategy):
    """Sort strategy by memory_model."""
    return strategy["memory_model"]


def sort_by_performance(strategy):
    """Sort strategy by performance potentially."""
    return (
        strategy["use_recompute"],
        -strategy["tensor_model_parallel_size"],
        (
            -strategy["sequence_parallel"]
            if strategy["sequence_parallel"] is not None
            else -float("inf")
        ),
        strategy["micro_batch_size"],
        -strategy["pipeline_model_parallel_size"],
        -strategy["data_parallel_size"],
        (
            strategy["recompute_num_layers"]
            if strategy["recompute_num_layers"] is not None
            else float("inf")
        ),
    )


def compare_by_recompute(strategy1, strategy2):
    """Compare two strategies by recompute. If strategy1 is larger memory, return True."""
    result = False
    # Strategy1 not use recompute, Strategy2 use recompute
    if not strategy1["use_recompute"] and strategy2["use_recompute"]:
        result = True
    # Strategy1 use recompute, Strategy2 use recompute
    elif strategy1["use_recompute"] and strategy2["use_recompute"]:
        # Block recompute
        if (
            strategy1["recompute_method"] == "block"
            and strategy2["recompute_method"] == "block"
        ):
            if strategy1["recompute_num_layers"] <= strategy2["recompute_num_layers"]:
                result = True
        elif (
            strategy1["recompute_method"] == "uniform"
            and strategy2["recompute_method"] == "uniform"
        ):
            if (
                strategy1["recompute_granularity"] == "selective"
                and strategy2["recompute_granularity"] == "full"
            ):
                result = True
    elif (
        strategy1["use_recompute"] == strategy2["use_recompute"]
        and strategy1["recompute_method"] == strategy2["recompute_method"]
        and strategy1["recompute_granularity"] == strategy2["recompute_granularity"]
        and strategy1["recompute_num_layers"] == strategy2["recompute_num_layers"]
    ):
        result = True

    return result


def convert_config_to_megatron_args(config, strategy):
    args = SimpleNamespace()
    flagscale_args = config.train.model
    args.hidden_size = flagscale_args.hidden_size
    args.num_attention_heads = flagscale_args.num_attention_heads
    args.num_layers = flagscale_args.num_layers
    args.use_flash_attn = config.train.system.get("use_flash_attn", False)

    if "kv_channels" not in flagscale_args:
        assert args.hidden_size % args.num_attention_heads == 0
        args.kv_channels = args.hidden_size // args.num_attention_heads
    else:
        args.kv_channels = flagscale_args["kv_channels"]

    if "group_query_attention" not in flagscale_args:
        args.group_query_attention = None
    else:
        args.group_query_attention = flagscale_args["group_query_attention"]

    if "num_experts" not in flagscale_args:
        args.num_experts = None
    else:
        args.num_experts = flagscale_args["num_experts"]

    if "swiglu" not in flagscale_args:
        args.swiglu = False
    else:
        args.swiglu = flagscale_args["swiglu"]

    if "multiple_of" not in flagscale_args:
        args.multiple_of = None
    else:
        args.multiple_of = flagscale_args["multiple_of"]

    if "hidden_dim_multiplier" not in flagscale_args:
        args.hidden_dim_multiplier = None
    else:
        args.hidden_dim_multiplier = flagscale_args["hidden_dim_multiplier"]

    if "ffn_hidden_size" not in flagscale_args:
        if args.swiglu:
            if args.multiple_of is not None:
                hidden_dim = int(4 * args.hidden_size * 2 / 3)
                if args.hidden_dim_multiplier is not None:
                    assert (
                        args.hidden_dim_multiplier > 0
                    ), "multiplier for hidden dim should be greater than zero"
                    hidden_dim = int(hidden_dim * args.hidden_dim_multiplier)
                args.ffn_hidden_size = args.multiple_of * (
                    (hidden_dim + args.multiple_of - 1) // args.multiple_of
                )
            else:
                args.ffn_hidden_size = int((4 * args.hidden_size * 2 / 3) / 64) * 64
        else:
            args.ffn_hidden_size = 4 * args.hidden_size
    else:
        args.ffn_hidden_size = flagscale_args["ffn_hidden_size"]

    if "make_vocab_size_divisible_by" not in flagscale_args:
        args.make_vocab_size_divisible_by = 128
    else:
        args.make_vocab_size_divisible_by = flagscale_args[
            "make_vocab_size_divisible_by"
        ]
    args.tensor_model_parallel_size = strategy["tensor_model_parallel_size"]
    if "padded_vocab_size" not in flagscale_args:
        # To append megatron path to PYTHONPATH
        autotuner_dir = os.path.dirname(__file__)
        great_grandparent_dir = os.path.dirname(
            os.path.dirname(os.path.dirname(autotuner_dir))
        )
        sys.path.insert(0, os.path.join(great_grandparent_dir, "megatron"))
        from megatron.training.tokenizer.tokenizer import _vocab_size_with_padding

        args.rank = -1
        args.padded_vocab_size = _vocab_size_with_padding(
            config.train.data.tokenizer.vocab_size, args
        )
    else:
        args.padded_vocab_size = flagscale_args["padded_vocab_size"]

    if "untie_embeddings_and_output_weights" not in flagscale_args:
        args.untie_embeddings_and_output_weights = None
    else:
        args.untie_embeddings_and_output_weights = flagscale_args[
            "untie_embeddings_and_output_weights"
        ]

    args.pipeline_model_parallel_size = strategy["pipeline_model_parallel_size"]
    args.data_parallel_size = strategy["data_parallel_size"]
    args.use_distributed_optimizer = strategy["use_distributed_optimizer"]
    args.seq_length = flagscale_args.seq_length
    args.micro_batch_size = strategy["micro_batch_size"]
    if strategy["num_layers_per_virtual_pipeline_stage"] is not None:
        args.virtual_pipeline_model_parallel_size = (
            flagscale_args.num_layers
            // args.pipeline_model_parallel_size
            // strategy["num_layers_per_virtual_pipeline_stage"]
        )
    else:
        args.virtual_pipeline_model_parallel_size = None

    args.sequence_parallel = strategy["sequence_parallel"]
    args.recompute_granularity = strategy["recompute_granularity"]
    args.recompute_method = strategy["recompute_method"]
    args.recompute_num_layers = strategy["recompute_num_layers"]

    return args
