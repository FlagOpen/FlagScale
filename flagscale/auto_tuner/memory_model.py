def default_model(strategy, config):
    """
    At present, we provide a memory formula based on llama2,
    where the optimizer is the adam optimizer,
    using mixed precision training
    without context parallel and expert parallel.
    """
    d = strategy["data_parallel_size"]
    use_dist_opt = strategy["use_distributed_optimizer"]
    t = strategy["tensor_model_parallel_size"]
    sp = 1 if strategy["sequence_parallel"] else 0
    pp = strategy["pipeline_model_parallel_size"]

    use_flash_attn = config.train.system.get("use_flash_attn", False)
    use_recompute = strategy["use_recompute"]
    recompute_method = strategy["recompute_method"]
    recompute_granularity = strategy["recompute_granularity"]
    recompute_num_layers = strategy["recompute_num_layers"]
    b = strategy["micro_batch_size"]
    h = config.train.model.hidden_size
    s = config.train.model.seq_length
    model_size_in_b = config.experiment.auto_tuner.memory_model.model_size_in_b

    model_state = (
        (12 / d + 4) / (t * pp) * model_size_in_b
        if use_dist_opt
        else 16 / (t * pp) * model_size_in_b
    )

    use_vpp = 1 if strategy["num_layers_per_virtual_pipeline_stage"] else 0
    vpp_scale = (
        (pp - 1)
        / (
            pp
            * config.train.model.num_layers
            / pp
            / strategy["num_layers_per_virtual_pipeline_stage"]
        )
        if use_vpp
        else 0
    )
    a = config.train.model.num_attention_heads
    L = config.train.model.num_layers

    act = None
    if not use_recompute and not use_flash_attn:
        act = (
            (1 + vpp_scale)
            * L
            / t
            * ((10 * (sp - sp * t + t) + 56 / 3) * s * b * h + 5 * a * s * s * b)
        )
    elif not use_recompute and use_flash_attn:
        act = (1 + vpp_scale) * L / t * ((8 * (sp - sp * t + t) + 38 / 3) * s * b * h)

    elif recompute_method == "uniform" and recompute_granularity == "full":
        act = (1 + vpp_scale) * 2 * s * b * h * L

    elif recompute_method == "block" and recompute_granularity == "full":
        act = (1 + vpp_scale) * 2 * s * b * h * recompute_num_layers + (
            (1 + vpp_scale)
            * (L / pp - recompute_num_layers)
            / t
            * ((10 * (sp - sp * t + t) + 56 / 3) * s * b * h + 5 * a * s * s * b)
        )

    elif recompute_granularity == "selective":
        act = (1 + vpp_scale) * L / t * ((8 * (sp - sp * t + t) + 38 / 3) * s * b * h)

    assert act is not None

    total = int(model_state * 1e3 + act / 1e6)
    return total
