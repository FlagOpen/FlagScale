from flagscale.runner.auto_tuner.utils import convert_config_to_megatron_args
from flagscale.train.theoretical_memory_usage import report_theoretical_memory


def default_model(strategy, config):
    """Use megatron built in memory model."""
    args = convert_config_to_megatron_args(config, strategy)
    num_microbatches = (
        config.train.model.global_batch_size
        // strategy["data_parallel_size"]
        // strategy["micro_batch_size"]
    )
    total_memory = report_theoretical_memory(args, num_microbatches=num_microbatches)
    return total_memory
