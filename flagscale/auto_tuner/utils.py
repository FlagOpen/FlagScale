def divisible(x, y):
    if x % y == 0:
        return True
    return False


def beside(keys, strategy, history):
    """Compare strategy with history strategies Whether same besides given keys"""
    from .search.searcher import __BUILT_IN_STRATEGY_DIMS__

    retrieval = []
    is_same = True
    for task in history:
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
    return (
        -strategy["tensor_model_parallel_size"],
        -strategy["pipeline_model_parallel_size"],
        -strategy["num_layers_per_virtual_pipeline_stage"],
        -strategy["use_distributed_optimizer"],
        strategy["micro_batch_size"],
        -strategy["use_recompute"],
    )


def sort_by_performance(strategy):
    return (
        -strategy["micro_batch_size"],
        strategy["use_recompute"],
        strategy["tensor_model_parallel_size"],
        strategy["pipeline_model_parallel_size"],
    )
