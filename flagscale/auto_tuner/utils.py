def divisible(x, y):
    if x % y == 0:
        return True
    return False


def beside(keys, strategy, history):
    """Compare strategy with history strategies Whether same besides given keys"""
    from .search.searcher import __BUILT_IN_STRATEGY_DIMS__

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
    return (
        -strategy["tensor_model_parallel_size"],
        -strategy["pipeline_model_parallel_size"],
        -strategy["use_recompute"],
        strategy["micro_batch_size"],
    )


def sort_by_performance(strategy):
    magic_number = 4
    return (
        -strategy["use_recompute"],
        (strategy["tensor_model_parallel_size"] % magic_number),
        (strategy["micro_batch_size"] % magic_number),
        strategy["pipeline_model_parallel_size"],
        (
            strategy["recompute_num_layers"]
            if strategy["recompute_num_layers"]
            else float("inf")
        ),
    )
