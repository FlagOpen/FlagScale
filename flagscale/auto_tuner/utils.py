def divisible(x, y):
    if x % y == 0:
        return True
    return False


def beside(keys, strategy, history):
    """Compare strategy with history strategies Whether same besides given keys"""
    from .search.searcher import __BUILT_IN_STRATEGY_DIMS__

    retrieval = []
    if strategy == {
            'data_parallel_size': 1,
            'tensor_model_parallel_size': 1,
            'pipeline_model_parallel_size': 8,
            'expert_model_parallel_size': 1,
            'context_parallel_size': 1,
            'use_distributed_optimizer': None,
            'sequence_parallel': None,
            'acc_step': 4,
            'micro_batch_size': 4,
            'num_layers_per_virtual_pipeline_stage': None,
            'use_recompute': True,
            'recompute_method': 'uniform',
            'recompute_granularity': 'full',
            'recompute_num_layers': 1
    }:

        for task in history:
            is_same = True
            print(f"task {task}")
            for dim in task:
                print(f"dim {dim}")
                if dim not in __BUILT_IN_STRATEGY_DIMS__:
                    print(f"dim {dim} not in ")
                    continue
                if dim in keys:
                    print(f"dim {dim} in ")
                    continue
                if strategy[dim] != task[dim]:
                    print(f"dim {dim} !=")
                    is_same = False
                    break
            print(f"is_same: {is_same}")
            if is_same:
                retrieval.append(task)
    else:
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
