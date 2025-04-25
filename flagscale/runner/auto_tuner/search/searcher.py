import copy
import itertools
import logging
import time

from functools import reduce
from pprint import pprint

from omegaconf import OmegaConf

from flagscale.runner.auto_tuner.memory_model import default_model
from flagscale.runner.auto_tuner.search.algorithm import GridAlgo
from flagscale.runner.auto_tuner.utils import divisible

BUILT_IN_STRATEGY_DIMS = [
    "data_parallel_size",
    "use_distributed_optimizer",
    "tensor_model_parallel_size",
    "sequence_parallel",
    "pipeline_model_parallel_size",
    "num_layers_per_virtual_pipeline_stage",
    "use_recompute",
    "recompute_method",
    "recompute_granularity",
    "recompute_num_layers",
    "micro_batch_size",
    "context_parallel_size",
    "expert_model_parallel_size",
]

_BUILT_IN_SERVE_STRATEGY_DIMS = [
    "tensor_model_parallel_size",
    "pipeline_model_parallel_size",
    "instance",
    "block_size",
    "max_num_batched_tokens",
    "max_num_seqs",
    "swap_space",
]

_DEFAULT_SERVE_TUNE_SPACE = {
    "block_size": [8, 16, 32],
    "max_num_batched_tokens": [512, 1024, 2048],
    "max_num_seqs": [64, 128, 256],
    "swap_space": [0, 4, 8, 16],
}


class Searcher:

    def __init__(self, config):
        # Build search space and set value of each dim
        self.logger = logging.getLogger("FlagScale-AutoTuner")
        self.config = config

        # Build search space
        start_time = time.time()
        self.space = self.build_space(self.config)
        end_time = time.time()
        self.logger.info(
            "Searcher: build search space in {:.2f} seconds and space is {}".format(
                end_time - start_time, self.space
            )
        )

        # Build strategies by Cartesian product search space
        start_time = time.time()
        self.strategies = self.build_strategies(self.space, self.config)
        end_time = time.time()
        self.logger.info(
            "Searcher: build {} candidate strategies in {:.2f} seconds.".format(
                len(self.strategies), end_time - start_time
            )
        )

        if "memory_model" in self.config.experiment.auto_tuner:
            # In the future, the memory model will be loaded by yaml
            model_name = self.config.experiment.auto_tuner.memory_model.get("model_name", "default")
            if model_name != "default":
                raise NotImplementedError(
                    "The memory model {} is not implemented yet.".format(model_name)
                )

            for strategy in self.strategies:
                strategy["memory_model"] = default_model(strategy, self.config)
                self.logger.info(
                    "Searcher: strategy is {}, memory model is {} MB".format(
                        strategy, strategy["memory_model"]
                    )
                )

        # Build search algorithm to explore strategies
        self.algo = self.build_algo(self.strategies, self.config)

    def _sort(self, key, dim, priority=None):
        """Sort the dim according to priority."""
        # NOTE: Vpp and expert degree will be sorted in the future
        if priority is not None:
            if key in ["sequence_parallel"]:
                dim.sort(reverse=True)
            if key in ["data_parallel_size", "recompute_granularity", "micro_batch_size"]:
                if priority == "memory":
                    dim.sort()
                elif priority == "performance":
                    dim.sort(reverse=True)
            elif key in [
                "use_distributed_optimizer",
                "tensor_model_parallel_size",
                "pipeline_model_parallel_size",
                "recompute_method",
                "recompute_num_layers",
                "context_parallel_size",
                "use_recompute",
            ]:
                if priority == "memory":
                    dim.sort(reverse=True)
                elif priority == "performance":
                    dim.sort()

    def build_space(self, config):
        """Set value of each dim and sort."""
        space = {}
        cards = config.experiment.auto_tuner.cards
        cards_per_node = config.experiment.auto_tuner.nproc_per_node
        num_layers = config.train.model.num_layers
        gbs = config.train.model.global_batch_size
        if "space" not in config.experiment.auto_tuner:
            config.experiment.auto_tuner.space = {}

        if "algo" not in self.config.experiment.auto_tuner:
            self.config.experiment.auto_tuner.algo = {"name": "grid", "priority": None}
        priority = config.experiment.auto_tuner.algo.get("priority", None)
        if config.experiment.auto_tuner.platform.get("airs_switch", False):
            priority = "memory"
        # Set data parallel degree
        space["data_parallel_size"] = (
            [i for i in range(1, cards + 1)]
            if "data_parallel_size" not in config.experiment.auto_tuner.space
            or config.experiment.auto_tuner.space.data_parallel_size == "auto"
            else config.experiment.auto_tuner.space.data_parallel_size
        )
        self._sort("data_parallel_size", space["data_parallel_size"], priority)

        # Set distributed optimizer
        space["use_distributed_optimizer"] = (
            [True, False]
            if "use_distributed_optimizer" not in config.experiment.auto_tuner.space
            or config.experiment.auto_tuner.space.use_distributed_optimizer == "auto"
            else config.experiment.auto_tuner.space.use_distributed_optimizer
        )
        self._sort("use_distributed_optimizer", space["use_distributed_optimizer"], priority)

        # Set tensor parallel degree
        space["tensor_model_parallel_size"] = (
            [i for i in range(1, cards_per_node + 1)]
            if "tensor_model_parallel_size" not in config.experiment.auto_tuner.space
            or config.experiment.auto_tuner.space.tensor_model_parallel_size == "auto"
            else config.experiment.auto_tuner.space.tensor_model_parallel_size
        )
        self._sort("tensor_model_parallel_size", space["tensor_model_parallel_size"], priority)

        # Set sequence parallel
        space["sequence_parallel"] = (
            [True, False]
            if "sequence_parallel" not in config.experiment.auto_tuner.space
            or config.experiment.auto_tuner.space.sequence_parallel == "auto"
            else config.experiment.auto_tuner.space.sequence_parallel
        )
        self._sort("sequence_parallel", space["sequence_parallel"], priority)

        # Set pipeline parallel degree
        space["pipeline_model_parallel_size"] = (
            [i for i in range(1, cards + 1)]
            if "pipeline_model_parallel_size" not in config.experiment.auto_tuner.space
            or config.experiment.auto_tuner.space.pipeline_model_parallel_size == "auto"
            else config.experiment.auto_tuner.space.pipeline_model_parallel_size
        )
        self._sort("pipeline_model_parallel_size", space["pipeline_model_parallel_size"], priority)

        # Set virtual pipeline parallel degree
        space["num_layers_per_virtual_pipeline_stage"] = (
            [i for i in range(0, num_layers + 1)]
            if "num_layers_per_virtual_pipeline_stage" not in config.experiment.auto_tuner.space
            or config.experiment.auto_tuner.space.num_layers_per_virtual_pipeline_stage == "auto"
            else config.experiment.auto_tuner.space.num_layers_per_virtual_pipeline_stage
        )
        self._sort(
            "num_layers_per_virtual_pipeline_stage",
            space["num_layers_per_virtual_pipeline_stage"],
            priority,
        )

        # Set use recompute
        space["use_recompute"] = (
            [True, False]
            if "use_recompute" not in config.experiment.auto_tuner.space
            or config.experiment.auto_tuner.space.use_recompute == "auto"
            else config.experiment.auto_tuner.space.use_recompute
        )
        self._sort("use_recompute", space["use_recompute"], priority)
        # Set recompute method
        space["recompute_method"] = (
            ["uniform", "block"]
            if "recompute_method" not in config.experiment.auto_tuner.space
            or config.experiment.auto_tuner.space.recompute_method == "auto"
            else config.experiment.auto_tuner.space.recompute_method
        )
        self._sort("recompute_method", space["recompute_method"], priority)

        # Set recompute granularity
        space["recompute_granularity"] = (
            ["full", "selective"]
            if "recompute_granularity" not in config.experiment.auto_tuner.space
            or config.experiment.auto_tuner.space.recompute_granularity == "auto"
            else config.experiment.auto_tuner.space.recompute_granularity
        )
        self._sort("recompute_granularity", space["recompute_granularity"], priority)

        # Set recompute num layers
        space["recompute_num_layers"] = (
            [i for i in range(1, num_layers + 1)]
            if "recompute_num_layers" not in config.experiment.auto_tuner.space
            or config.experiment.auto_tuner.space.recompute_num_layers == "auto"
            else config.experiment.auto_tuner.space.recompute_num_layers
        )
        self._sort("recompute_num_layers", space["recompute_num_layers"], priority)

        # Set micro batch size
        space["micro_batch_size"] = (
            [i for i in range(1, gbs + 1)]
            if "micro_batch_size" not in config.experiment.auto_tuner.space
            or config.experiment.auto_tuner.space.micro_batch_size == "auto"
            else config.experiment.auto_tuner.space.micro_batch_size
        )
        self._sort("micro_batch_size", space["micro_batch_size"], priority)

        # Set context parallel degree
        space["context_parallel_size"] = (
            [i for i in range(1, cards + 1)]
            if "context_parallel_size" not in config.experiment.auto_tuner.space
            or config.experiment.auto_tuner.space.context_parallel_size == "auto"
            else config.experiment.auto_tuner.space.context_parallel_size
        )
        self._sort("context_parallel_size", space["context_parallel_size"], priority)

        # Set expert parallel degree
        # NOTE: Expert parallel degree is not supported now
        space["expert_model_parallel_size"] = (
            [1]
            if "expert_model_parallel_size" not in config.experiment.auto_tuner.space
            or config.experiment.auto_tuner.space.expert_model_parallel_size == "auto"
            else config.experiment.auto_tuner.space.expert_model_parallel_size
        )
        self._sort("expert_model_parallel_size", space["expert_model_parallel_size"], priority)
        return space

    def build_strategies(self, space, config):
        """Build strategies by Cartesian product search space."""
        parallelism_part = self._product_parallel_dims(space, config)
        micro_batch_size_vpp_part = self._product_micro_batch_size_vpp_dims(
            parallelism_part, space, config
        )
        recompute_part = self._product_recompute_dims(micro_batch_size_vpp_part, space, config)

        return recompute_part

    def build_algo(self, strategies, config):
        name = self.config.experiment.auto_tuner.algo.name
        if name == "grid":
            return GridAlgo(strategies, self.config)
        else:
            raise NotImplementedError("Currently only grid search is supported.")

    def _product_parallel_dims(self, space, config):
        # Avoid space explosion after product
        product_parallelism_dims = []
        cards = config.experiment.auto_tuner.cards
        for data_parallel_size in space["data_parallel_size"]:
            dims = {}
            if not divisible(cards, data_parallel_size):
                continue
            # prune by local batch size
            gbs = config.train.model.global_batch_size
            if not divisible(gbs, data_parallel_size):
                continue
            for tensor_model_parallel_size in space["tensor_model_parallel_size"]:
                if not divisible(cards, tensor_model_parallel_size):
                    continue
                if not divisible(cards, data_parallel_size * tensor_model_parallel_size):
                    continue
                hidden_size = config.train.model.hidden_size
                num_attention_size = config.train.model.num_attention_heads
                if not divisible(hidden_size, tensor_model_parallel_size):
                    continue
                if not divisible(num_attention_size, tensor_model_parallel_size):
                    continue
                for pipeline_model_parallel_size in space["pipeline_model_parallel_size"]:
                    if not divisible(cards, pipeline_model_parallel_size):
                        continue
                    if not divisible(
                        cards,
                        data_parallel_size
                        * tensor_model_parallel_size
                        * pipeline_model_parallel_size,
                    ):
                        continue
                    num_layers = config.train.model.num_layers
                    if not divisible(num_layers, pipeline_model_parallel_size):
                        continue
                    for context_parallel_size in space["context_parallel_size"]:
                        if not divisible(cards, context_parallel_size):
                            continue
                        if not divisible(
                            cards,
                            data_parallel_size
                            * tensor_model_parallel_size
                            * pipeline_model_parallel_size
                            * context_parallel_size,
                        ):
                            continue
                        seq_length = config.train.model.seq_length
                        if not divisible(seq_length, context_parallel_size):
                            continue
                        for expert_model_parallel_size in space["expert_model_parallel_size"]:
                            if not divisible(cards, expert_model_parallel_size):
                                continue
                            if not divisible(
                                cards,
                                data_parallel_size
                                * tensor_model_parallel_size
                                * pipeline_model_parallel_size
                                * context_parallel_size
                                * expert_model_parallel_size,
                            ):
                                continue
                            if (
                                data_parallel_size
                                * tensor_model_parallel_size
                                * pipeline_model_parallel_size
                                * context_parallel_size
                                * expert_model_parallel_size
                                != cards
                            ):
                                continue

                            dims["data_parallel_size"] = data_parallel_size
                            dims["tensor_model_parallel_size"] = tensor_model_parallel_size
                            dims["pipeline_model_parallel_size"] = pipeline_model_parallel_size
                            dims["expert_model_parallel_size"] = expert_model_parallel_size
                            dims["context_parallel_size"] = context_parallel_size
                            copied_dims = copy.deepcopy(dims)
                            product_parallelism_dims.append(copied_dims)
        product_dist_opt_dims = []

        for use_distributed_optimizer in space["use_distributed_optimizer"]:
            dims = {}
            dims["use_distributed_optimizer"] = use_distributed_optimizer
            copied_dims = copy.deepcopy(dims)
            product_dist_opt_dims.append(copied_dims)

        product_sp_dims = []

        for sequence_parallel in space["sequence_parallel"]:
            dims = {}
            dims["sequence_parallel"] = sequence_parallel
            copied_dims = copy.deepcopy(dims)
            product_sp_dims.append(copied_dims)

        result = []
        unique_result = set()
        # append sp and dist opt
        for product_parallelism_dim in product_parallelism_dims:
            product_dim = {}
            product_dim.update(product_parallelism_dim)
            if product_parallelism_dim["data_parallel_size"] == 1:
                product_dim["use_distributed_optimizer"] = None
                if product_parallelism_dim["tensor_model_parallel_size"] == 1:
                    product_dim["sequence_parallel"] = None
                    self._append(result, unique_result, product_dim)

                else:
                    for product_sp_dim in product_sp_dims:
                        if product_sp_dim["sequence_parallel"]:
                            seq_length = config.train.model.seq_length
                            if not divisible(
                                seq_length, product_parallelism_dim["tensor_model_parallel_size"]
                            ):
                                continue
                        product_dim["sequence_parallel"] = product_sp_dim["sequence_parallel"]
                        self._append(result, unique_result, product_dim)
            else:
                for product_dist_opt_dim in product_dist_opt_dims:
                    product_dim["use_distributed_optimizer"] = product_dist_opt_dim[
                        "use_distributed_optimizer"
                    ]

                    if product_parallelism_dim["tensor_model_parallel_size"] == 1:
                        product_dim["sequence_parallel"] = None
                        self._append(result, unique_result, product_dim)
                    else:
                        for product_sp_dim in product_sp_dims:
                            product_dim["sequence_parallel"] = product_sp_dim["sequence_parallel"]
                            self._append(result, unique_result, product_dim)
        return result

    def _product_recompute_dims(self, micro_batch_size_vpp_part, space, config):
        result = []
        unique_result = set()
        for micro_batch_size_vpp in micro_batch_size_vpp_part:
            product_dim = {}
            product_dim.update(micro_batch_size_vpp)
            for use_recompute in space["use_recompute"]:
                product_dim["use_recompute"] = use_recompute
                for recompute_method in space["recompute_method"]:
                    if not use_recompute:
                        product_dim["recompute_method"] = None
                    else:
                        product_dim["recompute_method"] = recompute_method
                    for recompute_granularity in space["recompute_granularity"]:
                        if not use_recompute:
                            product_dim["recompute_granularity"] = None
                        else:
                            product_dim["recompute_granularity"] = recompute_granularity
                        if recompute_granularity == "selective":
                            product_dim["recompute_method"] = None
                        for recompute_num_layers in space["recompute_num_layers"]:
                            if (
                                not use_recompute
                                or product_dim["recompute_granularity"] == "selective"
                            ):
                                product_dim["recompute_num_layers"] = None
                            else:
                                layers_per_stage = (
                                    config.train.model.num_layers
                                    // product_dim["pipeline_model_parallel_size"]
                                )
                                if recompute_num_layers > layers_per_stage:
                                    continue
                                if recompute_method == "uniform":
                                    if not divisible(
                                        config.train.model.num_layers, recompute_num_layers
                                    ):
                                        continue
                                    if recompute_num_layers != layers_per_stage:
                                        continue
                                product_dim["recompute_num_layers"] = recompute_num_layers
                            self._append(result, unique_result, product_dim)
        return result

    def _product_micro_batch_size_vpp_dims(self, parallelism_part, space, config):
        """Just product micro_batch_size and vpp and pruned by parallel product dims."""
        result = []
        unique_result = set()

        gbs = config.train.model.global_batch_size
        num_layers = config.train.model.num_layers

        for parallelism in parallelism_part:
            product_dim = {}
            product_dim.update(parallelism)
            for micro_batch_size in space["micro_batch_size"]:
                data_parallel_size = parallelism["data_parallel_size"]
                local_batch_size = gbs // data_parallel_size
                # prune micro_batch_size
                if not divisible(local_batch_size, micro_batch_size):
                    continue
                product_dim["acc_step"] = local_batch_size // micro_batch_size
                product_dim["micro_batch_size"] = micro_batch_size
                for num_layers_per_virtual_pipeline_stage in space[
                    "num_layers_per_virtual_pipeline_stage"
                ]:
                    # Not use interleaved pipeline
                    if num_layers_per_virtual_pipeline_stage == 0:
                        product_dim["num_layers_per_virtual_pipeline_stage"] = None
                        self._append(result, unique_result, product_dim)
                    else:
                        pipeline_model_parallel_size = parallelism["pipeline_model_parallel_size"]
                        layers = config.train.model.num_layers
                        if (
                            pipeline_model_parallel_size <= 2
                            and num_layers_per_virtual_pipeline_stage >= 1
                        ):
                            continue

                        layers_per_pp_stage = layers // pipeline_model_parallel_size
                        if not divisible(
                            layers_per_pp_stage, num_layers_per_virtual_pipeline_stage
                        ):
                            continue

                        # Micro batches should divide pp size
                        if not divisible(
                            product_dim["micro_batch_size"], num_layers_per_virtual_pipeline_stage
                        ):
                            continue

                        if pipeline_model_parallel_size == 1:
                            product_dim["num_layers_per_virtual_pipeline_stage"] = None
                        else:
                            product_dim["num_layers_per_virtual_pipeline_stage"] = (
                                num_layers_per_virtual_pipeline_stage
                            )
                        self._append(result, unique_result, product_dim)

        return result

    def _append(self, result, unique_result, product_dim):
        sorted_items = tuple(sorted(product_dim.items()))
        if sorted_items not in unique_result:
            unique_result.add(sorted_items)
            copied_dim = copy.deepcopy(product_dim)
            result.append(copied_dim)

    def search(self):
        """Search once and return one strategy."""
        return self.algo.search()

    def has_done(self):
        """Return True if search is finished."""
        return self.algo.has_done()


class ServeSearcher(Searcher):
    def __init__(self, config):
        self._nodes_aware_dims = [
            item
            for item in _BUILT_IN_SERVE_STRATEGY_DIMS
            if item not in _DEFAULT_SERVE_TUNE_SPACE.keys()
        ]
        super(ServeSearcher, self).__init__(config)

    def _create_space_aware_nodes(self, space, cards):
        if cards == 1:
            for k in self._nodes_aware_dims:
                space[k] = 1

        fixed_dims = {}
        for idx, key in enumerate(self._nodes_aware_dims):
            if key in space and space[key] != "auto":
                fixed_dims[idx] = space[key]

        nodes_aware_strategies = self._find_combinations(
            cards, len(self._nodes_aware_dims), fixed_dims
        )
        for key_idx, key in enumerate(self._nodes_aware_dims):
            space[key] = list(set([v[key_idx] for v in nodes_aware_strategies]))
        return space, nodes_aware_strategies

    def _find_combinations(self, target, num_dims, fixed_dims={}, current=[]):
        results = []
        dim_index = len(current)

        if num_dims == 1:
            if dim_index in fixed_dims and target not in fixed_dims[dim_index]:
                return []
            return [[*current, target]]

        if dim_index in fixed_dims:
            candidates = fixed_dims[dim_index]
        else:
            candidates = range(1, target + 1)

        for i in candidates:
            if target % i == 0:
                results.extend(
                    self._find_combinations(target // i, num_dims - 1, fixed_dims, current + [i])
                )

        return results

    def _create_default_space(self, cards):
        space = dict.fromkeys(self._nodes_aware_dims, "auto")
        space.update(_DEFAULT_SERVE_TUNE_SPACE)
        return self._create_space_aware_nodes(space, cards)

    def _create_space(self, space, cards):
        if len(space) == 0:
            space, nodes_aware_strategies = self._create_default_space(cards)
            return space, nodes_aware_strategies

        space, nodes_aware_strategies = self._create_space_aware_nodes(space, cards)

        for key, value in space.items():
            if value == "auto":
                if key in _DEFAULT_SERVE_TUNE_SPACE.keys():
                    space[key] = _DEFAULT_SERVE_TUNE_SPACE[key]
            else:
                assert type(OmegaConf.to_object(value)) in [
                    tuple,
                    list,
                ], f"type of {key} in search space must be list or tuple, but now is {type(value)}"
        return space, nodes_aware_strategies

    def build_space(self, config):
        """
        the number of cards is fixed.
        """
        cards = config.experiment.auto_tuner.cards
        space = getattr(config.experiment.auto_tuner, "space", {})
        if len(space) != 0:
            self._nodes_aware_dims = [item for item in space if item in self._nodes_aware_dims]
        space, nodes_aware_strategies = self._create_space(space, cards)
        self._nodes_aware_strategies = nodes_aware_strategies

        config.experiment.auto_tuner.space = space
        if "algo" not in self.config.experiment.auto_tuner:
            self.config.experiment.auto_tuner.algo = {"name": "grid", "priority": None}
        return space

    def build_strategies(self, space, config):
        """Build strategies by Cartesian product search space."""
        node_unaware_tune_space = {
            key: value for key, value in space.items() if key in _DEFAULT_SERVE_TUNE_SPACE
        }
        values = list(node_unaware_tune_space.values())
        cartesian_product_unaware_values = list(itertools.product(*values))
        cartesian_product_values = list(
            itertools.product(self._nodes_aware_strategies, cartesian_product_unaware_values)
        )
        cartesian_product_values = [tuple(tuple(a) + b) for a, b in cartesian_product_values]
        strategies = [
            dict(zip(self.space.keys(), combination)) for combination in cartesian_product_values
        ]
        print("================== grid search space: ================== \n")
        pprint(strategies, indent=2)

        return strategies
