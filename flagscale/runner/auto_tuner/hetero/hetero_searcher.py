import copy
import itertools
import logging

from collections import Counter

from ..search.searcher import Searcher


class HeteroSearcher(Searcher):

    def __init__(self, config, resources):
        self.logger = logging.getLogger("FlagScale-AutoTuner")
        self.config = config
        self.resources = resources
        self.space = self.build_space(config)
        self.logger.info("HeteroSearcher: built search space, space is {}".format(self.space))

        self.strategies = self.build_strategies(self.space, self.config)
        self.algo = self.build_algo(self.strategies, self.config)

    def build_space(self, config):
        space = super().build_space(config)
        if "data_parallel_size" in space:
            del space["data_parallel_size"]
        user_space = config.experiment.auto_tuner.get("space", {})

        hetero_keys = [
            "hetero_pipeline_layer_split",
            "hetero_cards_per_stage",
            "hetero_device_types",
        ]

        pp_size_options = user_space.get("pipeline_model_parallel_size", [1])
        max_pp_size = max(pp_size_options) if pp_size_options else 1

        for i in range(max_pp_size):
            hetero_keys.append(f"hetero_dp_stage_{i}")

        for key in hetero_keys:
            if key in user_space:
                space[key] = user_space[key]

        return space

    def build_strategies(self, space, config):
        base_hetero_strategies = []
        logger = self.logger

        if not self.resources:
            logger.error("Hetero-tuning requires a hostfile, but no resources were found.")
            return []

        # card_counts_by_type -> {'A800': 16, 'B150': 32}
        card_counts_by_type = Counter(
            res['type'] for res in self.resources.values() if res.get('type')
        )
        logger.info(f"Physical hardware detected from hostfile: {card_counts_by_type}")

        # logical_stages -> ['A800', 'B150']
        logical_stages = space.get("hetero_device_types", [])
        if not logical_stages:
            logger.error("Hetero search space must define 'hetero_device_types'.")
            return []

        cards_per_stage = []
        for device_type in logical_stages:
            if device_type not in card_counts_by_type:
                logger.error(
                    f"Device type '{device_type}' defined in YAML not found in hostfile's types: {list(card_counts_by_type.keys())}"
                )
                return []
            cards_per_stage.append(card_counts_by_type[device_type])

        logger.info(
            f"Mapping logical stages {logical_stages} to physical cards_per_stage: {cards_per_stage}"
        )

        layer_splits_config = space.get("hetero_pipeline_layer_split", [])
        if layer_splits_config == "auto":
            self.logger.info("Auto-generating layer splits based on model and pipeline size.")
            num_layers = config.train.model.num_layers
            pp_size = len(logical_stages)
            if pp_size == 2:
                avg_layers = num_layers // pp_size
                search_range = space.get("hetero_split_search_range", 4)
                search_step = space.get("hetero_split_search_step", 2)

                generated_splits = set()
                for i in range(-search_range, search_range + 1, search_step):
                    split_a = avg_layers + i
                    split_b = num_layers - split_a
                    if split_a > 0 and split_b > 0:
                        generated_splits.add(tuple(sorted((split_a, split_b))))
                layer_splits = [list(s) for s in sorted(list(generated_splits))]
                self.logger.info(f"Auto-generated layer splits to search: {layer_splits}")
            elif pp_size > 1:
                layers_per_stage = [num_layers // pp_size] * pp_size
                remaining_layers = num_layers % pp_size
                for i in range(remaining_layers):
                    layers_per_stage[i] += 1
                layer_splits = [layers_per_stage]
                self.logger.info(
                    f"Auto-generated layer splits for {pp_size} stages (average): {layer_splits}"
                )
            else:
                layer_splits = [[num_layers]]
        else:
            layer_splits = layer_splits_config
        for split in layer_splits:
            if len(split) != len(logical_stages):
                logger.warning(
                    f"Layer split {split} length doesn't match PP size {len(logical_stages)}. Skipping."
                )
                continue

            if sum(split) != config.train.model.num_layers:
                logger.warning(f"Layer split {split} sum invalid. Skipping.")
                continue

            per_stage_mesh_options = []
            is_valid_split = True
            for i, stage_cards in enumerate(cards_per_stage):
                tp_options = space.get("tensor_model_parallel_size", [1])
                dp_options_key = f"hetero_dp_stage_{i}"

                stage_meshes = []
                if dp_options_key in space:
                    dp_options = space[dp_options_key]
                    for tp in tp_options:
                        for dp in dp_options:
                            if tp * dp == stage_cards:
                                mesh = [tp, 1, 1, dp, 1]
                                stage_meshes.append(mesh)
                else:
                    self.logger.info(
                        f"Auto-calculating DP options for stage {i} based on TP options."
                    )
                    for tp in tp_options:
                        if stage_cards % tp == 0:
                            dp = stage_cards // tp
                            mesh = [tp, 1, 1, dp, 1]
                            stage_meshes.append(mesh)

                if not stage_meshes:
                    is_valid_split = False
                    break
                per_stage_mesh_options.append(stage_meshes)

            if not is_valid_split:
                continue

            all_pipeline_meshes = list(itertools.product(*per_stage_mesh_options))
            for pipeline_mesh in all_pipeline_meshes:
                final_mesh_list = [item for mesh in pipeline_mesh for item in mesh]
                equivalent_dp = sum(mesh[3] for mesh in pipeline_mesh)

                strategy = {
                    "pipeline_model_parallel_size": len(split),
                    "tensor_model_parallel_size": pipeline_mesh[0][0],
                    "data_parallel_size": equivalent_dp,
                    "hetero_pipeline_layer_split": list(split),
                    "hetero_cards_per_stage": cards_per_stage,
                    "hetero_process_meshes": final_mesh_list,
                    "hetero_device_types": logical_stages,
                    "context_parallel_size": 1,
                    "expert_model_parallel_size": 1,
                    "sequence_parallel": False,
                    "use_distributed_optimizer": True if equivalent_dp > 1 else False,
                }
                base_hetero_strategies.append(strategy)

        if not base_hetero_strategies:
            return []
        mbs_combined = self._product_micro_batch_size_vpp_dims(
            base_hetero_strategies, space, config
        )
        final_strategies = self._product_recompute_dims(mbs_combined, space, config)

        self.logger.info(f"Generated {len(final_strategies)} complete heterogeneous strategies.")
        return final_strategies

    def _append(self, result, unique_result, product_dim):
        hashable_items = []
        for k, v in sorted(product_dim.items()):
            hashable_items.append((k, tuple(v) if isinstance(v, list) else v))

        sorted_items = tuple(hashable_items)

        if sorted_items not in unique_result:
            unique_result.add(sorted_items)
            result.append(copy.deepcopy(product_dim))
