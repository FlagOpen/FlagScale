import itertools
import logging

from typing import Dict, List

from omegaconf import DictConfig, ListConfig, OmegaConf

from flagscale.runner.auto_tuner.search.searcher import Searcher


def _generate_all_partitions_with_max_diff(n: int, k: int, max_diff: int):
    """
    Generates all integer partitions of n into k parts, with a constraint
    that the difference between the largest and smallest part does not exceed max_diff.
    """
    if k == 0:
        if n == 0:
            yield []
        return
    if k == 1:
        if n > 0:
            yield [n]
        return

    # Bound the search for the first element to satisfy the max_diff constraint
    for i in range((n + k - 1) // k, n - (k - 1) + 1):
        if n - i < (k - 1):
            continue
        for rest in _generate_all_partitions_with_max_diff(n - i, k - 1, max_diff):
            if not rest:
                if k - 1 == 0 and n - i == 0:
                    if i <= max_diff:
                        yield [i]
                continue

            # Keep partitions in descending order to avoid duplicates
            if i < rest[0]:
                continue

            # Check the max_diff constraint
            if i - rest[-1] > max_diff:
                continue

            yield [i] + rest


def _generate_valid_layer_splits(
    total_layers: int, mesh_pp_sizes: List[int], inter_mesh_max_diff: int, intra_mesh_max_diff: int
):
    """
    Generates valid layer splits using the new intelligent splitting algorithms.
    """
    num_meshes = len(mesh_pp_sizes)
    if num_meshes == 0:
        return

    # 1. Use the new partition function with max_diff for inter-mesh layer distribution
    partition_generator = _generate_all_partitions_with_max_diff(
        total_layers, num_meshes, inter_mesh_max_diff
    )

    for inter_mesh_partition in partition_generator:
        for mesh_layer_distribution in set(itertools.permutations(inter_mesh_partition)):
            possible_splits_per_mesh = []
            is_distribution_possible = True
            for i in range(num_meshes):
                layers_for_this_mesh = mesh_layer_distribution[i]
                local_pp = mesh_pp_sizes[i]
                if layers_for_this_mesh < local_pp:
                    is_distribution_possible = False
                    break

                # 2. Use the flexible split for intra-mesh distribution with a hardcoded max_diff of 2
                mesh_i_splits_generator = _generate_all_partitions_with_max_diff(
                    layers_for_this_mesh, local_pp, intra_mesh_max_diff
                )
                mesh_i_splits = list(mesh_i_splits_generator)

                if not mesh_i_splits:
                    is_distribution_possible = False
                    break

                possible_splits_per_mesh.append(mesh_i_splits)

            if is_distribution_possible:
                for combined_intra_splits_tuple in itertools.product(*possible_splits_per_mesh):
                    final_split = []
                    for single_mesh_split_list in combined_intra_splits_tuple:
                        final_split.extend(single_mesh_split_list)

                    if sum(final_split) == total_layers and len(final_split) == sum(mesh_pp_sizes):
                        yield final_split


class HeteroSearcher(Searcher):
    """
    A specialized searcher for heterogeneous environments.
    """

    def __init__(self, config: Dict, resources: Dict):
        self.resources = resources
        self.node_info = []
        self.mesh_templates = []
        self.device_types_in_template = []
        self.recompute_search_space = {}
        self.layer_split_constraints = {}
        super().__init__(config)

    def build_space(self, config: Dict) -> Dict:
        space = {}
        auto_tuner_config = config.experiment.auto_tuner
        hetero_space = auto_tuner_config.space
        if not self.resources:
            raise ValueError("Heterogeneous tuning requires a valid hostfile/resources dict.")
        for hostname, resource_info in self.resources.items():
            self.node_info.append(
                {
                    "name": hostname,
                    "type": resource_info.get("type", "default_gpu"),
                    "slots": resource_info.get("slots", 0),
                }
            )
        self.logger.info(f"Received node-aware hardware info: {self.node_info}")

        def safe_to_container(value):
            if isinstance(value, (DictConfig, ListConfig)):
                return OmegaConf.to_container(value, resolve=True)
            return value

        # Parse the new layer splitting constraint intra_mesh_max_diff from the config
        self.layer_split_constraints['inter_mesh_max_diff'] = safe_to_container(
            hetero_space.get("hetero_inter_mesh_max_layer_diff", "auto")
        )
        self.layer_split_constraints['intra_mesh_max_diff'] = safe_to_container(
            hetero_space.get("hetero_intra_mesh_max_layer_diff", "auto")
        )
        space["hetero_pipeline_layer_split"] = safe_to_container(
            hetero_space.get("hetero_pipeline_layer_split", "auto")
        )
        raw_mesh_templates = safe_to_container(hetero_space.get("hetero_process_meshes", []))
        if not raw_mesh_templates or len(raw_mesh_templates) % 5 != 0:
            raise ValueError(
                "'hetero_process_meshes' must be a non-empty list with a length divisible by 5."
            )
        self.mesh_templates = [
            raw_mesh_templates[i : i + 5] for i in range(0, len(raw_mesh_templates), 5)
        ]
        self.logger.info(f"Parsed {len(self.mesh_templates)} mesh templates from config.")
        self.device_types_in_template = config.train.system.hetero.get("hetero_device_types", [])
        if len(self.mesh_templates) != len(self.device_types_in_template):
            raise ValueError(
                f"Mismatch: The number of mesh templates ({len(self.mesh_templates)}) does not match "
                f"the number of hetero_device_types ({len(self.device_types_in_template)}). "
            )

        space["micro_batch_size"] = safe_to_container(hetero_space.get("micro_batch_size", [1]))
        space["use_distributed_optimizer"] = safe_to_container(
            hetero_space.get("use_distributed_optimizer", [True, False])
        )
        space["sequence_parallel"] = safe_to_container(
            hetero_space.get("sequence_parallel", [True, False])
        )
        self.recompute_search_space['use_recompute'] = safe_to_container(
            hetero_space.get("use_recompute", [True, False])
        )
        self.recompute_search_space['granularity'] = safe_to_container(
            hetero_space.get("recompute_granularity_per_stage_micro_batch", "auto")
        )
        self.recompute_search_space['method'] = safe_to_container(
            hetero_space.get("recompute_method_per_stage_micro_batch", "auto")
        )
        self.recompute_search_space['num_layers'] = safe_to_container(
            hetero_space.get("recompute_num_layers_per_stage_micro_batch", "auto")
        )

        space["num_layers_per_virtual_pipeline_stage"] = [0]
        if "algo" not in auto_tuner_config:
            auto_tuner_config.algo = {"name": "grid", "priority": None}
        return space

    # generate recompute configurations dynamically
    def _generate_recompute_configs(self, pp_size: int, num_micro_batches: int) -> List[Dict]:
        if pp_size == 0:
            return [{}]

        def get_options_for(key: str) -> List[list]:
            user_config = self.recompute_search_space.get(key)
            if user_config == "auto":
                auto_templates = [[[pp_size, 'ALL', 0]], [[pp_size, 'ALL', 1]]]
                if key == 'num_layers':
                    return [[[pp_size, 'ALL', 1]]]
                return auto_templates
            elif isinstance(user_config, list):
                valid_options = []
                for template_list in user_config:
                    total_stages_in_template = sum(item[0] for item in template_list)
                    if total_stages_in_template == pp_size:
                        valid_options.append(template_list)
                return valid_options
            return []

        granularity_options = get_options_for('granularity')
        method_options = get_options_for('method')
        num_layers_options = get_options_for('num_layers')

        if not granularity_options or not method_options or not num_layers_options:
            return [{}]

        all_recompute_combinations = []
        for gran_list, meth_list, num_list in itertools.product(
            granularity_options, method_options, num_layers_options
        ):

            def render_template(template_list):
                rendered_list = []
                for item in template_list:
                    rendered_item = [val if val != 'ALL' else num_micro_batches for val in item]
                    rendered_list.append(rendered_item)
                return rendered_list

            all_recompute_combinations.append(
                {
                    "recompute_granularity_per_stage_micro_batch": render_template(gran_list),
                    "recompute_method_per_stage_micro_batch": render_template(meth_list),
                    "recompute_num_layers_per_stage_micro_batch": render_template(num_list),
                }
            )

        return all_recompute_combinations if all_recompute_combinations else [{}]

    def build_strategies(self, space: Dict, config: Dict) -> List[Dict]:
        self.logger.info("Building comprehensive heterogeneous strategies (Node-Aware)...")

        all_assignments = self._find_valid_assignments(
            mesh_idx=0, available_nodes=self.node_info, current_assignments=[], config=config
        )
        self.logger.info(
            f"Generated {len(all_assignments)} core heterogeneous parallelism assignments."
        )

        # Refactored into a two-stage process
        # Stage 1: Build base strategies without recompute
        base_strategies_without_recompute = []
        gbs = config.train.model.global_batch_size
        total_layers = config.train.model.num_layers

        processed_base_parallels = set()

        for assignment in all_assignments:
            global_pp_size = sum(item['mesh'][4] for item in assignment)

            base_parallel_info = {
                "pipeline_model_parallel_size": global_pp_size,
                "hetero_process_meshes": [item['mesh'] for item in assignment],
                "hetero_device_types": [item['device_type'] for item in assignment],
            }
            layer_splits = []
            if space["hetero_pipeline_layer_split"] == 'auto':
                if global_pp_size > 0:
                    mesh_pp_sizes = [item['mesh'][4] for item in assignment]
                    inter_mesh_diff = self.layer_split_constraints['inter_mesh_max_diff']
                    if inter_mesh_diff == 'auto':
                        # Heuristic for auto mode
                        inter_mesh_diff = max(
                            (
                                config.train.model.num_layers // len(mesh_pp_sizes)
                                if len(mesh_pp_sizes) > 0
                                else config.train.model.num_layers
                            ),
                            4,
                        )

                    intra_mesh_diff = self.layer_split_constraints['intra_mesh_max_diff']
                    if intra_mesh_diff == 'auto':
                        intra_mesh_diff = total_layers
                    elif not isinstance(intra_mesh_diff, int) or intra_mesh_diff < 0:
                        self.logger.warning(
                            f"Invalid value for hetero_intra_mesh_max_layer_diff: {intra_mesh_diff}. Using default heuristic (max_diff={total_layers})."
                        )
                        intra_mesh_diff = total_layers
                    layer_splits_generator = _generate_valid_layer_splits(
                        total_layers, mesh_pp_sizes, inter_mesh_diff, intra_mesh_diff
                    )
                    layer_splits.extend(list(layer_splits_generator))
                else:
                    layer_splits.append([])
            elif isinstance(space["hetero_pipeline_layer_split"], list):
                for split in space["hetero_pipeline_layer_split"]:
                    if len(split) == global_pp_size and sum(split) == config.train.model.num_layers:
                        layer_splits.append(split)

            for split in layer_splits:
                dp_list = [mesh[3] for mesh in base_parallel_info["hetero_process_meshes"]]
                first_mesh_dp = dp_list[0] if dp_list else 1

                for mbs in space["micro_batch_size"]:
                    is_gbs_compatible = all((gbs % (dp * mbs) == 0) for dp in dp_list if dp > 0)
                    if not is_gbs_compatible:
                        continue

                    product = first_mesh_dp * mbs
                    is_hetero_dp_compatible = all(
                        (product % hetero_dp == 0) for hetero_dp in dp_list if hetero_dp > 0
                    )

                    if not is_hetero_dp_compatible:
                        self.logger.debug(
                            f"Pruning strategy (in generation): first_mesh_dp({first_mesh_dp}) * mbs({mbs}) = {product} is not divisible by all hetero_dp {dp_list}."
                        )
                        continue
                    all_dp_are_one = all(dp == 1 for dp in dp_list)
                    do_options = [False] if all_dp_are_one else space["use_distributed_optimizer"]
                    for use_do in do_options:
                        # Revised SP Logic
                        tp_list = [mesh[0] for mesh in base_parallel_info["hetero_process_meshes"]]
                        all_tp_are_one = all(tp == 1 for tp in tp_list)
                        tps_are_mixed = (
                            len(set(tp_list)) > 1
                        )  # True if TPs are different (e.g., [1, 2] or [2, 4])

                        # Use a set to prevent duplicate appends (e.g., when all_tp_are_one is True)
                        added_effective_sps_for_this_combo = set()

                        for sp_option in space["sequence_parallel"]:
                            effective_sp = sp_option  # Default to user's option

                            # Rule 1: TPs are mixed (e.g., [1, 2]) -> SP must be True
                            if tps_are_mixed:
                                if not sp_option:  # Prune sp=False if TPs are mixed
                                    self.logger.debug(
                                        f"Pruning SP Option={sp_option}: Invalid. TPs are mixed {tp_list}, sp must be True. Skipping."
                                    )
                                    continue
                                effective_sp = True  # Ensure it's True

                            # Rule 2: All TPs are 1
                            elif all_tp_are_one:
                                # SP must be False. Correct sp=True to sp=False.
                                effective_sp = False

                            # Rule 3: All TPs are > 1 AND Same (e.g., [2, 2])
                            else:
                                # SP can be True or False, respect sp_option
                                effective_sp = sp_option

                            # Prevent adding duplicates
                            # (This handles Rule 2 where sp=True and sp=False both result in effective_sp=False)
                            if effective_sp in added_effective_sps_for_this_combo:
                                self.logger.debug(
                                    f"Skipping append for effective_sp={effective_sp}, already added for this combo (e.g., all TP=1 case)."
                                )
                                continue
                            added_effective_sps_for_this_combo.add(effective_sp)

                            # Append Valid Strategy
                            base_strategies_without_recompute.append(
                                {
                                    **base_parallel_info,
                                    "hetero_pipeline_layer_split": split,
                                    "micro_batch_size": mbs,
                                    "use_distributed_optimizer": use_do,
                                    "sequence_parallel": effective_sp,
                                }
                            )
                            self.logger.debug(
                                f"Appended base strategy (MBS={mbs}, DO={use_do}, SP={effective_sp})"
                            )
        self.logger.info(f"Created {len(base_strategies_without_recompute)} base strategies.")

        # Stage 2: Dynamically append recompute configurations
        final_strategies = []
        for base_strategy in base_strategies_without_recompute:
            use_recompute_options = self.recompute_search_space['use_recompute']

            if False in use_recompute_options:
                final_strategies.append(
                    {
                        **base_strategy,
                        'use_recompute': False,
                        "recompute_granularity_per_stage_micro_batch": None,
                        "recompute_method_per_stage_micro_batch": None,
                        "recompute_num_layers_per_stage_micro_batch": None,
                    }
                )

            if True in use_recompute_options:
                pp_size = base_strategy['pipeline_model_parallel_size']
                mbs = base_strategy['micro_batch_size']
                dp = base_strategy['hetero_process_meshes'][0][3]

                if (mbs * dp) == 0:
                    num_micro_batches = gbs
                else:
                    num_micro_batches = gbs // (mbs * dp)
                if num_micro_batches == 0:
                    num_micro_batches = 1

                recompute_combinations = self._generate_recompute_configs(
                    pp_size, num_micro_batches
                )
                for recom_config in recompute_combinations:
                    if recom_config:
                        final_strategies.append(
                            {**base_strategy, 'use_recompute': True, **recom_config}
                        )

        # Finally, add compatibility keys for the Pruner
        for strategy in final_strategies:
            meshes = strategy.get("hetero_process_meshes", [])
            use_recompute = strategy.get('use_recompute', False)
            strategy.update(
                {
                    "num_layers_per_virtual_pipeline_stage": None,
                    "tensor_model_parallel_size": meshes[0][0] if meshes else 1,
                    "context_parallel_size": meshes[0][1] if meshes else 1,
                    "expert_model_parallel_size": meshes[0][2] if meshes else 1,
                    "data_parallel_size": meshes[0][3] if meshes else 1,
                    "decoder_first_pipeline_num_layers": None,
                    "decoder_last_pipeline_num_layers": None,
                    "recompute_method": "uniform" if use_recompute else None,
                    "recompute_granularity": "full" if use_recompute else None,
                    "recompute_num_layers": 1 if use_recompute else None,
                }
            )

        self.logger.info(
            f"Built a total of {len(final_strategies)} comprehensive candidate strategies."
        )
        unique_strategies_set = set()
        deduplicated_strategies = []
        final_strategy_counter = 0
        for strategy in final_strategies:
            final_strategy_counter += 1
            try:
                # Exclude keys that change per run or are internal/temporary results
                keys_to_exclude_for_hash = {
                    'idx',
                    'pruned',
                    'prune_reason',
                    'max_mem',
                    'performance',
                    'elapsed_time',
                    'start_time',
                    'stopped_by_tuner',
                    'error',
                }
                # Convert dict to hashable tuple (handle nested lists/tuples)
                items_to_hash = tuple(
                    sorted(
                        (
                            k,
                            (
                                tuple(tuple(x) if isinstance(x, list) else x for x in v)
                                if isinstance(v, list)
                                else v
                            ),
                        )
                        for k, v in strategy.items()
                        if k not in keys_to_exclude_for_hash
                    )
                )
                is_duplicate = items_to_hash in unique_strategies_set
                self.logger.debug(
                    f"  Checking final strategy {final_strategy_counter}/{len(final_strategies)}. Hashable: {items_to_hash}. Is duplicate? {is_duplicate}"
                )

                if not is_duplicate:
                    unique_strategies_set.add(items_to_hash)
                    deduplicated_strategies.append(strategy)
                # No 'else' needed, just skip appending if duplicate
            except TypeError as e:
                self.logger.warning(
                    f"  Could not hash final strategy {final_strategy_counter} for deduplication, keeping it. Error: {e}. Strategy: {strategy}"
                )
                deduplicated_strategies.append(strategy)  # Keep if hashing fails

        removed_count = len(final_strategies) - len(deduplicated_strategies)
        if removed_count > 0:
            self.logger.info(
                f"Removed {removed_count} duplicate strategy configurations during final check. "
                f"Final unique strategies count: {len(deduplicated_strategies)}."
            )
        else:
            self.logger.info("No duplicate strategies found during final check.")

        self.logger.info(
            f"build_strategies finished. Returning {len(deduplicated_strategies)} unique strategies."
        )
        return deduplicated_strategies

    def _get_search_values(self, template_val, max_val):
        if isinstance(template_val, int):
            return [template_val] if 0 < template_val <= max_val else []
        if isinstance(template_val, list):
            return sorted([v for v in template_val if 0 < v <= max_val])
        if template_val == 'auto':
            return [i for i in range(1, max_val + 1) if max_val % i == 0]
        return []

    def _is_candidate_valid(self, candidate, template):
        if isinstance(template, int):
            return candidate == template
        if isinstance(template, list):
            return candidate in template
        if template == 'auto':
            return True
        return False

    def _find_parallel_configs_for_mesh(self, mesh_template, nodes_for_mesh, config):
        total_gpus = sum(node['slots'] for node in nodes_for_mesh)
        if total_gpus == 0:
            return []
        tp_space = self._get_search_values(mesh_template[0], total_gpus)
        cp_space = self._get_search_values(mesh_template[1], total_gpus)
        ep_space = self._get_search_values(mesh_template[2], total_gpus)
        dp_template = mesh_template[3]
        pp_template = mesh_template[4]
        valid_configs = []
        for tp in tp_space:
            if config.train.model.hidden_size % tp != 0:
                continue
            for cp in cp_space:
                for ep in ep_space:
                    for pp in self._get_search_values(pp_template, total_gpus):
                        base_product = tp * cp * ep * pp
                        if total_gpus > 0 and total_gpus % base_product == 0:
                            required_dp = total_gpus // base_product
                            if self._is_candidate_valid(required_dp, dp_template):
                                valid_configs.append([tp, cp, ep, required_dp, pp])
        return valid_configs

    def _find_valid_assignments(self, mesh_idx, available_nodes, current_assignments, config):
        if mesh_idx == len(self.mesh_templates):
            return [current_assignments]
        results = []
        current_template = self.mesh_templates[mesh_idx]
        current_device_type = self.device_types_in_template[mesh_idx]
        # TODO: The current implementation strictly maps each mesh template to a specific device type.
        # A future, more advanced version could relax this constraint. The algorithm could
        # attempt to assign any group of available nodes to any mesh template as long as
        # parallelism constraints are met, enabling more flexible cross-hardware scheduling.
        candidate_nodes = [node for node in available_nodes if node['type'] == current_device_type]
        if not candidate_nodes:
            return []
        for k in range(1, len(candidate_nodes) + 1):
            for nodes_to_assign_tuple in itertools.combinations(candidate_nodes, k):
                nodes_to_assign = list(nodes_to_assign_tuple)
                parallel_configs = self._find_parallel_configs_for_mesh(
                    current_template, nodes_to_assign, config
                )
                if not parallel_configs:
                    continue
                assigned_node_names = {node['name'] for node in nodes_to_assign}
                remaining_nodes = [
                    node for node in available_nodes if node['name'] not in assigned_node_names
                ]
                for p_config in parallel_configs:
                    is_assignment_valid = True
                    if not config.train.model.get("untie_embeddings_and_output_weights", False):
                        if mesh_idx == len(self.mesh_templates) - 1 and current_assignments:
                            first_mesh_tp = current_assignments[0]['mesh'][0]
                            last_mesh_tp = p_config[0]
                            if first_mesh_tp != last_mesh_tp:
                                is_assignment_valid = False
                    if is_assignment_valid:
                        new_assignment = {
                            'mesh': p_config,
                            'device_type': current_device_type,
                            'nodes': nodes_to_assign,
                        }
                        sub_results = self._find_valid_assignments(
                            mesh_idx + 1,
                            remaining_nodes,
                            current_assignments + [new_assignment],
                            config,
                        )
                        results.extend(sub_results)
        return results
