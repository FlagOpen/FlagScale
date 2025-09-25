import itertools
import logging

from typing import Dict, List

from omegaconf import DictConfig, ListConfig, OmegaConf

from flagscale.runner.auto_tuner.search.searcher import Searcher


def _generate_all_partitions(n: int, k: int):
    """
    Generates all possible integer partitions of n into k parts.
    e.g., (7, 3) -> [5,1,1], [4,2,1], [3,3,1], [3,2,2]
    """
    if k == 1:
        if n > 0:
            yield [n]
        return
    # We can assign at least 1 layer to the first part, and at most n-(k-1)
    for i in range(1, n - (k - 1) + 1):
        for rest in _generate_all_partitions(n - i, k - 1):
            # To avoid duplicates like [1,2] and [2,1], we generate in descending order.
            if i >= rest[0]:
                yield [i] + rest


def _generate_balanced_split(n: int, k: int) -> List[int]:
    """
    Generates the single most arithmetically balanced split of n into k parts.
    The difference between the max and min values will not be greater than 1.
    """
    if k <= 0 or n < k:
        return []
    base = n // k
    rem = n % k
    return [base + 1] * rem + [base] * (k - rem)


def _generate_valid_layer_splits(total_layers: int, mesh_pp_sizes: List[int]):
    """
    Generates valid layer splits based on the new hierarchical logic.
    """
    num_meshes = len(mesh_pp_sizes)
    if num_meshes == 0:
        return

    # 1. Get all possible ways to distribute total_layers among the meshes
    for inter_mesh_partition in _generate_all_partitions(total_layers, num_meshes):
        # 2. Consider all permutations of that distribution
        for mesh_layer_distribution in set(itertools.permutations(inter_mesh_partition)):
            final_split = []
            is_distribution_possible = True

            # 3. For each mesh, check if its layer budget can be balanced internally
            for i in range(num_meshes):
                layers_for_this_mesh = mesh_layer_distribution[i]
                local_pp = mesh_pp_sizes[i]

                # Check if subdivision is possible
                if layers_for_this_mesh < local_pp:
                    is_distribution_possible = False
                    break

                # Perform the balanced subdivision
                intra_mesh_split = _generate_balanced_split(layers_for_this_mesh, local_pp)

                final_split.extend(intra_mesh_split)

            if is_distribution_possible:
                yield final_split


class HeteroSearcher(Searcher):
    """
    A specialized searcher for heterogeneous environments.
    """

    # __init__, build_space and other helpers remain the same as the last working version.
    def __init__(self, config: Dict, resources: Dict):
        self.resources = resources
        self.node_info = []
        self.mesh_templates = []
        self.device_types_in_template = []
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
        space["use_recompute"] = safe_to_container(hetero_space.get("use_recompute", [True, False]))
        space["recompute_method"] = safe_to_container(
            hetero_space.get("recompute_method", ["uniform", "block"])
        )
        space["recompute_granularity"] = safe_to_container(
            hetero_space.get("recompute_granularity", ["full", "selective"])
        )
        num_layers = config.train.model.num_layers
        space["recompute_num_layers"] = safe_to_container(
            hetero_space.get("recompute_num_layers", list(range(1, num_layers + 1)))
        )
        space["num_layers_per_virtual_pipeline_stage"] = [0]
        if "algo" not in auto_tuner_config:
            auto_tuner_config.algo = {"name": "grid", "priority": None}
        return space

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

    def build_strategies(self, space: Dict, config: Dict) -> List[Dict]:
        self.logger.info("Building comprehensive heterogeneous strategies (Node-Aware)...")

        all_assignments = self._find_valid_assignments(
            mesh_idx=0, available_nodes=self.node_info, current_assignments=[], config=config
        )
        self.logger.info(
            f"Generated {len(all_assignments)} core heterogeneous parallelism assignments."
        )

        parallelism_part = []
        global_pp_size_target = None
        # Manual mode for layer split now also implies a pp_size target
        if space["hetero_pipeline_layer_split"] != 'auto':
            if space["hetero_pipeline_layer_split"]:
                global_pp_size_target = len(space["hetero_pipeline_layer_split"][0])

        for assignment in all_assignments:
            global_pp_size = sum(item['mesh'][4] for item in assignment)

            if global_pp_size_target is not None and global_pp_size != global_pp_size_target:
                continue

            base_strategy = {
                "pipeline_model_parallel_size": global_pp_size,
                "hetero_process_meshes": [item['mesh'] for item in assignment],
                "hetero_device_types": [item['device_type'] for item in assignment],
            }

            #  Call the new hierarchical splitter logic ---
            if space["hetero_pipeline_layer_split"] == 'auto':
                if global_pp_size > 0:
                    mesh_pp_sizes = [item['mesh'][4] for item in assignment]
                    for split in _generate_valid_layer_splits(
                        config.train.model.num_layers, mesh_pp_sizes
                    ):
                        strat = base_strategy.copy()
                        strat["hetero_pipeline_layer_split"] = split
                        parallelism_part.append(strat)
            else:  # Manual mode validation
                for split in space["hetero_pipeline_layer_split"]:
                    if len(split) == global_pp_size and sum(split) == config.train.model.num_layers:
                        strat = base_strategy.copy()
                        strat["hetero_pipeline_layer_split"] = split
                        parallelism_part.append(strat)

        self.logger.info(
            f"Created {len(parallelism_part)} core strategies after considering layer splits."
        )
        final_strategies = self._product_with_other_dims(parallelism_part, space, config)
        self.logger.info(
            f"Built a total of {len(final_strategies)} comprehensive candidate strategies."
        )
        return final_strategies

    def _product_with_other_dims(self, parallelism_part, space, config):
        # This function remains unchanged
        full_strategies = []
        for base_strategy in parallelism_part:
            meshes = base_strategy["hetero_process_meshes"]
            if not meshes:
                continue

            dp_list = [mesh[3] for mesh in meshes]
            base_dp = dp_list[0] if dp_list else 1
            for mbs in space["micro_batch_size"]:
                gbs = config.train.model.global_batch_size
                is_dp_compatible = all((gbs % (dp * mbs) == 0) for dp in dp_list if dp > 0)
                if not is_dp_compatible:
                    continue
                all_dp_are_one = all(dp == 1 for dp in dp_list)
                do_options = [False] if all_dp_are_one else space["use_distributed_optimizer"]
                for use_do in do_options:
                    for sp in space["sequence_parallel"]:
                        all_tp_gt_one = all(mesh[0] > 1 for mesh in meshes)
                        if sp and not all_tp_gt_one:
                            continue
                        for vpp_layers in space["num_layers_per_virtual_pipeline_stage"]:
                            for use_recompute in space["use_recompute"]:
                                strategy_keys = {
                                    **base_strategy,
                                    "micro_batch_size": mbs,
                                    "use_distributed_optimizer": use_do,
                                    "sequence_parallel": sp,
                                    "use_recompute": use_recompute,
                                    "num_layers_per_virtual_pipeline_stage": (
                                        vpp_layers if vpp_layers > 0 else None
                                    ),
                                    "tensor_model_parallel_size": meshes[0][0],
                                    "context_parallel_size": meshes[0][1],
                                    "expert_model_parallel_size": meshes[0][2],
                                    "data_parallel_size": meshes[0][3],
                                    "decoder_first_pipeline_num_layers": None,
                                    "decoder_last_pipeline_num_layers": None,
                                }
                                if not use_recompute:
                                    strategy_keys.update(
                                        {
                                            "recompute_method": None,
                                            "recompute_granularity": None,
                                            "recompute_num_layers": None,
                                        }
                                    )
                                    full_strategies.append(strategy_keys)
                                else:
                                    for r_method in space["recompute_method"]:
                                        for r_granularity in space["recompute_granularity"]:
                                            new_strat = strategy_keys.copy()
                                            new_strat.update(
                                                {
                                                    "recompute_method": r_method,
                                                    "recompute_granularity": r_granularity,
                                                    "recompute_num_layers": space[
                                                        "recompute_num_layers"
                                                    ][0],
                                                }
                                            )
                                            full_strategies.append(new_strat)
        return full_strategies
