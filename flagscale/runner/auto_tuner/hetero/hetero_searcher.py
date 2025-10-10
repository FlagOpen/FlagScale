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
        if n - i < (k - 1): continue
        for rest in _generate_all_partitions_with_max_diff(n - i, k - 1, max_diff):
            if not rest:
                if k-1 == 0 and n-i == 0:
                     if i <= max_diff: yield [i]
                continue
            
            # Keep partitions in descending order to avoid duplicates
            if i < rest[0]:
                continue
            
            # Check the max_diff constraint
            if i - rest[-1] > max_diff:
                continue

            yield [i] + rest

def _generate_flexible_split(n: int, k: int, max_diff: int = 1) -> List[int]:
    """
    Generates a representative partition of n into k parts where the difference
    between parts is no more than max_diff. This is a heuristic approach.
    """
    if k <= 0 or n < k:
        return []
    
    base = n // k
    rem = n % k
    
    result = [base] * k
    for i in range(rem):
        result[i] += 1
    
    result.sort(reverse=True)
    current_diff = result[0] - result[-1]
    
    l, r = 0, k - 1
    while current_diff < max_diff and l < r:
        if result[l] - result[r] >= max_diff:
            break
        if r > 0 and result[r] - 1 < result[r-1]:
             r -= 1
             continue
        if l < k -1 and result[l] + 1 > result[l+1]:
            l += 1
            continue

        result[l] += 1
        result[r] -= 1
        result.sort(reverse=True)
        current_diff = result[0] - result[-1]

    if result[0] - result[-1] > max_diff:
        # Fallback to the most balanced split if the heuristic overshoots
        result = [base + 1] * rem + [base] * (k - rem)

    return result

def _generate_valid_layer_splits(total_layers: int, mesh_pp_sizes: List[int], inter_mesh_max_diff: int):
    """
    Generates valid layer splits using the new intelligent splitting algorithms.
    """
    num_meshes = len(mesh_pp_sizes)
    if num_meshes == 0:
        return
    
    # 1. Use the new partition function with max_diff for inter-mesh layer distribution
    partition_generator = _generate_all_partitions_with_max_diff(total_layers, num_meshes, inter_mesh_max_diff)

    for inter_mesh_partition in partition_generator:
        for mesh_layer_distribution in set(itertools.permutations(inter_mesh_partition)):
            final_split = []
            is_distribution_possible = True
            for i in range(num_meshes):
                layers_for_this_mesh = mesh_layer_distribution[i]
                local_pp = mesh_pp_sizes[i]
                if layers_for_this_mesh < local_pp:
                    is_distribution_possible = False
                    break
                
                # 2. Use the flexible split for intra-mesh distribution with a hardcoded max_diff of 2
                intra_mesh_split = _generate_flexible_split(layers_for_this_mesh, local_pp, max_diff=2)
                
                if not intra_mesh_split or sum(intra_mesh_split) != layers_for_this_mesh:
                    is_distribution_possible = False
                    break
                
                final_split.extend(intra_mesh_split)
            
            if is_distribution_possible:
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
        # Added attributes to store parsed search space for new features
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

        # Parse the new layer splitting constraint from the config
        self.layer_split_constraints['inter_mesh_max_diff'] = safe_to_container(
            hetero_space.get("hetero_inter_mesh_max_layer_diff", "auto")
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
        for gran_list, meth_list, num_list in itertools.product(granularity_options, method_options, num_layers_options):
            def render_template(template_list):
                rendered_list = []
                for item in template_list:
                    rendered_item = [val if val != 'ALL' else num_micro_batches for val in item]
                    rendered_list.append(rendered_item)
                return rendered_list

            all_recompute_combinations.append({
                "recompute_granularity_per_stage_micro_batch": render_template(gran_list),
                "recompute_method_per_stage_micro_batch": render_template(meth_list),
                "recompute_num_layers_per_stage_micro_batch": render_template(num_list)
            })
        
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

        for assignment in all_assignments:
            global_pp_size = sum(item['mesh'][4] for item in assignment)
            
            base_parallel_info = {
                "pipeline_model_parallel_size": global_pp_size,
                "hetero_process_meshes": [item['mesh'] for item in assignment],
                "hetero_device_types": [item['device_type'] for item in assignment],
            }
            
            # Handle layer splitting with new constraints
            layer_splits = []
            if space["hetero_pipeline_layer_split"] == 'auto':
                if global_pp_size > 0:
                    mesh_pp_sizes = [item['mesh'][4] for item in assignment]
                    inter_mesh_diff = self.layer_split_constraints['inter_mesh_max_diff']
                    if inter_mesh_diff == 'auto':
                        # Heuristic for auto mode
                        inter_mesh_diff = max(config.train.model.num_layers // len(mesh_pp_sizes) if len(mesh_pp_sizes)>0 else config.train.model.num_layers, 4)

                    layer_splits.extend(list(_generate_valid_layer_splits(config.train.model.num_layers, mesh_pp_sizes, inter_mesh_diff)))
                else:
                    layer_splits.append([])
            elif isinstance(space["hetero_pipeline_layer_split"], list):
                for split in space["hetero_pipeline_layer_split"]:
                    if len(split) == global_pp_size and sum(split) == config.train.model.num_layers:
                        layer_splits.append(split)

            for split in layer_splits:
                dp_list = [mesh[3] for mesh in base_parallel_info["hetero_process_meshes"]]
                for mbs in space["micro_batch_size"]:
                    is_dp_compatible = all((gbs % (dp * mbs) == 0) for dp in dp_list if dp > 0)
                    if not is_dp_compatible: continue

                    all_dp_are_one = all(dp == 1 for dp in dp_list)
                    do_options = [False] if all_dp_are_one else space["use_distributed_optimizer"]
                    for use_do in do_options:
                        for sp in space["sequence_parallel"]:
                            all_tp_gt_one = all(mesh[0] > 1 for mesh in base_parallel_info["hetero_process_meshes"])
                            if sp and not all_tp_gt_one: continue

                            base_strategies_without_recompute.append({
                                **base_parallel_info,
                                "hetero_pipeline_layer_split": split,
                                "micro_batch_size": mbs,
                                "use_distributed_optimizer": use_do,
                                "sequence_parallel": sp,
                            })

        self.logger.info(f"Created {len(base_strategies_without_recompute)} base strategies.")

        # Stage 2: Dynamically append recompute configurations
        final_strategies = []
        for base_strategy in base_strategies_without_recompute:
            use_recompute_options = self.recompute_search_space['use_recompute']
            
            if False in use_recompute_options:
                final_strategies.append({
                    **base_strategy,
                    'use_recompute': False,
                    "recompute_granularity_per_stage_micro_batch": None,
                    "recompute_method_per_stage_micro_batch": None,
                    "recompute_num_layers_per_stage_micro_batch": None,
                })
            
            if True in use_recompute_options:
                pp_size = base_strategy['pipeline_model_parallel_size']
                mbs = base_strategy['micro_batch_size']
                dp = base_strategy['hetero_process_meshes'][0][3]

                if (mbs * dp) == 0: num_micro_batches = gbs
                else: num_micro_batches = gbs // (mbs * dp)
                if num_micro_batches == 0: num_micro_batches = 1

                recompute_combinations = self._generate_recompute_configs(pp_size, num_micro_batches)
                for recom_config in recompute_combinations:
                    if recom_config:
                        final_strategies.append({
                            **base_strategy,
                            'use_recompute': True,
                            **recom_config
                        })
        
        # Finally, add compatibility keys for the Pruner
        for strategy in final_strategies:
            meshes = strategy.get("hetero_process_meshes", [])
            use_recompute = strategy.get('use_recompute', False)
            strategy.update({
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
            })

        self.logger.info(
            f"Built a total of {len(final_strategies)} comprehensive candidate strategies."
        )
        return final_strategies

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
