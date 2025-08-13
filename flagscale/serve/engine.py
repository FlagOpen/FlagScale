import importlib
import importlib.util
import inspect
import json
import logging
import os
import subprocess
import sys

from pathlib import Path
from typing import Any, Dict

import matplotlib.pyplot as plt
import numpy as np
import omegaconf
import ray
import uvicorn
import yaml

from dag_utils import check_and_get_port
from fastapi import FastAPI, HTTPException, Request
from pydantic import create_model
from ray import serve
from ray.serve.handle import DeploymentHandle

# from flagscale.logger import logger
logger = logging.getLogger("ray.serve")

logger.setLevel(logging.INFO)


def load_class_from_file(file_path: str, class_name: str):
    file_path = os.path.abspath(file_path)
    logger.info(f"Loading class {class_name} from file: {file_path}")
    sys.path.insert(0, os.path.dirname(file_path))
    module_name = os.path.splitext(os.path.basename(file_path))[0]
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None:
        raise ImportError(f"Cannot create module spec from {file_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, class_name):
        raise ImportError(f"Class {class_name} not found in {file_path}")
    return getattr(module, class_name)


def make_deployment(logic_cls, **deploy_kwargs):
    @serve.deployment(**deploy_kwargs)
    class WrappedModel:
        def __init__(self):
            self.logic = logic_cls()

        async def forward(self, *args, **kwargs):
            if inspect.iscoroutinefunction(self.logic.forward):
                return await self.logic.forward(*args, **kwargs)
            return self.logic.forward(*args, **kwargs)

    return WrappedModel


@serve.deployment
class FinalModel:
    def __init__(
        self,
        graph_config: Dict[str, Any],
        handles: Dict[str, DeploymentHandle],
        config: omegaconf.DictConfig,
    ):
        self.graph_config = graph_config
        self.handles = handles

        # determine return nodes
        all_nodes = set(graph_config.keys())
        dep_nodes = {dep for cfg in graph_config.values() for dep in cfg.get("depends", [])}
        self.roots = list(all_nodes - dep_nodes)
        assert len(self.roots) == 1, "Only one return node is allowed"
        request_config = config.experiment.runner.deploy.request
        self.request_base = create_model(
            "Request",
            **{
                field: (type_, ...)
                for field, type_ in zip(request_config.args, request_config.types)
            },
        )

    async def __call__(self, http_request):
        origin_request = await http_request.json()
        request_data = self.request_base(**origin_request).dict()

        results_cache = {}

        async def run_node(node_name, **input_data):
            if node_name in results_cache:
                return results_cache[node_name]

            node_cfg = self.graph_config[node_name]
            handle = self.handles[node_name]

            if node_cfg.get("depends"):
                dep_results = []
                for dep in node_cfg["depends"]:
                    res = await run_node(dep, **input_data)
                    dep_results.append(res)
                if len(dep_results) == 1:
                    result = await handle.forward.remote(dep_results[0])
                else:
                    result = await handle.forward.remote(*dep_results)
            else:
                result = await handle.forward.remote(**input_data)

            results_cache[node_name] = result
            return result

        final_results = {}
        root = self.roots[0]
        final_results[root] = await run_node(root, **request_data)
        return final_results[root]


def build_graph(config):
    connection_list = config.serve
    # Convert list of dicts with 'serve_id' to dict keyed by serve_id
    connection = {
        cfg["serve_id"]: {k: v for k, v in cfg.items() if k != "serve_id"}
        for cfg in connection_list
    }

    handles = {}
    deployments = {}
    for name, cfg in connection.items():
        logic_cls = load_class_from_file(cfg["module"], cfg["name"])
        resources = cfg.get("resources", {})
        ray_actor_options = {}
        if "num_gpus" in resources:
            ray_actor_options["num_gpus"] = resources["num_gpus"]
        deploy_kwargs = {
            "num_replicas": resources.get("num_replicas", 1),
            "ray_actor_options": ray_actor_options,
        }
        deployments[name] = make_deployment(logic_cls, **deploy_kwargs)
        handles[name] = deployments[name].bind()

    root_model = FinalModel.bind(connection, handles, config)
    return root_model


class ServeEngine:
    def __init__(self, config):
        self.config = config
        self.model_config = config.serve
        self.exp_config = config.experiment
        self.check_task(self.exp_config)
        self.tasks = {}

    def check_task(self, config):
        if not config.get("runner", {}).get("deploy", None):
            raise ValueError("key deploy is missing for deployment configuration.")
        self.check_dag()

    def check_dag(self, visibilization=True):
        # Ensure that all dependencies are valid
        dag = {}
        for model_alias, model_config in ((k, v) for d in self.model_config for k, v in d.items()):
            dependencies = []
            if "depends" in model_config:
                deps = model_config["depends"]
                if not isinstance(deps, (list, omegaconf.listconfig.ListConfig)):
                    deps = [deps]
                dependencies = deps
            dag[model_alias] = dependencies

            for dep in dependencies:
                if dep not in self.model_config["deploy"]["models"]:
                    raise ValueError(
                        f"Dependency {dep} for model {model_alias} not found in config['deploy']['models']"
                    )

        # Helper function to check for cycles using DFS
        def _is_cyclic(node, visited, stack):
            visited.add(node)
            stack.add(node)
            for neighbor in dag.get(node, []):
                if neighbor not in visited:
                    if _is_cyclic(neighbor, visited, stack):
                        return True
                elif neighbor in stack:
                    return True
            stack.remove(node)
            return False

        # Check for cycles
        visited = set()
        for node in dag:
            if node not in visited:
                if _is_cyclic(node, visited, set()):
                    raise ValueError(
                        "The graph contains cycles and is not a Directed Acyclic Graph (DAG)."
                    )

        def _visualize_dag_with_force_directed_layout(
            dag, file_name, iterations=100, k=1.0, t=1.0, cooling_factor=0.9
        ):
            nodes = list(dag.keys())
            n = len(nodes)

            # Initialize node positions
            positions = {node: np.random.rand(2) * 10 for node in nodes}

            for _ in range(iterations):
                # Calculate repulsive forces
                for i in range(n):
                    for j in range(i + 1, n):
                        node1, node2 = nodes[i], nodes[j]
                        delta = positions[node1] - positions[node2]
                        distance = np.linalg.norm(delta)
                        if distance > 1e-10:
                            f = (delta / distance) * (k**2 / distance)
                            positions[node1] += f
                            positions[node2] -= f

                # Calculate attractive forces
                for node, neighbors in dag.items():
                    for neighbor in neighbors:
                        delta = positions[node] - positions[neighbor]
                        distance = np.linalg.norm(delta)
                        if distance > 1e-10:
                            f = (delta / distance) * (distance / k)
                            positions[node] -= f
                            positions[neighbor] += f

                # Cool down
                t *= cooling_factor
                # Limit movement step
                for node in nodes:
                    move = np.random.randn(2) * t
                    positions[node] += move

            # Normalize positions
            all_positions = np.array([positions[node] for node in nodes])
            x_min, y_min = all_positions.min(axis=0)
            x_max, y_max = all_positions.max(axis=0)
            all_positions = (all_positions - [x_min, y_min]) / ([x_max - x_min, y_max - y_min])
            for i, node in enumerate(nodes):
                positions[node] = all_positions[i]

            # Create figure
            plt.figure(figsize=(8, 6))

            # Draw edges
            for node, neighbors in dag.items():
                x, y = positions[node]
                for neighbor in neighbors:
                    nx, ny = positions[neighbor]
                    plt.arrow(
                        x,
                        y,
                        nx - x,
                        ny - y,
                        head_width=0.04,
                        head_length=0.08,
                        fc="gray",
                        ec="gray",
                        length_includes_head=True,
                        alpha=0.8,
                        zorder=5,
                    )

            # Draw nodes
            for node, (x, y) in positions.items():
                plt.scatter(x, y, s=800, color="lightblue", edgecolors="black", zorder=3)
                plt.text(x, y, node, fontsize=12, ha="center", va="center", zorder=4)

            # Add title
            plt.title("Directed Acyclic Graph (DAG)", fontsize=14)

            # Set aspect ratio
            plt.axis("equal")

            # Hide axes
            plt.axis("off")

            # Save figure
            plt.savefig(file_name)
            plt.close()

        # Optionally visualize the DAG
        if visibilization:
            dag_img_path = os.path.join(self.exp_config.exp_dir, "dag.png")
            _visualize_dag_with_force_directed_layout(dag, dag_img_path)

    def init_task(self, pythonpath=""):
        hostfile = self.model_config.get("hostfile", None)
        address = "auto"
        exp_path = os.path.join(self.exp_config.exp_dir, "ray_workflow")
        ray_path = os.path.abspath(exp_path)
        if hostfile:
            head_ip, head_port = next(
                (
                    (node.master.ip, node.master.get("port", None))
                    for node in hostfile.nodes
                    if "master" in node
                ),
                (None, None),
            )
            if head_ip is None:
                raise ValueError(
                    f"Failed to start Ray cluster using hostfile {hostfile} due to master node missing. Please ensure that the file exists and has the correct format."
                )
            if head_port is None:
                port = check_and_get_port()
            else:
                port = check_and_get_port(target_port=int(head_port))
            cmd = ["ray", "start", "--head", f"--port={port}", f"--storage={ray_path}"]
            logger.info(f"head node command: {cmd}")
            head_result = subprocess.run(
                cmd, check=True, capture_output=True, text=True, encoding="utf-8", errors="replace"
            )
            if head_result.returncode != 0:
                logger.warning(
                    f"Head Node cmd {ssh_cmd} failed with return code {head_result.returncode}."
                )
                logger.warning(f"Output: {head_result.stdout}")
                logger.warning(f"Error: {head_result.stderr}")
                sys.exit(head_result.returncode)
            address = f"{head_ip}:{port}"

            for item in hostfile.nodes:
                if "node" in item:
                    node = item.node
                    if node.type == "gpu":
                        node_cmd = f"ray start --address={address} --num-gpus={node.slots}"

                    elif node.type == "cpu":
                        node_cmd = f"ray start --address={address} --num-cpus={node.slots}"
                    else:
                        resource = json.dumps({node.type: node.slots}).replace('"', '\\"')
                        node_cmd = f"ray start --address={address} --resources='{resource}'"
                    if self.exp_config.get("cmds", "") and self.exp_config.cmds.get(
                        "before_start", ""
                    ):
                        before_start_cmd = self.exp_config.cmds.before_start
                        node_cmd = (
                            f"export RAY_STORAGE={ray_path} && {before_start_cmd} && " + node_cmd
                        )

                    if node.get("port", None):
                        ssh_cmd = f'ssh -n -p {node.port} {node.ip} "{node_cmd}"'
                    else:
                        ssh_cmd = f'ssh -n {node.ip} "{node_cmd}"'

                    logger.info(f"worker node command: {cmd}")

                    result = subprocess.run(
                        ssh_cmd,
                        shell=True,
                        check=True,
                        capture_output=True,
                        text=True,
                        encoding="utf-8",
                        errors="replace",
                    )
                    if result.returncode != 0:
                        logger.warning(
                            f"SSH command {ssh_cmd} failed with return code {result.returncode}."
                        )
                        logger.warning(f"Output: {result.stdout}")
                        logger.warning(f"Error: {result.stderr}")
                        sys.exit(result.returncode)
        else:
            port = check_and_get_port()
            head_ip = "127.0.0.1"
            cmd = ["ray", "start", "--head", f"--port={port}", f"--storage={ray_path}"]
            logger.info(f"head node command: {cmd}")
            head_result = subprocess.run(
                cmd, check=True, capture_output=True, text=True, encoding="utf-8", errors="replace"
            )
            if head_result.returncode != 0:
                logger.warning(
                    f"local command {cmd} failed with return code {head_result.returncode}."
                )
                logger.warning(f"Output: {head_result.stdout}")
                logger.warning(f"Error: {head_result.stderr}")
                sys.exit(head_result.returncode)
            address = f"{head_ip}:{port}"

        logger.info(f" =========== pythonpath {pythonpath} -----------------------")
        if pythonpath:
            ray.init(address=address, runtime_env={"env_vars": {"PYTHONPATH": pythonpath}})
        else:
            ray.init(address=address)

    def run_task(self):
        graph = build_graph(self.config)
        serve.start(http_options={"port": self.exp_config.runner.deploy.get("port", 8000)})
        serve.run(
            graph,
            name=self.exp_config.exp_name,
            route_prefix=self.exp_config.runner.deploy.get("name", "/"),
            blocking=True,
        )
