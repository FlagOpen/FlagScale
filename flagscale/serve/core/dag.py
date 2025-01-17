import os, importlib
import sys
import uvicorn
import socket
import subprocess
import json


# import networkx as nx
import matplotlib.pyplot as plt
import ray
from ray import workflow
import omegaconf
import logging as logger

from pathlib import Path

from pydantic import create_model
from typing import Callable, Any
from fastapi import FastAPI, HTTPException, Request


def visualize_dag(dag, file_name):
    # Assign positions for nodes (simple linear layout for demonstration)
    positions = {}
    for idx, node in enumerate(dag):
        positions[node] = (idx * 2, 0)  # Space nodes horizontally

    plt.figure(figsize=(10, 6))

    # Keep track of drawn edges to avoid drawing the same edge twice
    drawn_edges = set()

    for node, neighbors in dag.items():
        x, y = positions[node]
        for neighbor in neighbors:
            nx, ny = positions[neighbor]
            if (node, neighbor) not in drawn_edges:
                # Draw arrow
                plt.arrow(
                    x, y, nx - x, ny - y,
                    head_width=0.15, head_length=0.3, fc="gray", ec="gray", length_includes_head=True, alpha=0.7
                )
                drawn_edges.add((node, neighbor))  # Mark edge as drawn
        # Draw the node
        plt.text(
            x, y, node, fontsize=12, ha="center", va="center",
            bbox=dict(facecolor="lightblue", edgecolor="black", boxstyle="round,pad=0.3")
        )

    plt.axis("off")
    plt.savefig(file_name, bbox_inches="tight", dpi=300)
    plt.close()


def _visualize_dag(dag, file_name):
    # Create a simple DAG visualization using matplotlib
    positions = {}
    for idx, node in enumerate(dag):
        positions[node] = (idx, 0)

    plt.figure(figsize=(8, 6))
    for node, neighbors in dag.items():
        x, y = positions[node]
        for neighbor in neighbors:
            nx, ny = positions[neighbor]
            plt.arrow(
                x, y, nx - x, ny - y,
                head_width=0.05, head_length=0.1, fc="gray", ec="gray", length_includes_head=True
            )
        plt.text(x, y, node, fontsize=10, ha="center", va="center", bbox=dict(facecolor="lightblue", alpha=0.5))

    plt.axis("off")
    plt.savefig(file_name, bbox_inches="tight")
    plt.close()

class Builder:
    def __init__(self, config):
        self.config = config.serve
        self.exp_config = config.experiment
        self.check_config(self.config)
        self.tasks = {}

    def check_config(self, config):
        if not config.get("deploy", None):
            raise ValueError("key deploy is missing for deployment configuration.")
        if not config.deploy.get("models", None):
            raise ValueError("key models is missing for building dag pipeline.")

    def find_final_node(self):
        whole_nodes = set(self.config["deploy"]["models"].keys())
        dependencies = set()

        for model_alias, model_config in self.config["deploy"]["models"].items():
            if len(model_config.get("depends", [])) > 0:
                dependencies.update(model_config.depends)

        output_node = whole_nodes - dependencies
        if len(output_node) != 1:
            raise ValueError(
                f"There should only have one final node but there are {len(output_node)} nodes {output_node}."
            )
        return list(output_node)[0]

    def check_and_get_port(self, target_port=None, host="0.0.0.0"):
        """
        Check if a specific port is free; if not, allocate a free port.
        :param target_port: The port number to check, default is None.
        :param host: The host address to check, default is "0.0.0.0".
        :return: A tuple (is_free, port), where `is_free` indicates if the target port is free,
                and `port` is the allocated port (target_port if free, or a new free port).
        """
        if target_port is None:
            # The same as Ray
            port = 6379
        else:
            port = target_port
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                # Try binding the target port
                s.bind((host, port))
                logger.info(f"Port {port} is free and can be used.")
                return port
            except OSError:
                # Target port is occupied, get a free port
                s.bind((host, 0))
                free_port = s.getsockname()[1]
                if target_port is None:
                    logger.info(
                        f"Port {port} is occupied. Allocated free port: {free_port}"
                    )
                else:
                    logger.warning(
                        f"Port {port} is occupied. Allocated free port: {free_port}"
                    )
                return free_port

    def check_dag(self, visibilization=True):
        # Ensure that all dependencies are valid
        dag = {}
        for model_alias, model_config in self.config["deploy"]["models"].items():
            dependencies = []
            if "depends" in model_config:
                deps = model_config["depends"]
                if not isinstance(deps, (list, omegaconf.listconfig.ListConfig)):
                    deps = [deps]
                dependencies = deps
            dag[model_alias] = dependencies

            for dep in dependencies:
                if dep not in self.config["deploy"]["models"]:
                    raise ValueError(
                        f"Dependency {dep} for model {model_alias} not found in config['deploy']['models']"
                    )

        # Helper function to check for cycles using DFS
        def is_cyclic(node, visited, stack):
            visited.add(node)
            stack.add(node)
            for neighbor in dag.get(node, []):
                if neighbor not in visited:
                    if is_cyclic(neighbor, visited, stack):
                        return True
                elif neighbor in stack:
                    return True
            stack.remove(node)
            return False

        # Check for cycles
        visited = set()
        for node in dag:
            if node not in visited:
                if is_cyclic(node, visited, set()):
                    if visibilization:
                        visualize_dag(dag, "dag1.png")
                    raise ValueError(
                        "The graph contains cycles and is not a Directed Acyclic Graph (DAG)."
                    )

        # Optionally visualize the DAG
        if visibilization:
            visualize_dag(dag, "dag.png")

    # def check_dag(self, visibilization=False):

    #     # Ensure that all dependencies are valid
    #     dag = {}
    #     for model_alias, model_config in self.config["deploy"]["models"].items():
    #         dependencies = []
    #         if "depends" in model_config:
    #             deps = model_config["depends"]
    #             if not isinstance(deps, (list, omegaconf.listconfig.ListConfig)):
    #                 deps = [deps]
    #             dependencies = deps
    #         else:
    #             dependencies = []
    #         dag[model_alias] = dependencies

    #         for dep in dependencies:
    #             if dep not in self.config["deploy"]["models"]:
    #                 raise ValueError(
    #                     f"Dependency {dep} for model {model_alias} not found in config['deploy']['models']"
    #                 )

    #     # Create a directed graph
    #     G = nx.DiGraph()

    #     # Add nodes and edges
    #     for node, neighbors in dag.items():
    #         for neighbor in neighbors:
    #             G.add_edge(node, neighbor)

    #     if visibilization or not nx.is_directed_acyclic_graph(G):
    #         pos = nx.spring_layout(G)
    #         plt.figure(figsize=(8, 6))
    #         nx.draw(
    #             G,
    #             pos,
    #             with_labels=True,
    #             node_color="lightblue",
    #             edge_color="gray",
    #             node_size=2000,
    #             font_size=15,
    #             arrows=True,
    #         )

    #         # Save the graph as an image without displaying it
    #         dag_file_name = "dag.png"
    #         plt.savefig(dag_file_name, bbox_inches="tight")
    #         plt.close()
    #         # Ensure that the graph is a DAG
    #         if not nx.is_directed_acyclic_graph(G):
    #             raise ValueError(
    #                 f"The graph contains cycles and is not a Directed Acyclic Graph (DAG). The dag can be visibilized at {dag_file_name}"
    #             )

    def init_cluster(self, pythonpath=""):

        hostfile = self.config.get("hostfile", None)
        address="auto"
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
                port = self.check_and_get_port()
            else:
                port = self.check_and_get_port(target_port=int(head_port))
            cmd = ["ray", "start", "--head", f"--port={port}", f"--storage={ray_path}"]
            logger.info(f"head node command: {cmd}")
            head_result = subprocess.run(
                cmd,
                check=True,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
            )
            if head_result.returncode != 0:
                logger.warning(
                    f"SSH command {ssh_cmd} failed with return code {head_result.returncode}."
                )
                logger.warning(f"Output: {head_result.stdout}")
                logger.warning(f"Error: {head_result.stderr}")
                sys.exit(head_result.returncode)
            address = f"{head_ip}:{port}"

            for item in hostfile.nodes:
                if "node" in item:
                    node = item.node
                    if node.type == "gpu":
                        node_cmd = (
                            f"ray start --address={address} --num-gpus={node.slots}"
                        )

                    elif node.type == "cpu":
                        node_cmd = (
                            f"ray start --address={address} --num-cpus={node.slots}"
                        )
                    else:
                        resource = json.dumps({node.type: node.slots}).replace('"', '\\"')
                        node_cmd = (
                            f"ray start --address={address} --resources='{resource}'"
                        )
                    if self.exp_config.get("cmds", "") and self.exp_config.cmds.get("before_start", ""):
                        before_start_cmd = self.exp_config.cmds.before_start
                        node_cmd = f"export RAY_STORAGE={ray_path} && {before_start_cmd} && " + node_cmd

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
        logger.warning(f" =========== pythonpath {pythonpath} -----------------------")
        if pythonpath:
            ray.init(
                address=address,
                runtime_env={"env_vars": {"PYTHONPATH": pythonpath}},
            )
        else:
            ray.init(address=address)

    def build_task(self):
        self.check_dag()
        pythonpath_tmp = set()
        for model_alias, model_config in self.config["deploy"]["models"].items():
            module_name = model_config["module"]
            path = Path(module_name)
            module_dir = str(path.parent)
            pythonpath_tmp.add(os.path.abspath(module_dir))
        pythonpath = ":".join(pythonpath_tmp)
        print(f" --------------------------- {pythonpath} ----------------------------- ", flush=True)
        self.init_cluster(pythonpath=pythonpath)

        for model_alias, model_config in self.config["deploy"]["models"].items():
            module_name = model_config["module"]
            model_name = model_config["name"]
            path = Path(module_name)
            module_tmp = path.stem
            module_dir = str(path.parent)
            sys.path.append(module_dir) 
            module = importlib.import_module(module_tmp)
            model = getattr(module, model_name)
            resources = model_config.resources
            num_gpus = resources.get("gpu", 0)
            num_cpus = resources.get("cpu", 1)
            customs = {res: resources[res] for res in resources if res not in ['gpu', 'cpu']}
            self.tasks[model_alias] = ray.remote(model).options(num_cpus=num_cpus, num_gpus=num_gpus, resources=customs)
            # tasks[model_alias] = ray.remote(num_gpus=num_gpus)(model)
            # models[model_alias] = model
        return

    def run_task(self, *input_data):
        assert len(self.tasks) > 0
        models_to_process = list(self.config["deploy"]["models"].keys())

        model_nodes = {}

        while models_to_process:
            progress = False
            for model_alias in list(models_to_process):
                model_config = self.config["deploy"]["models"][model_alias]
                dependencies = []
                if "depends" in model_config:
                    deps = model_config["depends"]
                    if not isinstance(deps, (list, omegaconf.listconfig.ListConfig)):
                        deps = [deps]
                    dependencies = deps
                else:
                    dependencies = []

                if all(dep in model_nodes for dep in dependencies):
                    if dependencies:
                        if len(dependencies) > 1:
                            inputs = [model_nodes[dep] for dep in dependencies]
                            model_nodes[model_alias] = self.tasks[model_alias].bind(
                                *inputs
                            )
                        else:
                            model_nodes[model_alias] = self.tasks[model_alias].bind(
                                model_nodes[dependencies[0]]
                            )
                    else:
                        if len(input_data)==0:
                            model_nodes[model_alias] = self.tasks[model_alias].bind()
                        else:
                            model_nodes[model_alias] = self.tasks[model_alias].bind(
                                *input_data
                            )
                    models_to_process.remove(model_alias)
                    progress = True
            if not progress:
                raise ValueError("Circular dependency detected in model configuration")

        logger.info(f" =========== deploy model_nodes ============= ", model_nodes)
        find_final_node = self.find_final_node()

        final_node = model_nodes[find_final_node]
        # pydot is required to plot DAG, install it with `pip install pydot`.
        # ray.dag.vis_utils.plot(final_node, "output.jpg")
        final_result = workflow.run(final_node)
        return final_result

    def run_router_task(self, method="post"):

        router_config = self.config["deploy"].get("service")

        assert router_config and len(router_config) > 0
        name = router_config["name"]
        port = router_config["port"]
        request_names = router_config["request"]["names"]
        request_types = router_config["request"]["types"]

        RequestData = create_model(
            "Request",
            **{field: (type_, ...) for field, type_ in zip(request_names, request_types)},
        )
        app = FastAPI()

        if method.lower() == "post":

            @app.post(name)
            async def route_handler(request_data: RequestData):
                input_data = tuple(getattr(request_data, field) for field in request_names)
                try:
                    response = self.run_task(*input_data)
                    return response
                except Exception as e:
                    raise HTTPException(status_code=400, detail=str(e))

        else:
            raise ValueError(f"Unsupported HTTP method: {method}")
        uvicorn.run(app, host="127.0.0.1", port=port)
