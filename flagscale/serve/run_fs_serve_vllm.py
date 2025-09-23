import logging

from ray import serve
from ray.serve.llm import LLMConfig, build_openai_app

from flagscale import serve as flagscale_serve  # Keep original configuration loading logic

# -------------------------- 1. Load existing configuration file --------------------------
flagscale_serve.load_args()
TASK_CONFIG = flagscale_serve.task_config


# Parse target model configuration from the config file (corresponds to serve_id: vllm_model in original YAML)
def get_model_config_from_task(serve_id: str = "vllm_model"):
    if not TASK_CONFIG.get("serve"):
        raise ValueError("No 'serve' section found in task config.")
    for item in TASK_CONFIG.serve:
        if item.get("serve_id") == serve_id:
            return item
    raise ValueError(f"Model config with serve_id={serve_id} not found.")


model_config = get_model_config_from_task()
engine_args = model_config.get("engine_args", {})
resources = model_config.get("resources", {})

# Define predefined key list for subsequent filtering
para_list = {
    # filter para
    "model",
    "served_model_name",
    "port",
    # defualt para
    "tensor_parallel_size",
    "pipeline_parallel_size",
    "gpu_memory_utilization",
    "max_model_len",
    "max_num_seqs",
    "enforce_eager",
    "enable_chunked_prefill",
    "enable_auto_tool_choice",
    "tool_call_parser",
}

# Initialize engine parameter dictionary with predefined options and default values
_engine_kwargs = {
    "tensor_parallel_size": engine_args.get("tensor_parallel_size", 1),
    "pipeline_parallel_size": engine_args.get("pipeline_parallel_size", 1),
    "gpu_memory_utilization": engine_args.get("gpu_memory_utilization", 0.9),
    "max_model_len": engine_args.get("max_model_len", 32768),
    "max_num_seqs": engine_args.get("max_num_seqs", 256),
    "enforce_eager": engine_args.get("enforce_eager", True),
    "enable_chunked_prefill": engine_args.get("enable_chunked_prefill", True),
    "enable_auto_tool_choice": engine_args.get("enable_auto_tool_choice", False),
    "tool_call_parser": engine_args.get("tool_call_parser", None),
}

# -------------------------- 2. Define Ray LLM configuration --------------------------
llm_config = LLMConfig(
    # Core configuration for model loading
    model_loading_config={
        "model_id": engine_args.get("served_model_name", engine_args["model"]),
        "model_source": engine_args["model"],
    },
    # Deployment resource configuration (corresponds to original resources field)
    deployment_config={
        "autoscaling_config": {
            "min_replicas": resources.get("num_replicas", 1),
            "max_replicas": resources.get("num_replicas", 1),
            "target_num_ongoing_requests_per_replica": 1000,
        }
    },
    # Engine-specific parameters (vLLM exclusive configuration, passed directly to vLLM engine)
    engine_kwargs=_engine_kwargs,
    # Runtime environment (add environment variables if needed)
    # This part is already handled by flagscale
    runtime_env={},
)

# -------------------------- 3. Start Ray Serve service --------------------------
if __name__ == "__main__":
    # Configure logging
    logger = logging.getLogger("ray.serve")
    logger.setLevel(logging.INFO)
    logger.addHandler(logging.FileHandler("serve.log"))

    # Add other key-value pairs from engine_args not included in predefined keys
    for key, value in engine_args.items():
        if key not in para_list:
            _engine_kwargs[key] = value
            logger.info("Adding untested engine arg: %s=%s", key, value)
    if engine_args.get("served_model_name", None):
        logger.info("Serving model name: %s", engine_args["served_model_name"])
    else:
        logger.warning(
            'No served_model_name specified in engine_args, using {engine_args["model"]}.'
        )

    # Start Ray Serve and set HTTP port (read from original config, default 8000)
    deploy_port = TASK_CONFIG.experiment.get("runner", {}).get("deploy", {}).get("port", 8000)
    serve.start(http_options={"host": "0.0.0.0", "port": deploy_port})  # Allow external access

    # Build OpenAI-compatible app and run
    app = build_openai_app({"llm_configs": [llm_config]})
    serve.run(
        app,
        name="vllm_service",  # Service name (corresponds to original SERVICE_NAME)
        blocking=True,  # Block main thread to keep service running
    )
