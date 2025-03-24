import asyncio
import logging
import time
from typing import Any, AsyncGenerator

import ray
from fastapi import FastAPI
from fastapi.responses import JSONResponse, Response, StreamingResponse

from vllm import SamplingParams
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.entrypoints.openai.protocol import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionStreamResponse,
    CompletionRequest,
    CompletionResponse,
    CompletionResponseStreamChoice,
    CompletionStreamResponse,
    UsageInfo,
)
from vllm.utils import random_uuid

try:
    import flag_scale
except Exception as e:
    pass

from flagscale import serve

serve.load_args()
TASK_CONFIG = serve.task_config

SERVICE_NAME = "vllm_service"


def get_model_config(model_name):
    if not TASK_CONFIG.get("serve", None):
        raise ValueError("No 'serve' section found in task config.")
    if not TASK_CONFIG.serve.get("model_args", None):
        raise ValueError("No 'model_args' section found in task config.")
    model_config = TASK_CONFIG.serve.model_args.get(model_name, None)

    if model_config:
        return model_config
    else:
        raise ValueError(f"No model config found for model {model_name}.")


def get_deploy_config(model_name):
    if not TASK_CONFIG.get("serve", None):
        raise ValueError("No 'serve' section found in task config.")
    if not TASK_CONFIG.serve.get("deploy", None):
        raise ValueError("No 'deploy' section found in task config.")
    resource_config = {}

    if TASK_CONFIG.serve.deploy.get(
        "models", None
    ) and TASK_CONFIG.serve.deploy.models.get(model_name, None):
        models_resource_config = TASK_CONFIG.serve.deploy.models.get(model_name, None)
        ray_actor_options = {}
        resource_set = {"num_gpus", "num_cpus"}
        for item in resource_set:
            if item in models_resource_config:
                ray_actor_options[item] = models_resource_config[item]
        if ray_actor_options:
            resource_config["ray_actor_options"] = ray_actor_options
        if "num_replicas" in models_resource_config:
            resource_config["num_replicas"] = models_resource_config["num_replicas"]
    if not resource_config:
        resource_config = {"num_replicas": 1}
    resource_config["max_ongoing_requests"] = 1000
    return resource_config


def get_sample_args(request):
    pre_args = {"temperature", "top_p", "top_k", "max_tokens", "logprobs"}
    items = request.model_dump(exclude_unset=True)
    sample_args = {key: items[key] for key in pre_args if key in items}
    if "max_completion_tokens" in items:
        sample_args["max_tokens"] = items["max_completion_tokens"]
    return sample_args


app = FastAPI()

from ray import serve

logger = logging.getLogger("ray.serve")

logger.setLevel(logging.INFO)
logger.addHandler(logging.FileHandler("serve.log"))


def check_health(service_name):
    status = serve.status()
    if service_name in status.applications:
        service_status = status.applications[service_name].status
        logger.info(f"service {service_name} status: {service_status}")
        if service_status == "RUNNING":
            return True
    logger.info(f"service {service_name} is not ready")
    return False


@serve.deployment(**get_deploy_config("vllm_model"))
class LLMActor:
    def __init__(self):
        engine_args = AsyncEngineArgs(**get_model_config("vllm_model"))
        self.engine = AsyncLLMEngine.from_engine_args(engine_args)

    def generate(self, prompt, sampling_params, request_id):
        return self.engine.generate(prompt, sampling_params, request_id)


@serve.deployment(num_replicas="auto", max_ongoing_requests=1000)
@serve.ingress(app)
class LLMService:
    def __init__(self, llm_actor):
        self.llm_actor = llm_actor
        self.ready = False

    @app.post("/v1/completions")
    async def generate_handler(self, request: CompletionRequest):
        logger.debug(f"========== Receive request {request}========== ")
        if not self.ready:
            self.ready = check_health(SERVICE_NAME)
            if not self.ready:
                return JSONResponse(
                    status_code=503,
                    content={
                        "message": "Service is not ready, please try again later."
                    },
                )
        prompt = request.prompt
        stream = request.stream
        request_id = "cmpl-" + random_uuid()
        sample_args = get_sample_args(request)
        sampling_params = SamplingParams(**sample_args)
        results_generator = self.llm_actor.generate.options(stream=True).remote(
            prompt,
            sampling_params,
            request_id,
        )

        if stream:

            async def stream_results() -> AsyncGenerator[bytes, None]:
                num_choices = 1 if request.n is None else request.n
                previous_num_tokens = [0] * num_choices
                num_prompt_tokens = 0

                async for request_output in results_generator:
                    prompt = request_output.prompt
                    assert prompt is not None
                    text_outputs = ""
                    for item in request_output.outputs:

                        i = item.index
                        previous_num_tokens[i] = len(item.token_ids)
                        text_outputs += item.text

                    chunk = CompletionStreamResponse(
                        id=request_id,
                        created=int(time.time()),
                        model=request.model,
                        choices=[
                            CompletionResponseStreamChoice(
                                index=0,
                                text=text_outputs,
                                logprobs=None,
                                finish_reason=None,
                                stop_reason=None,
                            )
                        ],
                    )
                    if request_output.prompt_token_ids is not None:
                        num_prompt_tokens = len(request_output.prompt_token_ids)
                    response_json = chunk.model_dump_json(exclude_unset=True)
                    yield f"data: {response_json}\n\n"

                if request.stream_options and request.stream_options.include_usage:
                    completion_tokens = sum(previous_num_tokens)
                    final_usage = UsageInfo(
                        prompt_tokens=num_prompt_tokens,
                        completion_tokens=completion_tokens,
                        total_tokens=num_prompt_tokens + completion_tokens,
                    )

                    final_usage_chunk = CompletionStreamResponse(
                        id=request_id,
                        object="text_completion",
                        created=int(time.time()),
                        choices=[],
                        model=request.model,
                        usage=final_usage,
                    )
                    final_usage_data = final_usage_chunk.model_dump_json(
                        exclude_unset=True, exclude_none=True
                    )
                    yield f"data: {final_usage_data}\n\n"

                yield "data: [DONE]\n\n"

            return StreamingResponse(stream_results(), media_type="text/event-stream")
        else:
            final_output = None
            try:
                async for request_output in results_generator:
                    final_output = request_output
            except asyncio.CancelledError:
                return Response(status_code=499)

            text_outputs = ""
            prompt_tokens = len(final_output.prompt_token_ids)
            completion_tokens = 0

            assert final_output is not None
            prompt = final_output.prompt
            assert prompt is not None

            for item in final_output.outputs:
                text_outputs += item.text
                completion_tokens += len(item.token_ids)

            ret = CompletionResponse(
                id=request_id,
                created=int(time.time()),
                model=request.model,
                choices=[
                    {
                        "text": text_outputs,
                        "index": 0,
                        "logprobs": None,
                        "finish_reason": None,
                    }
                ],
                usage={
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": prompt_tokens + completion_tokens,
                },
            )

            return JSONResponse(content=ret.model_dump())

    @app.post("/v1/chat/completions")
    async def generate_handler(self, request: ChatCompletionRequest):
        logger.debug(f"========== Receive request {request}========== ")
        if not self.ready:
            self.ready = check_health(SERVICE_NAME)
            if not self.ready:
                return JSONResponse(
                    status_code=503,
                    content={
                        "message": "Service is not ready, please try again later."
                    },
                )
        user_message = request.messages[-1]["content"]
        if isinstance(user_message, list):
            user_message = " ".join(
                [item["text"] for item in user_message if item["type"] == "text"]
            )

        stream = request.stream
        request_id = "cmpl-" + random_uuid()
        sample_args = get_sample_args(request)
        logger.debug(f"Request {request_id} sampling_params {sample_args}")
        sampling_params = SamplingParams(**sample_args)
        results_generator = self.llm_actor.generate.options(stream=True).remote(
            user_message,
            sampling_params,
            request_id,
        )

        if stream:

            async def stream_results() -> AsyncGenerator[bytes, None]:
                num_choices = 1 if request.n is None else request.n
                previous_num_tokens = [0] * num_choices
                num_prompt_tokens = 0

                async for request_output in results_generator:
                    prompt = request_output.prompt
                    assert prompt is not None
                    text_outputs = ""
                    for item in request_output.outputs:

                        i = item.index
                        previous_num_tokens[i] = len(item.token_ids)
                        text_outputs += item.text

                    chunk = ChatCompletionStreamResponse(
                        id=request_id,
                        created=int(time.time()),
                        model=request.model,
                        choices=[
                            {
                                "index": 0,
                                "delta": {"role": "assistant", "content": text_outputs},
                                "logprobs": None,
                                "finish_reason": None,
                                "stop_reason": None,
                            }
                        ],
                    )
                    if request_output.prompt_token_ids is not None:
                        num_prompt_tokens = len(request_output.prompt_token_ids)
                    response_json = chunk.model_dump_json(exclude_unset=True)
                    yield f"data: {response_json}\n\n"
                if request.stream_options and request.stream_options.include_usage:
                    completion_tokens = sum(previous_num_tokens)
                    final_usage = UsageInfo(
                        prompt_tokens=num_prompt_tokens,
                        completion_tokens=completion_tokens,
                        total_tokens=num_prompt_tokens + completion_tokens,
                    )

                    final_usage_chunk = ChatCompletionStreamResponse(
                        id=request_id,
                        object="chat.completion.chunk",
                        created=int(time.time()),
                        choices=[],
                        model=request.model,
                        usage=final_usage,
                    )
                    final_usage_data = final_usage_chunk.model_dump_json(
                        exclude_unset=True, exclude_none=True
                    )
                    yield f"data: {final_usage_data}\n\n"
                yield "data: [DONE]\n\n"

            logger.debug(f"Return reponse for request {request_id} ")
            return StreamingResponse(stream_results(), media_type="text/event-stream")
        else:
            final_output = None
            try:
                async for request_output in results_generator:
                    final_output = request_output
            except asyncio.CancelledError:
                return Response(status_code=499)

            assert final_output is not None
            prompt = final_output.prompt
            assert prompt is not None

            text_outputs = ""
            prompt_tokens = len(final_output.prompt_token_ids)
            completion_tokens = 0

            for item in final_output.outputs:
                text_outputs += item.text
                completion_tokens += len(item.token_ids)

            ret = ChatCompletionResponse(
                id=request_id,
                created=int(time.time()),
                model=request.model,
                choices=[
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": text_outputs},
                        "logprobs": None,
                        "finish_reason": None,
                    }
                ],
                usage={
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": prompt_tokens + completion_tokens,
                },
            )

            return JSONResponse(content=ret.model_dump())


if __name__ == "__main__":
    serve.start(
        http_options={"host": "0.0.0.0", "port": TASK_CONFIG.serve.deploy.service.port}
    )
    llm_actor = LLMActor.bind()
    serve.run(
        LLMService.bind(llm_actor),
        name=SERVICE_NAME,
        route_prefix="/",
        blocking=True,
    )
