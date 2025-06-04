import asyncio
import base64
import logging
import time

from io import BytesIO
from typing import Any, AsyncGenerator

import ray

from fastapi import FastAPI
from fastapi.responses import JSONResponse, Response, StreamingResponse
from PIL import Image
from transformers import AutoTokenizer

from vllm import SamplingParams
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.entrypoints.chat_utils import (
    apply_hf_chat_template,
    parse_chat_messages,
    resolve_chat_template_content_format,
)
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
from vllm.inputs import PromptType, SingletonPrompt, TextPrompt, TokensPrompt
from vllm.utils import random_uuid

try:
    import flag_scale
except Exception as e:
    pass

from flagscale import serve

serve.load_args()
TASK_CONFIG = serve.task_config

SERVICE_NAME = "vllm_service"


def decode_base64_to_image(base64_str: str) -> Image.Image:
    # If the string looks like "data:image;base64,AAAA..."
    # split off the prefix up to the comma
    if "," in base64_str:
        base64_str = base64_str.split(",", 1)[1]

    # Decode the Base64 string into bytes
    image_data = base64.b64decode(base64_str)

    # Convert the bytes into a PIL Image
    pil_img = Image.open(BytesIO(image_data)).convert("RGB")
    return pil_img


def parse_dict_arg(val: str):
    """Parses a string of comma-separated KEY=VALUE pairs into a dict."""
    if len(val) == 0:
        return None
    out = dict()
    for item in val.split(","):
        # split into key and value, lowercase & strip whitespace
        key, value = [p.lower().strip() for p in item.split("=")]
        if not key or not value:
            raise ValueError("Each item should be in the form KEY=VALUE")
        iv = int(value)
        # disallow conflicting duplicates
        if key in out and out[key] != iv:
            raise ValueError(f"Conflicting values specified for key: {key}")
        out[key] = iv
    return out


def get_engine_args(model_name):
    if not TASK_CONFIG.get("serve", None):
        raise ValueError("No 'serve' section found in task config.")

    model_config = None
    for item in TASK_CONFIG.serve:
        if item.get("serve_id", None) == model_name:
            model_config = item
            break
    if model_config is None:
        raise ValueError(f"No {model_name} configuration found in task config: {TASK_CONFIG}")

    common_args = model_config.get("engine_args", {})
    engine_args = model_config.get("engine_args_specific", {}).get("vllm", {})
    engine_args.update(common_args)

    if engine_args:
        limit_mm_per_prompt = engine_args.get("limit_mm_per_prompt", None)
        if isinstance(limit_mm_per_prompt, str):
            parsed_value = parse_dict_arg(engine_args["limit_mm_per_prompt"])
            engine_args["limit_mm_per_prompt"] = parsed_value
        engine_args.pop("port", None)
        return engine_args
    else:
        raise ValueError(f"No vllm args found for serve_id {model_name}.")


def get_deploy_config(model_name, device="gpu"):
    if not TASK_CONFIG.get("serve", None):
        raise ValueError("No 'serve' section found in task config.")

    model_config = None
    for item in TASK_CONFIG.serve:
        if item.get("serve_id", None) == model_name:
            model_config = item
            break
    if model_config is None:
        raise ValueError(f"No {model_name} configuration found in task config: {TASK_CONFIG}")

    resources = model_config.get("resources", {})
    if not resources:
        raise ValueError("No 'resources' section found in task config.")

    resource_config = {}
    ray_actor_options = {}

    resource_set = {"num_gpus", "num_cpus"}
    for item in resource_set:
        if item in resources:
            ray_actor_options[item] = resources[item]
    if ray_actor_options:
        resource_config["ray_actor_options"] = ray_actor_options

    if "num_replicas" in resources:
        resource_config["num_replicas"] = resources["num_replicas"]
    else:
        resource_config["num_replicas"] = 1

    if "num_gpus" not in resource_config.get("ray_actor_options", {}) and device == "gpu":
        tensor_parallel_size = (
            model_config.engine_args.get("tensor_parallel_size", None)
            or model_config.get("engine_args_specific", {})
            .get("vllm", {})
            .get("tensor_parallel_size", None)
            or 1
        )
        pipeline_parallel_size = (
            model_config.engine_args.get("pipeline_parallel_size", None)
            or model_config.get("engine_args_specific", {})
            .get("vllm", {})
            .get("pipeline_parallel_size", None)
            or 1
        )
        ray_actor_options["num_gpus"] = tensor_parallel_size * pipeline_parallel_size
        resource_config["ray_actor_options"] = ray_actor_options

    resource_config["max_ongoing_requests"] = 1000
    return resource_config


def get_sample_args(request):
    # same as args of vllm.SamplingParams
    pre_args = {
        "n",
        "best_of",
        "presence_penalty",
        "frequency_penalty",
        "repetition_penalty",
        "temperature",
        "top_p",
        "top_k",
        "min_p",
        "seed",
        "stop",
        "stop_token_ids",
        "bad_words",
        "include_stop_str_in_output",
        "ignore_eos",
        "max_tokens",
        "min_tokens",
        "logprobs",
        "prompt_logprobs",
        "detokenize",
        "skip_special_tokens",
        "spaces_between_special_tokens",
        "logits_processors",
        "truncate_prompt_tokens",
        "guided_decoding",
        "logit_bias",
        "allowed_token_ids",
    }
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
        engine_args = AsyncEngineArgs(**get_engine_args("vllm_model"))
        self.engine = AsyncLLMEngine.from_engine_args(engine_args)

    def get_model_config(self):
        return self.engine.get_model_config()

    def generate(self, prompt, sampling_params, request_id):
        return self.engine.generate(prompt, sampling_params, request_id)


# refer to openai-type endpoints of vLLM
@serve.deployment(num_replicas="auto", max_ongoing_requests=1000)
@serve.ingress(app)
class LLMService:
    def __init__(self, llm_actor):
        self.llm_actor = llm_actor
        self.ready = False
        self.tokenizer = AutoTokenizer.from_pretrained(
            get_engine_args("vllm_model")["model"], trust_remote_code=True
        )

    @app.post("/v1/completions")
    async def generate_handler(self, request: CompletionRequest):
        logger.info(f"========== /v1/completions Receive request ========== ")
        if not self.ready:
            self.ready = check_health(SERVICE_NAME)
            if not self.ready:
                return JSONResponse(
                    status_code=503,
                    content={"message": "Service is not ready, please try again later."},
                )
        prompt_data = request.prompt
        prompt = TextPrompt(prompt=prompt_data)

        stream = request.stream
        request_id = "cmpl-" + random_uuid()
        sample_args = get_sample_args(request)
        sampling_params = SamplingParams(**sample_args)
        results_generator = self.llm_actor.generate.options(stream=True).remote(
            prompt, sampling_params, request_id
        )

        if stream:

            async def stream_results() -> AsyncGenerator[bytes, None]:
                num_choices = 1 if request.n is None else request.n
                previous_num_tokens = [0] * num_choices
                num_prompt_tokens = 0
                length = 0

                async for request_output in results_generator:
                    prompt = request_output.prompt
                    assert prompt is not None
                    for item in request_output.outputs:

                        i = item.index

                        content = item.text
                        current_text = content[length:]
                        length = len(content)
                        previous_num_tokens[i] = len(item.token_ids)

                        finish_reason = item.finish_reason
                        stop_reason = item.stop_reason

                        chunk = CompletionStreamResponse(
                            id=request_id,
                            created=int(time.time()),
                            model=request.model,
                            choices=[
                                CompletionResponseStreamChoice(
                                    index=i,
                                    text=current_text,
                                    logprobs=None,
                                    finish_reason=finish_reason,
                                    stop_reason=stop_reason,
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
            finish_reason = None
            stop_reason = None

            assert final_output is not None
            prompt = final_output.prompt
            assert prompt is not None

            for item in final_output.outputs:
                text_outputs += item.text
                completion_tokens += len(item.token_ids)
                finish_reason = item.finish_reason
                stop_reason = item.stop_reason

            ret = CompletionResponse(
                id=request_id,
                created=int(time.time()),
                model=request.model,
                choices=[
                    {
                        "text": text_outputs,
                        "index": 0,
                        "logprobs": None,
                        "finish_reason": finish_reason,
                        "stop_reason": stop_reason,
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
        logger.info(f"==========/v1/chat/completions Receive request ========== ")
        if not self.ready:
            self.ready = check_health(SERVICE_NAME)
            if not self.ready:
                return JSONResponse(
                    status_code=503,
                    content={"message": "Service is not ready, please try again later."},
                )
        user_message = request.messages[-1]["content"]
        mm_data = dict()

        try:
            resolved_content_format = resolve_chat_template_content_format(
                chat_template=None,
                tools=None,
                given_format="auto",
                tokenizer=self.tokenizer,
                trust_remote_code=True,
            )
            logger.debug(f"========== tokenizer ========== {self.tokenizer}")
            model_config = await self.llm_actor.get_model_config.remote()
            conversation, mm_data = parse_chat_messages(
                request.messages,
                model_config,
                self.tokenizer,
                content_format=resolved_content_format,
            )
            logger.debug(f"========== mm_data ========== {mm_data}")

            formatted_text = apply_hf_chat_template(
                self.tokenizer,
                conversation=conversation,
                chat_template=None,
                tools=None,
                trust_remote_code=True,
                tokenize=False,
                add_generation_prompt=True,
            )

        except Exception as e:
            logger.warning(f"Failed to apply chat template: {str(e)}")
            if isinstance(user_message, list):
                user_message = " ".join(
                    [item["text"] for item in user_message if item["type"] == "text"]
                )
                mm_data = {
                    "image": [
                        decode_base64_to_image(item["image_url"]["url"])
                        for item in request.messages[-1]["content"]
                        if item["type"] == "image_url"
                    ]
                }
            formatted_text = user_message

        logger.debug(f"========== formatted_text ========== {formatted_text}")
        prompt_data = formatted_text
        prompt = TextPrompt(prompt=prompt_data)
        if mm_data:
            prompt["multi_modal_data"] = mm_data

        stream = request.stream
        request_id = "cmpl-" + random_uuid()
        sample_args = get_sample_args(request)
        logger.debug(f"Request {request_id} sampling_params {sample_args}")
        sampling_params = SamplingParams(**sample_args)
        results_generator = self.llm_actor.generate.options(stream=True).remote(
            prompt, sampling_params, request_id
        )

        if stream:

            async def stream_results() -> AsyncGenerator[bytes, None]:
                num_choices = 1 if request.n is None else request.n
                previous_num_tokens = [0] * num_choices
                num_prompt_tokens = 0
                length = 0

                async for request_output in results_generator:
                    prompt = request_output.prompt
                    assert prompt is not None
                    for item in request_output.outputs:

                        i = item.index
                        content = item.text
                        current_text = content[length:]
                        length = len(content)
                        previous_num_tokens[i] = len(item.token_ids)

                        finish_reason = item.finish_reason
                        stop_reason = item.stop_reason

                        chunk = ChatCompletionStreamResponse(
                            id=request_id,
                            object="chat.completion.chunk",
                            created=int(time.time()),
                            model=request.model,
                            choices=[
                                {
                                    "index": i,
                                    "delta": {"role": "assistant", "content": current_text},
                                    "logprobs": None,
                                    "finish_reason": finish_reason,
                                    "stop_reason": stop_reason,
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

            logger.info(f"Return stream reponse for request {request_id} ")
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
            finish_reason = None
            stop_reason = None

            for item in final_output.outputs:
                text_outputs += item.text
                completion_tokens += len(item.token_ids)
                finish_reason = item.finish_reason
                stop_reason = item.stop_reason

            ret = ChatCompletionResponse(
                id=request_id,
                created=int(time.time()),
                model=request.model,
                choices=[
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": text_outputs},
                        "logprobs": None,
                        "finish_reason": finish_reason,
                        "stop_reason": stop_reason,
                    }
                ],
                usage={
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": prompt_tokens + completion_tokens,
                },
            )
            logger.info(f"Return reponse for request {request_id} ")
            return JSONResponse(content=ret.model_dump())


if __name__ == "__main__":
    serve.start(
        http_options={
            "host": "0.0.0.0",
            "port": TASK_CONFIG.experiment.get("deploy", {}).get("port", 8000),
        }
    )
    llm_actor = LLMActor.bind()
    serve.run(LLMService.bind(llm_actor), name=SERVICE_NAME, route_prefix="/", blocking=True)
