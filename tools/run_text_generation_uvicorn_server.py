"""Sample Generate GPT"""
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),
                                             os.path.pardir)))
import socket
from megatron import get_args
from megatron import print_rank_0
from megatron.core import mpu
from megatron.checkpointing import load_checkpoint
from megatron.initialize import initialize_megatron
from megatron.model import GPTModel
from megatron.training import get_model
from megatron.arguments import core_transformer_config_from_args
import os
import uvicorn, json, datetime
import json 
from asgiref.sync import sync_to_async
from megatron.text_generation import generate_and_post_process
import time 
import torch 
import threading
import random
import sys 
from fastapi.responses import StreamingResponse

from tools.stream_conversation.conversation_convo_v2 import covert_prompt_to_input_ids_with_history

def get_tokenizer():
    from megatron.tokenizer.tokenizer import _AquilaTokenizer
    vocab_file = "examples/aquila/tokenizer/vocab.json"
    merge_file = "examples/aquila/tokenizer/merges.txt"
    special = "examples/aquila/tokenizer/special_tokens.txt"
    tokenizer = _AquilaTokenizer(vocab_file, merge_file, special)

    return tokenizer

tokenizer = get_tokenizer()
lock = threading.Lock()

vocab = tokenizer.vocab

id2word = {v:k for k, v in vocab.items()}

from tools.conversation import get_prompt
def make_sft_prompts(prompts):
    new_prompts = []
    for p in prompts:
        p = get_prompt(p)
        new_prompts.append(p)
    return new_prompts

def predict(model, prompts, seed, max_length, topk, topp, t, sft):
    
    if not isinstance(prompts, list):
        prompts = [prompts,]

    ## boolq needs to add cls token at first.
    if max_length == 0:
        prompts_addcls = []
        for p in prompts:
            prompts_addcls.append("[CLS]" + p)
        prompts = prompts_addcls

    completions = [{} for _ in range(len(prompts))]

    if sft:
        prompts = make_sft_prompts(prompts)

    input_length = max([len(tokenizer.tokenize(prompts[j])) for j in range(len(prompts))])

    with lock:  # Need to get lock to keep multiple threads from hitting code
        try:
            choice = torch.cuda.LongTensor([1])
            torch.distributed.broadcast(choice, 0)
            response, _, response_logprobs, _ = \
                            generate_and_post_process(
                            model,
                            prompts=prompts,
                            tokens_to_generate=max_length,
                            return_output_log_probs=True,
                            top_k_sampling=topk,
                            top_p_sampling=topp,
                            top_p_decay=0.0,
                            top_p_bound=0.0,
                            temperature=t,
                            add_BOS=False,
                            use_eod_token_for_early_termination=True,
                            stop_on_double_eol=False,
                            stop_on_eol=False,
                            prevent_newline_after_colon=False,
                            random_seed=seed,
                        )
            torch.cuda.empty_cache()

            for i in range(len(prompts)):
                prompt = prompts[i]

                truncation_length = len(prompt)
                response_i = response[i]

                response_logprobs_i = response_logprobs[i]

                if max_length != 0:
                    response_i: str = response_i[truncation_length:]

                    if response_i.endswith("</s>"):
                        response_logprobs_i = response_logprobs_i[:-1]
                        response_i = response_i.replace("</s>", "")
                    # else :
                    ids = tokenizer.tokenize(response_i)
                    response_logprobs_i = response_logprobs_i[-len(ids):]
                    if len(ids) == 0:
                        response_logprobs_i = []

                else :
                    ids = tokenizer.tokenize(response_i)
                    response_logprobs_i = [0] + response_logprobs_i
                convert_tokens = []
                for t in ids:
                    if t == 100006:
                        convert_tokens.append("[CLS]")
                    else :
                        convert_tokens.append(id2word.get(t, "[unk]"))

                print(len(response_logprobs_i), len(convert_tokens))
                
                completions[i]['text'] = response_i
                completions[i]['tokens'] = convert_tokens
                completions[i]['logprobs'] = response_logprobs_i
                tld = [{k: v} for k, v in zip(convert_tokens, response_logprobs_i)]
                completions[i]['top_logprobs_dicts'] = tld
        except Exception as e:
            print(f"Exception is {e}")
            print(f"occur a bug, please pay attention to it. return none result.")
            for i in range(len(prompts)):
                completions[i]['text'] = prompts[i]
                ids = tokenizer.tokenize(prompts[i])
                response_logprobs_i = [0.0] * len(ids)
                convert_tokens = []
                for t in ids:
                    if t == 100006:
                        convert_tokens.append("[CLS]")
                    else :
                        convert_tokens.append(id2word.get(t, "[unk]"))

                completions[i]['tokens'] = convert_tokens
                completions[i]['logprobs'] = response_logprobs_i
                tld = [{k: v} for k, v in zip(convert_tokens, response_logprobs_i)]
                completions[i]['top_logprobs_dicts'] = tld
                torch.cuda.empty_cache()

        return completions, input_length 

def killport(port):
    '''root authority is required'''
    try:
        command="kill -9 $(netstat -nlp | grep :"+str(port)+" | awk '{print $7}' | awk -F'/' '{{ print $1 }}')"
        os.system(command)
        print(f"killing {port} is succeed.")
    except:
        pass 
        print(f"{port} no need to kill")
    time.sleep(2)

previous_data = ""
flag_rc = False
flag_hm = False
unchanged_count = 0

def stop_signal(model_info, request_model_name, engine, prompt, server_port):
    print(f"model info is {model_info}, request model name is {request_model_name}, engine is {engine}, prompt is {prompt} ")
    global flag_rc, flag_hm, previous_data, unchanged_count

    if model_info == request_model_name and engine == "####Subjective_Inference_Ending####":
        flag_hm = True
    if  model_info == request_model_name and engine == "####eval_end####":
        flag_rc = True
    
    if flag_hm and flag_rc:
        if prompt == previous_data:
            unchanged_count += 1
        else:
            unchanged_count = 0
        print(f"Input unchanged for {unchanged_count} consecutive times.")
        if unchanged_count >= 120:
            print("Ternimating Singal Confirmed.")
            print("Input unchanged for 120 consecutive times. Terminating the server.")
            killport(server_port)
    previous_data = prompt

class UvicornServer:
    def __init__(self, model, server_port, model_info="aquila-34b") -> None:
        self.model = model
        self.server_port = server_port
        self.model_info = model_info

    def init_flask(self):
        from fastapi import FastAPI, Request

        app = FastAPI()

        @app.post("/batch_func")
        async def get_generate_h(request: Request):
            json_post_raw = await request.json()
            config = json.loads(json_post_raw)

            print("request come in")
            contexts = config["prompt"]
            topk= config.get("top_k_per_token", 20)
            topp = config.get("top_p", 0.9)
            t = config.get("temperature", 0.9)
            seed = config.get("seed", 123)
            sft = config.get("sft", False)

            ## determine if we need to stop the server
            stop_signal(self.model_info, request_model_name=config.get("model_name", ""), 
                        engine=config.get("engine", ""), prompt=contexts,
                        server_port=self.server_port)
            
            print(f"model info is {self.model_info}")
            s = time.time()
            res, input_length = await sync_to_async(predict)(self.model, contexts, seed, max_length=config['max_new_tokens'], topk=topk, topp=topp, t= t, sft=sft)
            e = time.time()
            print(f"spend time is {e - s}")
          
            result = {
                "completions": res,
                "input_length":input_length,
                "model_info":self.model_info}

            return result
    
        @app.post("/stream_func")
        async def get_generate_h(request: Request):
            json_post_raw = await request.json()
            config = json.loads(json_post_raw)

            print("request come in")
            prompts = config["prompt"]
            topk= config.get("top_k_per_token", 20)
            topp = config.get("top_p", 0.9)
            t = config.get("temperature", 0.9)
            seed = config.get("seed", 123)
            sft = config.get("sft", False)
            max_length=config['max_new_tokens']
            gene_time = config.get("time", 15)

            history = config.get("history", [])

            if seed == 0:
                seed = random.randint(0, 429496729)

            print(f"model info is {self.model_info}")

            assert type(prompts) is str
            if sft:
                prompts = covert_prompt_to_input_ids_with_history(prompts, history, tokenizer, 2048)
            
            prompts = [prompts,]

            with lock:  # Need to get lock to keep multiple threads from hitting code
                choice = torch.cuda.LongTensor([1])
                torch.distributed.broadcast(choice, 0)
                fun = generate_and_post_process(
                                            model,
                                            prompts=prompts,
                                            tokens_to_generate=max_length,
                                            return_output_log_probs=True,
                                            top_k_sampling=topk,
                                            top_p_sampling=topp,
                                            top_p_decay=0.0,
                                            top_p_bound=0.0,
                                            temperature=t,
                                            add_BOS=False,
                                            use_eod_token_for_early_termination=True,
                                            stop_on_double_eol=False,
                                            stop_on_eol=False,
                                            prevent_newline_after_colon=False,
                                            random_seed=seed,
                                            stream=True)
                torch.cuda.empty_cache()

                def trans():
                    start_time = time.time()
                    while True:
                        try:
                            yield next(fun)
                        except Exception as e:
                            print(f"e is {e}")
                            break

                        if time.time() - start_time > gene_time:
                            print("time up")
                            break

                return StreamingResponse(trans(), media_type="text/plain")
        

        return app

    def run(self):
        app = self.init_flask()
        uvicorn.run(app, host='0.0.0.0', port=self.server_port, workers=1)

def model_provider(pre_process=True, post_process=True):
    """Build the model."""

    config = core_transformer_config_from_args(get_args())

    print_rank_0('building GPT model ...')
    model = GPTModel(config, num_tokentypes=0, parallel_output=False, pre_process=pre_process, post_process=post_process)

    return model

def add_text_generate_args(parser):
    group = parser.add_argument_group(title='text generation')

    group.add_argument("--temperature", type=float, default=1.0,
                       help='Sampling temperature.')
    group.add_argument("--top_p", type=float, default=0.0,
                       help='Top p sampling.')
    group.add_argument("--top_k", type=int, default=0,
                       help='Top k sampling.')
    group.add_argument("--out-seq-length", type=int, default=1024,
                       help='Size of the output generated text.')
    return parser


if __name__ == "__main__":
    initialize_megatron(extra_args_provider=add_text_generate_args,
                        args_defaults={'tokenizer_type': 'GPT2BPETokenizer',
                                       'no_load_rng': True,
                                       'no_load_optim': True})

    args = get_args()
    print(f"args is {args}")
    if args.num_layers_per_virtual_pipeline_stage is not None:
        print("Interleaved pipeline schedule is not yet supported for text generation.")
        exit()
    print_rank_0("WARNING: Forcing exit_on_missing_checkpoint to True for text "
                 "generation.")
    args.exit_on_missing_checkpoint = True
    # Set up model and load checkpoint
    model = get_model(model_provider, wrap_with_ddp=False)

    print(f"get model successfully....")

    if args.load is not None:
        _ = load_checkpoint(model, None, None)
    print("loading model successfully....")

    assert len(model) == 1, "Above condition should have caught this"
    model = model[0]

    if mpu.is_pipeline_first_stage() and mpu.get_tensor_model_parallel_rank() == 0:
        server_ins = UvicornServer(model, server_port=int(args.server_port), model_info=args.model_info)
        server_ins.run()

    while True:
        choice = torch.cuda.LongTensor(1)
        torch.distributed.broadcast(choice, 0)
        if choice[0].item() == 1:
            try:
                generate_and_post_process(model)
                torch.cuda.empty_cache()

            except ValueError as ve:
                pass
