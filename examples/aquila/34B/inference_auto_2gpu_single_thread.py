
import argparse
import os 

parser = argparse.ArgumentParser(
        prog='34b-base-server',
    )
parser.add_argument('--server-port', required=True, type=int)
parser.add_argument('--master-process', required=True, type=int)
parser.add_argument('--device', default='0', type=str)
parser.add_argument('--iteration', required=False, type=int, default=-1)
parser.add_argument('--checkpoint-path', required=True, type=str)
parser.add_argument('--model-info', required=True, type=str)

args = parser.parse_args()

model_iteration = args.iteration
server_port = args.server_port
master_process = args.master_process
device_number = args.device
checkpoint_path = args.checkpoint_path
model_info = args.model_info

if model_iteration != -1:
    with open(os.path.join(checkpoint_path, "latest_checkpointed_iteration.txt"), "w") as f:
        f.write(str(model_iteration))

sh_content = """#!/bin/bash

DISTRIBUTED_ARGS="--nproc_per_node 2 \
                  --nnodes 1 \
                  --node_rank 0 \
                  --master_addr localhost \
                  --master_port {master_port}"

CHECKPOINT={checkpoint_path}
VOCAB_FILE=../examples/aquila/tokenizer/vocab.json
MERGE_FILE=../examples/aquila/tokenizer/merges.txt
SPECIAL_TOKENS_FILE=../examples/aquila/tokenizer/special_tokens.txt

export CUDA_DEVICE_MAX_CONNECTIONS=1

CUDA_VISIBLE_DEVICES={device_number} torchrun $DISTRIBUTED_ARGS tools/run_text_generation_uvicorn_server_single_thread.py \
       --server-port {server_port} \
       --model-info {model_info} \
       --make-vocab-size-divisible-by 64 \
       --use-flash-attn \
       --normalization RMSNorm \
       --tensor-model-parallel-size 2  \
       --pipeline-model-parallel-size 1  \
       --num-layers 60  \
       --hidden-size 6144  \
       --hidden-dim-multiplier 1.3 \
       --load ${CHECKPOINT}  \
       --disable-bias-linear \
       --norm-epsilon 1e-5 \
       --norm-init-weight 0.3 \
       --num-attention-heads 48  \
       --group-query-attention \
       --num-query-groups 8 \
       --max-position-embeddings 4096  \
       --use-rotary-position-embeddings \
       --no-position-embedding \
       --swiglu \
       --multiple-of 4096 \
       --untie-embeddings-and-output-weights \
       --tokenizer-type AquilaTokenizer  \
       --bf16  \
       --micro-batch-size 1  \
       --seq-length 4096  \
       --out-seq-length 3000  \
       --temperature 1.0  \
       --vocab-file $VOCAB_FILE  \
       --merge-file $MERGE_FILE  \
       --special-tokens-file $SPECIAL_TOKENS_FILE  \
       --top_p 0.9  \
       --seed 42
"""

sh_dir = "./../examples/aquila/34B/server"
os.makedirs(sh_dir, exist_ok=True)

sh_filename = os.path.join(sh_dir, f"{model_info}.sh")

print(f"server filename is {sh_filename}")

with open(sh_filename, "w") as f:
    sh_content = sh_content.replace("{master_port}", str(master_process))
    sh_content = sh_content.replace("{checkpoint_path}", checkpoint_path)
    sh_content = sh_content.replace("{device_number}", "'" + device_number + "'")
    sh_content = sh_content.replace("{server_port}", str(server_port))
    sh_content = sh_content.replace("{model_info}", model_info)
    
    f.write((sh_content))


import subprocess
import signal

run_cmd = f"sh {sh_filename}"

p = subprocess.Popen(run_cmd, shell=True, preexec_fn=os.setsid)
def signal_handler(signal, frame):
    os.killpg(os.getpgid(p.pid), 9)
signal.signal(signal.SIGINT, signal_handler)
p.wait()
print ('finish')
