#!/bin/bash

if [ -z "$1" ]
then
  code_id=0
else
  code_id=$1
fi

cd megatron

export PYTHONPATH=..:$PYTHONPATH

run_pytest() {
  local test_path=$1
  echo "Running $test_path"
  torchrun --nproc_per_node=8 -m pytest --import-mode=importlib --cov-append --cov-report=html:/workspace/report/$code_id/cov-report-megatron --cov=megatron/core -q -x -p no:warnings $test_path
  if [ $? -ne 0 ]; then
    echo "Pytest failed for $test_path"
    exit 1
  fi
}

run_pytest tests/unit_tests/data
run_pytest tests/unit_tests/dist_checkpointing/models/test_bert_model.py
run_pytest tests/unit_tests/dist_checkpointing/models/test_gpt_model.py
run_pytest tests/unit_tests/dist_checkpointing/models/test_grouped_mlp.py
run_pytest tests/unit_tests/dist_checkpointing/models/test_mlp_glu.py
run_pytest tests/unit_tests/dist_checkpointing/models/test_retro_model.py
run_pytest tests/unit_tests/dist_checkpointing/models/test_sequential_mlp.py
run_pytest tests/unit_tests/dist_checkpointing/models/test_t5_model.py
run_pytest tests/unit_tests/dist_checkpointing/test_*.py

run_pytest tests/unit_tests/distributed
run_pytest tests/unit_tests/fusions
run_pytest tests/unit_tests/inference
run_pytest tests/unit_tests/models
run_pytest tests/unit_tests/pipeline_parallel
run_pytest tests/unit_tests/tensor_parallel

run_pytest tests/unit_tests/transformer/moe/test_a2a_token_dispatcher.py
run_pytest tests/unit_tests/transformer/moe/test_aux_loss.py
run_pytest tests/unit_tests/transformer/moe/test_grouped_mlp.py
run_pytest tests/unit_tests/transformer/moe/test_routers.py
run_pytest tests/unit_tests/transformer/moe/test_sequential_mlp.py
run_pytest tests/unit_tests/transformer/moe/test_token_dispatcher.py

run_pytest tests/unit_tests/transformer/test_attention.py
run_pytest tests/unit_tests/transformer/test_mlp.py
run_pytest tests/unit_tests/transformer/test_module.py
run_pytest tests/unit_tests/transformer/test_retro_attention.py
run_pytest tests/unit_tests/transformer/test_spec_customization.py
run_pytest tests/unit_tests/transformer/test_transformer_block.py
run_pytest tests/unit_tests/transformer/test_transformer_layer.py

run_pytest tests/unit_tests/test_basic.py
run_pytest tests/unit_tests/test_imports.py
run_pytest tests/unit_tests/test_local_multi_tensor_fns.py
run_pytest tests/unit_tests/test_num_microbatches_calculator.py
run_pytest tests/unit_tests/test_optimizer.py

run_pytest tests/unit_tests/test_parallel_state.py
run_pytest tests/unit_tests/test_training.py
run_pytest tests/unit_tests/test_utils.py 