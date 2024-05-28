export PYTHONPATH=./megatron:$PYTHONPATH
export PYTHONPATH=./../../FlagScale/:$PYTHONPATH
cd megatron
# torchrun --nproc_per_node=8 -m pytest -q -x tests/unit_tests/data
torchrun --nproc_per_node=8 -m pytest -q -x tests/unit_tests/dist_checkpointing
torchrun --nproc_per_node=8 -m pytest -q -x tests/unit_tests/fusions
torchrun --nproc_per_node=8 -m pytest -q -x tests/unit_tests/models
torchrun --nproc_per_node=8 -m pytest -q -x tests/unit_tests/pipeline_parallel
torchrun --nproc_per_node=8 -m pytest -q -x tests/unit_tests/tensor_parallel
torchrun --nproc_per_node=8 -m pytest -q -x tests/unit_tests/transformer
torchrun --nproc_per_node=8 -m pytest -q -x tests/unit_tests/*.py