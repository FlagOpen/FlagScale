export PYTHONPATH=./megatron:$PYTHONPATH
export PYTHONPATH=./../../FlagScale/:$PYTHONPATH

cd megatron

torchrun --nproc_per_node=8 -m pytest -x tests/unit_tests/test_training.py \
                                            tests/unit_tests/test_parallel_state.py \
                                            tests/unit_tests/data/test_builder.py \
                                            tests/unit_tests/pipeline_parallel \
                                            tests/unit_tests/transformer \
                                            tests/unit_tests/models \
                                            tests/unit_tests/dist_checkpointing \
                                            tests/unit_tests/tensor_parallel \
                                            tests/unit_tests/dist_checkpointing/test_optimizer.py
                                            