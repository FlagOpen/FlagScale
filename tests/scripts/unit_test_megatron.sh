export PYTHONPATH=./megatron:$PYTHONPATH
export PYTHONPATH=./../../FlagScale/:$PYTHONPATH

cd megatron

# passed 
torchrun --nproc_per_node=8 -m pytest -q -x tests/unit_tests/data/test_preprocess_data.py
torchrun --nproc_per_node=8 -m pytest -q -x tests/unit_tests/data/test_preprocess_mmdata.py

torchrun --nproc_per_node=8 -m pytest -q -x tests/unit_tests/dist_checkpointing/models/test_t5_model.py
torchrun --nproc_per_node=8 -m pytest -q -x tests/unit_tests/dist_checkpointing/models/test_mlp_glu.py
torchrun --nproc_per_node=8 -m pytest -q -x tests/unit_tests/dist_checkpointing/test_async_save.py
torchrun --nproc_per_node=8 -m pytest -q -x tests/unit_tests/dist_checkpointing/test_mapping.py

torchrun --nproc_per_node=8 -m pytest -q -x tests/unit_tests/models/test_bert_model.py
torchrun --nproc_per_node=8 -m pytest -q -x tests/unit_tests/models/test_gpt_model.py
torchrun --nproc_per_node=8 -m pytest -q -x tests/unit_tests/models/test_multimodal_projector.py
torchrun --nproc_per_node=8 -m pytest -q -x tests/unit_tests/models/test_clip_vit_model.py
torchrun --nproc_per_node=8 -m pytest -q -x tests/unit_tests/models/test_llava_model.py
torchrun --nproc_per_node=8 -m pytest -q -x tests/unit_tests/models/test_t5_model.py

torchrun --nproc_per_node=8 -m pytest -q -x tests/unit_tests/pipeline_parallel
                   
torchrun --nproc_per_node=8 -m pytest -q -x tests/unit_tests/tensor_parallel/test_cross_entropy.py
torchrun --nproc_per_node=8 -m pytest -q -x tests/unit_tests/tensor_parallel/test_initialization.py
torchrun --nproc_per_node=8 -m pytest -q -x tests/unit_tests/tensor_parallel/test_mappings.py
torchrun --nproc_per_node=8 -m pytest -q -x tests/unit_tests/tensor_parallel/test_tensor_parallel_utils.py
torchrun --nproc_per_node=8 -m pytest -q -x tests/unit_tests/tensor_parallel/test_data.py
torchrun --nproc_per_node=8 -m pytest -q -x tests/unit_tests/tensor_parallel/test_layers.py
torchrun --nproc_per_node=8 -m pytest -q -x tests/unit_tests/tensor_parallel/test_random.py

torchrun --nproc_per_node=8 -m pytest -q -x tests/unit_tests/transformer/moe/test_grouped_mlp.py
torchrun --nproc_per_node=8 -m pytest -q -x tests/unit_tests/transformer/moe/test_routers.py
torchrun --nproc_per_node=8 -m pytest -q -x tests/unit_tests/transformer/moe/test_sequential_mlp.py
torchrun --nproc_per_node=8 -m pytest -q -x tests/unit_tests/transformer/moe/test_token_dispatcher.py
     
torchrun --nproc_per_node=8 -m pytest -q -x tests/unit_tests/transformer/test_transformer_block.py
torchrun --nproc_per_node=8 -m pytest -q -x tests/unit_tests/transformer/test_attention.py
torchrun --nproc_per_node=8 -m pytest -q -x tests/unit_tests/transformer/test_module.py
torchrun --nproc_per_node=8 -m pytest -q -x tests/unit_tests/transformer/test_spec_customization.py
torchrun --nproc_per_node=8 -m pytest -q -x tests/unit_tests/transformer/test_transformer_layer.py

torchrun --nproc_per_node=8 -m pytest -q -x tests/unit_tests/test_basic.py
torchrun --nproc_per_node=8 -m pytest -q -x tests/unit_tests/test_imports.py
torchrun --nproc_per_node=8 -m pytest -q -x tests/unit_tests/test_optimizer.py
torchrun --nproc_per_node=8 -m pytest -q -x tests/unit_tests/test_training.py
torchrun --nproc_per_node=8 -m pytest -q -x tests/unit_tests/test_utils.py
torchrun --nproc_per_node=8 -m pytest -q -x tests/unit_tests/test_parallel_state.py


# unpassed
# torchrun --nproc_per_node=8 -m pytest -q -x tests/unit_tests/data/test_builder.py
# torchrun --nproc_per_node=8 -m pytest -q -x tests/unit_tests/data/test_gpt_dataset.py
# torchrun --nproc_per_node=8 -m pytest -q -x tests/unit_tests/data/test_multimodal_dataset.py

# torchrun --nproc_per_node=8 -m pytest -q -x tests/unit_tests/dist_checkpointing/models/test_gpt_model.py
# torchrun --nproc_per_node=8 -m pytest -q -x tests/unit_tests/dist_checkpointing/models/test_retro_model.py
# torchrun --nproc_per_node=8 -m pytest -q -x tests/unit_tests/dist_checkpointing/models/test_bert_model.py
# torchrun --nproc_per_node=8 -m pytest -q -x tests/unit_tests/dist_checkpointing/models/test_sequential_mlp.py
# torchrun --nproc_per_node=8 -m pytest -q -x tests/unit_tests/dist_checkpointing/test_fully_parallel.py
# torchrun --nproc_per_node=8 -m pytest -q -x tests/unit_tests/dist_checkpointing/test_optimizer.py

# torchrun --nproc_per_node=8 -m pytest -q -x tests/unit_tests/fusions

# torchrun --nproc_per_node=8 -m pytest -q -x tests/unit_tests/models/test_base_embedding.py

# torchrun --nproc_per_node=8 -m pytest -q -x tests/unit_tests/transformer/moe/test_a2a_token_dispatcher.py
# torchrun --nproc_per_node=8 -m pytest -q -x tests/unit_tests/transformer/test_mlp.py
# torchrun --nproc_per_node=8 -m pytest -q -x tests/unit_tests/transformer/test_retro_attention.py