export PYTHONPATH=./megatron:$PYTHONPATH

pytest -x megatron/tests/unit_tests/test_basic.py \
       megatron/tests/unit_tests/test_imports.py \
       megatron/tests/unit_tests/test_utils.py \
       megatron/tests/unit_tests/data/test_mock_gpt_dataset.py  \
       megatron/tests/unit_tests/data/test_multimodal_dataset.py \
       megatron/tests/unit_tests/data/test_preprocess_mmdata.py \
       megatron/tests/unit_tests/data/test_preprocess_data.py \
       megatron/tests/unit_tests/fusions
       