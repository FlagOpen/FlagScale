export PYTHONPATH=./flagscale:$PYTHONPATH
torchrun --nproc_per_node=8 -m pytest -q -x tests/unit_tests/launcher