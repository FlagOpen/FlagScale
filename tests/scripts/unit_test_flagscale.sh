export PYTHONPATH=./flagscale:$PYTHONPATH

pytest -x tests/unit_tests/launcher/test_parse_hostfile.py