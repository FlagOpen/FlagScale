if [ -z "$1" ]
then
  code_id=0
else
  code_id=$1
fi


torchrun --nproc_per_node=8 -m pytest --import-mode=importlib --cov-append --cov-report=html:/workspace/report/$code_id/cov-report-flagscale --cov=flagscale -q -x tests/unit_tests/launcher