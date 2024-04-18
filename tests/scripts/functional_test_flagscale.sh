# if [ "$2" = "stop" ]; then
#     python run.py --config-path tests/functional_tests/$1/conf --config-name config action=stop
#     exit 0
# fi

OUT_DIR=./tests/functional_tests/$1/test_result
if [ -d "$OUT_DIR" ]; then
    echo "$OUT_DIR exist."
    rm -r $OUT_DIR
    sleep 3s
fi

python run.py --config-path tests/functional_tests/$1/conf --config-name config action=test
