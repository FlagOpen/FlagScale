python run.py --config-path tests/functional_tests/aquila/conf --config-name config action=test
pytest -s tests/functional_tests/test_result.py --test_reaults_path=./tests/functional_tests/aquila/test_result

python run.py --config-path tests/functional_tests/mixtral/conf --config-name config action=test
pytest -s tests/functional_tests/test_result.py --test_reaults_path=./tests/functional_tests/mixtral/test_result