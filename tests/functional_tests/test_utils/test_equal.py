import json
import os

import numpy as np
import pytest


def find_directory(start_path, target_dir_name):
    for root, dirs, files in os.walk(start_path):
        if target_dir_name in dirs:
            return os.path.join(root, target_dir_name)
    return None


@pytest.mark.usefixtures("test_path", "test_type", "test_model", "test_case")
def test_equal(test_path, test_type, test_model, test_case):
    # Construct the test_result_path using the provided fixtures
    test_result_path = os.path.join(
        test_path, test_type, test_model, "results_test", test_case
    )
    start_path = os.path.join(test_result_path, "logs/details/host_0_localhost")

    attempt_path = find_directory(start_path, "attempt_0")
    assert (
        attempt_path is not None
    ), f"Failed to find 'attempt_0' directory in {start_path}"

    results_path = os.listdir(attempt_path)
    results_path.sort()
    result_path = os.path.join(attempt_path, results_path[-1], "stdout.log")

    print("result_path:", result_path)

    assert os.path.exists(result_path), f"Failed to find 'stdout.log' at {result_path}"

    with open(result_path, "r") as file:
        lines = file.readlines()

    result_json = {}
    result_json["lm loss:"] = {"values": []}

    for line in lines:
        if " iteration" in line:
            line_split = line.strip().split("|")
            for key_value in line_split:
                if key_value.startswith(" lm loss:"):
                    result_json["lm loss:"]["values"].append(
                        float(key_value.split(":")[1])
                    )

    gold_value_path = os.path.join(
        test_path, test_type, test_model, "results_gold", test_case + ".json"
    )
    assert os.path.exists(
        gold_value_path
    ), f"Failed to find gold result JSON at {gold_value_path}"

    with open(gold_value_path, "r") as f:
        gold_result_json = json.load(f)

    print("\nResult checking")
    print("Result: ", result_json)
    print("Gold Result: ", gold_result_json)
    print(
        "The results are basically equal: ",
        np.allclose(
            gold_result_json["lm loss:"]["values"], result_json["lm loss:"]["values"]
        ),
    )

    assert np.allclose(
        gold_result_json["lm loss:"]["values"], result_json["lm loss:"]["values"]
    ), "Result not close to gold result"
