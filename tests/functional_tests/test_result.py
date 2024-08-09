import os, json, sys
import numpy as np

def find_directory(start_path, target_dir_name):
    for root, dirs, files in os.walk(start_path):
        if target_dir_name in dirs:
            return os.path.join(root, target_dir_name)
    return None

def test_train_result(test_path:str):

    test_train_reault_path = test_path + "/train_result_test"
    start_path = test_train_reault_path + "/logs/details/host_0_localhost"
    attempt_path = find_directory(start_path, "attempt_0")
    results_path = (os.listdir(attempt_path))
    results_path.sort()
    result_path  = attempt_path + "/" + results_path[-1] + "/stdout.log"

    with open(result_path, 'r') as file:
        lines = file.readlines()

    result_json = {}
    result_json["lm loss:"] = {"values":[]}

    for line in lines:
        if " iteration" in line:
            line_split = line.strip().split("|")
            for key_value in line_split:
                if key_value[0:9] == " lm loss:":
                    result_json["lm loss:"]["values"].append(float(key_value.split(':')[1]))

    gold_log_path = test_path + "/train_result_gold/loss.json"

    with open(gold_log_path, 'r') as f:
        gold_result_json = json.load(f)
    
    print("\nTrain result checking")
    print("Train loss: ", result_json)
    print("Gold train loss: ", gold_result_json)
    print("The results are basically equal: ", np.allclose(gold_result_json["lm loss:"]["values"], result_json["lm loss:"]["values"]))

    assert np.allclose(gold_result_json["lm loss:"]["values"], result_json["lm loss:"]["values"]), "result not close to gold result"
    

def test_analyze_result(test_path:str):

    test_analyze_reault_path = test_path + "/analyze_result_test"
    gold_analyze_reault_path = test_path + "/analyze_result_gold"

    with open(test_analyze_reault_path + "/parallelism_to_groups.json", 'r') as f:
        test_parallelism_to_groups = json.load(f)
    with open(gold_analyze_reault_path + "/parallelism_to_groups.json", 'r') as f:
        gold_parallelism_to_groups = json.load(f)
    
    print("\nAnalyze result checking")
    print("Analyze result: ", test_parallelism_to_groups)
    print("Gold Analyze result: ", gold_parallelism_to_groups)
    print("The results are equal: ", test_parallelism_to_groups == gold_parallelism_to_groups)

    assert test_parallelism_to_groups == gold_parallelism_to_groups, "result not equal to gold result"

    with open(test_analyze_reault_path + "/rank_to_parallelism_to_group_id.json", 'r') as f:
        test_rank_to_parallelism_to_group_id = json.load(f)
    with open(gold_analyze_reault_path + "/rank_to_parallelism_to_group_id.json", 'r') as f:
        gold_rank_to_parallelism_to_group_id = json.load(f)
    
    print("\nAnalyze result checking")
    print("Analyze result: ", test_rank_to_parallelism_to_group_id)
    print("Gold Analyze result: ", gold_rank_to_parallelism_to_group_id)
    print("The results are equal: ", test_rank_to_parallelism_to_group_id == gold_rank_to_parallelism_to_group_id)

    assert test_rank_to_parallelism_to_group_id == gold_rank_to_parallelism_to_group_id, "result not equal to gold result"    

if __name__ == '__main__':
    test_path = sys.argv[1]