import os, json, sys
import numpy as np

def test_result(test_reaults_path:str):

    host_path = test_reaults_path + "/logs/details/host_0_localhost"
    id_name      = os.listdir(host_path)[0]
    default_name = os.listdir(host_path + "/" + id_name)[0]
    attempt_name = os.listdir(host_path + "/" + id_name + "/" + default_name)[0]
    results_path = (os.listdir(host_path + "/" + id_name + "/" + default_name+ "/" + attempt_name))
    results_path.sort()
    result_path  = host_path + "/" + id_name + "/" + default_name+ "/" + attempt_name + "/" + results_path[-1] + "/stdout.log"

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

    gold_log_path = test_reaults_path + "/../gold_result/gold_result.json"

    with open(gold_log_path, 'r') as f:
        gold_result_json = json.load(f)
    
    print("\nresult checking")
    print("result: ", result_json)
    print(result_json)
    print("gold_result: ", gold_result_json)
    print("The results are basically equal: ", np.allclose(gold_result_json["lm loss:"]["values"], result_json["lm loss:"]["values"]))

    assert np.allclose(gold_result_json["lm loss:"]["values"], result_json["lm loss:"]["values"]), "result not close to gold result"
    

if __name__ == '__main__':
    test_reaults_path = sys.argv[1]
    