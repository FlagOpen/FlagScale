import hydra, os, subprocess
from omegaconf import DictConfig

@hydra.main(version_base=None, config_name="config")
def main(config : DictConfig) -> None:

    print("read test config from tests/conf/config.yaml")

    action = config["action"]
    
    if action == "stop":
        print("stoping all pytest")
        subprocess.call("tests/scripts/test_stop.sh", shell=True)
        print("stoping all functional_test_flagscale")
        functional_test_flagscale_list = config["functional_tests"]["functional_test_flagscale"]
        for test_name in functional_test_flagscale_list:
            subprocess.call("tests/scripts/functional_test_flagscale.sh " + test_name + " " + "stop", shell=True)
        return

    outputs_dir = config["outputs_dir"]
    if not os.path.exists(outputs_dir): 
        os.makedirs(outputs_dir)

    unit_tests_list = config["unit_tests"]
    for test_name in unit_tests_list:
        if unit_tests_list[test_name]:
            print("Running " + test_name + ", results are stored in " + outputs_dir + "/" + test_name + ".log")
            subprocess.call("tests/scripts/" + test_name + ".sh > " + outputs_dir + "/" + test_name + ".log", shell=True)

    functional_test_flagscale_list = config["functional_tests"]["functional_test_flagscale"]
    for test_name in functional_test_flagscale_list:
        base_test_name = "functional_test_flagscale"
        full_test_name = base_test_name + "_" + test_name + ""
        print("The results of " + full_test_name + " are stored in " + outputs_dir + "/" + full_test_name + ".log and tests/functional_tests/"+ test_name +"/test_result/logs/host_0_localhost.output")
        subprocess.call("tests/scripts/" + base_test_name + ".sh " + test_name + " > " + outputs_dir + "/" + full_test_name + ".log", shell=True)
    

if __name__ == "__main__":
    main()


