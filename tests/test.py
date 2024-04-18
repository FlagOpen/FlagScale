import hydra, os, subprocess, tempfile
from omegaconf import DictConfig


@hydra.main(version_base=None, config_name="config")
def main(config: DictConfig) -> None:

    print("Read test config from tests/conf/config.yaml")

    action = config["action"]

    if action == "stop":
        print("stoping all pytest")
        subprocess.call("tests/scripts/unit_test_stop.sh", shell=True)
        # print("stoping all functional_test_flagscale")
        # functional_test_flagscale_list = config["functional_tests"][
        #     "functional_test_flagscale"
        # ]
        # for test_name in functional_test_flagscale_list:
        #     subprocess.call(
        #         "tests/scripts/functional_test_flagscale.sh "
        #         + test_name
        #         + " "
        #         + "stop",
        #         shell=True,
        #     )
        return

    unit_tests_list = config["unit_tests"]
    for test_name in unit_tests_list:
        if unit_tests_list[test_name]:
            print(
                "Ready to start running tests: "
                + test_name
            )
            input("Press any key to start ...")
            subprocess.call(
                "tests/scripts/"
                + test_name
                + ".sh", 
                shell=True,
            )

    functional_test_flagscale_list = config["functional_tests"][
        "functional_test_flagscale"
    ]
    for test_name in functional_test_flagscale_list:
        base_test_name = "functional_test_flagscale"
        full_test_name = base_test_name + "_" + test_name + ""
        print(
                "Ready to start running tests: "
                + full_test_name
            )
        input("Press any key to start ...")
        subprocess.call(
            "tests/scripts/"
            + base_test_name
            + ".sh "
            + test_name, 
            shell=True,
        )


if __name__ == "__main__":
    main()
