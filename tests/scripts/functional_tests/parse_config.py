import argparse

import yaml


def parse_config(config_file, type, mission):
    with open(config_file, "r") as file:
        config = yaml.safe_load(file)

    # Check if type exists in the configuration
    if type not in config:
        raise ValueError(f"Test type '{type}' not found in configuration file.")

    # Check if mission exists within the specified type
    if mission not in config[type]:
        raise ValueError(
            f"Test mission '{mission}' not found under test type '{type}' in configuration file."
        )

    return config[type][mission]


def main():
    parser = argparse.ArgumentParser(description="Parse functional test configuration.")
    parser.add_argument(
        "--config", required=True, help="Path to the YAML configuration file."
    )
    parser.add_argument(
        "--type", required=True, help="Test type to run (e.g., 'train' or 'inference')."
    )
    parser.add_argument(
        "--mission",
        required=True,
        help="Test mission to run (e.g., 'aquila' or 'mixtral').",
    )

    args = parser.parse_args()

    try:
        result = parse_config(args.config, args.type, args.mission)
        print(result)
    except ValueError as e:
        print(str(e))
        exit(1)


if __name__ == "__main__":
    main()
