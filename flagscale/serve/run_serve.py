import sys

from flagscale.serve.arguments import parse_config
from flagscale.serve.core.dag import Builder


def main():
    config = parse_config()
    project_path = config["root_path"]
    sys.path.append(project_path)
    builder = Builder(config)
    tasks = builder.build_task()
    res = builder.run_task(tasks, input_data="Introduce Bruce Lee")
    print("**************** res ****************", res)


if __name__ == "__main__":
    main()
