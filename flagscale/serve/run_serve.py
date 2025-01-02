import sys

from flagscale.serve.arguments import parse_config
from flagscale.serve.core.dag import Builder


def main():
    config = parse_config()
    project_path = config["root_path"]
    sys.path.append(project_path)
    builder = Builder(config)
    builder.build_task()
    if config["deploy"].get("router"):
        builder.run_router_task()
    else:
        result = builder.run_task()


if __name__ == "__main__":
    main()
