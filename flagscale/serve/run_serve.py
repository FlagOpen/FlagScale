from flagscale.serve.arguments import parse_config
from flagscale.serve.engine import ServeEngine


def main():
    config = parse_config()
    engine = ServeEngine(config)
    engine.build_task()
    engine.run_router_task()


if __name__ == "__main__":
    main()
