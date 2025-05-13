# It creates a mapping conf of argument names to placeholder values and saves
# the configuration to a file named "template_conf.yaml".
import yaml
from vllm.entrypoints.openai.cli_args import create_parser_for_docs


def gen_template_conf():
    """
    This function generates a template configuration file for the VLLM serve command-line arguments.
    It extracts the argument names from the VLLM parser and creates a mapping configuration.
    The configuration is saved to a file named "template_conf.yaml".
    """
    pass
    serve_args = []
    parser = create_parser_for_docs()

    for i in parser._actions:
        for j in i.option_strings:
            if "--" in j:
                j = j.replace("--", "")
                j = j.replace("-", "_")
                serve_args.append(j)

    print("len of vllm serve args: ", len(serve_args))
    conf = {"NEW_BACKEND_NAME": {"key_mapping": {}, "kv_mapping_func": []}}
    for i in serve_args:
        conf["NEW_BACKEND_NAME"]["key_mapping"][i] = "TODO"
    conf["NEW_BACKEND_NAME"]["kv_mapping_func"].append("TODO")

    with open("template_conf.yaml", "w") as f:
        yaml.dump(conf, f, indent=4, sort_keys=True)


if __name__ == "__main__":
    gen_template_conf()
    print("./template_conf.yaml generated")
    print("Please update the keys with TODO values in the template_conf.yaml file.")
