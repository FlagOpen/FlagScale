import os
import yaml
import argparse
from omegaconf import OmegaConf, ListConfig
from vllm import LLM, SamplingParams


def get_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-path", type=str, required=True, help="Path to the configuration YAML file")
    args = parser.parse_args()

    config_path = args.config_path
    # Open the YAML file and convert it into a dictionary
    with open(config_path, 'r') as file:
        config_dict = yaml.safe_load(file)
    
    # Convert the dictionary into a DictConfig
    config = OmegaConf.create(config_dict)
    return config 


def get_prompts(prompts):
    print(prompts, type(prompts))
    if isinstance(prompts, str) and os.path.isfile(prompts):
        with open(prompts, 'r') as file:
            return [line.strip() for line in file.readlines()]
    elif isinstance(prompts, (list, ListConfig)):
        return prompts
    else:
        raise ValueError("Prompts should be either a list of strings or a path to a file containing a list of strings.")


def inference():
    # Get the configuration.
    config = get_config()

    # Get the prompts.
    prompts = get_prompts(config.generate.prompts)

    # Create a sampling params object.
    sampling_args = config.get("sampling", {})
    sampling_params = SamplingParams(**sampling_args)

    # Create an LLM.
    llm_args = config.get("llm", {})
    model = llm_args.pop("model", None)
    assert model is not None
    llm = LLM(model, **llm_args)

    # Generate texts from the prompts. 
    outputs = llm.generate(prompts, sampling_params)

    # Print the outputs.
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")


if __name__ == '__main__':
    # Run the inference
    inference()
