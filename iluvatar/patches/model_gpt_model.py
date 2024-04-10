import megatron
from megatron import print_rank_0


def print_device_type():
    device_type = "iluvatar" 
    if device_type:
        print_rank_0("=== Monkey-patching Device Type: {} ===".format(device_type))
    else:
        print_rank_0("=== Monkey-patching Device Type: None ===")


# This is used for monkey-patching demonstration.
megatron.model.gpt_model.print_device_type = print_device_type
