import megatron
from megatron import print_rank_0
from flagscale.patches_utils import add_patches_module

#[iluvatar] start of changes
def print_device_type():
    device_type = "iluvatar" 
    if device_type:
        print_rank_0("=== Monkey-patching Device Type: {} ===".format(device_type))
    else:
        print_rank_0("=== Monkey-patching Device Type: None ===")

#[iluvatar] end of changes

# This is used for monkey-patching demonstration.
module_path = "megatron.core.models.gpt.gpt_model"
module_dict = {"print_device_type",print_device_type}
add_patches_module(module_path,module_dict)
