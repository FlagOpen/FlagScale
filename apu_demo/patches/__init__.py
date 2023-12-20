from.device_init import apu_demo_init
apu_demo_init()

# Apply the following patch during the import time
from .model_gpt_model import print_device_type