import torch
import torch_musa
from functools import wraps

from .utils import report_memory, get_ltor_masks_and_position_ids
from .arguments import validate_args, _add_training_args, core_transformer_config_from_args, _add_distributed_args
from .checkpointing import _load_base_checkpoint, get_rng_state, read_metadata
from .initialize import _compile_dependencies, _initialize_distributed
from .model_transformer import FlashSelfAttention, bias_dropout_add_fused_train
from .memory import MemoryBuffer
from .core_utils import GlobalMemoryBuffer
from .zarr import _save_to_existing_array
from .core_pipeline_parallel_p2p_communication import _communicate, _communicate_shapes
from .core_tensor_parallel_data import _build_key_size_numel_dictionaries, broadcast_data
from .model_transformer import FlashSelfAttention, bias_dropout_add_fused_train