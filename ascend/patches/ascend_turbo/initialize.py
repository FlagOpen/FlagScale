from .ascend_turbo_cfg import ascend_turbo_cfg

from megatron.core.tensor_parallel.layers import ColumnParallelLinear, RowParallelLinear
from megatron.core.parallel_state import (
    get_tensor_model_parallel_group,
    get_tensor_model_parallel_world_size
)
from .mc2_linears_seq_parallel import ColumnSeqParallelLinear, RowSeqParallelLinear

def column_parallel_forward(self, input_):
    bias = self.bias if not self.skip_bias_add else None

    output = ColumnSeqParallelLinear.apply(
        input_,
        self.weight,
        bias,
        ascend_turbo_cfg.get_group()
    )

    output_bias = self.bias if self.skip_bias_add else None
    return output, output_bias

def row_parallel_forward(self, input_):
    output = RowSeqParallelLinear.apply(
        input_,
        self.weight,
        None,
        ascend_turbo_cfg.get_group()
    )

    if not self.skip_bias_add:
        output = output + self.bias if self.bias is not None else output
        output_bias = None
    else:
        output_bias = self.bias

    return output, output_bias


def initialize_cfg_from_farmework():
    ascend_turbo_cfg.set_group(get_tensor_model_parallel_group)
    ascend_turbo_cfg.set_world_size(get_tensor_model_parallel_world_size)

    ascend_turbo_cfg.set_column_parallel_linear(ColumnParallelLinear)
    ascend_turbo_cfg.set_row_parallel_linear(RowParallelLinear)
    ascend_turbo_cfg.parallel_linear_plugin(column_parallel_forward, row_parallel_forward)

def initialize_cfg_from_args(args):
    if not args.sequence_parallel or args.tensor_model_parallel_size is 1:
        return

    ascend_turbo_cfg.set_sequence_parallel(args.sequence_parallel)
    ascend_turbo_cfg.set_all_gather_recomputation(True)
    initialize_cfg_from_farmework()
