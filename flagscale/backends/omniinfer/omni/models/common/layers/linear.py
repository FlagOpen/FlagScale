# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

from typing import Dict, List, Optional, Tuple

from vllm.platforms import current_platform

import torch
import torch_npu
from torch.nn.parameter import Parameter, UninitializedParameter
# from omni.models.common.utils import is_ray_multi_servers_devices
from omni.adaptors.vllm.distributed.communication_op import tensor_model_parallel_reduce_scatter


from vllm.distributed import (divide, split_tensor_along_last_dim)
from vllm.model_executor.layers.linear import (WEIGHT_LOADER_V2_SUPPORTED,
                                               LinearBase,
                                               adjust_marlin_shard,
                                               adjust_scalar_to_fused_array)
from vllm import logger
from vllm.model_executor.layers.linear import ColumnParallelLinear, ReplicatedLinear
from vllm.model_executor.layers.linear import RowParallelLinear as RowParallelLinearGPU

from vllm.model_executor.layers.quantization.base_config import QuantizationConfig
from vllm.model_executor.parameter import (BasevLLMParameter,
                                           PackedColumnParameter,
                                           PackedvLLMParameter,
                                           PerTensorScaleParameter,
                                           RowvLLMParameter)
from vllm.model_executor.utils import set_weight_attrs

from vllm.distributed import (
    get_tensor_model_parallel_rank,
    tensor_model_parallel_all_reduce,
    tensor_model_parallel_all_gather,
    get_tp_group,
    get_world_group
)
from omni.adaptors.vllm.distributed.communication_op import (
    expert_parallel_all_gather,
    expert_parallel_all_reduce,
)
from omni.adaptors.vllm.distributed.parallel_state import get_mlp_tp_size, get_mlp_tp_rank

class AscendMergedColumnParallelLinear(LinearBase):
    def __init__(self,
                 input_size: int,
                 output_sizes: List[int],
                 bias: bool = True,
                 gather_output: bool = False,
                 skip_bias_add: bool = False,
                 params_dtype: Optional[torch.dtype] = None,
                 quant_config: Optional[QuantizationConfig] = None,
                 prefix: str = ""):
        self.output_sizes = output_sizes
        tp_size = get_mlp_tp_size()
        if not all(output_size % tp_size == 0 for output_size in output_sizes):
            raise RuntimeError("All output_sizes must be divisible by tp_size")

        output_size = sum(output_sizes)
        super().__init__(input_size, output_size, skip_bias_add, params_dtype,
                         quant_config, prefix)

        self.gather_output = gather_output

        # Divide the weight matrix along the last dimension.
        if self.quant_method is None:
            raise RuntimeError("self.quant_method must not be None")
        self.output_size_per_partition = divide(self.output_size, tp_size)
        self.output_partition_sizes = [self.output_size_per_partition]
        # If QKV or MergedColumn, use output size of each partition.
        if hasattr(self, "output_sizes"):
            self.output_partition_sizes = [
                divide(output_size, tp_size)
                for output_size in self.output_sizes
            ]
        output_sizes = [output_size]

        self.quant_method.create_weights(
            layer=self,
            input_size_per_partition=self.input_size,
            output_partition_sizes=self.output_partition_sizes,
            input_size=self.input_size,
            output_size=self.output_size,
            params_dtype=self.params_dtype,
            weight_loader=(
                self.weight_loader_v2 if self.quant_method.__class__.__name__
                in WEIGHT_LOADER_V2_SUPPORTED else self.weight_loader))
        if bias:
            self.bias = Parameter(
                torch.empty(self.output_size_per_partition,
                            dtype=params_dtype))
            set_weight_attrs(self.bias, {
                "output_dim": 0,
                "weight_loader": self.weight_loader,
            })
        else:
            self.register_parameter("bias", None)
        self.throw_dequant = True

    def forward(self, input_):
        bias = self.bias if not self.skip_bias_add else None

        # Matrix multiply.
        if self.quant_method is None:
            raise RuntimeError("self.quant_method must not be None")
        output_parallel = self.quant_method.apply(self, input_, bias)
        if self.gather_output:
            if not isinstance(output_parallel, torch.Tensor):
                raise RuntimeError("not support throw dequant when need gather output")
            # All-gather across the partitions.
            output = expert_parallel_all_gather(output_parallel)
        else:
            output = output_parallel
        output_bias = self.bias if self.skip_bias_add else None
        return output, output_bias

    def weight_loader(self,
                      param: Parameter,
                      loaded_weight: torch.Tensor,
                      loaded_shard_id: Optional[int] = None):

        # Special case for GGUF
        # initialize GGUF param after we know the quantize type
        is_gguf_weight = getattr(param, "is_gguf_weight", False)
        is_gguf_weight_type = getattr(param, "is_gguf_weight_type", False)
        if is_gguf_weight_type:
            param.data[loaded_shard_id].copy_(loaded_weight)
            param.shard_weight_type[loaded_shard_id] = loaded_weight.item()
            return

        if is_gguf_weight:
            tp_size = get_mlp_tp_size()
            tp_rank = get_mlp_tp_rank()

            output_dim = getattr(param, "output_dim", None)
            shard_size = loaded_weight.size(output_dim) // tp_size
            start_idx = tp_rank * shard_size

            loaded_weight = loaded_weight.narrow(output_dim, start_idx,
                                                 shard_size)

            param.shard_id.append(loaded_shard_id)
            param.shard_id_map[loaded_shard_id] = len(param.data_container)
            param.data_container.append(loaded_weight)
            if len(param.data_container) == 2:
                self.qweight = param.materialize_nested()
            return

        param_data = param.data
        output_dim = getattr(param, "output_dim", None)
        # Special case for AQLM codebooks.
        is_metadata = getattr(param, "is_metadata", False)
        # Special case for per-tensor scale to load scalar into fused array.
        needs_scalar_to_array = getattr(param, "needs_scalar_to_array", False)

        if loaded_shard_id is None:
            # Loaded weight is already fused on disk (qkv/mlp).
            if output_dim is None:
                if needs_scalar_to_array:
                    param_data, loaded_weight = adjust_scalar_to_fused_array(
                        param_data, loaded_weight, 0)

                if param_data.shape != loaded_weight.shape:
                    raise RuntimeError("param_data.shape != loaded_weight.shape")
                param_data.copy_(loaded_weight)
                return
            current_shard_offset = 0
            shard_offsets: List[Tuple[int, int, int]] = []
            for i, output_size in enumerate(self.output_sizes):
                shard_offsets.append((i, current_shard_offset, output_size))
                current_shard_offset += output_size
            packed_dim = getattr(param, "packed_dim", None)
            for shard_id, shard_offset, shard_size in shard_offsets:
                # Special case for Quantization.
                # If quantized, we need to adjust the offset and size to account
                # for the packing.
                if packed_dim == output_dim:
                    shard_size = shard_size // param.pack_factor
                    shard_offset = shard_offset // param.pack_factor
                    # Special case for Marlin.
                    shard_size, shard_offset = adjust_marlin_shard(
                        param, shard_size, shard_offset)

                loaded_weight_shard = loaded_weight.narrow(
                    output_dim, shard_offset, shard_size)
                self.weight_loader(param, loaded_weight_shard, shard_id)
            return

        if loaded_shard_id >= len(self.output_sizes):
            raise RuntimeError("loaded_shard_id must be less than the length of self.output_sizes")
        tp_size = get_mlp_tp_size()
        tp_rank = get_mlp_tp_rank()
        if output_dim is not None:
            shard_offset = sum(self.output_sizes[:loaded_shard_id]) // tp_size
            shard_size = self.output_sizes[loaded_shard_id] // tp_size
            # Special case for quantization.
            # If quantized, we need to adjust the offset and size to account
            # for the packing.
            packed_dim = getattr(param, "packed_dim", None)
            if packed_dim == output_dim:
                shard_size = shard_size // param.pack_factor
                shard_offset = shard_offset // param.pack_factor
                # Special case for Marlin.
                shard_size, shard_offset = adjust_marlin_shard(
                    param, shard_size, shard_offset)

            use_bitsandbytes_4bit = getattr(param, "use_bitsandbytes_4bit",
                                            False)
            if use_bitsandbytes_4bit:
                shard_size = loaded_weight.shape[output_dim]
                shard_offset = loaded_weight.shape[output_dim] * \
                    loaded_shard_id

            param_data = param_data.narrow(output_dim, shard_offset,
                                           shard_size)
            start_idx = tp_rank * shard_size
            # bitsandbytes loads the weights of the specific portion
            # no need to narrow here
            if not use_bitsandbytes_4bit:
                loaded_weight = loaded_weight.narrow(output_dim, start_idx,
                                                     shard_size)
        # Special case for AQLM codebooks.
        elif is_metadata:
            # metadata indicates fixed size concatenated along dim 0
            shard_size = loaded_weight.shape[0]
            shard_offset = loaded_shard_id * shard_size
            param_data = param_data.narrow(0, shard_offset, shard_size)

        # Special case for per-tensor scales in fused case.
        elif needs_scalar_to_array:
            param_data, loaded_weight = adjust_scalar_to_fused_array(
                param_data, loaded_weight, loaded_shard_id)

        else:
            ignore_warning = getattr(param, "ignore_warning", False)
            if not ignore_warning:
                logger.warning(
                    "Loading a weight without `output_dim` attribute in "
                    "MergedColumnParallelLinear, assume the weight is "
                    "the same for all partitions.")

        if param_data.shape != loaded_weight.shape:
            raise RuntimeError("param_data.shape != loaded_weight.shape")
        param_data.copy_(loaded_weight)

    def _load_fused_module_from_checkpoint(self, param: BasevLLMParameter,
                                           loaded_weight: torch.Tensor):
        """
        Handle special case for models where MLP layers are already
        fused on disk. In this case, we have no shard id. This function
        determmines the shard id by splitting these layers and then calls
        the weight loader using the shard id.

        An example of a model with these fused layers:
        https://huggingface.co/microsoft/Phi-3-mini-4k-instruct
        """

        current_shard_offset = 0
        shard_offsets: List[Tuple[int, int, int]] = []
        for i, output_size in enumerate(self.output_sizes):
            shard_offsets.append((i, current_shard_offset, output_size))
            current_shard_offset += output_size

        for shard_id, shard_offset, shard_size in shard_offsets:
            # Special case for Quantization.
            # If quantized, we need to adjust the offset and size to account
            # for the packing.
            if isinstance(param, (PackedColumnParameter, PackedvLLMParameter
                                  )) and param.packed_dim == param.output_dim:
                shard_size, shard_offset = \
                    param.adjust_shard_indexes_for_packing(
                    shard_size=shard_size, shard_offset=shard_offset)

            loaded_weight_shard = loaded_weight.narrow(param.output_dim,
                                                       shard_offset,
                                                       shard_size)
            self.weight_loader_v2(param, loaded_weight_shard, shard_id)

    def weight_loader_v2(self,
                         param: BasevLLMParameter,
                         loaded_weight: torch.Tensor,
                         loaded_shard_id: Optional[int] = None):
        if loaded_shard_id is None:
            if isinstance(param, PerTensorScaleParameter):
                param.load_merged_column_weight(loaded_weight=loaded_weight,
                                                shard_id=0)
                return
            elif type(param) in (RowvLLMParameter, BasevLLMParameter):
                param.load_merged_column_weight(loaded_weight=loaded_weight)
                return
            self._load_fused_module_from_checkpoint(param, loaded_weight)
            return

        if loaded_shard_id >= len(self.output_sizes):
            raise RuntimeError("loaded_shard_id must be less than len(self.output_sizes)")

        tp_size = get_mlp_tp_size()
        shard_offset = sum(self.output_sizes[:loaded_shard_id]) // tp_size
        shard_size = self.output_sizes[loaded_shard_id] // tp_size

        param.load_merged_column_weight(loaded_weight=loaded_weight,
                                        shard_id=loaded_shard_id,
                                        shard_offset=shard_offset,
                                        shard_size=shard_size)

class AscendRowParallelLinear(LinearBase):

    def __init__(self,
                 input_size: int,
                 output_size: int,
                 bias: bool = True,
                 input_is_parallel: bool = True,
                 skip_bias_add: bool = False,
                 params_dtype: Optional[torch.dtype] = None,
                 reduce_results: bool = True,
                 quant_config: Optional[QuantizationConfig] = None,
                 prefix: str = ""):
        super().__init__(input_size, output_size, skip_bias_add, params_dtype,
                         quant_config, prefix)

        self.input_is_parallel = input_is_parallel
        self.reduce_results = reduce_results

        # Divide the weight matrix along the last dimension.
        self.tp_size = get_mlp_tp_size()
        self.tp_rank = get_mlp_tp_rank()
        self.input_size_per_partition = divide(input_size, self.tp_size)
        if self.quant_method is None:
            raise RuntimeError("self.quant_method must not be None")

        self.quant_method.create_weights(
            layer=self,
            input_size_per_partition=self.input_size_per_partition,
            output_partition_sizes=[self.output_size],
            input_size=self.input_size,
            output_size=self.output_size,
            params_dtype=self.params_dtype,
            weight_loader=(
                self.weight_loader_v2 if self.quant_method.__class__.__name__
                in WEIGHT_LOADER_V2_SUPPORTED else self.weight_loader))
        if not reduce_results and (bias and not skip_bias_add):
            raise ValueError("When not reduce the results, adding bias to the "
                             "results can lead to incorrect results")

        if bias:
            self.bias = Parameter(
                torch.empty(self.output_size, dtype=params_dtype))
            set_weight_attrs(self.bias, {
                "output_dim": 0,
                "weight_loader": self.weight_loader,
            })
        else:
            self.register_parameter("bias", None)

    def weight_loader(self, param: Parameter, loaded_weight: torch.Tensor):
        tp_size = get_mlp_tp_size()
        tp_rank = get_mlp_tp_rank()
        input_dim = getattr(param, "input_dim", None)
        use_bitsandbytes_4bit = getattr(param, "use_bitsandbytes_4bit", False)

        # Special case for GGUF
        is_gguf_weight = getattr(param, "is_gguf_weight", False)
        is_gguf_weight_type = getattr(param, "is_gguf_weight_type", False)
        if is_gguf_weight_type:
            param.weight_type = loaded_weight.item()

        # Materialize GGUF UninitializedParameter
        if is_gguf_weight and isinstance(param, UninitializedParameter):
            weight_shape = list(loaded_weight.shape)
            if input_dim:
                weight_shape[input_dim] = weight_shape[input_dim] // tp_size
            param.materialize(tuple(weight_shape), dtype=loaded_weight.dtype)

        param_data = param.data
        # bitsandbytes loads the weights of the specific portion
        # no need to narrow here
        if input_dim is not None and not use_bitsandbytes_4bit:
            shard_size = param_data.shape[input_dim]
            start_idx = tp_rank * shard_size
            loaded_weight = loaded_weight.narrow(input_dim, start_idx,
                                                 shard_size)

        # Special case for loading scales off disk, which often do not
        # have a shape (such as in the case of AutoFP8).
        if len(loaded_weight.shape) == 0:
            loaded_weight = loaded_weight.reshape(1)

        if param_data.shape != loaded_weight.shape:
            raise RuntimeError("param_data.shape != loaded_weight.shape")
        param_data.copy_(loaded_weight)

    def weight_loader_v2(self, param: BasevLLMParameter,
                         loaded_weight: torch.Tensor):

        # Special case for loading scales off disk, which often do not
        # have a shape (such as in the case of AutoFP8).
        if len(loaded_weight.shape) == 0:
            if loaded_weight.numel() != 1:
                raise RuntimeError("loaded_weight.numel() != 1")
            loaded_weight = loaded_weight.reshape(1)

        param.load_row_parallel_weight(loaded_weight=loaded_weight)

    def forward(self, input_):
        if self.input_is_parallel:
            input_parallel = input_
        else:
            tp_rank = get_mlp_tp_rank()
            splitted_input = split_tensor_along_last_dim(
                input_, num_partitions=self.tp_size)
            input_parallel = splitted_input[tp_rank].contiguous()

        # Matrix multiply.
        if self.quant_method is None:
            raise RuntimeError("self.quant_method is None")
        # Only fuse bias add into GEMM for rank 0 (this ensures that
        # bias will not get added more than once in TP>1 case)
        bias_ = None if (self.tp_rank > 0 or self.skip_bias_add) else self.bias
        output_parallel = self.quant_method.apply(self,
                                                  input_parallel,
                                                  bias=bias_)
        if self.reduce_results and self.tp_size > 1:
            output = expert_parallel_all_reduce(output_parallel)
        else:
            output = output_parallel

        output_bias = self.bias if self.skip_bias_add else None

        return output, output_bias

    def extra_repr(self) -> str:
        s = f"input_features={self.input_size_per_partition}"
        s += f", output_features={self.output_size}"
        s += f", bias={self.bias is not None}"
        s += f", tp_size={self.tp_size}"
        s += f", reduce_results={self.reduce_results}"
        return s
    

class ColumnParallelLinearQuantGather(ColumnParallelLinear):
    def __init__(self, input_size, output_size, bias, quant_config, prefix):
        super().__init__(input_size=input_size,
                         output_size=output_size,
                         bias=bias,
                         quant_config=quant_config,
                         prefix=prefix)
 
    def forward(self, input_):
        bias = self.bias if not self.skip_bias_add else None
 
        # Matrix multiply.
        if self.quant_method is None:
            raise RuntimeError("self.quant_method is not None")
        output_parallel = self.quant_method.apply(self, input_, bias, inner_gather=True)
        if self.gather_output:
            # All-gather across the partitions.
            output = tensor_model_parallel_all_gather(output_parallel)
        else:
            output = output_parallel
        output_bias = self.bias if self.skip_bias_add else None
        return output, output_bias


class RowParallelLinear(RowParallelLinearGPU):
    def __init__(
            self,
            input_size: int,
            output_size: int,
            bias: bool = True,
            input_is_parallel: bool = True,
            skip_bias_add: bool = False,
            params_dtype: Optional[torch.dtype] = None,
            reduce_results: bool = True,
            quant_config: Optional[QuantizationConfig] = None,
            prefix: str = ""
    ):
        super().__init__(input_size, output_size, bias, input_is_parallel, skip_bias_add, params_dtype, reduce_results,
                         quant_config, prefix)
        # Adapt: adapt npu_mm_all_reduce_base op init info
        tp_rank = get_tensor_model_parallel_rank()
        local_rank = get_world_group().local_rank
        torch_npu.npu.set_device(local_rank)
        default_pg = get_tp_group().device_group
        if False: #is_ray_multi_servers_devices(self.tp_size):
            self.hcom_info = None
        else:
            self.hcom_info = default_pg._get_backend(torch.device(current_platform.device_type)).get_hccl_comm_name(tp_rank)
        self.quant_config = quant_config
        # Adapt end.

    @staticmethod
    def _get_token_num(input_parallel):
        # w8a8_dynamic_quant scenario use dict type
        if isinstance(input_parallel, Dict):
            input_parallel_tensor = input_parallel.get("x_int8")
        else:
            input_parallel_tensor = input_parallel
        token_num = input_parallel_tensor.shape[0] if input_parallel_tensor.ndim == 2 else \
            input_parallel_tensor.shape[0] * input_parallel_tensor.shape[1]
        return int(token_num)

    def forward(self, input_):
        # Set up backprop all-reduce.
        if self.input_is_parallel:
            input_parallel = input_
        else:
            tp_rank = get_tensor_model_parallel_rank()
            splitted_input = split_tensor_along_last_dim(
                input_, num_partitions=self.tp_size)
            input_parallel = splitted_input[tp_rank].contiguous()

        # Only fuse bias add into GEMM for rank 0 (this ensures that
        # bias will not get added more than once in TP>1 case)
        bias_ = None if (self.tp_rank > 0 or self.skip_bias_add) else self.bias

        output_parallel = self.quant_method.apply(self, input_parallel, bias=bias_)
        if self.reduce_results and self.tp_size > 1:
            output_ = tensor_model_parallel_all_reduce(output_parallel)
        else:
            output_ = output_parallel
        # Adapt end.
        output_bias = self.bias if self.skip_bias_add else None

        return output_, output_bias


class RowParallelLinearWithReduceScatter(RowParallelLinear):
    def __init__(self, *args, **kwargs):
        super(RowParallelLinearWithReduceScatter, self).__init__(*args, **kwargs)
        if self.bias is not None:
            raise RuntimeError("self.bias is not None")

    def forward(self, input_):
        if self.input_is_parallel:
            input_parallel = input_
        else:
            tp_rank = get_tensor_model_parallel_rank()
            splitted_input = split_tensor_along_last_dim(
                input_, num_partitions=self.tp_size)
            input_parallel = splitted_input[tp_rank].contiguous()

        # Matrix multiply.
        if self.quant_method is None:
            raise RuntimeError("self.quant_method is None")
        # Only fuse bias add into GEMM for rank 0 (this ensures that
        # bias will not get added more than once in TP>1 case)
        bias_ = None if (self.tp_rank > 0 or self.skip_bias_add) else self.bias
        output_parallel = self.quant_method.apply(self,
                                                  input_parallel,
                                                  bias=bias_)
        if self.reduce_results and self.tp_size > 1:
            output = tensor_model_parallel_reduce_scatter(output_parallel)
        else:
            output = output_parallel

        output_bias = self.bias if self.skip_bias_add else None

        return output, output_bias



class MergedReplicatedLinear(ReplicatedLinear):

    def __init__(self,
                 input_size: int,
                 output_sizes: List[int],
                 bias: bool = True,
                 skip_bias_add: bool = False,
                 params_dtype: Optional[torch.dtype] = None,
                 quant_config: Optional[QuantizationConfig] = None,
                 prefix: str = ""):
        self.output_sizes = output_sizes
        super().__init__(input_size=input_size,
                         output_size=sum(output_sizes),
                         bias=bias,
                         skip_bias_add=skip_bias_add,
                         params_dtype=params_dtype,
                         quant_config=quant_config,
                         prefix=prefix)

    def weight_loader(self,
                      param: Parameter,
                      loaded_weight: torch.Tensor,
                      loaded_shard_id: Optional[int] = None):

        # Special case for GGUF
        # initialize GGUF param after we know the quantize type
        is_gguf_weight = getattr(param, "is_gguf_weight", False)
        is_gguf_weight_type = getattr(param, "is_gguf_weight_type", False)
        if is_gguf_weight_type:
            param.data[loaded_shard_id].copy_(loaded_weight)
            param.shard_weight_type[loaded_shard_id] = loaded_weight.item()
            return

        if is_gguf_weight:
            tp_size = 1
            tp_rank = 0

            output_dim = getattr(param, "output_dim", None)
            shard_size = loaded_weight.size(output_dim) // tp_size
            start_idx = tp_rank * shard_size

            loaded_weight = loaded_weight.narrow(output_dim, start_idx,
                                                 shard_size)

            param.shard_id.append(loaded_shard_id)
            param.shard_id_map[loaded_shard_id] = len(param.data_container)
            param.data_container.append(loaded_weight)
            if len(param.data_container) == 2:
                self.qweight = param.materialize_nested()
            return

        param_data = param.data
        output_dim = getattr(param, "output_dim", None)
        # Special case for AQLM codebooks.
        is_metadata = getattr(param, "is_metadata", False)
        # Special case for per-tensor scale to load scalar into fused array.
        needs_scalar_to_array = getattr(param, "needs_scalar_to_array", False)

        if loaded_shard_id is None:
            # Loaded weight is already fused on disk (qkv/mlp).
            if output_dim is None:
                if needs_scalar_to_array:
                    param_data, loaded_weight = adjust_scalar_to_fused_array(
                        param_data, loaded_weight, 0)

                if param_data.shape != loaded_weight.shape:
                    raise RuntimeError("param_data.shape != loaded_weight.shape")
                param_data.copy_(loaded_weight)
                return
            current_shard_offset = 0
            shard_offsets: List[Tuple[int, int, int]] = []
            for i, output_size in enumerate(self.output_sizes):
                shard_offsets.append((i, current_shard_offset, output_size))
                current_shard_offset += output_size
            packed_dim = getattr(param, "packed_dim", None)
            for shard_id, shard_offset, shard_size in shard_offsets:
                # Special case for Quantization.
                # If quantized, we need to adjust the offset and size to account
                # for the packing.
                if packed_dim == output_dim:
                    shard_size = shard_size // param.pack_factor
                    shard_offset = shard_offset // param.pack_factor
                    # Special case for Marlin.
                    shard_size, shard_offset = adjust_marlin_shard(
                        param, shard_size, shard_offset)

                loaded_weight_shard = loaded_weight.narrow(
                    output_dim, shard_offset, shard_size)
                self.weight_loader(param, loaded_weight_shard, shard_id)
            return

        if loaded_shard_id >= len(self.output_sizes):
            raise RuntimeError("loaded_shard_id >= len(self.output_sizes)")
        tp_rank = 0
        tp_size = 1
        if output_dim is not None:
            shard_offset = sum(self.output_sizes[:loaded_shard_id]) // tp_size
            shard_size = self.output_sizes[loaded_shard_id] // tp_size
            # Special case for quantization.
            # If quantized, we need to adjust the offset and size to account
            # for the packing.
            packed_dim = getattr(param, "packed_dim", None)
            if packed_dim == output_dim:
                shard_size = shard_size // param.pack_factor
                shard_offset = shard_offset // param.pack_factor
                # Special case for Marlin.
                shard_size, shard_offset = adjust_marlin_shard(
                    param, shard_size, shard_offset)

            use_bitsandbytes_4bit = getattr(param, "use_bitsandbytes_4bit",
                                            False)
            if use_bitsandbytes_4bit:
                shard_size = loaded_weight.shape[output_dim]
                shard_offset = loaded_weight.shape[output_dim] * \
                    loaded_shard_id

            param_data = param_data.narrow(output_dim, shard_offset,
                                           shard_size)
            start_idx = tp_rank * shard_size
            # bitsandbytes loads the weights of the specific portion
            # no need to narrow here
            if not use_bitsandbytes_4bit:
                loaded_weight = loaded_weight.narrow(output_dim, start_idx,
                                                     shard_size)
        # Special case for AQLM codebooks.
        elif is_metadata:
            # metadata indicates fixed size concatenated along dim 0
            shard_size = loaded_weight.shape[0]
            shard_offset = loaded_shard_id * shard_size
            param_data = param_data.narrow(0, shard_offset, shard_size)

        # Special case for per-tensor scales in fused case.
        elif needs_scalar_to_array:
            param_data, loaded_weight = adjust_scalar_to_fused_array(
                param_data, loaded_weight, loaded_shard_id)

        else:
            ignore_warning = getattr(param, "ignore_warning", False)
            if not ignore_warning:
                logger.warning(
                    "Loading a weight without `output_dim` attribute in "
                    "MergedColumnParallelLinear, assume the weight is "
                    "the same for all partitions.")

        if param_data.shape != loaded_weight.shape:
            raise RuntimeError("param_data.shape != loaded_weight.shape")
        param_data.copy_(loaded_weight)
