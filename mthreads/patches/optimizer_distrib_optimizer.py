import torch

import megatron
from megatron.core import tensor_parallel
from megatron import print_rank_0

@classmethod
def build_model_and_main_param_groups(cls,
                                        model_gbuf_ranges,
                                        param_gbuf_map,
                                        opt_group_ranges):
    """
    Create main parameter groups needed for the optimizer step.

    These groups encompass both: 1) groups used by this class, for
    reducing/gather, and 2) groups used by the inner optimizer for the
    parameter update. Given that the conceptual grad buffer partitioning
    (created in earlier method) doesn't respect parameter boundaries,
    the optimizer operates on shards of the model parameters, rather than
    the full parameters.
    """

    # Parameter groups:
    #   model_float16_groups: original float16 parameters
    #   model_fp32_groups: original fp32 parameters
    #   shard_float16_groups: shards of original float16 parameters
    #   shard_fp32_groups: shards of original fp32 parameters
    #   shard_fp32_from_float16_groups: fp32 copy of float16 parameters
    model_float16_groups = []
    model_fp32_groups = []
    shard_float16_groups = []
    shard_fp32_groups = []
    shard_fp32_from_float16_groups = []

    # Allocate (or slice) each group's param shard.
    for group_index, group_range in enumerate(opt_group_ranges):

        # Params of this group.
        model_float16_params_this_group = []
        model_fp32_params_this_group = []
        shard_float16_params_this_group = []
        shard_fp32_params_this_group = []
        shard_fp32_from_float16_params_this_group = []
        model_float16_groups.append(model_float16_params_this_group)
        model_fp32_groups.append(model_fp32_params_this_group)
        shard_float16_groups.append(shard_float16_params_this_group)
        shard_fp32_groups.append(shard_fp32_params_this_group)
        shard_fp32_from_float16_groups.append(
            shard_fp32_from_float16_params_this_group)

        for model_param in group_range["params"]:

            assert model_param.requires_grad

            model_index, dtype = param_gbuf_map[model_param]
            gbuf_range = model_gbuf_ranges[model_index][dtype]
            param_range = gbuf_range["param_map"][model_param]["param"]

            # fp16, bf16 params.
            if model_param.type() in ['torch.musa.HalfTensor',
                                        'torch.musa.BFloat16Tensor']:

                # Clone model -> main.
                shard_model_param = model_param.detach().view(-1) \
                    [param_range.start:param_range.end]
                shard_main_param = shard_model_param.clone().float()
                tensor_parallel.copy_tensor_model_parallel_attributes(
                    shard_model_param, model_param)
                tensor_parallel.copy_tensor_model_parallel_attributes(
                    shard_main_param, model_param)
                if hasattr(model_param, 'shared'):
                    shard_model_param.shared = model_param.shared
                    shard_main_param.shared = model_param.shared

                # Add to group.
                model_float16_params_this_group.append(model_param)
                shard_float16_params_this_group.append(shard_model_param)
                shard_fp32_from_float16_params_this_group.append(shard_main_param)

            # fp32 params.
            elif model_param.type() == 'torch.musa.FloatTensor':
                print('floatoptimizer', flush=True)
                shard_model_param = model_param.view(-1) \
                    [param_range.start:param_range.end]
                model_fp32_params_this_group.append(model_param)
                shard_fp32_params_this_group.append(shard_model_param)
                tensor_parallel.copy_tensor_model_parallel_attributes(
                    shard_model_param, model_param)
                if hasattr(model_param, 'shared'):
                    shard_model_param.shared = model_param.shared

            else:
                raise TypeError('Wrapped parameters must be one of '
                                'torch.musa.FloatTensor,  '
                                'torch.musa.HalfTensor, or '
                                'torch.musa.BFloat16Tensor. '
                                'Received {}'.format(model_param.type()))

        # Update optimizer's params.
        group_range["orig_group"]["params"] = [
            *shard_fp32_params_this_group,
            *shard_fp32_from_float16_params_this_group,
        ]

    return (
        model_float16_groups,
        model_fp32_groups,
        shard_float16_groups,
        shard_fp32_groups,
        shard_fp32_from_float16_groups,
    )


# TODO(mthreads)
def DistributedOptimizer_load_state_dict(self, state_dict):
        """Load the state dict.

        As detailed in state_dict(), the state dict contains all non-
        parameter-related variables. This method is notably longer than
        state_dict(), because the Torch optimizers state has yet to be
        allocated at this point, and so we must do a cross referencing between
        the optimizers state (and the ordering it expects for parameter state)
        and this DP rank's shards. The optimizer at this point does not contain
        any tensor dimension information, so we must get these dimensions from
        the DP shards mapped during DistributedOptimizer.__init__().

        The tensor parameter state is loaded via load_parameter_state(), and
        so this method also must populate the loaded state dict with dummy
        tensor data (i.e., via torch.empty() below). This will be overwritten
        during load_parameter_state().

        ** Note: Torch optimizer's state structure. **
        The Torch optimizer stores its state in two levels. The top level is a
        list of groups, where each group contains a list of integer indexes
        (corresponding to parameters) that index into a master parameter list
        that is shared by all groups. As such, three values are necessary for
        maintaining this ordering:

        - group_index : The group to which a parameter belongs.
        - group_order : The index of a parameter within its group.
        - state_order : The index of a parameter within the shared parameter
            list.
        """

        # Get the Torch optimizer's state dict.
        # - This 'inner' optimizer at this point is unallocated, and only
        #   contains an integer odering of parameters within each group, and
        #   the ordering of parameters within its flattened parameter state
        #   list.
        inner_state_dict = self.optimizer.state_dict()
        state_dict_param_groups = [{
            **group,
            "params" : list(inner_state_dict["param_groups"][idx]["params"]),
        } for idx, group in enumerate(state_dict["optimizer"]["param_groups"])]

        # Allocate 'dummy' data for optimizer state (i.e., torch.empty() below)
        # - Real data is overwritten during load_parameter_state().
        state_dict_state = []
        for gbuf_range_maps in self.model_gbuf_ranges:
            for gbuf_range_map in gbuf_range_maps.values():
                for model_param, param_range_map in \
                    gbuf_range_map["param_map"].items():

                    # Get parameter ordering information (see method docstring
                    # for details).
                    group_index, group_order = \
                        self.model_param_group_index_map[model_param]
                    state_order = inner_state_dict["param_groups"] \
                        [group_index]["params"][group_order]

                    # Allocate dummy tensors.
                    numel = len(param_range_map["gbuf_world"])
                    init_shard = lambda : torch.empty(
                        (numel,),
                        dtype=torch.float32,
                        device='musa:{}'.format(torch.musa.current_device()))

                    state_dict_state.append((state_order, {
                        "exp_avg" : init_shard(),
                        "exp_avg_sq" : init_shard(),
                    }))

        # Sort by state order (see method docstring for details).
        state_dict_state.sort(key = lambda s : s[0])
        state_dict_state = {s[0]:s[1] for s in state_dict_state}

        # Optimizer.
        self.optimizer.load_state_dict({
            "state" : state_dict_state,
            "param_groups" : state_dict_param_groups,
        })

        # Grad scaler.
        if 'grad_scaler' not in state_dict:
            if self.fp16:
                print_rank_0('***WARNING*** found an old checkpoint, will not '
                             'load grad scaler ...')
        else:
            if self.grad_scaler:
                self.grad_scaler.load_state_dict(state_dict['grad_scaler'])
            else:
                print_rank_0('***WARNING*** fould the grad scaler in the '
                             'checkpoint but it is None in the class. '
                             'Skipping loading grad scaler ...')
                

megatron.optimizer.distrib_optimizer.DistributedOptimizer.build_model_and_main_param_groups = build_model_and_main_param_groups
megatron.optimizer.DistributedOptimizer.build_model_and_main_param_groups = build_model_and_main_param_groups
