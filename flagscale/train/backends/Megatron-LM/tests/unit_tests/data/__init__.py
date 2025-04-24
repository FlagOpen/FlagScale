def set_mock_args():
    from unittest import mock
    def init_mock_args(args):
        args.data_parallel_random_init = False
        args.virtual_pipeline_model_parallel_size = None
        args.bf16 = True
        args.accumulate_allreduce_grads_in_fp32 = False
        args.overlap_grad_reduce = False
        args.use_distributed_optimizer = True
        args.load = None
        args.save_param_index_maps_only = False
        args.rampup_batch_size = None
        args.global_batch_size = 8
        args.micro_batch_size = 1
        args.data_parallel_size = 8
        args.adlr_autoresume = False
        args.timing_log_option = 'minmax'
        args.timing_log_level = 0
        args.pretrained_checkpoint = None 
        return args

    with mock.patch('megatron.training.training.get_args', data_parallel_random_init=False) as mock_args:
        init_mock_args(mock_args.return_value)
        from megatron.training.global_vars import set_args
        set_args(mock_args.return_value)