
class AscendConfig:
    def __init__(self):
        self.ColumnParallelLinear = None
        self.RowParallelLinear = None
        self.group_func = None
        self.world_size_func = None

        self.sequence_parallel_enabled = True
        self.all_gather_recomputation = True

    def set_sequence_parallel(self, sequence_parallel):
        self.sequence_parallel = sequence_parallel

    def set_all_gather_recomputation(self, all_gather_recomputation):
        self.all_gather_recomputation = all_gather_recomputation

    def set_group(self, group_func):
        self.group_func = group_func

    def get_group(self):
        return self.group_func()

    def set_world_size(self, world_size_func):
        self.world_size_func = world_size_func

    def get_world_size(self):
        return self.world_size_func()

    def set_column_parallel_linear(self, column_parallel_linear):
        self.ColumnParallelLinear = column_parallel_linear

    def set_row_parallel_linear(self, row_parallel_linear):
        self.RowParallelLinear = row_parallel_linear

    def parallel_linear_plugin(self, column_parallel_forward, row_parallel_forward):
        self.ColumnParallelLinear.forward = column_parallel_forward
        self.RowParallelLinear.forward = row_parallel_forward

ascend_turbo_cfg = AscendConfig()
