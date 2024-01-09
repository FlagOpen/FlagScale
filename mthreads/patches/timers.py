import time
import torch
import torch_musa
import megatron

def start(self, barrier=False):
    """Start the timer."""
    assert not self._started, 'timer has already been started'
    if barrier:
        torch.distributed.barrier(group=self._barrier_group)
    torch.musa.synchronize()
    self._start_time = time.time()
    self._started = True


def stop(self, barrier=False):
    """Stop the timer."""
    assert self._started, 'timer is not started'
    if barrier:
        torch.distributed.barrier(group=self._barrier_group)
    torch.musa.synchronize()
    self._elapsed += (time.time() - self._start_time)
    self._started = False

def _get_elapsed_time_all_ranks(self, names, reset, barrier):
        """
        Assumptions:
            - All the ranks call this function.
            - `names` are identical on all ranks.
        If the above assumptions are not met, calling this function will
        result in hang.
        Arguments:
            - names: list of timer names
            - reset: reset the timer after recording the elapsed time
            - barrier: if set, do a global barrier before time measurments
        """

        # First make sure all the callers are in sync.
        if barrier:
            torch.distributed.barrier()

        world_size = torch.distributed.get_world_size()
        rank = torch.distributed.get_rank()

        # Here we can use gather on the rank we want to print the
        # timing, however, there is no gather_base support in
        # pytorch yet. It is simpler to deal with a single tensor
        # and since we are only gathering a small amount of data,
        # it should be ok to use all-gather instead of gather.
        rank_name_to_time = torch.zeros((world_size, len(names)),
                                        dtype=torch.float,
                                        device='musa:{}'.format(torch.musa.current_device()))
        for i, name in enumerate(names):
            if name in self._timers:
                # Here we don't need to pass the barrier flag as all
                # the processes are already in sync. This avoids the
                # issue of different timers having different barrier
                # groups inside their class.
                rank_name_to_time[rank, i] = self._timers[name].elapsed(
                    reset=reset)

        # See the note above for why we are not using gather.
        torch.distributed._all_gather_base(rank_name_to_time.view(-1),
                                           rank_name_to_time[rank, :].view(-1))

        return rank_name_to_time

megatron.timers.Timer.start = start
megatron.timers.Timer.stop = stop
megatron.timers._get_elapsed_time_all_ranks = _get_elapsed_time_all_ranks
