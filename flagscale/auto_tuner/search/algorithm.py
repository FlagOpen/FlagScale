from abc import ABC, abstractmethod


class Algo(ABC):
    def __init__(self, strategies, config):
        self.strategies = strategies
        self.config = config

    @abstractmethod
    def search(self):
        pass

    @abstractmethod
    def has_done(self):
        pass


class GridAlgo(Algo):

    def __init__(self, strategies, config):
        super().__init__(strategies, config)

        # Sort the strategies according to priority, the priorty can be memory or performance.
        # If memeory, the strategies will be sorted by memory usage, otherwise by performance.
        priority = self.config.auto_tuner.algo.get("priority", None)
        if priority is not None:
            if priority == "memory":
                from ..utils import sort_by_memory

                self.strategies.sort(key=sort_by_memory)
            elif priority == "performance":
                from ..utils import sort_by_performance

                self.strategies.sort(key=sort_by_performance)
            else:
                raise ValueError(
                    "Unknown priority: {}, priority only in [memory, performance]".format(
                        priority
                    )
                )

        self.idx = 0

    def search(self):
        """Return a task iteratively."""
        strategy = None
        if self.idx < len(self.strategies):
            strategy = self.strategies[self.idx]
            self.idx += 1
        return strategy

    def has_done(self):
        """Return True if the task space is empyt."""
        if self.idx >= len(self.strategies):
            return True
        return False
