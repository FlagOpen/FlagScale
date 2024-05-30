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
        self.idx = 0

    def checkout(self, mode):
        if mode == "memory":
            from ..utils import sort_by_memory
            if self.idx > 0 and self.idx < len(self.strategies):
                self.strategies = self.strategies[:self.idx] + sorted(
                    self.strategies[self.idx:], key=sort_by_memory)

        elif mode == "performance":
            from ..utils import sort_by_performance
            if self.idx > 0 and self.idx < len(self.strategies):
                self.strategies = self.strategies[:self.idx] + sorted(
                    self.strategies[self.idx:], key=sort_by_performance)

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
