from abc import ABC, abstractmethod
from torch.nn import Module

class BaseWrapper(Module, ABC):
    def __init__(self, name, layer):
        super(BaseWrapper, self).__init__(name, layer)
        self.name = name
        ### disable _enable_compress means only observer
        self._enable_compress = False

    def add_batch(self):
        raise NotImplementedError
    
    def compress(self):
        raise NotImplementedError
    
    @setattr
    def enable_compress(self):
        self._enable_compress = True

    @setattr
    def disable_compress(self):
        self._enable_compress = False