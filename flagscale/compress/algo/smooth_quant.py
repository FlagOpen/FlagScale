from llmcompressor.modifiers.utils.compression_wrapper import ModuleCompressionWrapper
from llmcompressor.modifiers.smoothquant import *

class SmoothQuantWrapper(ModuleCompressionWrapper):
    def __init__(self, name, layer):
        super().__init__(name=name, layer=layer)
        