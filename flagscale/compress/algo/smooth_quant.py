import torch
from flagscale.compress.blockwise_wrapper import BlockCompressionWrapper
from llmcompressor.modifiers.smoothquant import *
from llmcompressor.modifiers.smoothquant.utils import DEFAULT_SMOOTHQUANT_MAPPINGS

__all__ = ["SmoothQuantWrapper"]
class SmoothQuantWrapper(BlockCompressionWrapper):
    def __init__(self, name, layer):
        super().__init__(name=name, layer=layer)
        self.sq = SmoothQuantModifier()
        self.sq.ignore = [] if not self.sq.ignore else self.sq.ignore
        self.sq.mappings = self.sq._infer_mappings_from_model(self.layer)
        self.sq.resolved_mappings_ = self.sq._resolve_mappings(self.layer)
        self.sq.scales_ = {}
    
    def add_batch(self, inp: torch.Tensor, out: torch.Tensor):
        self.sq._setup_scale_hooks()

    def compress(self, smoothing_strength):
        self.sq.smoothing_strength = smoothing_strength
        self.sq._apply_smoothing(self.layer)
