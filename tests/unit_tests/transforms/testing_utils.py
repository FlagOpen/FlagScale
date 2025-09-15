from torch import nn

from flagscale.models.adapters.base_adapter import BaseAdapter
from flagscale.transforms.hook import ModelHook


def make_linear_backbone(in_features: int = 2, out_features: int = 2) -> nn.Module:
    return nn.Sequential(nn.Linear(in_features, out_features))


class DummyAdapter(BaseAdapter):
    def __init__(self, backbone: nn.Module) -> None:
        super().__init__(backbone)
        self._backbone = backbone

    def backbone(self) -> nn.Module:
        return self._backbone


class ObserveStore:
    def __init__(self) -> None:
        self.calls: list = []

    def set(self, name, value):  # signature matches usage in ModelHook.set_state_context
        self.calls.append(name)


class ObserveHook(ModelHook):
    def __init__(self, store: ObserveStore) -> None:
        super().__init__()
        self.register_stateful(store)
