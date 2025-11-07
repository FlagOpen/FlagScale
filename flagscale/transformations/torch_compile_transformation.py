from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import torch
import torch.fx as fx

from omegaconf import DictConfig
from torch import nn
from torch._dynamo.eval_frame import OptimizedModule
from torch._inductor.pattern_matcher import (
    Arg,
    CallFunction,
    Match,
    PatternMatcherPass,
    register_graph_pattern,
)
from torch.fx.operator_schemas import normalize_function

from flagscale.compilation.inductor_pass import InductorPass
from flagscale.runner.utils import logger
from flagscale.transformations.transformation import Selector, Transformation, build_selector

_sin_cos_matcher = PatternMatcherPass(pass_name="sin_cos_cat_pattern")

# Target pattern (before rewrite):
#     # concat sine and cosine embeddings
#     emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
#
#     # flip sine and cosine embeddings
#     if flip_sin_to_cos:
#         emb = torch.cat([emb[:, half_dim:], emb[:, :half_dim]], dim=-1)


def _is_bigint(value: Any) -> bool:
    return isinstance(value, int) and value > 10**12


def _as_tuple(value: Any) -> Optional[Tuple[Any, ...]]:
    if isinstance(value, (list, tuple)):
        return tuple(value)
    if hasattr(value, "__iter__"):
        try:
            return tuple(value)
        except TypeError:
            return None
    return None


def _extract_flip_components(
    cat_node: fx.Node,
) -> Optional[Tuple[fx.Node, fx.Node, fx.Node, Tuple[fx.Node, fx.Node]]]:
    if not (
        isinstance(cat_node, fx.Node)
        and cat_node.op == "call_function"
        and cat_node.target is torch.ops.aten.cat.default
    ):
        return None

    normalized = normalize_function(
        cat_node.target, cat_node.args, cat_node.kwargs  # type: ignore[arg-type]
    )
    if normalized is None:
        list_arg = cat_node.args[0] if cat_node.args else cat_node.kwargs.get("tensors")
        dim_arg = cat_node.args[1] if len(cat_node.args) > 1 else cat_node.kwargs.get("dim", -1)
    else:
        norm_args, norm_kwargs = normalized
        list_arg = norm_args[0] if len(norm_args) > 0 else norm_kwargs.get("tensors")
        dim_arg = norm_args[1] if len(norm_args) > 1 else norm_kwargs.get("dim", -1)

    if dim_arg != -1 or list_arg is None:
        return None

    slices = _as_tuple(list_arg)
    if slices is None or len(slices) != 2:
        return None

    slice0, slice1 = slices

    for slc in slices:
        if not (cat_node in slc.users and len(slc.users) == 1):
            return None
    for slc in slices:
        if not (
            isinstance(slc, fx.Node)
            and slc.op == "call_function"
            and slc.target is torch.ops.aten.slice.Tensor
            and len(slc.args) >= 4
            and isinstance(slc.args[1], int)
            and slc.args[1] == 1
        ):
            return None

    base0, base1 = slice0.args[0], slice1.args[0]
    if base0 is not base1 or not isinstance(base0, fx.Node):
        return None

    base = base0
    normalized_base = normalize_function(
        base.target, base.args, base.kwargs  # type: ignore[arg-type]
    )
    if normalized_base is None:
        base_list = base.args[0] if base.args else base.kwargs.get("tensors")
        base_dim = base.args[1] if len(base.args) > 1 else base.kwargs.get("dim", -1)
    else:
        base_args, base_kwargs = normalized_base
        base_list = base_args[0] if len(base_args) > 0 else base_kwargs.get("tensors")
        base_dim = base_args[1] if len(base_args) > 1 else base_kwargs.get("dim", -1)

    if base_dim != -1 or base_list is None:
        return None

    base_nodes = _as_tuple(base_list)
    if base_nodes is None or len(base_nodes) != 2:
        return None

    sin_node, cos_node = base_nodes
    if not (
        isinstance(sin_node, fx.Node)
        and sin_node.op == "call_function"
        and sin_node.target is torch.ops.aten.sin.default
        and isinstance(cos_node, fx.Node)
        and cos_node.op == "call_function"
        and cos_node.target is torch.ops.aten.cos.default
    ):
        return None

    start0, end0 = slice0.args[2], slice0.args[3]
    start1, end1 = slice1.args[2], slice1.args[3]

    forward = (
        isinstance(start0, fx.Node)
        and isinstance(end1, fx.Node)
        and start0 is end1
        and isinstance(start1, int)
        and start1 == 0
        and _is_bigint(end0)
    )
    reverse = (
        isinstance(start1, fx.Node)
        and isinstance(end0, fx.Node)
        and start1 is end0
        and isinstance(start0, int)
        and start0 == 0
        and _is_bigint(end1)
    )

    if not (forward or reverse):
        return None

    return base, sin_node, cos_node, (slice0, slice1)


def _sin_cos_extra_check(match: Match) -> bool:
    cat_node = match.nodes[0]
    return _extract_flip_components(cat_node) is not None


class SinCosCatPatternPass(InductorPass):
    def __call__(self, graph: torch.fx.Graph) -> None:
        _sin_cos_matcher.apply(graph)


@dataclass
class CompileOptions:
    """
    Options for the compilation.
    """

    mode: Optional[str] = "default"
    dynamic: Optional[bool] = True
    fullgraph: Optional[bool] = False
    disable: Optional[bool] = True


class MyCustomPass(InductorPass):
    def __call__(self, graph: torch.fx.Graph) -> None:
        print(">>> my custom pass")
        print(graph)


aten = torch.ops.aten

# shared inputs we want to capture once and reuse
base = Arg()  # %mul_17 feeding both sin and cos
split_idx = Arg()  # %floordiv reused by both slices

inner_cat = CallFunction(
    aten.cat.default,
    [CallFunction(aten.sin.default, base), CallFunction(aten.cos.default, base)],
    -1,
    _users=2,  # the cat feeds both slices
)

slice_hi = CallFunction(aten.slice.Tensor, inner_cat, 1, split_idx, 9223372036854775807)

slice_lo = CallFunction(aten.slice.Tensor, inner_cat, 1, 0, split_idx)

pattern = CallFunction(aten.cat.default, [slice_hi, slice_lo], -1)


@register_graph_pattern(pattern, pass_dict=_sin_cos_matcher)
def _rewrite_sin_cos_cat(match: Match, base, split_idx) -> None:
    print(f">>> my rewrite sin cos cat: {match.nodes}")

    sin_node, cos_node, inner_cat, slice_hi, slice_lo, cat_node = match.nodes

    graph = cat_node.graph

    # if not match.nodes:
    #     print(">>> no nodes")
    #     return

    # cat_node = match.nodes[0]
    # graph = cat_node.graph

    # normalized = normalize_function(
    #     cat_node.target,
    #     cat_node.args,
    #     cat_node.kwargs,  # type: ignore[arg-type]
    # )
    # if normalized is None:
    #     list_arg = cat_node.args[0] if cat_node.args else cat_node.kwargs.get("tensors")
    #     dim_value = (
    #         cat_node.args[1]
    #         if len(cat_node.args) > 1
    #         else cat_node.kwargs.get("dim", -1)
    #     )
    # else:
    #     norm_args, norm_kwargs = normalized
    #     list_arg = norm_args[0] if len(norm_args) > 0 else norm_kwargs.get("tensors")
    #     dim_value = norm_args[1] if len(norm_args) > 1 else norm_kwargs.get("dim", -1)

    # if dim_value is None:
    #     dim_value = -1

    # slices = _as_tuple(list_arg)
    # if slices is None or len(slices) != 2:
    #     print(">>> no slices")
    #     return

    # slice_hi, slice_lo = slices
    # if not (
    #     isinstance(slice_hi, fx.Node)
    #     and isinstance(slice_lo, fx.Node)
    #     and slice_hi.op == slice_lo.op == "call_function"
    #     and slice_hi.target is slice_lo.target is torch.ops.aten.slice.Tensor
    #     and len(slice_hi.args) >= 4
    #     and len(slice_lo.args) >= 4
    #     and slice_hi.args[0] is slice_lo.args[0]
    #     and isinstance(slice_hi.args[1], int)
    #     and slice_hi.args[1] == 1
    #     and isinstance(slice_lo.args[1], int)
    #     and slice_lo.args[1] == 1
    #     and slice_hi.args[2] is split_idx
    #     and _is_bigint(slice_hi.args[3])
    #     and isinstance(slice_lo.args[2], int)
    #     and slice_lo.args[2] == 0
    #     and slice_lo.args[3] is split_idx
    # ):
    #     print(">>> no slice hi or slice lo")
    #     return

    # inner_cat = slice_hi.args[0]
    # if not (
    #     isinstance(inner_cat, fx.Node)
    #     and inner_cat.op == "call_function"
    #     and inner_cat.target is torch.ops.aten.cat.default
    # ):
    #     print(">>> no inner cat")
    #     return

    # inner_normalized = normalize_function(
    #     inner_cat.target,
    #     inner_cat.args,
    #     inner_cat.kwargs,  # type: ignore[arg-type]
    # )
    # if inner_normalized is None:
    #     inner_list = (
    #         inner_cat.args[0] if inner_cat.args else inner_cat.kwargs.get("tensors")
    #     )
    #     inner_dim = (
    #         inner_cat.args[1]
    #         if len(inner_cat.args) > 1
    #         else inner_cat.kwargs.get("dim", -1)
    #     )
    # else:
    #     inner_args, inner_kwargs = inner_normalized
    #     inner_list = (
    #         inner_args[0] if len(inner_args) > 0 else inner_kwargs.get("tensors")
    #     )
    #     inner_dim = (
    #         inner_args[1] if len(inner_args) > 1 else inner_kwargs.get("dim", -1)
    #     )

    # bases = _as_tuple(inner_list)
    # if bases is None or len(bases) != 2:
    #     print(">>> no bases")
    #     return

    # sin_node, cos_node = bases
    # if not (
    #     isinstance(sin_node, fx.Node)
    #     and sin_node.op == "call_function"
    #     and sin_node.target is torch.ops.aten.sin.default
    #     and len(sin_node.args) == 1
    #     and sin_node.args[0] is base
    #     and isinstance(cos_node, fx.Node)
    #     and cos_node.op == "call_function"
    #     and cos_node.target is torch.ops.aten.cos.default
    #     and len(cos_node.args) == 1
    #     and cos_node.args[0] is base
    # ):
    #     print(">>> no sin or cos node")
    #     return

    # if inner_dim != -1:
    #     print(">>> inner dim is not -1")
    #     return

    with graph.inserting_before(cat_node):
        new_cat = graph.call_function(torch.ops.aten.cat.default, args=([cos_node, sin_node], -1))
        new_cat.meta.update(cat_node.meta)

    logger.info(f"SinCosCatPattern: rewrite applied at {cat_node}")
    cat_node.replace_all_uses_with(new_cat)

    for node in (slice_hi, slice_lo):
        if len(node.users) == 0 and node in graph.nodes:
            graph.erase_node(node)

    if len(inner_cat.users) == 0 and inner_cat in graph.nodes:
        graph.erase_node(inner_cat)

    if len(cat_node.users) == 0 and cat_node in graph.nodes:
        graph.erase_node(cat_node)

    graph.eliminate_dead_code()


class PassManager(InductorPass):
    def __init__(self, passes: Iterable[Callable[[fx.Graph], None]]):
        self.passes: List[Callable[[fx.Graph], None]] = list(passes)

    def __call__(self, graph: torch.fx.Graph) -> None:
        print(">>> pass manager")
        for p in self.passes:
            print(">>> pass manager pass name: ", p.__class__.__name__)
            p(graph)
        print(f"after pass manager: {graph}")

    def uuid(self) -> str:
        state = {f"{type(p).__name__}_{idx}": p.uuid() for idx, p in enumerate(self.passes)}
        return InductorPass.hash_dict(state)


def init_backend():
    passes = [SinCosCatPatternPass()]
    pass_manager = PassManager(passes)

    def backend(
        gm: fx.GraphModule,
        example_inputs: Sequence[Optional[Union[torch.Tensor, int, torch.SymInt]]],
    ) -> Any:
        previous = torch._inductor.config.post_grad_custom_post_pass
        torch._inductor.config.post_grad_custom_post_pass = pass_manager
        try:
            from torch._inductor.compile_fx import compile_fx

            return compile_fx(gm, example_inputs)
        finally:
            torch._inductor.config.post_grad_custom_post_pass = previous

    return backend


# TODO(yupu): Check if we need to support multiple modules with different passes.
class TorchCompileTransformation(Transformation):
    def __init__(
        self, options: Optional[Dict[str, Any]] = None, targets: Optional[DictConfig] = None
    ):
        super().__init__()
        self._options: CompileOptions = CompileOptions(**(options or {}))
        self._selector: Selector = build_selector(targets)

    def preflight(self) -> bool:
        return torch.__version__ >= "2.6"

    def targets(self, scope: nn.Module) -> Iterable[Tuple[str, nn.Module]]:
        return self._selector(scope)

    def apply(self, module: nn.Module) -> bool:
        logger.info(f"Applying torch compile for {module.__class__.__name__}")

        if self._options.disable or isinstance(module, OptimizedModule):
            logger.info(
                f"Skipping torch compile for {module.__class__.__name__}: disabled or already compiled"
            )
            return True

        module.compile(
            backend=init_backend(),
            mode=self._options.mode,
            dynamic=self._options.dynamic,
            fullgraph=self._options.fullgraph,
        )

        return True
