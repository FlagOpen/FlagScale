from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Set, Tuple

import torch.nn as nn

from flagscale.models.adapters import BaseAdapter
from flagscale.runner.utils import logger
from flagscale.transforms.transform import Transform, TransformPhase


@dataclass
class _PhasePlan:
    """
    A plan of transforms to be applied to the model by phase.
    """

    # Transforms to be applied before `torch.compile`
    pre: List[Transform]
    # The `torch.compile` transform
    compile: List[Transform]
    # Transforms to be applied after `torch.compile`
    post: List[Transform]


# TODO(yupu): Optionally support `strict=False` mode, where invalid transforms are pruned along with dependents
class TransformManager:
    """Orders and executes transforms by phase: pre_compile → compile → post_compile"""

    def __init__(self, transforms: Sequence[Transform]) -> None:
        # Mapping: name -> transform
        self._all: Dict[str, Transform] = {t.spec().name: t for t in transforms}

    def _partition(
        self, names: Set[str]
    ) -> Tuple[List[Transform], List[Transform], List[Transform]]:
        """Partition the transforms into pre-compile, compile, and post-compile phases.

        Args:
            names: The names of the transforms to partition.

        Returns:
            A tuple of lists of transforms for each phase.
        """

        pre: List[Transform] = []
        comp: List[Transform] = []
        post: List[Transform] = []

        for n in names:
            t = self._all[n]
            ph = t.spec().phase
            if ph == "pre_compile":
                pre.append(t)
            elif ph == "compile":
                comp.append(t)
            elif ph == "post_compile":
                post.append(t)
            else:
                raise ValueError(f"Unknown phase: {ph}")

        return pre, comp, post

    def _validate(self, model: nn.Module | BaseAdapter | List[nn.Module]) -> Set[str]:
        """Check if the given transforms can be applied to the model.

        Args:
            model: The model to apply the transforms to.

        Returns:
            A set of valid transform names.
        """

        names: Set[str] = set(self._all.keys())
        name_to_phase: Dict[str, TransformPhase] = {n: self._all[n].spec().phase for n in self._all}

        for n in list(names):
            t = self._all[n]

            # Check if the transform's requirements are met
            ok = t.preflight()
            if not ok:
                # TODO(yupu): Specify the reason
                raise ValueError(f"Transform {n} not supported.")

            # Check if the transform supports the model
            ok = t.supports(model)
            if not ok:
                raise ValueError(f"Transform {n} not supported for this model.")

            # Check if the required transforms are present
            reqs = t.spec().requires
            missing = [r for r in reqs if r not in names]
            if missing:
                raise ValueError(f"Transform {n} requires missing transforms: {missing}")

            # Check if the required transforms are in the same phase or earlier
            # allowed: req same phase or earlier; not allowed: req in later phase
            rank = {"pre_compile": 0, "compile": 1, "post_compile": 2}
            for r in reqs:
                dep_phase = name_to_phase[n]
                req_phase = name_to_phase[r]
                if rank[req_phase] > rank[dep_phase]:
                    raise ValueError(
                        f"Transform {n} requires {r} but {r} is in a later phase ({req_phase} > {dep_phase})"
                    )

            # Check if the forbidden transforms are present
            for f in t.spec().forbids:
                if f in names:
                    raise ValueError(f"Transform {n} and Transform {f} cannot be applied together")

        return names

    def _sort_phase(self, transforms: List[Transform]) -> List[Transform]:
        """
        Topological sort the transforms in the same phase.

        Args:
            transforms: The transforms to sort.

        Returns:
            A list of sorted transforms.
        """

        if not transforms:
            return []

        name_to_t: Dict[str, Transform] = {t.spec().name: t for t in transforms}
        indeg: Dict[str, int] = {n: 0 for n in name_to_t}
        adj: Dict[str, Set[str]] = {n: set() for n in name_to_t}

        # Build edges
        for t in transforms:
            ts = t.spec()
            # Requires edge: req -> t (within phase only; cross-phase validated earlier)
            for r in ts.requires:
                # Just to be safe, avoid self-loop
                if r in name_to_t and r != ts.name:
                    adj[r].add(ts.name)
                    indeg[ts.name] += 1
            # After: x -> t
            for x in ts.after:
                # Just to be safe, avoid self-loop
                if x in name_to_t and x != ts.name:
                    adj[x].add(ts.name)
                    indeg[ts.name] += 1
            # Before: t -> x
            for x in ts.before:
                # Just to be safe, avoid self-loop
                if x in name_to_t and x != ts.name:
                    adj[ts.name].add(x)
                    indeg[x] += 1

        # Topological sort with priority tie-breaker
        def prio_key(n: str) -> Tuple[int, str]:
            s = name_to_t[n].spec()
            return (s.priority, n)

        # The ready queue with indegree 0
        ready: List[str] = [n for n, d in indeg.items() if d == 0]
        ready.sort(key=prio_key, reverse=True)
        order: List[str] = []

        while ready:
            n = ready.pop(0)
            order.append(n)
            for m in sorted(adj[n]):
                indeg[m] -= 1
                if indeg[m] == 0:
                    ready.append(m)
                    ready.sort(key=prio_key, reverse=True)

        if len(order) != len(name_to_t):
            missing = set(name_to_t) - set(order)
            raise ValueError(f"Cyclic constraints in phase among: {sorted(missing)}")

        return [name_to_t[n] for n in order]

    def plan(self, model: nn.Module | BaseAdapter | List[nn.Module]) -> _PhasePlan:
        """Plan the transforms to be applied to the model.

        Args:
            model: The model to apply the transforms to.

        Returns:
            A plan of transforms to be applied to the model by phase.
        """

        active = self._validate(model)
        pre, comp, post = self._partition(active)

        return _PhasePlan(
            pre=self._sort_phase(pre), compile=self._sort_phase(comp), post=self._sort_phase(post)
        )

    def apply(
        self, model: nn.Module | BaseAdapter | List[nn.Module], *, dry_run: bool = False
    ) -> None | str:
        """Apply the transforms in the order specified by the plan.

        Args:
            model: The model to apply the transforms to. model can be a single nn.Module, BaseAdapter, or List[nn.Module].
            dry_run: If True, only plan the transforms and return the plan.

        Returns:
            None if dry_run is False, otherwise a string describing the plan.
        """

        plan = self.plan(model)
        if not dry_run:
            # TODO(yupu): Check if the transform is applied successfully
            for t in plan.pre:
                t.apply(model)
            for t in plan.compile:
                t.apply(model)
            for t in plan.post:
                t.apply(model)
        else:
            info = ""
            info += "Pre-compile transforms:\n"
            for t in plan.pre:
                info += f"  {t.spec().name}\n"
            info += "Compile transforms:\n"
            for t in plan.compile:
                info += f"  {t.spec().name}\n"
            info += "Post-compile transforms:\n"
            for t in plan.post:
                info += f"  {t.spec().name}\n"

            dry_run_info = f"DryRun plan(\n{info})"
            logger.info(dry_run_info)

            return dry_run_info
