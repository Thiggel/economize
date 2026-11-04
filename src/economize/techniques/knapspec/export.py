"""The exported KnapSpec drafter: one fixed skip set as a backend-neutral ``economize_knapspec`` block.

A draft step of :class:`~economize.techniques.knapspec.KnapSpecTechnique` embeds the token, runs the model with the
attention of some layers and the MLP of others skipped, and reads logits with the model's final norm and head. A
layer with both sublayers skipped is skipped whole, so the step is the jump drafter's path with ``m = u = 0``, the
parameter-free ``"embedding"`` predictor and the whole-skipped layers as holes
(:mod:`economize.techniques.layer_skip.export`), plus two sets of partially run layers. The block nests that jump
specification and adds them:

* ``jump``: the nested :class:`~economize.techniques.jump.export.JumpDrafterSpec` block (holes = whole-skipped
  layers below ``r``; ``r`` = the last layer the step runs a sublayer of);
* ``skip_attention``: layers whose attention is skipped and whose MLP runs (``h + M(h)``);
* ``skip_mlp``: layers whose MLP is skipped and whose attention runs (``h + A(h)``);
* ``draft_length``: the draft length the search chose with this set (informational: engines draft their own ``k``).

The technique re-searches its skip set while decoding; an engine serves the one set that was current at export.
The jump block is nested (never written at the top level) so that an engine dispatching on ``economize_jump`` does
not run a KnapSpec export as a layer-skip drafter with every partial layer run whole. The drafter owns no tensors
except an identity readout when ``r`` is below the last layer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from ...errors import ConfigError
from ..jump.export import JumpDrafterSpec
from ..layer_skip.export import jump_config, jump_state_dict

__all__ = ["CONFIG_KEY", "FORMAT_VERSION", "KnapSpecDrafterSpec", "drafter_state"]

#: Key of the KnapSpec block in ``config.json``.
CONFIG_KEY = "economize_knapspec"
#: Highest version of the block this economize reads.
FORMAT_VERSION = 1


@dataclass(frozen=True)
class KnapSpecDrafterSpec:
    """The ``economize_knapspec`` block (see the module docstring).

    Attributes:
        path: The layers a draft step runs at least one sublayer of, as a jump specification (``m = u = 0``).
        skip_attention: Layers on the path whose attention is skipped (1-indexed, ascending).
        skip_mlp: Layers on the path whose MLP is skipped (1-indexed, ascending; disjoint from ``skip_attention``).
        draft_length: Draft tokens per cycle the search chose with this set.
    """

    path: JumpDrafterSpec
    skip_attention: tuple[int, ...] = ()
    skip_mlp: tuple[int, ...] = ()
    draft_length: int = 1

    def __post_init__(self) -> None:
        attention = tuple(sorted({int(i) for i in self.skip_attention}))
        mlp = tuple(sorted({int(i) for i in self.skip_mlp}))
        object.__setattr__(self, "skip_attention", attention)  # frozen dataclass: normalize once
        object.__setattr__(self, "skip_mlp", mlp)
        path = set(self.path.jump_layers)
        if not set(attention) <= path or not set(mlp) <= path:
            raise ConfigError(f"skip_attention {list(attention)} and skip_mlp {list(mlp)} must lie on the draft "
                              f"path {list(self.path.jump_layers)}")
        if set(attention) & set(mlp):
            raise ConfigError(f"layers {sorted(set(attention) & set(mlp))} skip both sublayers: they are holes of "
                              f"the path, not partial layers")
        if self.draft_length < 1:
            raise ConfigError(f"draft_length must be positive, got {self.draft_length}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "format_version": FORMAT_VERSION,
            "jump": self.path.to_dict(),
            "skip_attention": list(self.skip_attention),
            "skip_mlp": list(self.skip_mlp),
            "draft_length": self.draft_length,
        }

    @classmethod
    def from_dict(cls, block: dict[str, Any]) -> KnapSpecDrafterSpec:
        version = int(block.get("format_version", 1))
        if version > FORMAT_VERSION:
            raise ConfigError(f"the {CONFIG_KEY!r} block has format version {version}; this economize reads up to "
                              f"{FORMAT_VERSION}: upgrade economize")
        return cls(
            path=JumpDrafterSpec.from_dict(block["jump"]),
            skip_attention=tuple(int(i) for i in block.get("skip_attention", ())),
            skip_mlp=tuple(int(i) for i in block.get("skip_mlp", ())),
            draft_length=int(block.get("draft_length", 1)),
        )

    @classmethod
    def from_hf_config(cls, config: Any) -> KnapSpecDrafterSpec:
        """Read the block from a configuration object or dict (top level, or in ``transformer_layer_config``)."""
        data = config if isinstance(config, dict) else getattr(config, "__dict__", {})
        nested = data.get("transformer_layer_config") or {}
        block = data.get(CONFIG_KEY) or nested.get(CONFIG_KEY)
        if not block:
            raise ConfigError(f"the configuration has no {CONFIG_KEY!r} block: export it with economize")
        return cls.from_dict(block)

    @classmethod
    def of(cls, technique: Any) -> KnapSpecDrafterSpec:
        """The spec of the current skip set of an attached :class:`~economize.techniques.knapspec.KnapSpecTechnique`.

        Raises:
            ConfigError: The technique has no skip set yet (:attr:`~KnapSpecTechnique.plan` is ``None`` before the
                first search; a generation leaves the set of its last search in place, or set one by hand), or the
                set skips nothing.
        """
        plan = technique.plan
        if plan is None:
            raise ConfigError("the KnapSpec technique has no skip set yet: run a generation (its search sets "
                              "technique.plan) or assign a SkipPlan to technique.plan before exporting")
        adapter = technique.adapter
        run = plan.run_layers(adapter.num_layers)
        whole = plan.attention & plan.mlp
        holes = tuple(i for i in sorted(whole) if i < max(run))
        path = JumpDrafterSpec(jump_config(run), adapter.num_layers, adapter.hidden_size, skip_layers=holes)
        spec = cls(path, tuple(sorted(plan.attention - whole)), tuple(sorted(plan.mlp - whole)), plan.draft_length)
        if not (whole or spec.skip_attention or spec.skip_mlp):
            raise ConfigError("the KnapSpec skip set is empty: the draft step would be the full model")
        return spec


def drafter_state(technique: Any) -> dict[str, torch.Tensor]:
    """The drafter's tensors under the jump technique's names: an identity ``readout.*`` when ``r < L``, else none.

    Raises:
        ConfigError: As :meth:`KnapSpecDrafterSpec.of`.
    """
    KnapSpecDrafterSpec.of(technique)  # refuses a technique without a skip set (run_layers would be all layers)
    return jump_state_dict(technique)
