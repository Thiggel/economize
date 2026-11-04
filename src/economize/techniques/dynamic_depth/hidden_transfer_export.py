"""The exported Hidden Transfer drafter: a backend-neutral ``economize_hidden_transfer`` block and the maps' tensors.

After a pass that ended at position ``n``, draft step ``i`` (``i = 1..k``) of
:class:`~economize.techniques.dynamic_depth.HiddenTransferTechnique` maps the model's state after layer ``t_i`` at
``n`` by ``W_i`` to a pseudo state at ``n + i``, runs layers ``t_i + 1..L`` on it (attending to the cache) and drafts
token ``n + i + 1`` with the model's final norm and LM head. Every step reads states of the same position ``n``;
none reads the drafted tokens.

The layers a step runs are a suffix of the jump drafter's path with ``m = u = t_1``, ``r = L`` and the
parameter-free ``"embedding"`` predictor (:mod:`economize.techniques.jump.export`), so the block nests that jump
specification (``path``) and an engine that serves jump drafters reuses the path's layers, sharing and attention
metadata; step ``i`` starts at layer ``t_i + 1`` of it.

The block holds:

* ``jump``: the nested :class:`~economize.techniques.jump.export.JumpDrafterSpec` block (the path ``t_1+1..L``);
* ``transfer_layers``: ``t_1..t_k`` (1-indexed, nondecreasing, below ``L``);
* ``param_dtype``: ``"float32"`` or ``"model"``, the dtype the maps run in.

The jump block is nested (and never written at the top level) so that an engine dispatching on the top-level
``economize_jump`` key does not mistake the export for a plain jump drafter. The maps' tensors are keyed
``hidden_transfer.transfers.{i}.weight`` (``i = 0..k-1``) as in the technique's ``state_dict``.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import pairwise
from typing import Any

import torch

from ...errors import ConfigError
from ..jump.config import JumpConfig
from ..jump.export import JumpDrafterSpec

__all__ = ["CONFIG_KEY", "FORMAT_VERSION", "WEIGHTS_PREFIX", "HiddenTransferDrafterSpec", "drafter_state"]

#: Key of the Hidden Transfer block in ``config.json``.
CONFIG_KEY = "economize_hidden_transfer"
#: Prefix of the maps' tensors in ``model.safetensors``.
WEIGHTS_PREFIX = "hidden_transfer."
#: Highest version of the block this economize reads.
FORMAT_VERSION = 1

_DTYPES = ("float32", "model")


def _path_config(first_layer: int) -> JumpConfig:
    """The jump configuration of the path ``t_1+1..L`` (``m = u = t_1``, the embedding predictor, frozen)."""
    return JumpConfig(target_layer=first_layer, source_layer=first_layer, exit_layer=None, predictor="embedding",
                      adaptation="frozen")


@dataclass(frozen=True)
class HiddenTransferDrafterSpec:
    """The ``economize_hidden_transfer`` block (see the module docstring).

    Attributes:
        num_layers: Depth ``L`` of the model.
        hidden_size: Width ``d`` of the model.
        transfer_layers: ``t_1..t_k``: step ``i`` maps the state after layer ``t_i``.
        param_dtype: ``"float32"`` (the maps run in float32) or ``"model"`` (in the model's dtype).
    """

    num_layers: int
    hidden_size: int
    transfer_layers: tuple[int, ...]
    param_dtype: str = "float32"

    def __post_init__(self) -> None:
        layers = tuple(int(layer) for layer in self.transfer_layers)
        object.__setattr__(self, "transfer_layers", layers)  # frozen dataclass: normalize once
        if not layers:
            raise ConfigError("transfer_layers must name at least one layer")
        if any(not 1 <= layer < self.num_layers for layer in layers):
            raise ConfigError(f"transfer_layers must lie in 1..{self.num_layers - 1}, got {list(layers)}")
        if any(b < a for a, b in pairwise(layers)):
            raise ConfigError(f"transfer_layers must be nondecreasing, got {list(layers)}")
        if self.param_dtype not in _DTYPES:
            raise ConfigError(f"param_dtype must be one of {_DTYPES}, got {self.param_dtype!r}")

    @property
    def num_steps(self) -> int:
        """``k``, the most draft tokens per round."""
        return len(self.transfer_layers)

    @property
    def source_layers(self) -> tuple[int, ...]:
        """The distinct transfer layers, ascending: the states a round reads (the export's auxiliary states)."""
        return tuple(sorted(set(self.transfer_layers)))

    @property
    def path(self) -> JumpDrafterSpec:
        """The layers ``t_1+1..L`` as a jump drafter specification (``m = u = t_1``, ``r = L``)."""
        return JumpDrafterSpec(_path_config(self.transfer_layers[0]), self.num_layers, self.hidden_size)

    def first_layer(self, step: int) -> int:
        """The first layer draft step ``step`` (1-based) runs: ``t_step + 1``."""
        if not 1 <= step <= self.num_steps:
            raise ConfigError(f"draft step {step} is outside 1..{self.num_steps}")
        return self.transfer_layers[step - 1] + 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "format_version": FORMAT_VERSION,
            "jump": self.path.to_dict(),
            "transfer_layers": list(self.transfer_layers),
            "param_dtype": self.param_dtype,
        }

    @classmethod
    def from_dict(cls, block: dict[str, Any]) -> HiddenTransferDrafterSpec:
        version = int(block.get("format_version", 1))
        if version > FORMAT_VERSION:
            raise ConfigError(f"the {CONFIG_KEY!r} block has format version {version}; this economize reads up to "
                              f"{FORMAT_VERSION}: upgrade economize")
        path = JumpDrafterSpec.from_dict(block["jump"])
        spec = cls(num_layers=path.num_layers, hidden_size=path.hidden_size,
                   transfer_layers=tuple(block["transfer_layers"]), param_dtype=str(block.get("param_dtype",
                                                                                             "float32")))
        if (path.target_layer, path.exit_layer, path.skip_layers) != (spec.transfer_layers[0], spec.num_layers, ()):
            raise ConfigError(f"the {CONFIG_KEY!r} block's jump path ({path.target_layer + 1}..{path.exit_layer}) "
                              f"does not start after its first transfer layer {spec.transfer_layers[0]}")
        return spec

    @classmethod
    def from_hf_config(cls, config: Any) -> HiddenTransferDrafterSpec:
        """Read the block from a configuration object or dict (top level, or in ``transformer_layer_config``)."""
        data = config if isinstance(config, dict) else getattr(config, "__dict__", {})
        nested = data.get("transformer_layer_config") or {}
        block = data.get(CONFIG_KEY) or nested.get(CONFIG_KEY)
        if not block:
            raise ConfigError(f"the configuration has no {CONFIG_KEY!r} block: export it with economize")
        return cls.from_dict(block)

    @classmethod
    def of(cls, technique: Any) -> HiddenTransferDrafterSpec:
        """The spec of an attached :class:`~economize.techniques.dynamic_depth.HiddenTransferTechnique`."""
        adapter = technique.adapter
        return cls(num_layers=adapter.num_layers, hidden_size=adapter.hidden_size,
                   transfer_layers=tuple(technique.transfer_layers), param_dtype=technique.config.param_dtype)


def drafter_state(technique: Any) -> dict[str, torch.Tensor]:
    """The maps' tensors (detached), keyed as in the technique's ``state_dict`` (``transfers.{i}.weight``)."""
    return {name: tensor.detach() for name, tensor in technique.state_dict().items()}
