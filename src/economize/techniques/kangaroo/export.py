"""The exported Kangaroo drafter: a backend-neutral ``economize_kangaroo`` block and the adapter's tensors.

A draft step of :class:`~economize.techniques.kangaroo.KangarooTechnique` runs the model's layers ``1..l`` on the
token's embedding, then the adapter (one attention layer over its own keys and values and two RMS norms) and the
model's LM head without the final norm. The first part is the jump drafter's path with ``m = u = 0``, the
parameter-free ``"embedding"`` predictor and ``r = l`` (:mod:`economize.techniques.jump.export`), so the export
nests a jump specification in its block and every engine that serves jump drafters reuses that path.

The block holds:

* ``jump``: the nested :class:`~economize.techniques.jump.export.JumpDrafterSpec` block (the path ``1..l``);
* the adapter's geometry: ``num_heads``, ``num_kv_heads``, ``head_dim``, ``rms_norm_eps``, ``attention_bias``;
* ``rope_layer``: the model layer whose rotary table the adapter uses (the last full-attention layer, else 1).

The jump block is nested (and never written at the top level) so that an engine dispatching on the top-level
``economize_jump`` key does not mistake a Kangaroo export for a plain jump drafter. The adapter's tensors are keyed
``kangaroo.<name>`` as in :class:`~economize.techniques.kangaroo.KangarooAdapter`'s ``state_dict``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from ...errors import ConfigError
from ..jump.config import JumpConfig
from ..jump.export import JumpDrafterSpec

__all__ = ["CONFIG_KEY", "FORMAT_VERSION", "WEIGHTS_PREFIX", "KangarooDrafterSpec", "adapter_state"]

#: Key of the Kangaroo block in ``config.json``.
CONFIG_KEY = "economize_kangaroo"
#: Prefix of the adapter's tensors in ``model.safetensors``.
WEIGHTS_PREFIX = "kangaroo."
#: Highest version of the block this economize reads.
FORMAT_VERSION = 1


def _path_config(exit_layer: int) -> JumpConfig:
    """The jump configuration of the shallow path ``1..l`` (``m = u = 0``, the embedding predictor, frozen)."""
    return JumpConfig(target_layer=0, source_layer=0, exit_layer=exit_layer, predictor="embedding",
                      adaptation="frozen")


@dataclass(frozen=True)
class KangarooDrafterSpec:
    """The ``economize_kangaroo`` block (see the module docstring).

    Attributes:
        path: The shallow path as a jump drafter specification (``m = u = 0``, ``r = l``).
        num_heads: Attention heads of the adapter.
        num_kv_heads: Key/value heads of the adapter.
        head_dim: Head width of the adapter.
        rms_norm_eps: Epsilon of the adapter's two norms.
        attention_bias: Whether the q/k/v projections have biases.
        rope_layer: The model layer (1-indexed) whose rotary table rotates the adapter's queries and keys.
    """

    path: JumpDrafterSpec
    num_heads: int
    num_kv_heads: int
    head_dim: int
    rms_norm_eps: float
    attention_bias: bool
    rope_layer: int

    def __post_init__(self) -> None:
        if self.num_heads % self.num_kv_heads:
            raise ConfigError(f"num_heads {self.num_heads} is not a multiple of num_kv_heads {self.num_kv_heads}")
        if not 1 <= self.rope_layer <= self.path.num_layers:
            raise ConfigError(f"rope_layer {self.rope_layer} is outside the model's {self.path.num_layers} layers")

    @property
    def exit_layer(self) -> int:
        """``l``, the last model layer of a draft step."""
        return self.path.exit_layer

    def to_dict(self) -> dict[str, Any]:
        return {
            "format_version": FORMAT_VERSION,
            "jump": self.path.to_dict(),
            "num_heads": self.num_heads,
            "num_kv_heads": self.num_kv_heads,
            "head_dim": self.head_dim,
            "rms_norm_eps": self.rms_norm_eps,
            "attention_bias": self.attention_bias,
            "rope_layer": self.rope_layer,
        }

    @classmethod
    def from_dict(cls, block: dict[str, Any]) -> KangarooDrafterSpec:
        version = int(block.get("format_version", 1))
        if version > FORMAT_VERSION:
            raise ConfigError(f"the {CONFIG_KEY!r} block has format version {version}; this economize reads up to "
                              f"{FORMAT_VERSION}: upgrade economize")
        return cls(
            path=JumpDrafterSpec.from_dict(block["jump"]),
            num_heads=int(block["num_heads"]),
            num_kv_heads=int(block["num_kv_heads"]),
            head_dim=int(block["head_dim"]),
            rms_norm_eps=float(block["rms_norm_eps"]),
            attention_bias=bool(block["attention_bias"]),
            rope_layer=int(block["rope_layer"]),
        )

    @classmethod
    def from_hf_config(cls, config: Any) -> KangarooDrafterSpec:
        """Read the block from a configuration object or dict (top level, or in ``transformer_layer_config``)."""
        data = config if isinstance(config, dict) else getattr(config, "__dict__", {})
        nested = data.get("transformer_layer_config") or {}
        block = data.get(CONFIG_KEY) or nested.get(CONFIG_KEY)
        if not block:
            raise ConfigError(f"the configuration has no {CONFIG_KEY!r} block: export it with economize")
        return cls.from_dict(block)

    @classmethod
    def of(cls, technique: Any) -> KangarooDrafterSpec:
        """The spec of an attached :class:`~economize.techniques.kangaroo.KangarooTechnique`."""
        adapter, net = technique.adapter, technique.adapter_net
        full = _last_full_attention_layer(adapter)
        return cls(
            path=JumpDrafterSpec(_path_config(technique.exit_layer), adapter.num_layers, adapter.hidden_size),
            num_heads=net.num_heads,
            num_kv_heads=net.num_kv_heads,
            head_dim=net.head_dim,
            rms_norm_eps=float(net.input_layernorm.eps),
            attention_bias=net.q_proj.bias is not None,
            rope_layer=full or 1,
        )


def _last_full_attention_layer(adapter: Any) -> int | None:
    from .technique import _last_full_attention_layer as find

    return find(adapter)


def adapter_state(technique: Any) -> dict[str, torch.Tensor]:
    """The adapter's tensors (detached), keyed as in :class:`~economize.techniques.kangaroo.KangarooAdapter`."""
    return {name: tensor.detach() for name, tensor in technique.adapter_net.state_dict().items()}
