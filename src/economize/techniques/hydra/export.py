"""The exported Hydra drafter: a backend-neutral ``economize_hydra`` block and the technique's tensors.

A draft round of :class:`~economize.techniques.hydra.HydraTechnique` reads the base state ``P_t`` of the last
verified token ``t`` (the model's last hidden state, after the final norm with ``head="own"``, or the output of the
Hydra++ prefix layer over those states) and runs head ``j`` on ``[P_t; E[x_{t+1}]; ...; E[x_{t+j}]]``, the embeddings
of the sampled token and the ``j - 1`` tokens drafted before it. Only the prefix layer attends (over the base states
of the verified tokens); the heads are MLPs, so an engine needs no attention for the draft steps.

The block holds:

* ``num_layers``, ``hidden_size``, ``vocab_size``: the model's depth, width and vocabulary;
* ``num_heads`` (``K``, the most draft tokens per round), ``head_layers`` (residual blocks per head), ``head``
  (``"own"``: each head has its own vocabulary projection; ``"shared"``: the heads end in the model's final norm and
  LM head) and ``param_dtype`` (``"float32"`` or ``"model"``);
* ``prefix`` and, with a prefix layer, its geometry: ``num_attention_heads``, ``num_kv_heads``, ``head_dim``,
  ``intermediate_size``, ``rms_norm_eps``, ``attention_bias`` and ``rope_layer`` (the model layer whose rotary table
  the prefix layer uses: the last full-attention layer, else 1).

The tensors are keyed ``hydra.<name>`` as in the technique's ``state_dict`` (``prefix.*``, ``heads.{j}.blocks.{i}.*``,
``heads.{j}.proj.*``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

import torch

from ...errors import ConfigError

__all__ = ["CONFIG_KEY", "FORMAT_VERSION", "WEIGHTS_PREFIX", "HydraDrafterSpec", "drafter_state"]

#: Key of the Hydra block in ``config.json``.
CONFIG_KEY = "economize_hydra"
#: Prefix of the technique's tensors in ``model.safetensors``.
WEIGHTS_PREFIX = "hydra."
#: Highest version of the block this economize reads.
FORMAT_VERSION = 1

_HEADS = ("own", "shared")
_DTYPES = ("float32", "model")


@dataclass(frozen=True)
class HydraDrafterSpec:
    """The ``economize_hydra`` block (see the module docstring).

    Attributes:
        num_layers: Depth of the model.
        hidden_size: Width ``d`` of the model.
        vocab_size: Rows of the model's LM head.
        num_heads: ``K``, the number of heads.
        head_layers: Residual blocks per head.
        head: ``"own"`` or ``"shared"``.
        param_dtype: ``"float32"`` or ``"model"``.
        prefix: Whether the base state is the output of the prefix layer (Hydra++).
        num_attention_heads: Attention heads of the prefix layer (0 without one).
        num_kv_heads: Key/value heads of the prefix layer (0 without one).
        head_dim: Head width of the prefix layer (0 without one).
        intermediate_size: Width of the prefix layer's gated MLP (0 without one).
        rms_norm_eps: Epsilon of the prefix layer's norms.
        attention_bias: Whether the prefix layer's q/k/v projections have biases.
        rope_layer: The model layer (1-indexed) whose rotary table rotates the prefix layer's queries and keys.
    """

    num_layers: int
    hidden_size: int
    vocab_size: int
    num_heads: int
    head_layers: int
    head: str
    param_dtype: str
    prefix: bool = False
    num_attention_heads: int = 0
    num_kv_heads: int = 0
    head_dim: int = 0
    intermediate_size: int = 0
    rms_norm_eps: float = 1e-6
    attention_bias: bool = False
    rope_layer: int = 1

    def __post_init__(self) -> None:
        if self.num_heads < 1 or self.head_layers < 1:
            raise ConfigError(f"a Hydra drafter needs num_heads >= 1 and head_layers >= 1, got {self.num_heads} and "
                              f"{self.head_layers}")
        if self.head not in _HEADS:
            raise ConfigError(f"head must be one of {_HEADS}, got {self.head!r}")
        if self.param_dtype not in _DTYPES:
            raise ConfigError(f"param_dtype must be one of {_DTYPES}, got {self.param_dtype!r}")
        if not self.prefix:
            return
        if min(self.num_attention_heads, self.num_kv_heads, self.head_dim, self.intermediate_size) < 1:
            raise ConfigError("a Hydra++ prefix layer needs positive num_attention_heads, num_kv_heads, head_dim and "
                              "intermediate_size")
        if self.num_attention_heads % self.num_kv_heads:
            raise ConfigError(f"num_attention_heads {self.num_attention_heads} is not a multiple of num_kv_heads "
                              f"{self.num_kv_heads}")
        if not 1 <= self.rope_layer <= self.num_layers:
            raise ConfigError(f"rope_layer {self.rope_layer} is outside the model's {self.num_layers} layers")

    def to_dict(self) -> dict[str, Any]:
        return {
            "format_version": FORMAT_VERSION,
            "num_layers": self.num_layers,
            "hidden_size": self.hidden_size,
            "vocab_size": self.vocab_size,
            "num_heads": self.num_heads,
            "head_layers": self.head_layers,
            "head": self.head,
            "param_dtype": self.param_dtype,
            "prefix": self.prefix,
            "num_attention_heads": self.num_attention_heads,
            "num_kv_heads": self.num_kv_heads,
            "head_dim": self.head_dim,
            "intermediate_size": self.intermediate_size,
            "rms_norm_eps": self.rms_norm_eps,
            "attention_bias": self.attention_bias,
            "rope_layer": self.rope_layer,
        }

    @classmethod
    def from_dict(cls, block: dict[str, Any]) -> HydraDrafterSpec:
        version = int(block.get("format_version", 1))
        if version > FORMAT_VERSION:
            raise ConfigError(f"the {CONFIG_KEY!r} block has format version {version}; this economize reads up to "
                              f"{FORMAT_VERSION}: upgrade economize")
        return cls(
            num_layers=int(block["num_layers"]),
            hidden_size=int(block["hidden_size"]),
            vocab_size=int(block["vocab_size"]),
            num_heads=int(block["num_heads"]),
            head_layers=int(block["head_layers"]),
            head=str(block["head"]),
            param_dtype=str(block["param_dtype"]),
            prefix=bool(block.get("prefix", False)),
            num_attention_heads=int(block.get("num_attention_heads", 0)),
            num_kv_heads=int(block.get("num_kv_heads", 0)),
            head_dim=int(block.get("head_dim", 0)),
            intermediate_size=int(block.get("intermediate_size", 0)),
            rms_norm_eps=float(block.get("rms_norm_eps", 1e-6)),
            attention_bias=bool(block.get("attention_bias", False)),
            rope_layer=int(block.get("rope_layer", 1)),
        )

    @classmethod
    def from_hf_config(cls, config: Any) -> HydraDrafterSpec:
        """Read the block from a configuration object or dict (top level, or in ``transformer_layer_config``)."""
        data = config if isinstance(config, dict) else getattr(config, "__dict__", {})
        nested = data.get("transformer_layer_config") or {}
        block = data.get(CONFIG_KEY) or nested.get(CONFIG_KEY)
        if not block:
            raise ConfigError(f"the configuration has no {CONFIG_KEY!r} block: export it with economize")
        return cls.from_dict(block)

    @classmethod
    def of(cls, technique: Any) -> HydraDrafterSpec:
        """The spec of an attached :class:`~economize.techniques.hydra.HydraTechnique`."""
        adapter, config = technique.adapter, technique.config
        lm_weight = cast(torch.Tensor, adapter.lm_head.weight)
        common: dict[str, Any] = {
            "num_layers": adapter.num_layers,
            "hidden_size": adapter.hidden_size,
            "vocab_size": int(lm_weight.shape[0]),
            "num_heads": int(config.num_heads),
            "head_layers": len(technique.heads[0].blocks),
            "head": str(config.head),
            "param_dtype": str(config.param_dtype),
        }
        prefix = technique.prefix
        if prefix is None:
            return cls(**common)
        return cls(
            **common,
            prefix=True,
            num_attention_heads=prefix.num_heads,
            num_kv_heads=prefix.num_kv_heads,
            head_dim=prefix.head_dim,
            intermediate_size=int(prefix.mlp.gate_proj.out_features),
            rms_norm_eps=float(prefix.input_layernorm.eps),
            attention_bias=prefix.q_proj.bias is not None,
            rope_layer=_last_full_attention_layer(adapter) or 1,
        )


def _last_full_attention_layer(adapter: Any) -> int | None:
    from ..eagle3.technique import _last_full_attention_layer as find

    return find(adapter)


def drafter_state(technique: Any) -> dict[str, torch.Tensor]:
    """The technique's tensors (detached), keyed as in its ``state_dict`` (``prefix.*``, ``heads.*``)."""
    return {name: tensor.detach() for name, tensor in technique.state_dict().items()}
