"""The exported ReDrafter drafter: a backend-neutral ``economize_redrafter`` block and the technique's tensors.

A draft round of :class:`~economize.techniques.redrafter.RedrafterTechnique` reads the context ``h_t`` of the last
verified token ``t`` (the model's last hidden state, after the final norm with ``head="own"``), starts the recurrent
state at ``s_0 = 0`` and, for ``k = 1, 2, ...``, sets ``s_k = SiLU(W E[x_{t+k}] + U s_{k-1} + b)`` and drafts token
``t + k + 1`` from the head on ``[h_t; s_k]`` (top-k warped). Nothing attends, so an engine needs no attention for
the draft steps. This block describes the chain; the beam search of ``beam_width > 1`` is a decoding choice of the
Hugging Face backend and is not part of the export.

The block holds:

* ``num_layers``, ``hidden_size``, ``vocab_size``: the model's depth, width and vocabulary;
* ``width`` (of the residual blocks, ``2 d`` without an input projection), ``head_layers`` (residual blocks),
  ``head`` (``"own"``: its own vocabulary projection; ``"shared"``: a map back to ``d`` read by the model's final norm
  and LM head), ``param_dtype`` (``"float32"`` or ``"model"``), ``top_k`` (the warp, ``0``: none) and ``num_drafts``
  (``beam_length - 1``, the most draft tokens per round).

The tensors are keyed ``redrafter.<name>`` as in the technique's ``state_dict`` (``rnn_w.*``, ``rnn_u.*``,
``input_proj.*``, ``blocks.{i}.linear.*``, ``proj.weight`` or ``down.weight``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

import torch

from ...errors import ConfigError

__all__ = ["CONFIG_KEY", "FORMAT_VERSION", "WEIGHTS_PREFIX", "RedrafterDrafterSpec", "drafter_state"]

#: Key of the ReDrafter block in ``config.json``.
CONFIG_KEY = "economize_redrafter"
#: Prefix of the technique's tensors in ``model.safetensors``.
WEIGHTS_PREFIX = "redrafter."
#: Highest version of the block this economize reads.
FORMAT_VERSION = 1

_HEADS = ("own", "shared")
_DTYPES = ("float32", "model")


@dataclass(frozen=True)
class RedrafterDrafterSpec:
    """The ``economize_redrafter`` block (see the module docstring).

    Attributes:
        num_layers: Depth of the model.
        hidden_size: Width ``d`` of the model and of the recurrent state.
        vocab_size: Rows of the model's LM head.
        width: Width of the residual blocks (an input projection maps ``[h_t; s_k]`` to it unless it is ``2 d``).
        head_layers: Residual blocks of the head (may be 0).
        head: ``"own"`` or ``"shared"``.
        param_dtype: ``"float32"`` or ``"model"``.
        top_k: The top-k warp of the head's logits (``0``: none).
        num_drafts: The most draft tokens per round (``beam_length - 1``).
    """

    num_layers: int
    hidden_size: int
    vocab_size: int
    width: int
    head_layers: int
    head: str
    param_dtype: str
    top_k: int
    num_drafts: int

    def __post_init__(self) -> None:
        if self.head_layers < 0 or self.width < 1 or self.num_drafts < 1 or self.top_k < 0:
            raise ConfigError(f"a ReDrafter drafter needs head_layers >= 0, width >= 1, num_drafts >= 1 and "
                              f"top_k >= 0, got {self.head_layers}, {self.width}, {self.num_drafts} and {self.top_k}")
        if self.head not in _HEADS:
            raise ConfigError(f"head must be one of {_HEADS}, got {self.head!r}")
        if self.param_dtype not in _DTYPES:
            raise ConfigError(f"param_dtype must be one of {_DTYPES}, got {self.param_dtype!r}")

    @property
    def input_projection(self) -> bool:
        """Whether an input projection maps ``[h_t; s_k]`` (``2 d``) to :attr:`width`."""
        return self.width != 2 * self.hidden_size

    def to_dict(self) -> dict[str, Any]:
        return {
            "format_version": FORMAT_VERSION,
            "num_layers": self.num_layers,
            "hidden_size": self.hidden_size,
            "vocab_size": self.vocab_size,
            "width": self.width,
            "head_layers": self.head_layers,
            "head": self.head,
            "param_dtype": self.param_dtype,
            "top_k": self.top_k,
            "num_drafts": self.num_drafts,
        }

    @classmethod
    def from_dict(cls, block: dict[str, Any]) -> RedrafterDrafterSpec:
        version = int(block.get("format_version", 1))
        if version > FORMAT_VERSION:
            raise ConfigError(f"the {CONFIG_KEY!r} block has format version {version}; this economize reads up to "
                              f"{FORMAT_VERSION}: upgrade economize")
        return cls(
            num_layers=int(block["num_layers"]),
            hidden_size=int(block["hidden_size"]),
            vocab_size=int(block["vocab_size"]),
            width=int(block["width"]),
            head_layers=int(block["head_layers"]),
            head=str(block["head"]),
            param_dtype=str(block["param_dtype"]),
            top_k=int(block["top_k"]),
            num_drafts=int(block["num_drafts"]),
        )

    @classmethod
    def from_hf_config(cls, config: Any) -> RedrafterDrafterSpec:
        """Read the block from a configuration object or dict (top level, or in ``transformer_layer_config``)."""
        data = config if isinstance(config, dict) else getattr(config, "__dict__", {})
        nested = data.get("transformer_layer_config") or {}
        block = data.get(CONFIG_KEY) or nested.get(CONFIG_KEY)
        if not block:
            raise ConfigError(f"the configuration has no {CONFIG_KEY!r} block: export it with economize")
        return cls.from_dict(block)

    @classmethod
    def of(cls, technique: Any) -> RedrafterDrafterSpec:
        """The spec of an attached :class:`~economize.techniques.redrafter.RedrafterTechnique`."""
        adapter, config = technique.adapter, technique.config
        d = int(adapter.hidden_size)
        return cls(
            num_layers=int(adapter.num_layers),
            hidden_size=d,
            vocab_size=int(cast(torch.Tensor, adapter.lm_head.weight).shape[0]),
            width=int(config.exit_dim or 2 * d),
            head_layers=len(technique.blocks),
            head=str(config.head),
            param_dtype=str(config.param_dtype),
            top_k=int(config.top_k),
            num_drafts=int(config.num_drafts),
        )


def drafter_state(technique: Any) -> dict[str, torch.Tensor]:
    """The technique's tensors (detached), keyed as in its ``state_dict``."""
    return {name: tensor.detach() for name, tensor in technique.state_dict().items()}
