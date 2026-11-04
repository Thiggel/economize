"""Configuration of DFlash block drafting."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar

from ...core.config import TechniqueConfig
from ...errors import ConfigError
from ...modules.dtype import COPY_DTYPES
from ...modules.rotary import rope_frequencies
from ...validation import (
    check_choice,
    check_indices,
    check_non_negative,
    check_number,
    check_positive,
    check_unit_interval,
)

if TYPE_CHECKING:
    from ...arch import ArchAdapter

__all__ = ["DFLASH_LAYER_TYPES", "DFlashConfig", "default_target_layers"]

#: Attention kinds of the draft layers, named as in the official ``config.json`` files.
DFLASH_LAYER_TYPES = ("sliding_attention", "full_attention")


def default_target_layers(num_layers: int, count: int = 5) -> tuple[int, ...]:
    """The reference's spread of ``count`` feature layers over a model of ``num_layers`` blocks
    (``build_target_layer_ids`` of z-lab/dflash: evenly from the second block to the fourth from last), as outputs of
    1-indexed blocks (the reference's 0-indexed id ``i`` reads the output of block ``i + 1``), clamped to ``1..L``
    and without repeats for very shallow models."""
    if count == 1:
        ids = [num_layers // 2]
    else:
        start, end = 1, num_layers - 3
        ids = [round(start + i * (end - start) / (count - 1)) for i in range(count)]
    return tuple(sorted({min(max(i + 1, 1), num_layers) for i in ids}))


@dataclass(frozen=True)
class DFlashConfig(TechniqueConfig):
    """Hyperparameters of a DFlash block drafter (Chen et al., 2026, arXiv 2602.06036).

    The drafter fuses the model's hidden states at ``target_layers`` (outputs of those 1-indexed blocks) of every
    committed token with ``fc`` and ``hidden_norm`` into a context vector, which each draft layer projects into
    keys and values of its own. One forward over the block ``[x_p, mask, ..., mask]`` (``block_size`` tokens at the
    positions ``p .. p + block_size - 1``, ``x_p`` the last token the model emitted) attends to the context of the
    positions before ``p`` and, inside the block, bidirectionally (full layers) or causally (sliding layers by
    default). The output at the ``i``-th mask, through the drafter's ``norm`` and the model's LM head, drafts the
    token at ``p + i``, so one forward drafts ``block_size - 1`` tokens.

    Attributes:
        target_layers: Blocks whose outputs are fused (``None``: :func:`default_target_layers` with 5 layers).
        block_size: Tokens of the drafted block, the anchor included (16 for z-lab's Gemma-4 drafter, 8 for
            RedHat's). A cycle drafts at most ``block_size - 1`` tokens.
        num_layers: Draft decoder layers.
        mask_token_id: Token id of the masks of the block (``4`` for Gemma-4's official drafters).
        intermediate_size: Width of the draft layers' MLPs (``None``: the model's).
        num_heads: Attention heads of the draft layers (``None``: the model's).
        num_kv_heads: Key/value heads of the draft layers (``None``: the model's).
        head_dim: Head width of the draft layers (``None``: the model's).
        layer_types: ``"sliding_attention"`` or ``"full_attention"`` per draft layer (``None``: all sliding but
            the last).
        sliding_window: Window of the sliding layers, on absolute positions.
        sliding_causal: Sliding layers attend causally inside the block (z-lab and the ``speculators`` default);
            full layers always attend bidirectionally inside the block.
        rope_theta: Base of the draft layers' rotary table.
        rope_scaling: Scaling of the rotary table, a Hugging Face ``rope_scaling`` dict. ``None``: unscaled.
        partial_rotary_factor: Fraction of each head the rotary table rotates.
        max_position_embeddings: The ``max_position_embeddings`` the table's scaling refers to.
        rms_norm_eps: Epsilon of the drafter's RMS norms.
        attention_bias: Biases on the q/k/v/o projections.
        embed_scale: Factor on the model's raw token embeddings (the reference's ``input_embedding_scale``; the
            official Gemma-4 drafters read unscaled embeddings, without Gemma's ``sqrt(d)``).
        logit_softcap: Final logit soft cap of the drafter: ``None`` applies the model's own logit transform
            (Gemma's soft cap), ``0`` none, a positive value ``c`` the cap ``c * tanh(z / c)``. Greedy drafts do
            not depend on it.
        param_dtype: Storage and compute dtype of the drafter's parameters: ``"float32"`` (default) or
            ``"model"`` (the model's dtype, as the official checkpoints are stored). Softmaxes and norms run in
            float32 and logits are returned in float32.
    """

    technique: ClassVar[str] = "dflash"

    target_layers: tuple[int, ...] | None = None
    block_size: int = 16
    num_layers: int = 5
    mask_token_id: int = 0
    intermediate_size: int | None = None
    num_heads: int | None = None
    num_kv_heads: int | None = None
    head_dim: int | None = None
    layer_types: tuple[str, ...] | None = None
    sliding_window: int = 2048
    sliding_causal: bool = True
    rope_theta: float = 1_000_000.0
    rope_scaling: Mapping[str, Any] | None = None
    partial_rotary_factor: float = 1.0
    max_position_embeddings: int | None = None
    rms_norm_eps: float = 1e-6
    attention_bias: bool = False
    embed_scale: float = 1.0
    logit_softcap: float | None = None
    param_dtype: str = "float32"

    def __post_init__(self) -> None:
        if self.target_layers is not None:
            layers = tuple(int(layer) for layer in self.target_layers)
            check_indices("target_layers", layers, allow_empty=False)
            object.__setattr__(self, "target_layers", layers)
        check_positive("block_size", self.block_size)
        if self.block_size < 2:
            raise ConfigError(f"block_size must be at least 2 (the anchor and one mask), got {self.block_size}")
        check_positive("num_layers", self.num_layers)
        check_non_negative("mask_token_id", self.mask_token_id)
        for name in ("intermediate_size", "num_heads", "num_kv_heads", "head_dim"):
            value = getattr(self, name)
            if value is not None:
                check_positive(name, value)
        if self.num_heads is not None and self.num_kv_heads is not None and self.num_heads % self.num_kv_heads:
            raise ConfigError(f"num_heads {self.num_heads} is not a multiple of num_kv_heads {self.num_kv_heads}")
        if self.layer_types is not None:
            types = tuple(self.layer_types)
            for kind in types:
                check_choice("layer_types", kind, DFLASH_LAYER_TYPES)
            if len(types) != self.num_layers:
                raise ConfigError(f"layer_types names {len(types)} layers, num_layers is {self.num_layers}")
            object.__setattr__(self, "layer_types", types)
        check_positive("sliding_window", self.sliding_window)
        check_number("rope_theta", self.rope_theta, low=0, closed_low=False)
        check_unit_interval("partial_rotary_factor", self.partial_rotary_factor, closed_low=False, closed_high=True)
        if self.rope_scaling is not None:
            object.__setattr__(self, "rope_scaling", dict(self.rope_scaling))
        # an unsupported rope type or invalid parameters raise here, before any weights are built
        rope_frequencies(self.head_dim or 64, self.rope_theta, self.rope_scaling,
                         partial_rotary_factor=self.partial_rotary_factor,
                         max_position_embeddings=self.max_position_embeddings)
        check_number("rms_norm_eps", self.rms_norm_eps, low=0, closed_low=False)
        check_number("embed_scale", self.embed_scale, low=0, closed_low=False)
        if self.logit_softcap is not None:
            check_number("logit_softcap", self.logit_softcap, low=0)
        check_choice("param_dtype", self.param_dtype, COPY_DTYPES)

    def layers(self, num_layers: int) -> tuple[int, ...]:
        """The fused layers for a model with ``num_layers`` blocks."""
        return self.target_layers if self.target_layers is not None else default_target_layers(num_layers)

    def types(self) -> tuple[str, ...]:
        """The attention kind of every draft layer."""
        if self.layer_types is not None:
            return self.layer_types
        return ("sliding_attention",) * (self.num_layers - 1) + ("full_attention",)

    def validate(self, adapter: ArchAdapter) -> None:
        layers = self.layers(adapter.num_layers)
        if max(layers) > adapter.num_layers:
            raise ConfigError(f"target_layers {layers} exceed the model's {adapter.num_layers} layers")
        if self.mask_token_id >= adapter.vocab_size:
            raise ConfigError(f"mask_token_id {self.mask_token_id} is outside the vocabulary ({adapter.vocab_size})")
        heads = self.num_heads or int(adapter.config.num_attention_heads)
        kv_heads = self.num_kv_heads or (adapter.num_kv_heads if self.num_heads is None else heads)
        if heads % kv_heads:
            raise ConfigError(f"the draft layers' {heads} heads are not a multiple of their {kv_heads} key/value heads")
