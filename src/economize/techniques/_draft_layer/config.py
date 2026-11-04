"""Hyperparameters every one-layer draft decoder shares (EAGLE-1/2, EAGLE-3)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from ...core.config import TechniqueConfig
from ...errors import ConfigError
from ...modules.dtype import COPY_DTYPES
from ...modules.rotary import rope_frequencies
from ...validation import check_choice, check_number, check_positive, check_unit_interval
from .._drafting.formats import JSONObject

if TYPE_CHECKING:
    from ...arch import ArchAdapter

__all__ = ["DraftLayerConfig"]


@dataclass(frozen=True)
class DraftLayerConfig(TechniqueConfig):
    """The draft decoder layer's shape, rotary table and training options.

    Subclasses set ``technique`` and add what feeds the layer (EAGLE-3's fused auxiliary layers and own head).
    Every ``None`` takes the model's value.

    Attributes:
        intermediate_size: Width of the draft layer's MLP (``None``: the model's).
        num_heads: Attention heads of the draft layer (``None``: the model's).
        num_kv_heads: Key/value heads of the draft layer (``None``: the model's).
        head_dim: Head width of the draft layer (``None``: the model's).
        attention_bias: Biases on the q/k/v projections (``None``: as the model's first layer).
        rope_theta: Base of a rotary table of the draft layer's own (``None``: the model's rotary table, which
            needs the model's head width). Official checkpoints set it from their ``config.json``.
        rope_scaling: Scaling of the own rotary table, a Hugging Face ``rope_scaling`` dict (``rope_type`` and its
            parameters; any type of ``transformers``' ``ROPE_INIT_FUNCTIONS`` but ``"dynamic"``). ``None``: unscaled.
        partial_rotary_factor: Fraction of each head the own rotary table rotates.
        max_position_embeddings: The ``max_position_embeddings`` the own table's scaling refers to (``yarn`` and
            ``longrope`` derive their factor from it; the default ``original_max_position_embeddings``).
        rms_norm_eps: Epsilon of the draft layer's RMS norms (``None``: the model's).
        zero_source_first: Also give position ``0`` a draft K/V entry, with a zero source. EAGLE starts at the
            second token, whose source is the first token's features.
        train_steps: Draft steps of the training-time test that ``train_logits`` unrolls (EAGLE-3 and SpecForge
            use 7).
        checkpoint_steps: Recompute each step of the training-time test in the backward pass (activation
            checkpointing) when gradients are on: the unrolled steps keep only their outputs and own keys/values,
            so activation memory is that of one step instead of all ``train_steps``, for one more forward of the
            draft layer. Gradients are unchanged; inference ignores it.
        param_dtype: Storage and compute dtype of the drafter's parameters: ``"float32"`` (default), or
            ``"model"`` (the model's dtype, e.g. bfloat16, to be trained with float32 master weights in
            :class:`economize.train.OffloadedAdamW`, as mixed-precision EAGLE training does). Norms and softmaxes
            still run in float32 and logits are returned in float32.
    """

    intermediate_size: int | None = None
    num_heads: int | None = None
    num_kv_heads: int | None = None
    head_dim: int | None = None
    attention_bias: bool | None = None
    rope_theta: float | None = None
    rope_scaling: JSONObject | None = None
    partial_rotary_factor: float = 1.0
    max_position_embeddings: int | None = None
    rms_norm_eps: float | None = None
    zero_source_first: bool = False
    train_steps: int = 7
    checkpoint_steps: bool = False
    param_dtype: str = "float32"

    def __post_init__(self) -> None:
        for name in ("intermediate_size", "num_heads", "num_kv_heads", "head_dim"):
            value = getattr(self, name)
            if value is not None:
                check_positive(name, value)
        if self.num_heads is not None and self.num_kv_heads is not None and self.num_heads % self.num_kv_heads:
            raise ConfigError(f"num_heads {self.num_heads} is not a multiple of num_kv_heads {self.num_kv_heads}")
        if self.rope_theta is not None:
            check_number("rope_theta", self.rope_theta, low=0, closed_low=False)
        check_unit_interval("partial_rotary_factor", self.partial_rotary_factor, closed_low=False, closed_high=True)
        if self.rope_scaling is not None or self.partial_rotary_factor != 1.0:
            if self.rope_theta is None:
                raise ConfigError("rope_scaling and partial_rotary_factor shape the own rotary table: set rope_theta")
            if self.rope_scaling is not None:
                object.__setattr__(self, "rope_scaling", dict(self.rope_scaling))
            # an unsupported type or invalid parameters raise here, before any weights are built
            rope_frequencies(self.head_dim or 64, self.rope_theta, self.rope_scaling,
                             partial_rotary_factor=self.partial_rotary_factor,
                             max_position_embeddings=self.max_position_embeddings)
        if self.rms_norm_eps is not None:
            check_number("rms_norm_eps", self.rms_norm_eps, low=0, closed_low=False)
        check_positive("train_steps", self.train_steps)
        check_choice("param_dtype", self.param_dtype, COPY_DTYPES)

    def validate(self, adapter: ArchAdapter) -> None:
        heads = self.num_heads or int(adapter.config.num_attention_heads)
        kv_heads = self.num_kv_heads or (adapter.num_kv_heads if self.num_heads is None else heads)
        if heads % kv_heads:
            raise ConfigError(f"the draft layer's {heads} heads are not a multiple of its {kv_heads} key/value heads")
        if self.rope_theta is None and (self.head_dim or adapter.head_dim) != adapter.head_dim:
            raise ConfigError("a draft head width other than the model's needs rope_theta (its own rotary table)")
