"""Configuration of Kangaroo drafting."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar

from ...core.config import TechniqueConfig
from ...errors import ConfigError
from ...modules.dtype import COPY_DTYPES
from ...validation import check_choice, check_number, check_positive, check_unit_interval

if TYPE_CHECKING:
    from ...arch import ArchAdapter

__all__ = ["KangarooConfig"]


@dataclass(frozen=True)
class KangarooConfig(TechniqueConfig):
    """Hyperparameters of a Kangaroo drafter (Liu et al., NeurIPS 2024, arXiv 2404.18911).

    A draft step runs the model's first ``exit_layer`` (``l``) layers on the last token, then the adapter (one
    attention layer with its own KV cache and two RMS norms, no MLP) and the model's LM head. Only the adapter is
    trained. Kangaroo's second early exit, which stops drafting once the drafter's top-1 probability falls below a
    threshold (0.6 in the paper), is the decoder's dynamic draft length (``DecodeConfig.min_confidence``). It is on by
    default: :func:`economize.apply` without ``decoding`` drafts up to ``max_draft`` tokens per cycle and stops at
    ``stop_threshold`` (see :meth:`KangarooTechnique.decode_config`).

    Attributes:
        exit_layer: ``l``, the last model layer of a draft step (``1 <= l < L``; the paper's 2 for 7B models).
        num_heads: Attention heads of the adapter (``None``: the model's last full-attention layer's).
        num_kv_heads: Key/value heads of the adapter (``None``: as ``num_heads``' source).
        head_dim: Head width of the adapter (``None``: the model's; another width needs a rotary table of that
            width, which the model does not have).
        rms_norm_eps: Epsilon of the adapter's norms (``None``: the model's).
        attention_bias: Biases on the q/k/v projections (``None``: as the model's first layer).
        param_dtype: Storage and compute dtype of the adapter: ``"float32"`` (default) or ``"model"``.
        stop_threshold: Top-1 probability below which a cycle stops drafting (the paper's 0.6; ``None``: draft the
            full cycle).
        max_draft: Draft tokens per cycle at most (the paper's ``gamma = 6``, so cycles of 7 tokens).
    """

    technique: ClassVar[str] = "kangaroo"

    exit_layer: int = 2
    num_heads: int | None = None
    num_kv_heads: int | None = None
    head_dim: int | None = None
    rms_norm_eps: float | None = None
    attention_bias: bool | None = None
    param_dtype: str = "float32"
    stop_threshold: float | None = 0.6
    max_draft: int = 6

    def __post_init__(self) -> None:
        check_positive("exit_layer", self.exit_layer)
        check_positive("max_draft", self.max_draft)
        if self.stop_threshold is not None:
            check_unit_interval("stop_threshold", self.stop_threshold, closed_high=True)
        for name in ("num_heads", "num_kv_heads", "head_dim"):
            value = getattr(self, name)
            if value is not None:
                check_positive(name, value)
        if self.num_heads is not None and self.num_kv_heads is not None and self.num_heads % self.num_kv_heads:
            raise ConfigError(f"num_heads {self.num_heads} is not a multiple of num_kv_heads {self.num_kv_heads}")
        if self.rms_norm_eps is not None:
            check_number("rms_norm_eps", self.rms_norm_eps, low=0, closed_low=False)
        check_choice("param_dtype", self.param_dtype, COPY_DTYPES)

    def validate(self, adapter: ArchAdapter) -> None:
        if self.exit_layer >= adapter.num_layers:
            raise ConfigError(
                f"exit_layer must be below the model's {adapter.num_layers} layers, got {self.exit_layer}"
            )
