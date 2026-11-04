"""The ``minference`` technique: long prompts attend over a per-head sparse pattern fitted at prefill (MInference).

MInference (Jiang et al., NeurIPS 2024) gives every attention head one of three patterns, A-shape, vertical-slash
or block-sparse, and fits the pattern's indices to every prompt from the prompt's own queries and keys
(:mod:`economize.attention.minference`). Only the prefill changes: decoding steps, chunked prefills and prompts
shorter than ``min_queries`` attend densely, and the cache keeps every entry. The saving is prefill compute (time to
first token), which the ``"economize"`` attention kernel realizes on CUDA by reading only the indexed keys
(:func:`~economize.attention.sparse_prefill_attention`); on the CPU the same attention runs as a masked fallback.

The technique attaches a :class:`~economize.attention.MInferencePrefill` to the attention module of every covered
layer (:func:`minference_layers`): full-attention layers above ``dense_layers`` and inside ``layers``. Sliding-window
layers attend densely in any case, as their window already bounds the work.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar

import torch.nn as nn

from ...attention.minference import PATTERNS, HeadPattern, MInferencePrefill, load_patterns
from ...attention.prefill import attach_prefill, detach_prefill
from ...attention.registry import ATTENTION_KERNELS, accepts_sparse_prefill
from ...core.config import TechniqueConfig
from ...core.metadata import Family, Saving, TechniqueInfo
from ...core.technique import Technique, register_technique
from ...errors import ConfigError, IncompatibleTechniquesError
from ...validation import check_layers
from .._kv import layer_indices
from ..prune.technique import DepthPruneTechnique

if TYPE_CHECKING:
    from ...arch import ArchAdapter
    from ...core.technique import AttachContext

__all__ = ["MInferenceConfig", "MInferenceTechnique", "minference_layers"]


@dataclass(frozen=True)
class MInferenceConfig(TechniqueConfig):
    """MInference: sparse prefill attention with a per-head pattern fitted to every prompt.

    Attributes:
        pattern: The pattern of every head when no ``pattern_file`` is given: ``"vertical_slash"`` (the pattern
            the reference's search picks for nearly every head of LLaMA-3 and Qwen2), ``"block_sparse"`` or
            ``"a_shape"``.
        pattern_file: A per-head pattern file of the reference (``minference/configs/*.json``: one object per
            layer mapping every query head to ``[type, v, s, score]``), as produced by its offline search.
        vertical: Vertical lines of a ``"vertical_slash"`` head (1000 as in the reference's default).
        slash: Slash lines of a ``"vertical_slash"`` head (6096 as in the reference's default).
        top_k: Key blocks of 64 per query block of a ``"block_sparse"`` head (100 as in the reference).
        sinks: Attention sinks of an ``"a_shape"`` head (128 as in the reference).
        local: Local window of an ``"a_shape"`` head (3968 as in the reference).
        dense_layers: The first layers that attend densely.
        layers: 1-indexed layers MInference acts on (empty: every layer); ``dense_layers`` and sliding-window
            layers are left out in any case.
        min_queries: Shortest prompt (tokens per row) that attends sparsely; shorter prompts attend densely. The
            default 65 makes a prompt of one block of 64 queries dense, as its pattern would keep every key.
        ratio: Multiplier of the ``v`` and ``s`` of a ``pattern_file`` (the reference's ``minference_ratio``).
    """

    technique: ClassVar[str] = "minference"
    pattern: str = "vertical_slash"
    pattern_file: str | None = None
    vertical: int = 1000
    slash: int = 6096
    top_k: int = 100
    sinks: int = 128
    local: int = 3968
    dense_layers: int = 0
    layers: tuple[int, ...] = ()
    min_queries: int = 65
    ratio: float = 1.0

    def __post_init__(self) -> None:
        if self.pattern not in PATTERNS:
            raise ConfigError(f"minference pattern must be one of {sorted(PATTERNS)}, got {self.pattern!r}")
        if self.vertical < 0 or self.slash < 1 or self.sinks < 0 or self.local < 1:
            raise ConfigError("minference needs vertical >= 0, slash >= 1, sinks >= 0 and local >= 1")
        if self.top_k < 1:
            raise ConfigError(f"minference top_k must be positive, got {self.top_k}")
        if self.dense_layers < 0:
            raise ConfigError(f"minference dense_layers must be non-negative, got {self.dense_layers}")
        if self.min_queries < 2:
            raise ConfigError(f"minference min_queries must be at least 2, got {self.min_queries}")
        if not self.ratio > 0:
            raise ConfigError(f"minference ratio must be positive, got {self.ratio}")

    def head_pattern(self) -> HeadPattern:
        """The pattern of every head without a ``pattern_file``."""
        if self.pattern == "a_shape":
            return HeadPattern("a_shape", self.sinks, self.local)
        return HeadPattern(self.pattern, self.vertical, self.slash)

    def patterns(self, adapter: ArchAdapter) -> tuple[tuple[HeadPattern, ...], ...]:
        """The head patterns of every layer (one row per layer of the model)."""
        heads = int(adapter.config.num_attention_heads)
        if self.pattern_file is None:
            return ((self.head_pattern(),) * heads,) * adapter.num_layers
        rows = load_patterns(self.pattern_file, self.ratio)
        if len(rows) != adapter.num_layers or any(len(row) != heads for row in rows):
            raise ConfigError(
                f"MInference pattern file {self.pattern_file!r} has {len(rows)} layers of "
                f"{sorted({len(row) for row in rows})} heads; the model has {adapter.num_layers} layers of {heads}"
            )
        return rows

    def validate(self, adapter: ArchAdapter) -> None:
        check_layers(self.layers, adapter)
        implementation = str(getattr(adapter.config, "_attn_implementation", None) or "eager")
        if not accepts_sparse_prefill(implementation):
            readers = [name for name in ATTENTION_KERNELS.names() if accepts_sparse_prefill(name)]
            raise IncompatibleTechniquesError(
                f"minference needs an attention implementation that attends through its patterns "
                f"({', '.join(readers) or 'none'}; after economize.attention.install()), got {implementation!r}"
            )
        self.patterns(adapter)
        if not minference_layers(self, adapter):
            raise ConfigError("minference covers no layer (dense_layers, layers and sliding-window layers leave none)")


def minference_layers(config: MInferenceConfig, adapter: ArchAdapter) -> tuple[int, ...]:
    """The 1-indexed layers MInference acts on: full-attention layers with an attention module, above
    ``dense_layers`` and inside ``layers``."""
    chosen = layer_indices(config.layers, adapter)
    covered = []
    for index in range(config.dense_layers + 1, adapter.num_layers + 1):
        if chosen is not None and index - 1 not in chosen:
            continue
        if adapter.attention_module(index) is None or adapter.attention_spec(index).kind != "full":
            continue
        covered.append(index)
    return tuple(covered)


@register_technique
class MInferenceTechnique(Technique):
    """MInference sparse prefill (see :class:`MInferenceConfig`). Training-free; no parameters.

    It changes neither the weights nor the cache, so it combines with every other family; KV techniques see the
    same prompt keys and values as without it.
    """

    name: ClassVar[str] = MInferenceConfig.technique
    config_class: ClassVar[type[MInferenceConfig]] = MInferenceConfig
    info: ClassVar[TechniqueInfo] = TechniqueInfo(
        family=Family.SPARSE_PREFILL,
        saves=Saving.COMPUTE | Saving.LATENCY,
        lossless=False,
    )
    config: MInferenceConfig

    def __init__(self, config: MInferenceConfig, adapter: Any) -> None:
        super().__init__(config, adapter)
        self.covered = minference_layers(config, adapter)
        rows = config.patterns(adapter)
        self._prefills = {
            index: MInferencePrefill(rows[index - 1], config.top_k, config.min_queries) for index in self.covered
        }
        self._attached: list[nn.Module] = []

    @classmethod
    def conflict(
        cls, config: TechniqueConfig | None, other: type[Technique], other_config: TechniqueConfig | None
    ) -> str | None:
        """Also refuses depth pruning next to a ``pattern_file``, whose rows address the unpruned layers."""
        reason = super().conflict(config, other, other_config)
        pattern_file = getattr(config, "pattern_file", None)
        if reason is None and pattern_file is not None and issubclass(other, DepthPruneTechnique):
            reason = (
                f"{cls.name} cannot be combined with {other.name}: its pattern file has one row per layer of the "
                "unpruned model; apply it to the saved pruned model with the removed layers' rows dropped"
            )
        return reason

    def attach(self, model: nn.Module, adapter: ArchAdapter, context: AttachContext | None = None) -> None:
        self.adapter = adapter
        self.refresh()

    def refresh(self) -> None:
        self.detach()
        for index, prefill in self._prefills.items():
            module = self.adapter.attention_module(index)
            if module is not None:
                attach_prefill(module, prefill)
                self._attached.append(module)

    def detach(self) -> None:
        for module in self._attached:
            detach_prefill(module)
        self._attached = []
