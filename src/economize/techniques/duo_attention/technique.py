"""The ``duo_attention`` technique: full attention for retrieval heads, a StreamingLLM window for streaming heads.

DuoAttention (Xiao et al., ICLR 2025) splits the KV heads of every layer into retrieval heads, which keep the whole
cache, and streaming heads, which keep the first ``sinks`` entries and the ``recent`` most recent ones. The split
comes from gate values trained per KV head (read by :mod:`.pattern`, trained by :mod:`.objective`); the prompt is
attended in full by every head, and the streaming heads' caches are trimmed afterwards, so during decoding a
streaming head attends to the sinks, the last ``recent`` committed entries and the entries of the running pass.

The technique is the positional ``"streaming"`` method of :mod:`economize.techniques.kv_evict` restricted to the
streaming heads: :class:`DuoAttentionCache` is a :class:`~economize.techniques.kv_evict.WindowEvictionCache` whose
evicted range applies only to the query heads of streaming KV heads. It keeps one cut per row, reads nothing back
from the device and runs under CUDA graphs. Eviction is simulated as in ``kv_evict``: the buffers keep every entry
and the byte accounting counts the entries the streaming heads still see. With ``attn_implementation="economize"``
the decode kernel skips every K/V tile that no query head of a KV head sees, so a streaming head reads only the
tiles of its sinks and its window.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

import torch

from ...core.config import TechniqueConfig
from ...core.metadata import Family, Saving, Simulation, TechniqueInfo
from ...core.technique import Technique, register_technique
from ...errors import ConfigError
from ...runtime import KVCache
from ...runtime.transfer import device_tensor
from ...validation import check_number
from .._kv import CacheWrapper, layer_indices
from ..kv_evict.technique import KVEvictConfig, KVEvictTechnique, WindowEvictionCache
from ..prune.technique import DepthPruneTechnique
from .pattern import full_attention_heads, read_pattern, read_scores

if TYPE_CHECKING:
    from ...arch import ArchAdapter

__all__ = ["DuoAttentionCache", "DuoAttentionConfig", "DuoAttentionTechnique", "streaming_heads"]


@dataclass(frozen=True)
class DuoAttentionConfig(KVEvictConfig):
    """DuoAttention: a trained split of the KV heads into retrieval and streaming heads.

    Attributes:
        pattern: The gate values per layer and KV head: a pattern directory of the reference
            (``full_attention_heads.tsv``), a path to such a file, or the values inline (one tuple per decoder
            layer, one value per KV head of the layer).
        sparsity: Share of streaming heads among the heads of the covered layers; the threshold is that quantile of
            their gate values. 0.5 as in the reference's examples; the paper keeps a quarter of the heads full for
            multi-head attention and half for grouped-query attention.
        threshold: Used when ``sparsity`` is ``None``: heads with a gate value at or above it keep full attention.
        sinks: First entries a streaming head always sees (64 as in the reference).
        recent: Most recent committed entries a streaming head sees (256 as in the reference).
        layers: 1-indexed layers whose streaming heads evict (empty: the ``"layers"`` that the pattern directory's
            ``config.json`` lists, else every layer); the heads of other layers keep full attention, and the sparsity
            counts only the covered layers' heads. A pattern trained by
            :func:`~economize.techniques.duo_attention.train_pattern` on a model with sliding-window layers lists
            its global layers there.

    The remaining :class:`~economize.techniques.kv_evict.KVEvictConfig` fields belong to the ``"streaming"`` method
    and keep their defaults. :meth:`from_pattern` reads a pattern directory together with its sink and recent
    sizes.
    """

    technique: ClassVar[str] = "duo_attention"
    pattern: str | tuple[tuple[float, ...], ...] = ()
    sparsity: float | None = 0.5
    threshold: float | None = None
    sinks: int | None = 64

    def __post_init__(self) -> None:
        if self.method != "streaming":
            raise ConfigError(f"duo_attention evicts like StreamingLLM (method 'streaming'), got {self.method!r}")
        if self.sinks is None:
            raise ConfigError("duo_attention needs a number of sinks")
        super().__post_init__()
        if isinstance(self.pattern, Path):
            object.__setattr__(self, "pattern", str(self.pattern))
        if not isinstance(self.pattern, str):  # from_dict hands inner lists back as lists
            try:
                rows = tuple(tuple(float(value) for value in row) for row in self.pattern)
            except (TypeError, ValueError):
                raise ConfigError("duo_attention pattern must be a path or rows of gate values") from None
            object.__setattr__(self, "pattern", rows)
        if not self.pattern:
            raise ConfigError("duo_attention needs a head pattern (a pattern directory, a .tsv file or gate values)")
        if (self.sparsity is None) == (self.threshold is None):
            raise ConfigError("duo_attention takes either a sparsity or a threshold (set the other to None)")
        if self.sparsity is not None:
            check_number("sparsity", self.sparsity, low=0, high=1, closed_high=True)
        if self.threshold is not None:
            check_number("threshold", self.threshold)

    @classmethod
    def from_pattern(cls, path: str | Path, **fields: Any) -> DuoAttentionConfig:
        """The configuration of a reference pattern directory ``path``: its gate values, and its ``config.json``'s
        ``sink_size`` and ``recent_size`` unless ``fields`` sets ``sinks`` or ``recent``."""
        files = read_pattern(path)
        settings: dict[str, Any] = {}
        if files.sinks is not None:
            settings["sinks"] = files.sinks
        if files.recent is not None:
            settings["recent"] = files.recent
        return cls(pattern=str(path), **{**settings, **fields})

    def covered_layers(self) -> tuple[int, ...]:
        """The 1-indexed layers whose streaming heads evict: :attr:`layers`, else the trained layers recorded in the
        pattern directory's ``config.json`` (empty: every layer)."""
        if self.layers or not isinstance(self.pattern, str) or not Path(self.pattern).is_dir():
            return self.layers
        return read_pattern(self.pattern).layers

    def scores(self) -> tuple[tuple[float, ...], ...]:
        """The gate values per decoder layer and KV head (read from the file of a path pattern), clipped to
        ``[0, 1]``."""
        if isinstance(self.pattern, str):
            return read_scores(self.pattern)
        return tuple(tuple(min(max(value, 0.0), 1.0) for value in row) for row in self.pattern)

    def validate(self, adapter: ArchAdapter) -> None:
        super().validate(adapter)
        streaming_heads(self, adapter)


def streaming_heads(config: DuoAttentionConfig, adapter: ArchAdapter) -> dict[int, tuple[bool, ...]]:
    """Per cache layer (0-indexed) with at least one streaming head, ``True`` for each streaming KV head.

    Rows of layers that attend over another layer's cache (KV sharing) are not read: a shared cache layer follows the
    row of the layer that writes it.

    Raises:
        ConfigError: The pattern does not have one row per decoder layer and one value per KV head.
    """
    scores = config.scores()
    if len(scores) != adapter.num_layers:
        raise ConfigError(f"duo_attention pattern has {len(scores)} layers, the model {adapter.num_layers}")
    covered = layer_indices(config.covered_layers(), adapter)
    rows: dict[int, tuple[float, ...]] = {}
    for index in range(1, adapter.num_layers + 1):
        if adapter.kv_source(index) != index or (covered is not None and index - 1 not in covered):
            continue
        kv_heads = adapter.head_geometry(index).kv_heads
        if len(scores[index - 1]) != kv_heads:
            raise ConfigError(
                f"duo_attention pattern has {len(scores[index - 1])} values for layer {index}, which has "
                f"{kv_heads} KV heads"
            )
        rows[index - 1] = scores[index - 1]
    full = full_attention_heads(list(rows.values()), config.sparsity, config.threshold)
    return {layer: tuple(not f for f in heads) for layer, heads in zip(rows, full, strict=True) if not all(heads)}


class DuoAttentionCache(WindowEvictionCache):
    """The cache of one generation under DuoAttention: a :class:`~economize.techniques.kv_evict.WindowEvictionCache`
    whose evicted range applies only to the streaming heads.

    Attributes:
        streaming: Per covered cache layer, ``True`` for each streaming KV head (layers without one are not covered).
    """

    def __init__(self, inner: KVCache, owner: DuoAttentionTechnique) -> None:
        self.streaming = owner.streaming
        self._groups = {layer: owner.query_group(layer) for layer in self.streaming}
        self._heads: dict[int, torch.Tensor] = {}  # per cache layer: [1, H_q, 1, 1] on the cache's device
        super().__init__(inner, owner)

    def covers(self, layer_idx: int) -> bool:
        return layer_idx in self.streaming and super().covers(layer_idx)

    def evicting_heads(self, layer_idx: int, device: torch.device) -> torch.Tensor | None:
        heads = self._heads.get(layer_idx)
        if heads is None or heads.device != device:
            self._place_heads(device)
            heads = self._heads[layer_idx]
        return heads

    def _place_heads(self, device: torch.device) -> None:
        """Copy the streaming query heads of every covered layer to ``device`` (kept at fixed addresses)."""
        for layer, heads in self.streaming.items():
            current = self._heads.get(layer)
            if current is None or current.device != device:
                query = [h for h in heads for _ in range(self._groups[layer])]
                self._heads[layer] = device_tensor(query, device, torch.bool).view(1, -1, 1, 1)

    def _sync(self, device: torch.device | None = None) -> torch.Tensor | None:
        target = super()._sync(device)
        if target is not None:
            self._place_heads(target.device)  # outside any recorded graph: the pass boundary
        return target

    def _window_bytes(
        self, layer_idx: int, keys: torch.Tensor, committed: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        whole = self.dense_bytes(keys, committed)
        streaming = sum(self.streaming[layer_idx])
        hidden = self._hidden_counts(committed) * streaming * 2.0 * keys.shape[3] * keys.element_size()
        return whole - hidden, whole

    def kept(self, layer_idx: int) -> torch.Tensor | None:
        kept = super().kept(layer_idx)
        if kept is None:
            return None
        full = ~torch.tensor(self.streaming[layer_idx])
        return kept | full[None, :, None]


@register_technique
class DuoAttentionTechnique(KVEvictTechnique):
    """DuoAttention: retrieval heads keep the full KV cache, streaming heads keep attention sinks and a recent window.

    See :class:`DuoAttentionConfig`. Training-free given a trained pattern; no parameters. The objective
    ``"duo_attention_gates"`` (:func:`~economize.techniques.duo_attention.train_pattern`) trains a pattern for any
    model.

    Attributes:
        streaming: Per cache layer with at least one streaming head, ``True`` for each streaming KV head.
    """

    name: ClassVar[str] = DuoAttentionConfig.technique
    config_class: ClassVar[type[DuoAttentionConfig]] = DuoAttentionConfig
    info = TechniqueInfo(
        family=Family.KV_EVICTION,
        saves=Saving.KV_MEMORY | Saving.COMPUTE,
        lossless=False,
        simulated=Simulation(
            Saving.KV_MEMORY,
            "keeps the streaming heads' evicted entries in the cache (the decode kernel of "
            "attn_implementation=\"economize\" skips them, other attention implementations mask them)",
        ),
    )
    config: DuoAttentionConfig

    def __init__(self, config: DuoAttentionConfig, adapter: ArchAdapter) -> None:
        super().__init__(config, adapter)
        self.streaming = streaming_heads(config, adapter)

    @classmethod
    def conflict(
        cls, config: TechniqueConfig | None, other: type[Technique], other_config: TechniqueConfig | None
    ) -> str | None:
        """Also refuses depth pruning, which removes layers before the pattern is read: the pattern's rows address
        the layers of the model it was trained on."""
        reason = super().conflict(config, other, other_config)
        if reason is None and issubclass(other, DepthPruneTechnique):
            reason = (
                f"{cls.name} cannot be combined with {other.name}: its head pattern has one row per layer of the "
                f"unpruned model; save the pruned model and apply {cls.name} to it with the removed layers' rows "
                "dropped from the pattern"
            )
        return reason

    def make_wrapper(self, cache: KVCache) -> CacheWrapper:
        return DuoAttentionCache(cache, self)

    def describe(self) -> dict[str, Any]:
        """The window of the streaming heads and how many KV heads stream."""
        heads = sum(
            self.adapter.head_geometry(i).kv_heads
            for i in range(1, self.adapter.num_layers + 1)
            if self.adapter.kv_source(i) == i
        )
        streaming = sum(sum(row) for row in self.streaming.values())
        return {
            **super().describe(),
            "streaming_heads": streaming,
            "kv_heads": heads,
            "sparsity": streaming / heads if heads else 0.0,
        }
