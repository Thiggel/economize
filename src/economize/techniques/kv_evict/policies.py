"""Eviction methods: which committed cache entries an :class:`~.technique.EvictionCache` keeps.

A method sees, per cache layer, the keep mask ``[B, H_kv, num_slots]`` (``True`` for entries attention may read)
and, if it asks for them, accumulated attention scores of the same shape. It acts at the two boundaries of every
exact pass (see :class:`PassInfo`) by clearing entries of the keep mask, and only below the pass's frontier, where
every entry is a committed token. New methods are registered in :data:`EVICTION_METHODS`::

    @EVICTION_METHODS.register("random")
    class RandomEviction(EvictionMethod):
        options = frozenset({"recent"})  # the KVEvictConfig.method_options it reads

        def before_pass(self, keep, scores, info):
            old = candidates(keep, self.sinks, info.frontier - self.config.recent)
            keep &= ~(old & (torch.rand(keep.shape, device=keep.device) < 0.5))
"""

from __future__ import annotations

from abc import ABC
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar

import torch
import torch.nn.functional as F

from ...registry import Registry
from ...runtime import Phase

if TYPE_CHECKING:
    from .technique import KVEvictConfig

__all__ = [
    "EVICTION_METHODS",
    "EvictionMethod",
    "H2OEviction",
    "H2OPrefillEviction",
    "PassInfo",
    "PyramidKVEviction",
    "SnapKVEviction",
    "StreamingEviction",
    "candidates",
    "keep_top",
]


@dataclass(frozen=True)
class PassInfo:
    """One exact pass, as eviction methods see it.

    Attributes:
        phase: The decoding phase of the pass, as :func:`economize.runtime.current_phase` reports it (a
            :data:`~economize.runtime.Phase`, or ``None`` outside the decoding loop).
        frontier: ``[B]`` first slot the pass writes per row (``-1`` for rows that only write padding). Every slot
            below it holds a committed token.
        end: ``[B]`` one past the last slot the pass writes per row (``0`` for padding-only rows).
        layer: The 0-indexed cache layer whose keep mask the method acts on (``-1`` when unknown).
        num_layers: The model's decoder layers (``0`` when unknown); with :attr:`layer` it places a layer in the
            stack, for methods whose budget depends on depth (:class:`PyramidKVEviction`).
    """

    phase: Phase | None
    frontier: torch.Tensor
    end: torch.Tensor
    layer: int = -1
    num_layers: int = 0

    @property
    def prefill(self) -> bool:
        """Whether this is the prompt pass, where every query is a committed token."""
        return self.phase == "prefill"

    def to(self, device: torch.device) -> PassInfo:
        return PassInfo(self.phase, self.frontier.to(device), self.end.to(device), self.layer, self.num_layers)

    def at(self, layer: int, num_layers: int) -> PassInfo:
        """The same pass, as cache layer ``layer`` of a model with ``num_layers`` decoder layers sees it."""
        return PassInfo(self.phase, self.frontier, self.end, layer, num_layers)


class EvictionMethod(ABC):  # noqa: B024  (every hook has a default; ABC marks it as a base to subclass)
    """A rule for evicting committed cache entries.

    Attributes:
        needs_scores: Whether the method reads accumulated attention scores (it then gets a ``scores`` tensor and
            :meth:`score_floor` decides which queries add to it).
    """

    needs_scores: ClassVar[bool] = False
    #: The method-specific :class:`KVEvictConfig` fields (``KVEvictConfig.method_options``) the method reads; the
    #: config rejects a non-default value of any other.
    options: ClassVar[frozenset[str]] = frozenset()
    #: Sinks kept when the config leaves ``sinks`` unset.
    default_sinks: ClassVar[int] = 0
    #: Whether the method evicts only prompt entries, once, after the prompt pass (and never a generated entry).
    #: The kept prompt entries are then fixed for the rest of the generation, so a cache can store them compacted
    #: (:mod:`economize.techniques.kv_evict_fused`).
    selects_once: ClassVar[bool] = False
    #: Whether the method evicts by slot alone: the same entries of every KV head, every slot ``sinks <= slot <
    #: window_cut(frontier)`` of a row, without scores. Its cache then keeps one cut per row instead of keep masks
    #: (:class:`~.technique.WindowEvictionCache`), reads nothing back from the device, and runs under CUDA graphs.
    positional: ClassVar[bool] = False

    def __init__(self, config: KVEvictConfig) -> None:
        self.config = config
        self.sinks = self.default_sinks if config.sinks is None else config.sinks

    def score_floor(self, info: PassInfo) -> torch.Tensor | None:
        """``[B]`` lowest query slot whose attention adds to the scores during this pass, or ``None`` (no scores)."""
        return None

    def before_pass(self, keep: torch.Tensor, scores: torch.Tensor | None, info: PassInfo) -> None:
        """Evict entries (clear ``keep`` in place) before the pass's queries read the cache."""

    def after_pass(self, keep: torch.Tensor, scores: torch.Tensor | None, info: PassInfo) -> None:
        """Evict entries after the pass wrote every layer."""

    def window_cut(self, frontier: int) -> int:
        """A :attr:`positional` method: the first slot kept after the sinks once an exact pass starts at
        ``frontier`` (every slot from the sinks up to it is evicted). It must agree with :meth:`before_pass`."""
        raise NotImplementedError(f"{type(self).__name__} is not a positional eviction method")

    def describe(self) -> dict[str, Any]:
        """The method's effective budget, for reports."""
        return {"method": self.config.method, "sinks": self.sinks}


#: Eviction methods by :attr:`KVEvictConfig.method`.
EVICTION_METHODS: Registry[type[EvictionMethod]] = Registry("kv eviction method", "economize.kv_eviction")


def candidates(keep: torch.Tensor, low: int | torch.Tensor, high: torch.Tensor) -> torch.Tensor:
    """Kept entries at slots ``low <= slot < high[b]`` of ``keep`` ``[B, H, N]`` (``low`` an int or ``[B]``)."""
    slot = torch.arange(keep.shape[-1], device=keep.device)
    low = torch.as_tensor(low, device=keep.device)
    low = low.view(-1, 1, 1) if low.dim() else low
    return keep & (slot >= low) & (slot < high.to(keep.device).view(-1, 1, 1))


def keep_top(keep: torch.Tensor, candidate: torch.Tensor, scores: torch.Tensor, budget: int | torch.Tensor) -> None:
    """Of the ``candidate`` entries of every row and head, keep the ``budget`` with the highest ``scores`` and evict
    the others (in place). ``budget`` is one number for every row or a ``[B]`` tensor of per-row budgets. Ties go to
    the earlier slot, so the choice does not depend on the buffer length (max pooling makes plateaus of equal
    scores)."""
    if not torch.is_tensor(budget):
        if budget <= 0:
            keep &= ~candidate
            return
        k = min(budget, keep.shape[-1])
        rank: torch.Tensor | None = None
    else:
        budget = budget.to(keep.device).clamp(min=0)
        k = min(int(budget.max()) if budget.numel() else 0, keep.shape[-1])
        if k <= 0:
            keep &= ~candidate
            return
        rank = torch.arange(k, device=keep.device) < budget.view(-1, 1, 1)  # [B, 1, k]
    ranked = scores.float().masked_fill(~candidate, float("-inf"))
    order = ranked.sort(dim=-1, descending=True, stable=True).indices[..., :k]
    src: torch.Tensor | bool = True if rank is None else rank.expand(*order.shape).contiguous()
    chosen = torch.zeros_like(candidate).scatter_(-1, order, src)
    keep &= ~candidate | chosen


@EVICTION_METHODS.register("streaming")
class StreamingEviction(EvictionMethod):
    """StreamingLLM (Xiao et al., ICLR 2024): the first ``sinks`` entries (attention sinks) and the ``recent``
    most recent committed entries; everything in between is evicted."""

    default_sinks: ClassVar[int] = 4
    positional: ClassVar[bool] = True
    options = frozenset({"recent"})

    def before_pass(self, keep: torch.Tensor, scores: torch.Tensor | None, info: PassInfo) -> None:
        keep &= ~candidates(keep, self.sinks, info.frontier - self.config.recent)

    def window_cut(self, frontier: int) -> int:
        return frontier - self.config.recent

    def describe(self) -> dict[str, Any]:
        return {**super().describe(), "recent": self.config.recent}


@EVICTION_METHODS.register("h2o")
class H2OEviction(EvictionMethod):
    """H2O (Zhang et al., NeurIPS 2023): the ``recent`` most recent committed entries plus, per KV head, the
    ``heavy`` older entries with the largest attention received so far (summed over all queries of all exact
    passes and over the query heads that share the KV head). An evicted entry never returns."""

    needs_scores: ClassVar[bool] = True
    options = frozenset({"recent", "heavy"})

    def score_floor(self, info: PassInfo) -> torch.Tensor | None:
        return torch.zeros_like(info.frontier)

    def before_pass(self, keep: torch.Tensor, scores: torch.Tensor | None, info: PassInfo) -> None:
        assert scores is not None
        old = candidates(keep, self.sinks, info.frontier - self.config.recent)
        keep_top(keep, old, scores, self.config.heavy)

    def describe(self) -> dict[str, Any]:
        return {**super().describe(), "recent": self.config.recent, "heavy": self.config.heavy}


@EVICTION_METHODS.register("h2o_prefill")
class H2OPrefillEviction(EvictionMethod):
    """H2O as the KVCache-Factory benchmarks run it (``H2OKVCluster``): once, after the prompt is prefilled, keep
    per KV head the ``recent`` last prompt entries plus the ``heavy`` older prompt entries with the largest
    attention received from all prompt queries. Generated entries are all kept and never scored.

    :class:`H2OEviction` follows the paper instead and keeps evicting during decoding with scores that keep
    accumulating, which loses more quality at the same budget. Prompts no longer than ``recent + heavy`` (plus the
    sinks) are left whole, as in the reference. The reference selects per query head by default
    (``kv_cache_granularity="query_head"``); here the scores of a GQA group are summed and the group shares one
    selection, as in :class:`SnapKVEviction`.
    """

    needs_scores: ClassVar[bool] = True
    selects_once: ClassVar[bool] = True
    options = frozenset({"recent", "heavy"})

    def score_floor(self, info: PassInfo) -> torch.Tensor | None:
        return torch.zeros_like(info.frontier) if info.prefill else None

    def after_pass(self, keep: torch.Tensor, scores: torch.Tensor | None, info: PassInfo) -> None:
        if not info.prefill:
            return
        assert scores is not None
        prefix_end = info.end - self.config.recent
        keep_top(keep, candidates(keep, self.sinks, prefix_end), scores, self.config.heavy)

    def describe(self) -> dict[str, Any]:
        return {**super().describe(), "recent": self.config.recent, "heavy": self.config.heavy}


@EVICTION_METHODS.register("snapkv")
class SnapKVEviction(EvictionMethod):
    """SnapKV (Li et al., NeurIPS 2024): once, after the prompt is prefilled, keep per KV head the ``heavy``
    prompt entries that the last ``observation_window`` prompt queries attend to most (scores pooled over
    ``kernel_size`` neighbouring slots) plus the observation window itself. Generated entries are all kept.

    The scores are the model's own attention probabilities, so each window query's softmax runs over every key it
    sees (prefix and causal window), as in the reference's ``SnapKVCluster``, which also normalizes over the whole
    row before it sums the prefix columns. The keep mask lives per KV head, so under GQA the scores are summed
    over the query heads of a group and the group shares one selection of ``heavy`` entries (a per-query-head
    selection would need a cache that stores the KV head once per query head).
    """

    needs_scores: ClassVar[bool] = True
    selects_once: ClassVar[bool] = True
    options = frozenset({"heavy", "observation_window", "kernel_size", "pooling"})

    def score_floor(self, info: PassInfo) -> torch.Tensor | None:
        if not info.prefill:
            return None
        return info.end - self.config.observation_window

    def after_pass(self, keep: torch.Tensor, scores: torch.Tensor | None, info: PassInfo) -> None:
        if not info.prefill:
            return
        assert scores is not None
        prefix_end = info.end - self.config.observation_window
        prefix = candidates(torch.ones_like(keep), 0, prefix_end)
        pooled = self.pool(scores.float().masked_fill(~prefix, 0.0))
        keep_top(keep, candidates(keep, self.sinks, prefix_end), pooled, self.config.heavy)

    def pool(self, scores: torch.Tensor) -> torch.Tensor:
        """``scores`` ``[B, H, N]`` smoothed over ``kernel_size`` neighbouring slots (SnapKV's clustering step)."""
        k = self.config.kernel_size
        if k == 1:
            return scores
        flat = scores.reshape(-1, 1, scores.shape[-1])
        if self.config.pooling == "max":
            pooled = F.max_pool1d(flat, k, stride=1, padding=k // 2)
        else:
            pooled = F.avg_pool1d(flat, k, stride=1, padding=k // 2)
        return pooled.reshape(scores.shape)

    def describe(self) -> dict[str, Any]:
        return {
            **super().describe(),
            "heavy": self.config.heavy,
            "observation_window": self.config.observation_window,
            "kernel_size": self.config.kernel_size,
            "pooling": self.config.pooling,
        }


@EVICTION_METHODS.register("pyramidkv")
class PyramidKVEviction(SnapKVEviction):
    """PyramidKV (Cai et al., 2024): SnapKV's prompt selection with a budget that shrinks linearly with depth.

    The scores, pooling and observation window are those of :class:`SnapKVEviction`; only the number of prompt
    entries a layer keeps differs. With ``C = heavy + observation_window`` (the reference's ``max_capacity_prompt``),
    ``w = observation_window``, prompt length ``q`` of the row, 0-indexed cache layer ``l`` of ``L`` decoder layers,
    the budget follows ``PyramidKVCluster`` of KVCache-Factory::

        lo = (C - w) // beta;  hi = 2 (C - w) - lo
        if hi >= q - w:  hi = q - w;  lo = 2 (C - w) - hi
        budget_l = hi - l * ((hi - lo) // (L - 1))

    so the layers keep ``L (C - w)`` prompt entries in total, between ``hi`` at the bottom and about ``lo`` at the
    top. A prompt shorter than ``C`` is left whole, and one shorter than ``2 (C - w)`` keeps ``C - w`` entries in
    every layer (both as in the reference). The selection is per KV head, the scores of a GQA group summed.
    """

    options = SnapKVEviction.options | {"beta"}

    def after_pass(self, keep: torch.Tensor, scores: torch.Tensor | None, info: PassInfo) -> None:
        if not info.prefill:
            return
        assert scores is not None
        if info.layer < 0 or info.num_layers <= 0:
            raise RuntimeError("PyramidKV needs the layer of the pass (PassInfo.layer and num_layers)")
        prefix_end = info.end - self.config.observation_window
        prefix = candidates(torch.ones_like(keep), 0, prefix_end)
        pooled = self.pool(scores.float().masked_fill(~prefix, 0.0))
        lengths = info.end.clamp(min=0).tolist()
        budget = torch.tensor([self.budget(q, info.layer, info.num_layers) for q in lengths], device=keep.device)
        keep_top(keep, candidates(keep, self.sinks, prefix_end), pooled, budget)

    def budget(self, prompt: int, layer: int, num_layers: int) -> int:
        """Prompt entries (outside the observation window) that ``layer`` keeps of a ``prompt``-token prompt."""
        window = self.config.observation_window
        base = self.config.heavy  # C - w
        if prompt < base + window:
            return max(prompt - window, 0)  # the reference keeps a short prompt whole
        if prompt < 2 * base:
            return base
        low = base // self.config.beta
        high = 2 * base - low
        if high >= prompt - window:
            high = prompt - window
            low = 2 * base - high
        step = (high - low) // (num_layers - 1) if num_layers > 1 else 0
        return high - layer * step

    def describe(self) -> dict[str, Any]:
        return {**super().describe(), "beta": self.config.beta}
