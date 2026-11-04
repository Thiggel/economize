"""The ``kv_evict`` technique: a KV cache that forgets committed entries (StreamingLLM, H2O, SnapKV).

Every generation's cache is wrapped in an :class:`EvictionCache`, which keeps a keep mask ``[B, H_kv, slots]``
per layer next to the K/V buffers. At the boundaries of every exact pass (see
:mod:`economize.techniques._kv.wrapper`) the configured :class:`~.policies.EvictionMethod` clears entries of the
mask below the pass's frontier. A forward pre-hook on every decoder layer intersects the layer's attention mask with
the keep mask of its cache layer (per KV head, expanded to the query heads), so attention never reads an evicted
entry, in exact passes and in draft steps alike.

Score-based methods (H2O, SnapKV) accumulate the attention probabilities that the queries of exact passes give to
every slot: from the returned weights under ``attn_implementation="eager"``, and through an
:class:`~.tap.AttentionTap` under ``"sdpa"`` (active for the duration of each exact pass). Only committed queries
count. A prompt's queries are committed at once; the queries of a later exact pass may include draft tokens that
verification rejects, so their attention waits in :class:`PendingScores` until the next exact pass, whose frontier
says which of them were accepted (tree compaction and dropped rows move the pending entries along).

Positional methods (StreamingLLM) evict the same slots of every head, a range that only depends on each row's
frontier. Their caches (:class:`WindowEvictionCache`) keep one cut per row on the host and on the device
instead of keep masks: no pass boundary reads the device, and the wrapper is CUDA-graph safe.

Eviction is simulated. The buffers keep every entry, positions are not reassigned, and the byte accounting
(:class:`~economize.techniques._kv.wrapper.RowStats`) counts the entries that remain visible, as a cache that
frees evicted slots would hold them.
"""

from __future__ import annotations

from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar

import torch
from torch import nn

from ...core.config import TechniqueConfig
from ...core.metadata import Family, Saving, Simulation, TechniqueInfo
from ...core.technique import AttachContext, register_technique
from ...errors import ConfigError, UnsupportedModelError
from ...runtime import KVCache, current_phase
from ...runtime.transfer import device_tensor
from ...validation import check_choice, check_layers, check_non_negative, check_options, check_positive
from .._kv import CacheWrapper, KVCacheTechnique, SlotStore, check_positional, find_wrapper, innermost, layer_indices
from .policies import EVICTION_METHODS, EvictionMethod, PassInfo
from .tap import AttentionTap

if TYPE_CHECKING:
    from ...arch import ArchAdapter

__all__ = ["POOLINGS", "EvictionCache", "KVEvictConfig", "KVEvictTechnique", "PendingScores", "WindowEvictionCache"]

POOLINGS = ("avg", "max")


@dataclass(frozen=True)
class KVEvictConfig(TechniqueConfig):
    """KV-cache eviction.

    Attributes:
        method: Name in :data:`~.policies.EVICTION_METHODS`: ``"streaming"`` (StreamingLLM), ``"h2o"``,
            ``"h2o_prefill"`` (H2O applied once to the prompt, as KVCache-Factory runs it), ``"snapkv"`` or
            ``"pyramidkv"`` (SnapKV with budgets that shrink with depth).
        sinks: First entries of every row that are never evicted (``None``: 4 for ``"streaming"``, else 0).
        recent: Most recent committed entries that are never evicted (``"streaming"``, ``"h2o"`` and
            ``"h2o_prefill"``, which counts them within the prompt). A window at
            least as long as the generation evicts nothing, which is exact.
        heavy: Older entries kept per KV head by score: H2O's heavy hitters, or the prompt entries SnapKV selects.
            An absolute number of entries; SnapKV's budget excludes the observation window (the reference's
            ``max_capacity_prompt`` is ``heavy + observation_window``).
        observation_window: SnapKV: last prompt tokens whose queries score the prompt (always kept). 32 as in
            SnapKV; PyramidKV's per-layer budgets use 8.
        kernel_size: SnapKV: width of the pooling over neighbouring slots (odd; 1 disables pooling).
        pooling: SnapKV: ``"avg"`` or ``"max"`` pooling. The defaults (max pooling, kernel 7) follow the paper's
            experiments; ``pooling="avg", kernel_size=5`` reproduces the SnapKV repository's fallback.
        beta: PyramidKV: ratio of the bottom layer's prompt budget to the top layer's (about ``2 beta - 1``; 20 as
            in the paper and KVCache-Factory).
        layers: 1-indexed layers whose cache is evicted (empty: every layer).

    The method options (:attr:`method_options`) that the chosen method does not read
    (:attr:`EvictionMethod.options <economize.techniques.kv_evict.EvictionMethod.options>`) must keep their defaults.
    """

    technique: ClassVar[str] = "kv_evict"
    #: Fields only some eviction methods read.
    method_options: ClassVar[tuple[str, ...]] = (
        "recent", "heavy", "observation_window", "kernel_size", "pooling", "beta",
    )
    method: str = "streaming"
    sinks: int | None = None
    recent: int = 256
    heavy: int = 256
    observation_window: int = 32
    kernel_size: int = 7
    pooling: str = "max"
    beta: int = 20
    layers: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        check_choice("method", self.method, tuple(EVICTION_METHODS.names()))
        if self.sinks is not None:
            check_non_negative("sinks", self.sinks)
        check_non_negative("recent", self.recent)
        check_non_negative("heavy", self.heavy)
        check_positive("observation_window", self.observation_window)
        check_positive("kernel_size", self.kernel_size)
        if self.kernel_size % 2 == 0:
            raise ConfigError(f"kernel_size must be odd, got {self.kernel_size}")
        check_choice("pooling", self.pooling, POOLINGS)
        check_positive("beta", self.beta)
        check_layers(self.layers)
        check_options(self, f"method={self.method!r}", EVICTION_METHODS.get(self.method).options, self.method_options)

    def validate(self, adapter: ArchAdapter) -> None:
        check_layers(self.layers, adapter)
        check_positional(adapter, self.technique)
        if EVICTION_METHODS.get(self.method).needs_scores and adapter.attn_implementation not in ("sdpa", "eager"):
            raise UnsupportedModelError(
                f"kv_evict method {self.method!r} reads attention scores, which needs attn_implementation "
                f"'sdpa' or 'eager', got {adapter.attn_implementation!r}"
            )


@dataclass
class PendingScores:
    """Attention that the queries of one exact pass gave to one cache layer, kept per query until it is known
    which queries were accepted.

    Attributes:
        layer: Cache layer.
        query_slots: ``[B, Q]`` slot of every query (the trash slot for queries that can no longer be accepted).
        received: ``[B, H_kv, Q, num_slots]`` attention each query gave to every slot, summed over the query heads
            of a KV head (zero for queries below the score floor or at padding).
    """

    layer: int
    query_slots: torch.Tensor
    received: torch.Tensor

    def accepted(self, frontier: torch.Tensor) -> torch.Tensor:
        """``[B, Q]`` float mask of the queries below ``frontier`` (``[B]``, ``-1`` for an unknown frontier)."""
        return (self.query_slots < frontier.to(self.query_slots.device)[:, None]).float()

    def gather(self, rows: torch.Tensor, src: torch.Tensor, dst: torch.Tensor, trash: int) -> None:
        """Follow the cache's tree compaction (:meth:`KVCache.gather`): the slots ``src`` of ``rows`` move to
        ``dst``, and the queries they overwrite can no longer be accepted."""
        r = rows.to(self.received.device)[:, None]
        src, dst = src.to(self.received.device), dst.to(self.received.device)
        self.received[r, :, :, dst] = self.received[r, :, :, src]
        slots = self.query_slots[rows.to(self.query_slots.device)]
        moved = slots[:, :, None] == src[:, None, :]  # [R, Q, J]
        overwritten = (slots[:, :, None] == dst[:, None, :]).any(-1)
        target = dst.gather(1, moved.int().argmax(-1))
        trash_slot = torch.full_like(slots, trash)
        slots = torch.where(moved.any(-1), target, torch.where(overwritten, trash_slot, slots))
        self.query_slots[rows.to(self.query_slots.device)] = slots

    def reorder(self, rows: torch.Tensor) -> None:
        """Keep the batch rows ``rows`` in this order."""
        self.query_slots = self.query_slots.index_select(0, rows.to(self.query_slots.device))
        self.received = self.received.index_select(0, rows.to(self.received.device))


class EvictionCache(CacheWrapper):
    """The cache of one generation under KV eviction (see the module docstring).

    Attributes:
        owner: The technique that made the cache (its layer hooks act only on caches they own).
        keep: Per cache layer, ``[B, H_kv, num_slots]`` booleans; ``False`` marks evicted entries.
        scores: Per cache layer, accumulated attention ``[B, H_kv, num_slots]`` (score-based methods only).
        evicted: Whether any entry was evicted yet (attention masks are left untouched until then).
    """

    def __init__(self, inner: KVCache, owner: KVEvictTechnique) -> None:
        super().__init__(inner)
        self.owner = owner
        self.method: EvictionMethod = owner.method
        self.layers = layer_indices(owner.config.layers, owner.adapter)
        self.keep, self.scores = SlotStore(), SlotStore()
        self.evicted = False
        self.layer: int | None = None  # cache layer of the decoder layer that is running
        self._floor: torch.Tensor | None = None  # lowest query slot that adds to the scores in this pass
        self._info: PassInfo | None = None
        self._pending: list[PendingScores] = []  # attention of the last exact pass's queries, not yet accepted
        # Methods that select once: per layer the kept prompt entries of every row (summed over the KV heads) and the
        # prompt ends, both float64 on the CPU, so that the accounting of later passes needs no device reads.
        self._prompt_kept: dict[int, torch.Tensor] = {}
        self._prompt_ends: torch.Tensor | None = None

    # ---- writes ------------------------------------------------------------------------------------------------
    def transform_update(
        self, key_states: torch.Tensor, value_states: torch.Tensor, layer_idx: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.covers(layer_idx):
            full = (self.inner.batch_size, key_states.shape[1], self.num_slots)
            self.keep.ensure(layer_idx, full, dtype=torch.bool, device=key_states.device, fill=True)
            if self._scoring():
                self.scores.ensure(layer_idx, full, dtype=torch.float32, device=key_states.device)
                zeros = key_states.new_zeros(key_states.shape[:3], dtype=torch.float32)
                self.scores.write(layer_idx, self.active_slots, zeros, self.active_rows)
        return key_states, value_states

    # ---- pass boundaries ---------------------------------------------------------------------------------------
    def pass_begin(self, frontier: list[int | None]) -> None:
        if self.method.selects_once and current_phase() != "prefill":
            # The selection is fixed: the pass only adds to the statistics. Its hooks would do nothing, and no score
            # is pending (the prompt pass adds its attention directly), so the pass's slots are never read here,
            # which ``external_pass`` (a CUDA graph replay) needs, since they are set only inside the graph.
            self._info = None
            self.account(frontier, self._layer_bytes)
            return
        slots = self.active_slots
        if slots.dim() == 1:
            slots = slots[None].expand(self.inner.active_batch_size, -1)
        real = slots != self.trash_slot
        end = torch.where(real, slots + 1, torch.zeros_like(slots)).amax(1)
        start = torch.tensor([-1 if c is None else c for c in frontier], device=slots.device)
        info = self._info = PassInfo(current_phase(), start, end)
        if info.prefill:
            self._prompt_kept, self._prompt_ends = {}, None
        self._commit_pending(start)
        layers = self.owner.adapter.num_layers
        for layer_idx, keep in self.keep.tensors.items():
            self.method.before_pass(keep, self._scores(layer_idx), info.at(layer_idx, layers).to(keep.device))
        if not self.method.selects_once:  # a method that selects once evicts only after the prompt pass
            self._note_evictions()
        self.account(frontier, self._layer_bytes)
        if self.method.needs_scores:
            self._floor = self.method.score_floor(info)

    def pass_scope(self) -> AbstractContextManager[Any]:
        """An :class:`~.tap.AttentionTap` while the pass's ``sdpa`` attention adds to the scores."""
        if self._floor is not None and self.owner.adapter.attn_implementation == "sdpa":
            return AttentionTap(self)
        return nullcontext()

    def pass_end(self) -> None:
        info, self._info, self._floor = self._info, None, None
        if info is None:
            return
        layers = self.owner.adapter.num_layers
        for layer_idx, keep in self.keep.tensors.items():
            self.method.after_pass(keep, self._scores(layer_idx), info.at(layer_idx, layers).to(keep.device))
        if self.method.selects_once:
            if info.prefill:
                self._note_evictions()
                self._count_prompt(info.end.clamp(min=0))
        else:
            self._note_evictions()

    def _count_prompt(self, end: torch.Tensor) -> None:
        """Remember the kept prompt entries of every layer after the prompt pass of a method that selects once."""
        self._prompt_ends = end.cpu().double()
        for layer_idx, keep in self.keep.tensors.items():
            below = torch.arange(keep.shape[-1], device=keep.device) < end.to(keep.device)[:, None]
            self._prompt_kept[layer_idx] = (keep & below[:, None, :]).sum((1, 2)).cpu().double()

    def _scoring(self) -> bool:
        """Whether this pass's writes get score slots. A method that selects once reads its scores only after the
        prompt pass, so later passes leave them alone (and a CUDA graph never writes into them)."""
        return self.method.needs_scores and (not self.method.selects_once or current_phase() == "prefill")

    def _scores(self, layer_idx: int) -> torch.Tensor | None:
        return self.scores[layer_idx] if layer_idx in self.scores else None  # noqa: SIM401  (SlotStore has no get)

    def _commit_pending(self, frontier: torch.Tensor) -> None:
        """Add the attention of the previous exact pass's accepted queries (those below ``frontier``) to the scores.

        Rows with an unknown frontier (``-1``: the pass only writes padding for them) keep their pending queries.
        """
        known = frontier >= 0
        kept: list[PendingScores] = []
        for entry in self._pending:
            scores = self.scores[entry.layer]
            accepted = entry.accepted(frontier).to(scores.device)
            scores += torch.einsum("bhqk,bq->bhk", entry.received.to(scores.device), accepted)
            entry.query_slots[known.to(entry.query_slots.device)] = self.trash_slot
            if bool((entry.query_slots != self.trash_slot).any()):
                kept.append(entry)
        self._pending = kept

    def _note_evictions(self) -> None:
        if not self.evicted:
            self.evicted = any(bool((~keep).any()) for keep in self.keep.tensors.values())

    def _layer_bytes(
        self, layer_idx: int, keys: torch.Tensor, committed: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        whole = self.dense_bytes(keys, committed)
        if layer_idx not in self.keep:
            return whole, whole
        keep = self.keep[layer_idx]
        prompt = self._prompt_kept.get(layer_idx)
        ends = self._prompt_ends
        if prompt is not None and ends is not None and bool(((committed == 0) | (committed >= ends)).all()):
            # every entry after the prompt is kept (the method selected once); padding rows are not accounted
            kept = prompt + keep.shape[1] * (committed - ends).clamp(min=0)
            return kept * 2.0 * keys.shape[3] * keys.element_size(), whole
        below = torch.arange(keep.shape[-1], device=keep.device) < committed.to(keep.device)[:, None]
        kept = (keep & below[:, None, :]).sum((1, 2)).cpu().double()
        return kept * 2.0 * keys.shape[3] * keys.element_size(), whole

    # ---- what attention sees -----------------------------------------------------------------------------------
    def kept(self, layer_idx: int) -> torch.Tensor | None:
        """``[B, H_kv, num_slots]`` booleans of cache layer ``layer_idx``, ``False`` for evicted entries (``None``
        before the layer was written or for a layer the cache does not evict)."""
        return self.keep[layer_idx] if layer_idx in self.keep else None  # noqa: SIM401  (SlotStore has no get)

    @property
    def masks_attention(self) -> bool:
        """Whether :meth:`attention_mask` applies the keep masks (once something was evicted)."""
        return self.evicted

    def attention_mask(self, layer_idx: int, mask: torch.Tensor | None) -> torch.Tensor | None:
        """``mask`` (``[B, 1 or H_q, Q, slots]``, boolean or additive) restricted to the kept entries of cache layer
        ``layer_idx``. Padding queries keep their mask; ``mask`` itself is never modified. An eager pass while
        nothing was evicted (as in the prompt pass) gets ``mask`` back, which spares a long prompt its
        ``[B, H_q, Q, slots]`` copy; a recorded pass always applies the keep masks."""
        if mask is None or not self.masks_attention or layer_idx not in self.keep:
            return mask
        if not self.evicted and not _capturing(mask):
            return mask
        keep = self.keep.view(layer_idx, self.active_rows).to(mask.device)
        keep = keep.repeat_interleave(self.owner.query_group(layer_idx), dim=1)[:, :, None, : mask.shape[-1]]
        slots = self.active_slots.to(mask.device)
        padding = (slots == self.trash_slot).view(-1, 1, slots.shape[-1], 1)
        hidden = ~keep & ~padding
        if mask.dtype == torch.bool:
            return mask & ~hidden
        return mask.masked_fill(hidden, torch.finfo(mask.dtype).min)

    # ---- attention scores (AttentionSink) ----------------------------------------------------------------------
    def wants_probs(self) -> bool:
        return self._floor is not None and self.layer is not None and self.layer in self.scores

    def first_query(self, queries: int) -> int:
        """The first of the running layer's ``queries`` query rows with a nonzero weight (``queries`` if none)."""
        weight = self._query_weight(queries)
        columns = weight.any(0).nonzero()
        return int(columns[0]) if columns.numel() else queries

    def observe(self, probs: torch.Tensor, query_start: int = 0) -> None:
        """Add the attention ``probs`` ``[B, H_q, Q', slots]`` of the running layer's query rows
        ``query_start .. query_start + Q'`` to its scores."""
        if not self.wants_probs():
            return
        assert self.layer is not None
        scores = self.scores[self.layer]
        queries = probs.shape[-2]
        weight = self._query_weight(query_start + queries, probs.shape[0])[:, query_start:]
        kv_heads = scores.shape[1]
        assert self._info is not None
        if self._info.prefill:  # every prompt query is committed
            received = torch.einsum("bhqk,bq->bhk", probs.float().to(scores.device), weight)
            batch, q_heads, keys = received.shape
            received = received.view(batch, kv_heads, q_heads // kv_heads, keys).sum(2)
            rows = self.active_rows
            if rows is None:
                scores[:, :, : keys] += received
            else:
                scores[:, :, : keys].index_add_(0, rows.to(scores.device), received)
            return
        slots = self._query_slots(probs.shape[0])[:, query_start : query_start + queries]
        self._pending.append(self._pending_entry(probs.float().to(scores.device) * weight[:, None, :, None], slots))

    def _query_slots(self, batch: int | None = None) -> torch.Tensor:
        """``[B', Q]`` cache slots of the pass's queries in the active rows."""
        assert self.layer is not None
        slots = self.active_slots.to(self.scores[self.layer].device)
        if slots.dim() == 1:
            slots = slots[None].expand(self.inner.active_batch_size if batch is None else batch, -1)
        return slots

    def _query_weight(self, queries: int, batch: int | None = None) -> torch.Tensor:
        """``[B', queries]`` weight (1 or 0) of the pass's first ``queries`` queries in the active rows: committed
        queries at or above the score floor count."""
        assert self.layer is not None and self._floor is not None
        slots = self._query_slots(batch)[:, :queries]
        floor = self._floor.to(slots.device)
        rows = self.active_rows
        if rows is not None:
            floor = floor.index_select(0, rows.to(slots.device))
        return ((slots != self.trash_slot) & (slots >= floor[:, None])).float()

    def _pending_entry(self, probs: torch.Tensor, slots: torch.Tensor) -> PendingScores:
        """A :class:`PendingScores` of the running layer from weighted ``probs`` ``[B', H_q, Q, K]`` of the active
        rows, whose queries sit at ``slots`` ``[B', Q]``; the other rows get no queries."""
        assert self.layer is not None
        scores = self.scores[self.layer]
        batch, q_heads, queries, keys = probs.shape
        kv_heads = scores.shape[1]
        per_kv = probs.view(batch, kv_heads, q_heads // kv_heads, queries, keys).sum(2)
        received = scores.new_zeros(scores.shape[0], kv_heads, queries, scores.shape[2])
        query_slots = torch.full((scores.shape[0], queries), self.trash_slot, dtype=torch.long, device=scores.device)
        rows = self.active_rows
        index = torch.arange(scores.shape[0], device=scores.device) if rows is None else rows.to(scores.device)
        received[index, :, :, :keys] = per_kv
        query_slots[index] = slots.to(scores.device, torch.long)
        return PendingScores(self.layer, query_slots, received)

    # ---- layout changes ----------------------------------------------------------------------------------------
    def on_gather(self, rows: torch.Tensor, src: torch.Tensor, dst: torch.Tensor) -> None:
        self.keep.gather(rows, src, dst)
        self.scores.gather(rows, src, dst)
        for entry in self._pending:
            entry.gather(rows, src, dst, self.trash_slot)

    def on_reorder(self, rows: torch.Tensor) -> None:
        self._reorder_keep(rows)
        self.scores.reorder(rows)
        index = rows.cpu()
        self._prompt_kept = {layer: kept[index] for layer, kept in self._prompt_kept.items()}
        if self._prompt_ends is not None:
            self._prompt_ends = self._prompt_ends[index]
        for entry in self._pending:
            entry.reorder(rows)

    def _reorder_keep(self, rows: torch.Tensor) -> None:
        self.keep.reorder(rows)

    def on_restart(self) -> None:
        self.keep, self.scores = SlotStore(), SlotStore()
        self.evicted = False
        self.layer = None
        self._floor = None
        self._info = None
        self._pending = []
        self._prompt_kept, self._prompt_ends = {}, None


class WindowEvictionCache(EvictionCache):
    """The cache of one generation under a :attr:`positional <.policies.EvictionMethod.positional>` method.

    Such a method evicts, in every covered layer and KV head, the slots ``sinks <= slot < cut[b]`` of row ``b``,
    where ``cut[b]`` is the largest :meth:`~.policies.EvictionMethod.window_cut` of the row's exact-pass frontiers so
    far. The cache keeps these cuts as a host list and as a ``[B]`` device tensor and builds the hidden range of an
    attention mask from them, so it needs no keep masks, its pass boundaries read nothing back from the device, and
    its byte accounting is computed on the host. It decodes exactly as :class:`EvictionCache` with the method's
    keep masks.

    Over the static cache of CUDA graphs the wrapper lives as long as the cache: one cut tensor per batch size (at a
    fixed address that the recorded mask reads), refilled at every pass boundary, and attention always applies the
    cuts, so the recorded passes do not depend on whether a row has evicted yet. Tree compaction moves only slots at
    or above the frontier, above every cut, so it leaves the cuts alone.
    """

    def __init__(self, inner: KVCache, owner: KVEvictTechnique) -> None:
        super().__init__(inner, owner)
        self._static = self.wraps_static_cache
        self._cuts: list[int] = [0] * inner.batch_size
        self._cut_sets: dict[int, torch.Tensor] = {}  # per batch size: the device cuts

    @property
    def masks_attention(self) -> bool:
        return self.evicted or self._static

    def transform_update(
        self, key_states: torch.Tensor, value_states: torch.Tensor, layer_idx: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return key_states, value_states

    def pass_begin(self, frontier: list[int | None]) -> None:
        self._cuts = [
            cut if f is None else max(cut, self.method.window_cut(f))
            for cut, f in zip(self._cuts, frontier, strict=True)
        ]
        self.evicted = self.evicted or any(cut > self.method.sinks for cut in self._cuts)
        self._sync()
        self.account(frontier, self._window_bytes)

    def pass_end(self) -> None:
        pass

    def _device(self) -> torch.device | None:
        keys = getattr(innermost(self.inner), "keys", None)
        if isinstance(keys, dict) and keys:
            return next(iter(keys.values())).device
        return None

    def _sync(self, device: torch.device | None = None) -> torch.Tensor | None:
        """Copy the host cuts into the device tensor of the current batch size (made on first use)."""
        device = device or self._device()
        if device is None:
            return None
        values = device_tensor(self._cuts, device)
        target = self._cut_sets.get(len(self._cuts))
        if target is None or target.device != device:
            target = self._cut_sets[len(self._cuts)] = values
        else:
            target.copy_(values)
        return target

    def _hidden_counts(self, committed: torch.Tensor) -> torch.Tensor:
        """``[B]`` float64 evicted entries per KV head below ``committed`` (``[B]`` float64)."""
        cut = torch.tensor(self._cuts, dtype=torch.float64)[: committed.shape[0]]
        return (torch.minimum(cut, committed) - self.method.sinks).clamp(min=0)

    def _window_bytes(
        self, layer_idx: int, keys: torch.Tensor, committed: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        whole = self.dense_bytes(keys, committed)
        return self.dense_bytes(keys, committed - self._hidden_counts(committed)), whole

    def kept(self, layer_idx: int) -> torch.Tensor | None:
        buffers = self.positional_buffers()
        if not self.covers(layer_idx) or layer_idx not in buffers:
            return None
        kv_heads = buffers[layer_idx].shape[1]
        slot = torch.arange(self.num_slots)
        cut = torch.tensor(self._cuts)
        hidden = (slot >= self.method.sinks) & (slot < cut[:, None])
        return ~hidden[:, None, :].expand(-1, kv_heads, -1)

    def attention_mask(self, layer_idx: int, mask: torch.Tensor | None) -> torch.Tensor | None:
        """``mask`` (``[B, 1 or H_q, Q, slots]``, boolean or additive) without the evicted range of every row.
        Padding queries keep their mask; ``mask`` itself is never modified. An eager pass while no row has evicted
        (as in the prompt pass) gets ``mask`` back, which spares a long prompt its ``[B, H_q, Q, slots]`` copy; a
        recorded pass always applies the cuts."""
        if mask is None or not self.masks_attention or not self.covers(layer_idx):
            return mask
        if not _capturing(mask) and all(cut <= self.method.sinks for cut in self._cuts):
            return mask
        cut = self._cut_sets.get(len(self._cuts))
        if cut is None or cut.device != mask.device:
            cut = self._sync(mask.device)
            assert cut is not None
        rows = self.active_rows
        if rows is not None:
            cut = cut.index_select(0, rows.to(cut.device))
        slot = torch.arange(mask.shape[-1], device=mask.device)
        hidden = (slot >= self.method.sinks) & (slot < cut[:, None])  # [B', slots]
        slots = self.active_slots.to(mask.device)
        padding = (slots == self.trash_slot).view(-1, 1, slots.shape[-1], 1)
        hidden = hidden[:, None, None, :] & ~padding
        heads = self.evicting_heads(layer_idx, mask.device)
        if heads is not None:
            hidden = hidden & heads
        if mask.dtype == torch.bool:
            return mask & ~hidden
        return mask.masked_fill(hidden, torch.finfo(mask.dtype).min)

    def evicting_heads(self, layer_idx: int, device: torch.device) -> torch.Tensor | None:
        """``[1, H_q, 1, 1]`` booleans of the query heads whose window cache layer ``layer_idx`` evicts, or
        ``None`` for every head (the default). A subclass that returns a mask must keep it at a fixed address on
        ``device`` while the cache lives (CUDA graphs read it)."""
        return None

    def on_gather(self, rows: torch.Tensor, src: torch.Tensor, dst: torch.Tensor) -> None:
        pass

    def on_reorder(self, rows: torch.Tensor) -> None:
        self._cuts = [self._cuts[r] for r in rows.tolist()]
        self._sync()

    def on_restart(self) -> None:
        super().on_restart()
        self._cuts = [0] * self.batch_size
        self._sync()


@register_technique
class KVEvictTechnique(KVCacheTechnique):
    """KV-cache eviction (see :class:`KVEvictConfig`). Training-free; no parameters.

    A :attr:`positional <.policies.EvictionMethod.positional>` method (StreamingLLM) gets a
    :class:`WindowEvictionCache` and runs under CUDA graphs; the others keep per-head keep masks and decode eagerly.
    """

    name: ClassVar[str] = KVEvictConfig.technique
    config_class: ClassVar[type[KVEvictConfig]] = KVEvictConfig
    info = TechniqueInfo(
        family=Family.KV_EVICTION,
        saves=Saving.KV_MEMORY | Saving.COMPUTE,
        lossless=False,
        simulated=Simulation(Saving.KV_MEMORY | Saving.COMPUTE, "masks the evicted entries, which stay in the cache"),
    )
    order: ClassVar[int] = 20
    config: KVEvictConfig

    def __init__(self, config: KVEvictConfig, adapter: ArchAdapter) -> None:
        super().__init__(config, adapter)
        self.method = EVICTION_METHODS.get(config.method)(config)
        if self.method.positional:
            self.graph_safe = True  # type: ignore[misc]  # per method, see WindowEvictionCache
        self.running: EvictionCache | None = None  # the cache of the decoder layer that is running
        self._handles: list[Any] = []

    def make_wrapper(self, cache: KVCache) -> CacheWrapper:
        if self.method.positional:
            return WindowEvictionCache(cache, self)
        return EvictionCache(cache, self)

    def query_group(self, cache_layer: int) -> int:
        """Query heads that share one KV head in cache layer ``cache_layer`` (0-indexed)."""
        geometry = self.adapter.head_geometry(cache_layer + 1)
        return geometry.heads // geometry.kv_heads

    def attach(self, model: nn.Module, adapter: ArchAdapter, context: AttachContext | None = None) -> None:
        self._install()

    def detach(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles = []

    def refresh(self) -> None:
        self.detach()
        self._install()

    def _install(self) -> None:
        eager = self.method.needs_scores and self.adapter.attn_implementation == "eager"
        for index in range(1, self.adapter.num_layers + 1):
            layer = self.adapter.layer(index)
            pre, post = _layer_hooks(self, index)
            self._handles.append(layer.register_forward_pre_hook(pre, with_kwargs=True))
            self._handles.append(layer.register_forward_hook(post, with_kwargs=True, always_call=True))
            attention = self.adapter.attention_module(index)
            if eager and attention is not None:
                self._handles.append(attention.register_forward_hook(_weights_hook(self)))

    def describe(self) -> dict[str, Any]:
        """The effective eviction budget."""
        return self.method.describe()


def _capturing(mask: torch.Tensor) -> bool:
    """Whether a CUDA graph is being recorded (the recorded pass must not depend on what was evicted so far)."""
    return mask.is_cuda and torch.cuda.is_current_stream_capturing()


def _layer_hooks(technique: KVEvictTechnique, index: int) -> tuple[Any, Any]:
    """Pre- and post-hooks of decoder layer ``index``. Closures (not bound methods) so that a deep copy of the layer
    keeps hooks that point at the same technique."""

    def pre(module: nn.Module, args: tuple, kwargs: dict) -> tuple[tuple, dict] | None:
        cache = find_wrapper(kwargs.get("past_key_values"), EvictionCache, owner=technique)
        if cache is None:
            return None
        cache.layer = technique.adapter.kv_source(index) - 1
        technique.running = cache
        mask = kwargs.get("attention_mask")
        masked = cache.attention_mask(cache.layer, mask)
        if masked is mask:
            return None
        return args, {**kwargs, "attention_mask": masked}

    def post(module: nn.Module, args: tuple, kwargs: dict, output: Any) -> None:
        cache = find_wrapper(kwargs.get("past_key_values"), EvictionCache, owner=technique)
        if cache is not None:
            cache.layer = None
        technique.running = None

    return pre, post


def _weights_hook(technique: KVEvictTechnique) -> Any:
    """Forward hook of an eager attention module: its returned weights go to the running layer's cache."""

    def hook(module: nn.Module, args: tuple, output: Any) -> None:
        weights = output[1] if isinstance(output, tuple) and len(output) > 1 else None
        cache = technique.running
        if weights is not None and cache is not None:
            cache.observe(weights)

    return hook
