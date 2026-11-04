"""The state of one :meth:`~economize.decoding.CycleDecoder.decode_batch` call and its batch bookkeeping.

A :class:`Generation` holds the token buffer, the cache and the per-row counters of a ragged batch. The functions of
:mod:`.passes`, :mod:`.commit`, :mod:`.drafting` and :mod:`.plain` read and update it; :meth:`Generation.drop`
removes finished rows from the batch and from every component that keeps per-row state (cache, drafter, layer
policy, recurrent snapshot).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import torch

from ..errors import ConfigError
from ..runtime import KVCache, Phase, RecurrentSnapshot, RingSpec, device_tensor
from .cost import CostKind, CostLog
from .protocols import TokenSelector
from .request import group_rows
from .result import DecodeResult
from .stop import StopChecker

if TYPE_CHECKING:
    from ..runtime import LayerRunner
    from .config import DecodeConfig
    from .loop import CycleDecoder

__all__ = ["Generation", "window_ring"]

#: Page size of the window cache (:class:`~economize.runtime.RingSpec`); FlashAttention 2 reads paged buffers in
#: multiples of 256 slots.
RING_PAGE = 256


class Generation:
    """The state of one :meth:`~economize.decoding.CycleDecoder.decode_batch` call.

    Lists indexed by ``b`` describe the rows still in the batch; lists indexed by ``row`` (``selectors``, ``stops``,
    ``max_new``, ``prompt_lengths``, ``costs``, ``acceptance``, ``results``) are per original row, and ``rows[b]`` maps
    one to the other.

    :attr:`snapshot` follows one cycle of a model with recurrent layers: :func:`~.drafting.draft` saves the
    recurrent states before the drafter runs and restores them after it, and keeps the snapshot. The next exact pass
    then advances the states over every token it sees, rejected drafts and padding included, and
    :func:`~.commit.replay` (called from :func:`~.commit.verify_and_commit`) restores the snapshot, re-runs the
    committed tokens of the rows that went too far and clears it.

    Args:
        decoder: The decoder whose components this generation uses.
        prompts: One 1-D prompt per row.
        max_new: Most new tokens per row.
        selectors: Token selector per row.
        stops: Stop condition per row.
        selector: The default selector, kept for drafters that read :attr:`DraftRequest.selector`.
    """

    def __init__(
        self,
        decoder: CycleDecoder,
        prompts: list[torch.Tensor],
        max_new: list[int],
        selectors: list[TokenSelector],
        stops: list[StopChecker],
        selector: TokenSelector,
    ) -> None:
        runner = decoder.runner
        self.runner, self.config, self.verifier, self.policy = runner, decoder.config, decoder.verifier, decoder.policy
        self.observers, self.layer_policy = decoder.observers, decoder.layer_policy
        self.selector, self.selectors, self.stops, self.max_new = selector, selectors, stops, max_new
        self.uniform = all(s is selectors[0] for s in selectors)
        self.num_layers = runner.adapter.num_layers
        self.device = runner.adapter.device
        n_cycle = self.config.cycle_length
        self.drafter = decoder.active_drafter
        self.capture = decoder.capture  # the same layers the decoder's graphs capture
        self.phase: Phase = "decode" if n_cycle == 1 else "refresh"

        prompts = [p.to(self.device).view(-1) for p in prompts]
        if any(p.numel() == 0 for p in prompts):
            raise ConfigError("every prompt needs at least one token")
        self.prompt_lengths = [p.numel() for p in prompts]
        span = self.drafter.draft_capacity(n_cycle) + 1 if self.drafter is not None else 1
        self.capacity = max(self.prompt_lengths) + max(max(max_new), 0) + max(n_cycle, span) + 1
        batch = len(prompts)
        self.seq = torch.zeros(batch, self.capacity + 1, dtype=torch.long, device=self.device)
        for b, prompt in enumerate(prompts):
            self.seq[b, : prompt.numel()] = prompt
        self.rows = list(range(batch))  # original index of every row still in the batch
        self.ends = list(self.prompt_lengths)  # committed tokens of every row
        self.starts = [0] * batch  # position of every row's first token that no exact pass has seen
        self.costs = [CostLog() for _ in range(batch)]
        self.acceptance: list[list[tuple[int, int]]] = [[] for _ in range(batch)]
        self.confidence: list[list[torch.Tensor | None]] = [[] for _ in range(batch)]  # per acceptance entry
        self.match: list[list[torch.Tensor | None]] = [[] for _ in range(batch)]  # per acceptance entry
        self.results: list[DecodeResult | None] = [None] * batch
        self.snapshot: RecurrentSnapshot | None = None
        self.prefilled = False  # whether the layer policy has seen this generation's rows (set by the prefill)
        self.graphs = decoder.graphs
        ring = window_ring(runner, self.config, max(n_cycle, span)) if self.config.window_cache else None
        if ring is not None and ring.buffer_slots >= self.capacity + 1:
            ring = None  # the prompt and its continuation fit in the window: a ring would not save memory
        if self.graphs is not None:
            cache: KVCache = self.graphs.cache_for(self.capacity, batch, ring)
        else:
            cache = runner.new_cache(self.capacity, batch, ring=ring)
        self.cache: KVCache = decoder.wrap_cache(cache) if decoder.wrap_cache is not None else cache
        if self.graphs is not None:
            self.graphs.bind(self.cache)

    @property
    def scratch(self) -> int:
        """Column of :attr:`seq` that padding tokens read (never a committed token)."""
        return self.seq.shape[1] - 1

    def groups(self) -> list[tuple[TokenSelector, list[int]]]:
        """The rows of the batch grouped by selector."""
        if self.uniform:
            return [(self.selectors[self.rows[0]], list(range(len(self.rows))))]
        return group_rows([self.selectors[row] for row in self.rows])

    def remaining(self, b: int) -> int:
        """New tokens row ``b`` may still commit."""
        row = self.rows[b]
        return self.max_new[row] - (self.ends[b] - self.prompt_lengths[row])

    def record_exact(self, tokens: Sequence[int], starts: Sequence[int]) -> None:
        """Log one exact pass of all ``L`` layers over ``tokens[b]`` tokens from position ``starts[b]``, for every row
        ``b`` of the batch."""
        for row, n, start in zip(self.rows, tokens, starts, strict=True):
            self.costs[row].record(CostKind.EXACT, n, start, 1, self.num_layers)

    def advance(self, b: int, token: int) -> bool:
        """Commit ``token`` (already in ``seq``) to row ``b``; return whether the row is finished."""
        position = self.ends[b]
        self.ends[b] += 1
        if self.remaining(b) <= 0:
            return True
        return self.stops[self.rows[b]](token, self.seq[b], position)

    def drop(self, finished: list[bool]) -> None:
        """Store the results of finished rows and remove them from the batch (and from the per-row state of the
        cache, the drafter, the layer policy and the recurrent snapshot)."""
        if not any(finished):
            return
        keep = [b for b, done in enumerate(finished) if not done]
        for b, done in enumerate(finished):
            if done:
                self.finish(b)
        rows = device_tensor(keep, self.device)
        self.rows = [self.rows[b] for b in keep]
        self.ends = [self.ends[b] for b in keep]
        self.starts = [self.starts[b] for b in keep]
        if not keep:
            return
        self.seq = self.seq.index_select(0, rows)
        self.cache.reorder(rows)
        if self.drafter is not None:
            self.drafter.reorder(rows)
        if self.layer_policy is not None and self.prefilled:
            self.layer_policy.reorder(rows)
        if self.snapshot is not None:

            def pick(state: torch.Tensor | None) -> torch.Tensor | None:
                return None if state is None else state.index_select(0, rows.to(state.device))

            self.snapshot = {layer: (pick(first), pick(second)) for layer, (first, second) in self.snapshot.items()}

    def finish(self, b: int) -> None:
        """Store the result of row ``b``."""
        row = self.rows[b]
        prompt_length = self.prompt_lengths[row]
        self.results[row] = DecodeResult(
            tokens=self.seq[b, prompt_length : self.ends[b]].clone(),
            cost=self.costs[row],
            acceptance=self.acceptance[row],
            confidence=_host_cycles(self.confidence[row]),
            match=_host_cycles(self.match[row]),
            prompt_length=prompt_length,
            num_layers=self.num_layers,
        )

    def outputs(self) -> list[DecodeResult]:
        """One result per original row, in order (every row must have finished)."""
        assert all(result is not None for result in self.results)
        return list(self.results)  # type: ignore[arg-type]


def _host_cycles(cycles: list[torch.Tensor | None]) -> list[Any]:
    """Per-draft values of one row's verified cycles (``[k]`` or ``[k, m]`` each) on the host, with one transfer
    (``[]`` for a cycle without them, and no entries at all when none was recorded)."""
    known = [c for c in cycles if c is not None]
    if not known:
        return []
    values = torch.cat([c.float() for c in known]).tolist()
    out, i = [], 0
    for c in cycles:
        n = 0 if c is None else c.shape[0]
        out.append(values[i : i + n])
        i += n
    return out


def window_ring(runner: LayerRunner, config: DecodeConfig, ahead: int) -> RingSpec | None:
    """The window cache of the model's sliding layers for :attr:`DecodeConfig.window_cache`, or ``None`` for a model
    without sliding layers. Passes hold up to :attr:`DecodeConfig.prefill_chunk` queries and drafts run up to
    ``ahead`` tokens past the committed ones.

    Raises:
        ConfigError: The sliding layers have different windows, or one of them attends over the cache of a layer
            that is not sliding (a KV-sharing layer).
    """
    adapter = runner.adapter
    sliding = [i for i in range(1, adapter.num_layers + 1) if adapter.attention_spec(i).kind == "sliding"]
    if not sliding:
        return None
    windows: set[int] = set()
    for i in sliding:
        window = adapter.attention_spec(i).window
        if window is None:
            raise ConfigError(f"window_cache: sliding layer {i} has no window")
        windows.add(window)
    if len(windows) > 1:
        raise ConfigError(f"window_cache needs one window for every sliding layer, got {sorted(windows)}")
    for i in sliding:
        source = adapter.kv_source(i)
        if adapter.attention_spec(source).kind != "sliding":
            raise ConfigError(
                f"window_cache: sliding layer {i} reads the cache of layer {source}, which is not sliding"
            )
    queries = max(config.prefill_chunk or 1, ahead + 1)  # DecodeConfig refuses window_cache without prefill_chunk
    layers = frozenset(adapter.kv_source(i) - 1 for i in sliding)
    return RingSpec.for_pass(layers, windows.pop(), queries=queries, ahead=ahead, page=RING_PAGE)
