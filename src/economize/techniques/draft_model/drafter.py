"""The drafter of two-model speculative decoding: a separate draft LM with its own KV cache."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import torch

from ...decoding.cost import CostKind, LayerFootprint
from ...decoding.graphs import static_cache_class
from ...decoding.protocols import Draft, Drafter, draft_footprint
from ...runtime import CacheSpec
from .._drafting import GraphedStep

if TYPE_CHECKING:
    from ...decoding.request import DraftRequest
    from ...runtime import ExactPass, KVCache, LayerRunner

__all__ = ["DraftModelDrafter"]

#: Cost-log kind of the committed tokens a draft LM processes before its first draft of a cycle (an alias of
#: :data:`~economize.decoding.CostKind.DRAFT_CATCHUP`).
CATCHUP = CostKind.DRAFT_CATCHUP
#: The network name of the draft LM's events in the cost log (``CostModel(draft=...)`` prices it).
MODULE = "draft_model"


class DraftModelDrafter(Drafter):
    """Drafts with a separate LM for a batch of rows.

    Per row the drafter remembers how many positions the draft LM's cache holds (``done``), which token it read at
    each of them and where the row's last catch-up ended (``anchor``): the positions before the anchor hold
    committed tokens, the later ones drafts. Before a row drafts, the drafts it read are compared with the committed
    tokens, and the cache stays valid up to the first draft that differs (the verifier accepts a prefix of the
    drafts; a row that skipped cycles, e.g. because a hybrid's lookup drafted it, may have committed other tokens
    there). The first call of a cycle processes the owed tokens ``valid .. end - 1`` of all rows in one padded pass
    (rows owe one token after a rejection, two after a fully accepted cycle and the prompt in the first cycle) and
    yields the first draft; later calls process one draft token each.

    Cost log: every call records a ``"draft"`` event of one token over layers ``1..L_draft`` of the network
    ``"draft_model"`` (:attr:`footprint`; :attr:`layer_span` is ``(1, 0)``, no layer of the model); the owed
    tokens beyond the first are recorded as a ``"draft_catchup"`` event of the same network before it.

    CUDA graphs (:meth:`enable_graphs`): the one-token draft steps are replayed from graphs, the catch-up of a cycle
    (its width varies) runs eagerly. The draft LM's cache is then the static variant of its cache kind, kept across
    generations and replaced (dropping the graphs) only when a generation needs more positions, and the per-row
    token record lives in one persistent buffer per batch size.

    Args:
        runner: The layer runner of the draft LM.
        vocab_size: Vocabulary size of the model; draft logits are cut or padded with ``-inf`` to it.

    Attributes:
        clamped_tokens: How many model tokens beyond the draft LM's vocabulary the draft LM read since
            :meth:`begin` (it reads its last token in their place, so the drafts that follow them are poor).
    """

    def __init__(self, runner: LayerRunner, vocab_size: int) -> None:
        self.draft_runner = runner
        self.vocab_size = vocab_size
        self.draft_vocab = runner.adapter.vocab_size
        self.layer_span = (1, 0)
        self.footprint = LayerFootprint(1, runner.adapter.num_layers, MODULE)
        self._cache: KVCache | None = None
        self._done: list[int] = []  # per row, positions the draft LM's cache holds
        self._anchor: list[int] = []  # per row, positions read by catch-ups (committed tokens)
        self._seen: torch.Tensor | None = None  # [B, capacity + 1] token read at each position; last column scratch
        self._clamped: torch.Tensor | None = None  # count of clamped tokens, on the device (no synchronisation)
        self._step_fn = GraphedStep(self._step)
        self._seen_buffers: dict[int, torch.Tensor] = {}  # graph mode: ``_seen`` per batch size

    @property
    def clamped_tokens(self) -> int:
        return 0 if self._clamped is None else int(self._clamped)

    # ---- generation lifecycle ----------------------------------------------------------------------------------
    def enable_graphs(self, wrap: Callable[[Callable[..., Any]], Callable[..., Any]]) -> bool:
        if static_cache_class(self.draft_runner.adapter.cache_kind) is None:
            return False
        self._step_fn.enable(wrap)
        return True

    def begin(self, runner: LayerRunner, cache: KVCache, batch_size: int) -> None:
        device = self.draft_runner.adapter.device
        self._done = [0] * batch_size
        self._anchor = [0] * batch_size
        if not self._step_fn.enabled:
            self._cache = self.draft_runner.new_cache(cache.capacity, batch_size)
            self._seen = torch.full((batch_size, cache.capacity + 1), -1, dtype=torch.long, device=device)
            self._clamped = torch.zeros((), dtype=torch.long, device=device)
            return
        if self._cache is None or self._cache.capacity < cache.capacity:
            static = static_cache_class(self.draft_runner.adapter.cache_kind)
            assert static is not None
            self._cache = static(CacheSpec(cache.capacity, batch_size, self.draft_runner.adapter.cache_kind))
            self._seen_buffers.clear()
            self._clamped = torch.zeros((), dtype=torch.long, device=device)
            self._step_fn.reset()
        else:
            self._cache.activate(batch_size)  # type: ignore[attr-defined]
        assert self._clamped is not None
        self._clamped.zero_()
        self._seen = self._seen_buffer(batch_size).fill_(-1)

    def _seen_buffer(self, batch_size: int) -> torch.Tensor:
        """Graph mode: the persistent ``_seen`` buffer of ``batch_size`` rows."""
        assert self._cache is not None
        if batch_size not in self._seen_buffers:
            self._seen_buffers[batch_size] = torch.full(
                (batch_size, self._cache.capacity + 1), -1, dtype=torch.long, device=self.draft_runner.adapter.device
            )
        return self._seen_buffers[batch_size]

    def observe(self, exact: ExactPass, columns: torch.Tensor) -> None:
        pass

    def reorder(self, rows: torch.Tensor) -> None:
        if self._cache is None or self._seen is None:
            return
        rows = rows.to(self.draft_runner.adapter.device)
        self._cache.reorder(rows)
        seen = self._seen.index_select(0, rows)
        self._seen = self._seen_buffer(len(rows)).copy_(seen) if self._step_fn.enabled else seen
        order = rows.tolist()
        self._done = [self._done[i] for i in order]
        self._anchor = [self._anchor[i] for i in order]

    # ---- drafting ---------------------------------------------------------------------------------------------
    def propose(self, request: DraftRequest) -> Draft:
        if self._cache is None:
            raise RuntimeError("DraftModelDrafter.propose needs begin() first")
        if request.drafting:
            request.extend(self._catch_up(request))
        while request.drafting:
            active = request.active
            logits = self.step(request.tokens, request.positions, request.valid)
            positions = request.positions.tolist()
            for b, on in enumerate(active):
                if on:
                    self._done[b] = positions[b] + 1
            request.extend(logits)
        return request.draft()

    def step(self, tokens: torch.Tensor, positions: torch.Tensor, valid: torch.Tensor | None) -> torch.Tensor:
        return self._step_fn(tokens, positions, valid)

    def _step(self, tokens: torch.Tensor, positions: torch.Tensor, valid: torch.Tensor | None) -> torch.Tensor:
        return self._run(tokens[:, None], positions[:, None], None if valid is None else valid[:, None])[:, 0]

    def _verified(self, request: DraftRequest) -> list[int]:
        """Per row, how many leading positions of the draft LM's cache hold the committed tokens.

        Positions before the row's anchor were read by a catch-up. The drafts read after it are valid up to the first
        one that differs from the committed token at its position.
        """
        assert self._seen is not None
        width = max(done - anchor for done, anchor in zip(self._done, self._anchor))
        if width <= 0:
            return list(self._done)
        device = self._seen.device
        anchor = torch.tensor(self._anchor, device=device)
        positions = anchor[:, None] + torch.arange(width, device=device)
        held = positions < torch.tensor(self._done, device=device)[:, None]
        positions = positions.clamp(max=self._seen.shape[1] - 1)
        committed = request.seq.to(device).gather(1, positions.clamp(max=request.scratch))
        same = (self._seen.gather(1, positions) == committed) & held
        return (anchor + same.long().cumprod(1).sum(1)).tolist()

    def _catch_up(self, request: DraftRequest) -> torch.Tensor:
        """Process every active row's owed tokens and return the logits of its first draft ``[B, V]``."""
        active = request.active
        starts = [min(verified, end - 1) for verified, end in zip(self._verified(request), request.ends)]
        owed = [end - start if on else 0 for start, end, on in zip(starts, request.ends, active)]
        width = max(owed)
        device = request.device
        offsets = torch.arange(width, device=device)
        start = torch.tensor(starts, device=device)
        count = torch.tensor(owed, device=device)
        positions = start[:, None] + offsets
        valid = offsets < count[:, None]
        tokens = request.seq.gather(1, positions.clamp(max=request.scratch))
        last = (count - 1).clamp(min=0)
        logits = self._run(tokens, positions.masked_fill(~valid, 0), valid, keep=last[:, None])
        for b, on in enumerate(active):
            if on:
                if owed[b] > 1:
                    request.costs[b].record_footprint(CATCHUP, owed[b] - 1, starts[b], draft_footprint(self))
                self._done[b] = self._anchor[b] = request.ends[b]
        return logits[:, 0]

    def _run(self, tokens: torch.Tensor, positions: torch.Tensor, valid: torch.Tensor | None,
             keep: torch.Tensor | None = None) -> torch.Tensor:
        """Draft-LM logits ``[B, S, V]`` (cut or padded to the model's vocabulary) for ``tokens`` ``[B, S]``.

        ``keep`` (``[B, K]`` indices into ``S``) computes the logits of those columns only (``[B, K, V]``): a
        catch-up over a long prompt then never materializes ``S`` rows of vocabulary-wide logits.
        """
        assert self._cache is not None and self._seen is not None
        runner = self.draft_runner
        device = runner.adapter.device
        tokens, positions = tokens.to(device), positions.to(device)
        valid = None if valid is None else valid.to(device)
        written = positions if valid is None else positions.masked_fill(~valid, self._seen.shape[1] - 1)
        self._seen.scatter_(1, written, tokens)
        if self.draft_vocab < self.vocab_size and self._clamped is not None:
            beyond = tokens >= self.draft_vocab
            self._clamped += (beyond if valid is None else beyond & valid).sum()
        ctx = runner.context(self._cache, positions, valid, trusted=True)
        hidden, _ = runner.run(runner.embed(tokens.clamp(max=self.draft_vocab - 1), ctx), ctx, 1,
                               runner.adapter.num_layers)
        if keep is not None:
            hidden = hidden.gather(1, keep.to(hidden.device)[..., None].expand(-1, -1, hidden.shape[-1]))
        logits = runner.logits(hidden).float()
        if self.draft_vocab > self.vocab_size:
            return logits[..., : self.vocab_size]
        if self.draft_vocab < self.vocab_size:
            pad = logits.new_full((*logits.shape[:-1], self.vocab_size - self.draft_vocab), float("-inf"))
            return torch.cat([logits, pad], -1)
        return logits
