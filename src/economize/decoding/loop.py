"""The draft-and-refresh decoding loop shared by every drafting technique.

One cycle, for every row of a batch at once:

1. An exact pass of the unmodified model runs, in parallel, over every token of a row that has not been through an
   exact pass yet: the last exact token and the drafts made after it (a chain, or the nodes of a draft tree). It
   writes exact K/V into the cache and yields the logits of the next position for each of these tokens.
2. The verifier compares the drafts with these logits (lossless decoding) or keeps them (lossy decoding) and
   chooses one further token from the exact logits.
3. The drafter takes the exact hidden states of the last accepted token and proposes up to ``cycle_length - 1``
   tokens cheaply, writing approximate K/V into the cache. The next exact pass overwrites them.

Rows are ragged. The prefill is left-padded (row ``b``'s prompt ends in the last column); later passes are
right-padded (every row starts in column 0 at its own position). Rows that finish leave the batch. Models with
recurrent layers (hybrids) restore their recurrent states around drafting and re-run the committed tokens of rows
whose pass included rejected drafts or padding (:meth:`~economize.runtime.LayerRunner.replay`).

With ``cycle_length = 1`` the loop is ordinary greedy or sampled decoding with a static cache.

With ``DecodeConfig.cuda_graphs`` the exact chain and tree passes, and the draft steps of drafters that opt in, are
replayed from CUDA graphs (:mod:`economize.decoding.graphs`). Prefills stay eager.

This module holds the decoder and its cycle; the parts of a cycle live in :mod:`.generation` (the state of one
call and its batch bookkeeping), :mod:`.passes` (exact passes), :mod:`.commit` (verification and commits),
:mod:`.drafting` (the drafting step), :mod:`.plain` (drafter-free decoding with windowed host reads) and
:mod:`.stop` (stop conditions).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import torch

from ..errors import ConfigError
from ..runtime import CACHES, KVCache, LayerPolicy, LayerRunner
from .commit import verify_and_commit
from .config import DecodeConfig
from .drafting import draft
from .generation import Generation
from .graphs import DecodeGraphs
from .passes import chain_pass, prefill, tree_pass
from .plain import run_plain
from .policies import POLICIES
from .protocols import Draft, Drafter, TokenSelector, Verifier, VerifyObserver
from .result import DecodeResult
from .selectors import GreedySelector
from .stop import StopChecker, StopCondition
from .verifiers import VERIFIERS

__all__ = ["CycleDecoder", "DecodeResult", "StopCondition", "VerifyObserver"]


class CycleDecoder:
    """Decodes with the unmodified model plus an optional drafter.

    Args:
        runner: Runs the model's layers.
        drafter: Proposes tokens between exact passes. Required when ``config.cycle_length > 1``.
        config: Decoding options.
        layer_policy: Applied to every exact pass (dynamic-depth techniques).
        observers: Called with a :class:`VerifyEvent` per row after every exact pass.
        wrap_cache: Wraps the KV cache of every generation (KV-cache techniques, e.g. quantized or sparse caches).
        graph_safe: Whether the attached techniques allow CUDA graphs (``config.cuda_graphs``). Graphs also need a
            CUDA model on one device, a cache kind with a static variant (:meth:`DecodeGraphs.supports_cache`), no
            recurrent layers, no ``layer_policy``, and no ``wrap_cache`` unless it has a true ``graph_safe``
            attribute (a :class:`~economize.core.cache_wrapping.CacheWrapping` of graph-safe KV-cache techniques);
            otherwise the decoder runs eagerly.

    Attributes:
        active_drafter: The drafter that runs, ``None`` with ``cycle_length = 1`` (ordinary decoding ignores it).
        capture: Layers whose hidden states every exact pass returns: the observed layers of :attr:`active_drafter`.
            The eager passes and the graphed passes capture exactly these.
        graphs: The decoder's CUDA graphs and static cache, ``None`` when it runs eagerly. They are kept across
            :meth:`decode_batch` calls, so reuse one decoder for repeated generations.

    Raises:
        ConfigError: ``cycle_length > 1`` without a drafter, ``lossless=True`` with a verifier that commits drafts,
            a verifier that both commits drafts and claims to verify trees, or a drafter that drafts trees
            (:meth:`~economize.decoding.Drafter.drafts_trees`) with a verifier that cannot verify them. Unknown verifier
            options raise from the verifier.
    """

    def __init__(
        self,
        runner: LayerRunner,
        drafter: Drafter | None,
        config: DecodeConfig,
        *,
        layer_policy: LayerPolicy | None = None,
        observers: Sequence[VerifyObserver] = (),
        wrap_cache: Callable[[KVCache], KVCache] | None = None,
        graph_safe: bool = True,
    ) -> None:
        if config.cycle_length > 1 and drafter is None:
            raise ConfigError("cycle_length > 1 needs a drafting technique; apply one or use cycle_length=1")
        self.runner, self.drafter, self.config = runner, drafter, config
        self.layer_policy, self.observers, self.wrap_cache = layer_policy, tuple(observers), wrap_cache
        self.verifier = VERIFIERS.get(config.verifier_name)(config)
        self.policy = POLICIES.get(config.draft_policy_name)(config)
        if drafter is not None and config.cycle_length > 1:
            drafter = self.policy.wrap(drafter, self.verifier)
            self.drafter = drafter
        _check_combination(config, self.verifier, drafter)
        self.active_drafter: Drafter | None = drafter if config.cycle_length > 1 else None
        if config.prefill_chunk is not None:
            self._check_chunked_prefill()
        self.capture: tuple[int, ...] = (
            tuple(self.active_drafter.observed_layers) if self.active_drafter is not None else ()
        )
        self.graphs: DecodeGraphs | None = None
        if config.cuda_graphs and graph_safe and self._graphable():
            self.graphs = DecodeGraphs(runner, self.capture, "decode" if config.cycle_length == 1 else "refresh")
            if self.active_drafter is not None:
                self.active_drafter.enable_graphs(self.graphs.wrap)

    def _check_chunked_prefill(self) -> None:
        """Raise :class:`ConfigError` for a decoder that cannot run :attr:`DecodeConfig.prefill_chunk` (and
        :attr:`DecodeConfig.window_cache`), before any decoding."""
        config, drafter = self.config, self.active_drafter
        reasons = []
        if self.layer_policy is not None:
            reasons.append("a layer policy (it decides per pass over whole prompts)")
        if self.observers:
            reasons.append("verify observers (they read every prompt position)")
        if self.runner.has_recurrent_layers:
            reasons.append("recurrent layers")
        if drafter is not None and not drafter.streams_prompt:
            reasons.append(f"{type(drafter).__name__}, which reads the hidden states of every prompt position")
        if config.window_cache:
            if drafter is not None and drafter.drafts_trees(self.verifier):
                reasons.append(f"window_cache with {type(drafter).__name__}, which drafts trees")
            cache = CACHES.get(self.runner.adapter.cache_kind)
            if not cache.supports_ring or ((static := cache.static_variant) is not None and not static.supports_ring):
                reasons.append(f"window_cache with the {self.runner.adapter.cache_kind!r} cache, which has no ring")
        if reasons:
            raise ConfigError("prefill_chunk cannot be combined with " + "; ".join(reasons))

    def _graphable(self) -> bool:
        """Whether the exact passes of this decoder can be CUDA graphs."""
        adapter = self.runner.adapter
        if self.layer_policy is not None or self.runner.has_recurrent_layers:
            return False
        if self.wrap_cache is not None and not getattr(self.wrap_cache, "graph_safe", False):
            return False
        if adapter.device.type != "cuda" or not DecodeGraphs.supports_cache(adapter.cache_kind):
            return False
        return len({p.device for p in adapter.model.parameters()}) == 1

    def decode(
        self,
        prompt: torch.Tensor,
        max_new_tokens: int,
        selector: TokenSelector | None = None,
        stop: StopCondition = None,
    ) -> DecodeResult:
        """Generate up to ``max_new_tokens`` tokens after ``prompt`` (a 1-D LongTensor of token ids)."""
        return self.decode_batch([prompt], max_new_tokens, selector, stop)[0]

    def decode_batch(
        self,
        prompts: Sequence[torch.Tensor] | torch.Tensor,
        max_new_tokens: int | Sequence[int],
        selector: TokenSelector | None = None,
        stop: StopCondition = None,
        *,
        selectors: Sequence[TokenSelector | None] | None = None,
        stops: Sequence[StopCondition] | None = None,
    ) -> list[DecodeResult]:
        """Generate tokens after each prompt (1-D LongTensors of any lengths, or the rows of a 2-D tensor) and
        return one result per prompt, in order.

        Args:
            prompts: The prompts.
            max_new_tokens: Most new tokens, for all rows or one number per row.
            selector: Token selector of every row (greedy when ``None``).
            stop: Stop condition of every row.
            selectors: One selector per row; a ``None`` entry uses ``selector``. Rows that share a selector object
                are drafted and verified together.
            stops: One stop condition per row; a ``None`` entry uses ``stop``.
        """
        rows = list(prompts.unbind(0)) if isinstance(prompts, torch.Tensor) and prompts.dim() == 2 else list(prompts)
        if not rows:
            return []
        default = selector or GreedySelector()
        chosen = _per_row(selectors, len(rows), "selectors")
        row_selectors = [s if s is not None else default for s in chosen] if chosen else [default] * len(rows)
        if self.config.cycle_length > 1 and self.verifier.requires_greedy and not all(s.greedy for s in row_selectors):
            raise ConfigError(
                f"verifier {self.config.verifier_name!r} is exact only for a deterministic token selector; "
                "use lossless=False to sample"
            )
        shared = StopChecker(stop)
        chosen_stops = _per_row(stops, len(rows), "stops") or [None] * len(rows)
        row_stops = [shared if s is None else StopChecker(s) for s in chosen_stops]
        if isinstance(max_new_tokens, int):
            limits = [max_new_tokens] * len(rows)
        else:
            limits = [int(n) for n in _per_row(max_new_tokens, len(rows), "max_new_tokens") or ()]
        return _run(Generation(self, rows, limits, row_selectors, row_stops, default))


def _check_combination(config: DecodeConfig, verifier: Verifier, drafter: Drafter | None) -> None:
    """Raise :class:`ConfigError` for a verifier and drafter that cannot decode together, before any decoding."""
    name = config.verifier_name
    if config.lossless is True and verifier.commits_drafts:
        raise ConfigError(f"lossless=True contradicts verifier {name!r}, which keeps drafts without verifying them; "
                          "drop one of the two options")
    if verifier.commits_drafts and verifier.supports_trees:
        raise ConfigError(f"verifier {name!r} commits drafts as they are proposed, so no exact pass verifies a draft "
                          "tree; a verifier sets at most one of commits_drafts and supports_trees")
    if config.cycle_length > 1 and drafter is not None and drafter.drafts_trees(verifier) \
            and not verifier.supports_trees:
        raise ConfigError(f"{type(drafter).__name__} drafts trees, which verifier {name!r} cannot verify; use a "
                          "verifier that verifies trees (lossless greedy, rejection, typical) or a chain drafter")


def _per_row(values: Sequence[Any] | None, count: int, name: str) -> list[Any] | None:
    if values is None:
        return None
    values = list(values)
    if len(values) != count:
        raise ConfigError(f"{name} needs one entry per prompt ({count}), got {len(values)}")
    return values


def _run(gen: Generation) -> list[DecodeResult]:
    """Decode every row of ``gen`` to its end: exact pass, verification and commits, drafting, repeated."""
    if gen.drafter is not None:  # the drafter sees every prompt, rows without new tokens leave at once
        gen.drafter.begin(gen.runner, gen.cache, len(gen.rows))
    gen.drop([gen.max_new[row] <= 0 for row in gen.rows])
    if not gen.rows:
        return gen.outputs()
    if gen.drafter is None and not gen.observers and gen.config.sync_interval > 1:
        return run_plain(gen)
    current = prefill(gen)
    while True:
        finished, last_columns = verify_and_commit(gen, current)
        if gen.drafter is not None and not all(finished):
            gen.drafter.observe(current.exact, last_columns)
            gen.drafter.observe_logits(current.logits)
        gen.drop(finished)
        if not gen.rows:
            break
        proposal = Draft.empty(len(gen.rows), gen.device)
        if gen.drafter is not None:
            proposal = draft(gen)
            if not gen.rows:
                break
        current = tree_pass(gen, proposal) if proposal.is_tree else chain_pass(gen, proposal)
    return gen.outputs()
