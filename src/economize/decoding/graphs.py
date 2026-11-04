"""CUDA graphs for the decoding loop (``DecodeConfig.cuda_graphs``).

A draft step of a cheap drafter and a short exact pass launch hundreds of small kernels each, so at small batch
sizes decoding is bound by kernel launches. :class:`DecodeGraphs` records the exact chain and draft-tree passes of a
:class:`~economize.decoding.CycleDecoder` and the draft steps of drafters that opt in (:meth:`Drafter.enable_graphs`)
as CUDA graphs and replays them. The decoder keeps one :class:`DecodeGraphs` for its whole life, so graphs recorded
in one generation are replayed in the next.

Everything a graph touches stays at a fixed address: the KV cache is the static variant of the model's cache kind
(:func:`static_cache_class`, a :class:`~economize.runtime.StaticKVCache` for ``"positional"``; one buffer set per batch
size, reused across generations, its capacity rounded up so that most generations fit
the same cache), drafters keep their per-row state in persistent buffers, and the inputs of every step are copied
into the graph's own input buffers. One graph is recorded per input signature (batch size, pass or tree width).
Recording happens on the third call with a signature; the first two run eagerly and allocate what the step allocates
lazily.

KV-cache techniques whose wrappers are ``graph_safe`` (:class:`~economize.techniques._kv.CacheWrapper`) run under
the graphs too: the generation binds the wrapped cache (:meth:`DecodeGraphs.bind`), every replayed exact pass runs
inside the wrapper's ``external_pass`` (its pass boundaries, eagerly), and the graphs are dropped whenever the cache's
``graph_state`` changes.

All graphs share one memory pool. This is safe because they never run concurrently and every graph's outputs stay
referenced by its :class:`~economize.runtime.StaticStep`. Outputs of a replay are overwritten by the next replay of the
same graph; the loop reads them before that.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import torch

from ..errors import ConfigError
from ..runtime import (
    CACHES,
    CacheSpec,
    ContextLayout,
    ExactPass,
    GraphPool,
    KVCache,
    LayerRunner,
    Phase,
    RingSpec,
    StaticStep,
    TreeLayout,
    phase,
)

__all__ = ["DecodeGraphs", "GraphWrap", "static_cache_class"]

#: What :meth:`Drafter.enable_graphs` receives: turns a step function of tensors into its graphed version.
GraphWrap = Callable[[Callable[..., Any]], StaticStep]


def static_cache_class(kind: str) -> type[KVCache] | None:
    """The cache class CUDA graphs use for the cache kind ``kind`` (a key of :data:`~economize.runtime.CACHES`), or
    ``None`` when the kind has no static variant and decoding stays eager.

    A cache class opts in with its ``static_variant`` class attribute
    (:attr:`economize.runtime.KVCache.static_variant`):
    a :class:`~economize.runtime.KVCache` subclass that never rebinds its buffers and has ``activate(batch_size)`` (see
    :class:`~economize.runtime.StaticKVCache`, the variant of ``"positional"``). A new graph-capable cache kind is
    therefore added by registering its class with a ``static_variant``, without changes here.
    """
    if kind not in CACHES:
        return None
    return CACHES.get(kind).static_variant


class DecodeGraphs:
    """The CUDA graphs and the static KV cache of one decoder.

    Args:
        runner: Runs the model's layers.
        capture: Layers whose hidden states the exact passes return (the drafter's observed layers).
        pass_phase: Phase of the exact passes after the prefill (``"decode"`` or ``"refresh"``).
        granularity: Cache capacities are rounded up to a multiple of this, so that generations of similar lengths
            share one cache and its graphs (the attention cost of a step grows with the capacity).
    """

    def __init__(
        self, runner: LayerRunner, capture: Sequence[int], pass_phase: Phase, *, granularity: int = 128
    ) -> None:
        self.runner = runner
        self.capture = tuple(capture)
        self.phase: Phase = pass_phase
        self.granularity = granularity
        self.cache_kind = runner.adapter.cache_kind
        static = static_cache_class(self.cache_kind)
        if static is None:
            raise ConfigError(f"cache kind {self.cache_kind!r} has no static variant for CUDA graphs")
        self._cache_class = static
        self._pool = GraphPool()
        self._cache: Any = None  # an instance of the static variant
        self._bound: KVCache | None = None  # what the passes run on: the static cache or a wrapper of it
        self._state: tuple[Any, ...] = ()  # the bound cache's graph_state when the graphs were recorded
        self._exact = self.wrap(self._exact_pass)
        self._tree = self.wrap(self._tree_pass)

    @staticmethod
    def supports_cache(kind: str) -> bool:
        """Whether exact passes over a cache of kind ``kind`` can be graphed (:func:`static_cache_class`)."""
        return static_cache_class(kind) is not None

    @property
    def num_graphs(self) -> int:
        """Graphs recorded so far (over all steps)."""
        return self._pool.num_graphs

    def wrap(self, fn: Callable[..., Any]) -> StaticStep:
        """``fn`` (a function of tensors without host synchronization) as a graphed step of this decoder.

        The step's graphs are dropped whenever the cache is rebuilt.
        """
        return self._pool.step(fn)

    def cache_for(self, capacity: int, batch_size: int, ring: RingSpec | None = None) -> KVCache:
        """The static cache for a generation of ``batch_size`` rows that needs ``capacity`` positions, with the
        window cache ``ring`` for its sliding layers if given.

        A generation that needs more positions than the current cache holds (or another ring) gets a new cache, and
        every graph recorded on the old one is dropped.
        """
        cache = self._cache
        if cache is None or cache.capacity < capacity or cache.spec.ring != ring:
            rounded = -(-capacity // self.granularity) * self.granularity
            self._reset()
            spec = CacheSpec(rounded, batch_size, self.cache_kind, ring)
            self._cache = self._bound = cache = self._cache_class(spec)
        else:
            cache.activate(batch_size)
        return cache

    def bind(self, cache: KVCache) -> None:
        """Run the passes of the next generation on ``cache``: the static cache of :meth:`cache_for` or a graph-safe
        wrapper of it. Graphs recorded over another cache object are dropped, so a wrapper that lives as long as the
        static cache keeps them."""
        if cache is not self._bound:
            self._reset()
            self._bound = cache
        self._check_state()

    def _reset(self) -> None:
        self._pool.reset()
        self._state = self._bound.graph_state() if self._bound is not None else ()

    def _check_state(self) -> None:
        """Drop every graph when the bound cache's :meth:`~economize.runtime.KVCache.graph_state` changed."""
        assert self._bound is not None
        state = self._bound.graph_state()
        if state != self._state:
            self._pool.reset()
            self._state = state

    def exact(self, tokens: torch.Tensor, positions: torch.Tensor, valid: torch.Tensor, picks: torch.Tensor,
              frontier: list[int | None]) -> tuple[ExactPass, torch.Tensor]:
        """An exact chain pass over ``tokens`` ``[B, S]`` at ``positions`` (``valid`` marks real tokens) and the
        float logits of the final states at the columns ``picks`` ``[B, K]``, replayed from a graph. Row ``b`` writes
        from slot ``frontier[b]`` on (its first position).

        The returned pass has a :class:`~economize.runtime.LayerContext` like an eager one (drafters read the pass's
        positions and padding in :meth:`Drafter.observe`); its tensors are the graph's buffers.
        """
        assert self._bound is not None
        with self._bound.external_pass(frontier):
            self._check_state()
            final, hidden, logits, layout, token_inputs = self._exact(tokens, positions, valid, picks)
        ctx = self.runner.context_from_layout(self._bound, layout, token_inputs)
        return ExactPass(final, dict(hidden), ctx), logits

    def tree(self, tokens: torch.Tensor, parents: torch.Tensor, base: torch.Tensor, valid: torch.Tensor,
             frontier: list[int | None]) -> tuple[ExactPass, torch.Tensor]:
        """An exact pass over the nodes of draft trees (``tokens``, ``parents`` and ``valid`` ``[B, T]``, the trees
        stored from slot ``base`` ``[B]`` on, see :meth:`~economize.runtime.TreeLayout.build`) and the float logits of
        every node, replayed from a graph (one per batch size and tree width). ``frontier`` is ``base`` as a list."""
        assert self._bound is not None
        with self._bound.external_pass(frontier):
            self._check_state()
            final, hidden, logits, layout, token_inputs = self._tree(tokens, parents, base, valid)
        ctx = self.runner.context_from_layout(self._bound, layout, token_inputs)
        return ExactPass(final, dict(hidden), ctx), logits

    def _tree_pass(
        self, tokens: torch.Tensor, parents: torch.Tensor, base: torch.Tensor, valid: torch.Tensor
    ) -> tuple[torch.Tensor, dict[int, torch.Tensor], torch.Tensor, ContextLayout, dict[str, torch.Tensor]]:
        assert self._bound is not None
        runner = self.runner
        tree = TreeLayout.build(parents, base, valid, trash_slot=self._bound.trash_slot)
        ctx = runner.tree_context(self._bound, tree, valid, trusted=True)
        with phase(self.phase):
            exact = runner.forward(tokens, ctx, capture=self.capture)
        return exact.final, exact.hidden, runner.logits(exact.final).float(), ctx.layout, ctx.token_inputs

    def _exact_pass(
        self, tokens: torch.Tensor, positions: torch.Tensor, valid: torch.Tensor, picks: torch.Tensor
    ) -> tuple[torch.Tensor, dict[int, torch.Tensor], torch.Tensor, ContextLayout, dict[str, torch.Tensor]]:
        assert self._bound is not None
        runner = self.runner
        ctx = runner.context(self._bound, positions, valid, trusted=True, consecutive=True)  # chain passes only
        with phase(self.phase):
            exact = runner.forward(tokens, ctx, capture=self.capture)
        final = exact.final.gather(1, picks[:, :, None].to(exact.final.device).expand(-1, -1, exact.final.shape[-1]))
        return exact.final, exact.hidden, runner.logits(final).float(), ctx.layout, ctx.token_inputs
