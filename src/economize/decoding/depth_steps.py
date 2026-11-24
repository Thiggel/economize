"""Decode steps of a dynamic-depth technique that leave skipped layers out of the computation.

The eager path of a :class:`~economize.runtime.LayerPolicy` (:meth:`~economize.runtime.LayerPolicy.decide` before
every layer) works on token masks on the device, so it cannot run under CUDA graphs, and a skipped layer still costs
the kernel launches of its decision. :class:`DepthSteps` runs a one-token decode step of a single row through the
policy's step interface instead (:attr:`~economize.runtime.LayerPolicy.supports_steps`):

* the policy plans each layer on the host (``planned``: run, skip, fill the K/V only, or a technique's own action);
* a layer whose action depends on the token's state gets a *probe* (a router score, or the scores of a group of
  layers): the layers planned before it run as one graph that ends with the probe, the host reads the probe's values
  and the policy decides;
* an action that emits a signal (CALM's exit confidence) ends its graph the same way;
* the layers after the last probe or signal and the LM head run as one final graph.

The graphs are recorded per *segment*, the sequence of ``(layer, action)`` pairs it runs plus how it ends, so a step
replays a handful of graphs and synchronizes once per probe or signal. Skipped layers launch no kernel at all. The
segments share one *prelude* graph per step that embeds the token and builds every attention mask, rotary table and
position-id tensor the layers read; the segments read these buffers, so they do not depend on the step's position.

Without graphs (a CPU model, or a split one) the same host loop runs eagerly, which is how the tests compare the
steps with the eager path. With the decoder's :class:`~economize.decoding.DecodeGraphs` the graphs share its memory
pool and are dropped with its other graphs whenever the static cache is rebuilt.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING

import torch
from torch import nn

from ..errors import ConfigError
from ..runtime import ExactPass, LayerContext, LayerPolicy, LayerRunner, StepAction, phase
from ..runtime.graphs import StaticStep, capture
from .passes import Pass
from .protocols import Draft

if TYPE_CHECKING:
    from torch.cuda import _POOL_HANDLE

    from ..runtime import KVCache
    from .generation import Generation
    from .graphs import DecodeGraphs

__all__ = ["DepthSteps"]

#: A segment: the ``(layer, action)`` pairs it runs, the layer it probes at its end (or ``None``) and whether it
#: ends with the LM head (else with the signal of its last action).
_Key = tuple[tuple[tuple[int, StepAction], ...], int | None, bool]


class _LayerOps:
    """:class:`~economize.runtime.StepOps` over the context of one step."""

    def __init__(self, runner: LayerRunner, ctx: LayerContext, overrides: Mapping[int, nn.Module]) -> None:
        self.adapter, self.ctx, self.overrides = runner.adapter, ctx, overrides

    def _layer(self, index: int) -> nn.Module:
        layer = self.overrides.get(index)
        return layer if layer is not None else self.adapter.layer(index)

    def run(self, index: int, hidden: torch.Tensor) -> torch.Tensor:
        return self.adapter.call_layer(index, self._layer(index), hidden, self.ctx)

    def fill(self, index: int, hidden: torch.Tensor) -> None:
        self.adapter.fill_kv(index, self._layer(index), hidden, self.ctx)


class DepthSteps:
    """One-token decode steps of one row under a layer policy that implements the step interface.

    Args:
        runner: Runs the model's layers (all on one device).
        policy: The decoder's layer policy (``supports_steps``).
        graphs: The decoder's CUDA graphs; ``None`` runs the steps eagerly.
        overrides: Modules that replace decoder layers (by 1-based index), as in
            :meth:`~economize.runtime.LayerRunner.run` (a draft view's layer copies).
        head: Turns the output of the last layer ``[1, 1, d]`` into what a step returns (default: the model's
            float logits ``[1, 1, V]``); a drafter passes its own head (``[1, V]`` draft logits).

    Attributes:
        graphed: Whether the steps are recorded as CUDA graphs (``graphs`` given and the model on a CUDA device;
            else the same steps run eagerly, on the decoder's static cache if it has one).
        pool: Memory pool of the step graphs (set by the decoder's :class:`~economize.runtime.GraphPool`).
    """

    def __init__(
        self,
        runner: LayerRunner,
        policy: LayerPolicy,
        graphs: DecodeGraphs | None,
        *,
        overrides: Mapping[int, nn.Module] | None = None,
        head: Callable[[torch.Tensor], torch.Tensor] | None = None,
    ) -> None:
        if not policy.supports_steps:
            raise ConfigError(f"{type(policy).__name__} does not implement the step interface")
        self.runner, self.policy, self.graphs = runner, policy, graphs
        self.overrides = dict(overrides or {})
        self.head = head
        self.graphed = graphs is not None and runner.adapter.device.type == "cuda"
        self.num_layers = runner.adapter.num_layers
        self.pool: _POOL_HANDLE | None = None
        self._segments: dict[_Key, tuple[torch.cuda.CUDAGraph, torch.Tensor]] = {}
        self._prelude: torch.cuda.CUDAGraph | None = None
        self._prelude_cache: KVCache | None = None
        self._ctx: LayerContext | None = None
        self._hidden: torch.Tensor | None = None
        self._tokens: torch.Tensor | None = None
        self._positions: torch.Tensor | None = None
        self._memo: tuple[int, int, int] = (0, 0, 0)
        if graphs is not None:
            graphs._pool.attach(self)

    # ---- the PoolMember interface ------------------------------------------------------------------------------
    def reset(self) -> None:
        """Drop every recorded graph (the decoder's cache was rebuilt or its graph state changed)."""
        self._segments.clear()
        self._prelude = self._prelude_cache = None
        self._ctx = self._hidden = None

    @property
    def num_graphs(self) -> int:
        return len(self._segments) + (self._prelude is not None)

    # ---- steps -------------------------------------------------------------------------------------------------
    @staticmethod
    def applies(gen: Generation, draft: Draft) -> bool:
        """Whether the next pass of ``gen`` is a step this engine runs: one row, one pending token, no drafts."""
        return len(gen.rows) == 1 and gen.ends[0] - gen.starts[0] == 1 and not draft.lengths[0]

    def step(self, gen: Generation) -> Pass:
        """The exact pass over the pending token of the single row of ``gen``."""
        start = gen.starts[0]
        with phase(gen.phase), gen.cache.external_pass([start]):
            logits = self._run(gen.cache, gen.seq[:, start:start + 1], start)
        gen.record_exact([1], [start])
        assert self._ctx is not None and self._hidden is not None
        exact = ExactPass(self._hidden, {}, self._ctx)
        return Pass(exact, logits, Draft.empty(1, gen.device), [start], [0], [1], 1)

    def draft(self, cache: KVCache, token: torch.Tensor, position: int) -> torch.Tensor:
        """A draft step: the output of :attr:`head` for the token ``[1, 1]`` at ``position``, whose K/V the layers
        it runs write into ``cache`` (the next exact pass overwrites them). Runs in the caller's phase."""
        return self._run(cache, token, position)

    # ---- internals ---------------------------------------------------------------------------------------------
    def _run(self, cache: KVCache, token: torch.Tensor, position: int) -> torch.Tensor:
        """One step of the token ``[1, 1]`` at ``position``: plan, probe and run the layers, return the head's
        output."""
        policy, num_layers = self.policy, self.num_layers
        counts = [(0, 0, 0)] * (num_layers + 1)
        policy.begin_step(position)
        if self.graphs is not None:
            self.graphs._check_state()  # may drop the graphs (reset)
        self._begin(cache, token, position)
        pending: list[tuple[int, StepAction]] = []
        for index in range(1, num_layers + 1):
            action = policy.planned(index)
            if action is None:
                value = self._segment((tuple(pending), index, False))
                pending = []
                action = policy.after_probe(index, tuple(float(v) for v in value.tolist()))
            pending.append((index, action))
            counts[index] = (1, int(action.runs), int(action.fills_kv and not action.runs))
            if action.emits_signal:
                signal = self._segment((tuple(pending), None, False))
                pending = []
                policy.after_op(index, action, float(signal.item()))
        output = self._segment((tuple(pending), None, True))
        policy.end_step(torch.tensor(counts, dtype=torch.long))
        return output

    def _begin(self, cache: KVCache, token: torch.Tensor, position: int) -> None:
        """Embed the step's token and build its context (:attr:`_ctx`, :attr:`_hidden`)."""
        device = self.runner.adapter.device
        if not self.graphed:
            positions = torch.full((1, 1), position, dtype=torch.long, device=device)
            self._ctx = self.runner.context(cache, positions, None, trusted=True, consecutive=True)
            self._hidden = self.runner.embed(token, self._ctx)
            return
        if self._prelude is None or self._prelude_cache is not cache:
            self.reset()
            self._tokens = token.clone()
            self._positions = torch.full((1, 1), position, dtype=torch.long, device=device)
            valid = torch.ones(1, 1, dtype=torch.bool, device=device)
            StaticStep._side_stream(lambda _: self._prelude_body(cache, valid), self._tokens)  # lazy allocations
            graph = torch.cuda.CUDAGraph()
            self._ctx, self._hidden = capture(graph, lambda: self._prelude_body(cache, valid), device, pool=self.pool)
            self._prelude, self._prelude_cache = graph, cache
            self._memo = self._memo_sizes()
        assert self._tokens is not None and self._positions is not None
        self._tokens.copy_(token)
        self._positions.fill_(position)
        self._prelude.replay()

    def _prelude_body(self, cache: KVCache, valid: torch.Tensor) -> tuple[LayerContext, torch.Tensor]:
        assert self._tokens is not None and self._positions is not None
        runner, adapter = self.runner, self.runner.adapter
        ctx = runner.context(cache, self._positions, valid, trusted=True, consecutive=True)
        hidden = runner.embed(self._tokens, ctx)
        for index in range(1, self.num_layers + 1):  # every buffer the layers read, built here once per step
            spec = adapter.attention_spec(index)
            ctx.mask(spec, hidden.device)
            ctx.rotary(spec.rope, hidden)
            ctx.position_ids(hidden.device)
        return ctx, hidden

    def _memo_sizes(self) -> tuple[int, int, int]:
        ctx = self._ctx
        assert ctx is not None
        return len(ctx._masks), len(ctx._rotary), len(ctx._position_ids)

    def _segment(self, key: _Key) -> torch.Tensor:
        """Run the segment ``key`` and return its probe value, signal or float logits."""
        if not self.graphed:
            return self._segment_body(key)
        recorded = self._segments.get(key)
        if recorded is not None:
            recorded[0].replay()
            return recorded[1]
        device = self.runner.adapter.device
        assert self._positions is not None
        output = StaticStep._side_stream(lambda _: self._segment_body(key), self._positions)  # the step's own run
        graph = torch.cuda.CUDAGraph()
        static = capture(graph, lambda: self._segment_body(key), device, pool=self.pool)
        if self._memo_sizes() != self._memo:
            raise RuntimeError("a layer read a context buffer the step prelude did not build; it would go stale "
                               "in the recorded graphs")
        self._segments[key] = (graph, static)
        return output

    def _segment_body(self, key: _Key) -> torch.Tensor:
        ops, probe, logits = key
        ctx, hidden = self._ctx, self._hidden
        assert ctx is not None and hidden is not None
        layer_ops = _LayerOps(self.runner, ctx, self.overrides)
        signal: torch.Tensor | None = None
        x = hidden
        with ctx.cache.writing_at(ctx.slots):
            for index, action in ops:
                x, signal = self.policy.step_op(index, action, x, layer_ops)
            if probe is not None:
                out = self.policy.probe(probe, x)
            elif logits:
                out = self.runner.logits(x).float() if self.head is None else self.head(x)
            else:
                assert signal is not None, "a segment that ends without probe or logits ends with a signal"
                out = signal
            if x is not hidden:
                hidden.copy_(x)
        return out
