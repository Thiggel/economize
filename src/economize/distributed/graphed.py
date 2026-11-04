"""CUDA graphs for speculative speculative decoding with a jump drafter.

The workers of :mod:`economize.distributed.self_drafting` run every pass eagerly. At batch size 1 a verification of
``k + 1`` tokens and a draft step of layers ``m+1..r`` are bound by kernel launches, as in single-device decoding
(:class:`~economize.decoding.DecodeGraphs`). The classes here record these passes as CUDA graphs and replay them:

* :class:`GraphedTargetWorker` replays the target's verification passes (one graph per width; the prompt pass stays
  eager), on a static KV cache kept across generations.
* :class:`GraphedJumpWorker` replays the drafter's chain steps and the tree passes of a speculation (one graph per
  tree shape), on its own static cache.
* :class:`GraphedLinkedSpeculativeDecoder` is :class:`~economize.distributed.linked.LinkedSpeculativeDecoder` with a
  graphed target; :class:`GraphedSelfSpeculativeDecoder` pairs it with :class:`GraphedJumpWorker`.

Replays do not wait for the device, so in ``mode="async"`` the drafter's speculation still overlaps the target's
verification. The rounds and the output are those of the eager workers. On the CPU every step runs eagerly on the
same static caches.

Example::

    worker = GraphedJumpWorker(replica, fanout=2)  # replica on cuda:1
    decoder = GraphedSelfSpeculativeDecoder(target, worker, SSDConfig(draft_length=6, mode="async"))
    result = decoder.generate(prompt_ids, max_new_tokens=256)
"""

from __future__ import annotations

import contextlib
from collections.abc import Callable, Iterable, Iterator

import torch
from torch import nn

from ..decoding.cost import CostLog
from ..decoding.graphs import static_cache_class
from ..errors import ConfigError
from ..runtime import CacheSpec, GraphPool, KVCache, LayerRunner, StaticKVCache, StaticStep
from ..runtime.context import LayerContext
from ..runtime.mask import TreeLayout
from .drafters import DRAFTER_WORKERS, DrafterWorkers
from .linked import CapturingTargetWorker, LinkedDrafter, LinkedSpeculativeDecoder
from .self_drafting import JumpWorker, SelfSpeculativeDecoder, States
from .ssd import SSDConfig

__all__ = [
    "GraphedJumpWorker",
    "GraphedLinkedSpeculativeDecoder",
    "GraphedSelfSpeculativeDecoder",
    "GraphedTargetWorker",
    "WorkerGraphs",
]



class WorkerGraphs:
    """The static KV cache and the graphed steps of one SSD worker.

    Args:
        runner: Runs the worker's model.
        granularity: Cache capacities are rounded up to a multiple of this, so that generations of similar lengths
            share one cache and its graphs.

    Raises:
        ConfigError: The model's cache kind has no static variant.
    """

    def __init__(self, runner: LayerRunner, *, granularity: int = 128) -> None:
        self.runner = runner
        self.device = runner.adapter.device
        self.granularity = granularity
        self.cache_kind = runner.adapter.cache_kind
        static = static_cache_class(self.cache_kind)
        if static is None or not issubclass(static, StaticKVCache):
            raise ConfigError(f"cache kind {self.cache_kind!r} has no static variant for CUDA graphs")
        self._cache_class: type[StaticKVCache] = static
        self.cuda = self.device.type == "cuda"
        self._pool = GraphPool(self.device, enabled=self.cuda)
        self.cache: StaticKVCache | None = None

    @property
    def num_graphs(self) -> int:
        return self._pool.num_graphs

    def wrap(self, fn: Callable[..., object]) -> StaticStep:
        """``fn`` as a graphed step on this worker's cache (its graphs are dropped when the cache is rebuilt)."""
        return self._pool.step(fn)

    @property
    def current(self) -> StaticKVCache:
        """The cache of the running generation (the last :meth:`cache_for`), which the graphed steps run on."""
        assert self.cache is not None, "cache_for() comes first"
        return self.cache

    def cache_for(self, capacity: int) -> StaticKVCache:
        """The static cache of one row with at least ``capacity`` positions."""
        cache = self.cache
        if cache is None or cache.capacity < capacity:
            rounded = -(-capacity // self.granularity) * self.granularity
            self._pool.reset()
            self.cache = cache = self._cache_class(CacheSpec(rounded, 1, self.cache_kind))
        else:
            cache.activate(1)
        return cache

    @contextlib.contextmanager
    def on_device(self) -> Iterator[None]:
        """The worker's device as the current CUDA device (graphs are captured on the current device's stream)."""
        if not self.cuda:
            yield
            return
        with torch.cuda.device(self.device):
            yield


class GraphedTargetWorker(CapturingTargetWorker):
    """A :class:`~economize.distributed.self_drafting.CapturingTargetWorker` that replays its verification passes
    from CUDA graphs.

    Args:
        model: The target model on its device.
        layers: Layers whose states the passes keep (the drafter's source layers; empty for a plain target).
        max_width: Passes over more tokens than this run eagerly (the prompt pass always does).
        granularity: See :class:`WorkerGraphs`.
    """

    def __init__(self, model: nn.Module, layers: Iterable[int] = (), *, max_width: int = 64,
                 granularity: int = 128) -> None:
        super().__init__(model, layers)
        self.max_width = max_width
        self.graphs = WorkerGraphs(self.runner, granularity=granularity)
        self._layers = tuple(sorted(self.layers))
        self._pass = self.graphs.wrap(self._pass_fn)

    def _reset(self, capacity: int) -> None:
        self.cache = self.graphs.cache_for(capacity)
        self.cost = CostLog()

    def _chain(self, tokens: torch.Tensor, position: int) -> torch.Tensor:
        if position == 0 or tokens.numel() > self.max_width:
            return super()._chain(tokens, position)
        self.cost.record(self.kind, tokens.numel(), position, 1, self.num_layers)
        positions = torch.arange(position, position + tokens.numel(), device=self.device)[None]
        with self.graphs.on_device():
            logits, hidden = self._pass(tokens.view(1, -1), positions)
            self.start = position
            self.hidden = {layer: state[0].clone() for layer, state in zip(self._layers, hidden)}
            return logits[0].clone()

    def _pass_fn(self, tokens: torch.Tensor, positions: torch.Tensor) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
        ctx = self.runner.context(self.graphs.current, positions, trusted=True)
        exact = self.runner.forward(tokens, ctx, capture=self.layers)
        return self.runner.logits(exact.final), tuple(exact.hidden[layer] for layer in self._layers)


class GraphedJumpWorker(JumpWorker):
    """A :class:`~economize.distributed.self_drafting.JumpWorker` that replays its draft steps from CUDA graphs.

    Args:
        model: A replica of the target with a jump technique attached, on the drafter's device.
        fanout: Candidates for the appended token per outcome.
        granularity: See :class:`WorkerGraphs`.
    """

    def __init__(self, model: nn.Module, fanout: int = 2, *, granularity: int = 128) -> None:
        super().__init__(model, fanout)
        self.graphs = WorkerGraphs(self.runner, granularity=granularity)
        self._chain_graph = self.graphs.wrap(self._chain_fn)
        self._tree_graph = self.graphs.wrap(self._tree_fn)

    # ---- the draft step ------------------------------------------------------------------------------------------
    def _new_cache(self, capacity: int) -> KVCache:
        return self.graphs.cache_for(capacity)

    def _chain_step(self, token: torch.Tensor, position: int, states: States) -> tuple[torch.Tensor, States]:
        self.cost.record(self.kind, 1, position, self.first, self.last)
        positions = torch.arange(position, position + 1, device=self.device)[None]
        sources = [states[layer].view(1, 1, -1) for layer in self.sources]
        with self.graphs.on_device():
            rows, x, captured = self._chain_graph(token.view(1, 1), positions, *sources)
            rows, out = self._outputs(rows, x, captured, [states])
        return rows[0], out[0]

    def _tree_step(self, tree: TreeLayout, base: int, nodes: torch.Tensor, valid: torch.Tensor, tokens: torch.Tensor,
                   parents: list[States]) -> tuple[torch.Tensor, list[States]]:
        self.cost.record(self.kind, tokens.numel(), base, self.first, self.last)
        sources = [torch.stack([p[layer] for p in parents])[None] for layer in self.sources]
        with self.graphs.on_device():
            rows, x, captured = self._tree_graph(tokens.view(1, -1), nodes, valid, tree.base, tree.slots,
                                                 tree.positions, tree.visible, *sources)
            return self._outputs(rows, x, captured, parents)

    def _chain_fn(self, tokens: torch.Tensor, positions: torch.Tensor,
                  *sources: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, tuple[torch.Tensor, ...]]:
        ctx = self.runner.context(self.graphs.current, positions, trusted=True)
        return self._jump_fn(tokens, ctx, sources)

    def _tree_fn(self, tokens: torch.Tensor, nodes: torch.Tensor, valid: torch.Tensor, base: torch.Tensor,
                 slots: torch.Tensor, positions: torch.Tensor, visible: torch.Tensor,
                 *sources: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, tuple[torch.Tensor, ...]]:
        tree = TreeLayout(base, slots, positions, visible)
        ctx = self.runner.tree_context(self.graphs.current, tree, trusted=True).select(nodes, valid)
        return self._jump_fn(tokens, ctx, sources)

    def _jump_fn(self, tokens: torch.Tensor, ctx: LayerContext,
                 sources: tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, torch.Tensor, tuple[torch.Tensor, ...]]:
        """The jump step of :meth:`JumpWorker._jump` without the host-side bookkeeping: logits ``[1, S, V]``, the
        predicted state after layer ``m`` and the captured window states, as tensors."""
        t = self.technique
        embedding = self.runner.embed(tokens, ctx)
        x = t.predict(list(sources), embedding)
        with t.jump_path():
            hidden, captured = self.runner.run(x, ctx, self.first, self.last, overrides=self._overrides,
                                               capture=self._captured)
        return t.logits(hidden).to(self.device), x, tuple(captured[layer] for layer in self._captured)

    def _outputs(self, rows: torch.Tensor, x: torch.Tensor, captured: tuple[torch.Tensor, ...],
                 parents: list[States]) -> tuple[torch.Tensor, list[States]]:
        """Copies of a replay's outputs (the next replay overwrites them) and every node's source states."""
        t = self.technique
        rows, x = rows.clone(), x.clone()
        states_after = {layer: state.to(self.device).clone() for layer, state in zip(self._captured, captured)}
        out: list[States] = []
        for s, parent in enumerate(parents):
            states = dict(parent)
            if t.target_layer in self._updated:
                states[t.target_layer] = x[0, s]
            for layer, state in states_after.items():
                states[layer] = state[0, s]
            out.append(states)
        return rows, out


class GraphedLinkedSpeculativeDecoder(LinkedSpeculativeDecoder):
    """:class:`~economize.distributed.linked.LinkedSpeculativeDecoder` whose target replays its verification passes
    from CUDA graphs (:class:`GraphedTargetWorker`). Pair it with a worker that graphs its draft steps by naming that
    worker class in :attr:`worker_class`.

    Args:
        target: The model whose output is reproduced, on its device.
        drafter: The graphed worker on another device.
        config: Decoding settings.
        max_width: Verification passes over more tokens run eagerly.
    """

    def __init__(self, target: nn.Module, drafter: LinkedDrafter, config: SSDConfig | None = None, *,
                 max_width: int = 64) -> None:
        self.max_width = max_width
        super().__init__(target, drafter, config)

    def _target_worker(self, target: nn.Module) -> CapturingTargetWorker:
        return GraphedTargetWorker(target, self.drafter.observed_layers, max_width=self.max_width)


class GraphedSelfSpeculativeDecoder(GraphedLinkedSpeculativeDecoder, SelfSpeculativeDecoder):
    """:class:`~economize.distributed.self_drafting.SelfSpeculativeDecoder` with CUDA graphs on both devices.

    Args:
        target: The model whose output is reproduced, on its device.
        drafter: The graphed jump worker on another device.
        config: Decoding settings.
        max_width: Verification passes over more tokens run eagerly.

    Raises:
        ConfigError: ``drafter`` is not a :class:`GraphedJumpWorker`.
    """

    worker_class = GraphedJumpWorker
    drafter: GraphedJumpWorker


DRAFTER_WORKERS.register("jump", DrafterWorkers(
    worker=JumpWorker, decoder=SelfSpeculativeDecoder,
    graphed_worker=GraphedJumpWorker, graphed_decoder=GraphedSelfSpeculativeDecoder,
))
