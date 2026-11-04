"""Speculative speculative decoding with a draft-layer drafter (EAGLE-1/2, EAGLE-3) on its own device.

:class:`DraftLayerWorker` runs the draft layer of any :class:`~economize.techniques._draft_layer.DraftLayerTechnique`
(:mod:`economize.techniques.eagle`, :mod:`economize.techniques.eagle3`) on a second device and speculates like the other
:class:`~economize.distributed.LinkedWorker` drafters: while the target verifies the spine it drafts the next chain
behind the likeliest outcomes, one tree pass of the draft layer per depth. The draft layer has its own keys and values:
the entry of position ``p`` is computed from the source of ``p - 1`` (the technique's ``fuse`` of the exact states of
its ``source_layers`` for a committed token, the draft layer's own output for a drafted one) and the embedding of the
token at ``p``.

:class:`DraftLayerSpeculativeDecoder` links the worker to the target. Its target keeps the states of the source
layers (and the embeddings) from every verification. Before the worker adopts a speculation or drafts after a miss it
fuses them for the committed positions and rewrites their draft keys and values, as the single-device drafter does in
``observe`` (the states move from one device to the other, ``len(source_layers) + 1`` vectors per token). In
``mode="sync"`` the rounds are those of single-device speculative decoding with the same drafter; in
``mode="async"`` the chain behind a hit speculation was drafted from the drafter's own states of the branch. Either
way the output is exactly the target's.

:class:`GraphedDraftLayerWorker` and :class:`GraphedDraftLayerSpeculativeDecoder` replay the draft steps and the
target's verification passes from CUDA graphs (see :mod:`economize.distributed.graphed`). The four classes are
registered in :data:`~economize.distributed.DRAFTER_WORKERS` for ``"eagle"`` and ``"eagle3"``; a new draft-layer
technique registers them under its own name. ``Eagle3Worker`` and the other ``Eagle3*`` names are aliases.

Example::

    replica = economize.load(AutoModelForCausalLM.from_pretrained(name, dtype=torch.bfloat16).to("cuda:1"), ckpt)
    decoder = DraftLayerSpeculativeDecoder(target, DraftLayerWorker(replica, fanout=2), SSDConfig(draft_length=6))
    result = decoder.generate(prompt_ids, max_new_tokens=256)
"""

from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING

import torch
from torch import nn

from ..decoding.cost import CostKind, CostLog
from ..errors import ConfigError
from ..modules.kv_cache import DraftKVCache
from ..runtime import GraphPool
from .drafters import DRAFTER_WORKERS, DrafterWorkers
from .graphed import GraphedLinkedSpeculativeDecoder
from .linked import LinkedSpeculativeDecoder, LinkedWorker, TreePass, attached_drafting_technique
from .ssd import SSDConfig
from .workers import Adoption, Chooser, Drafts, Spine, TreeTemplate

if TYPE_CHECKING:
    from ..techniques._draft_layer import DraftLayerTechnique

__all__ = [
    "DRAFT_LAYER_WORKERS",
    "DraftLayerSpeculativeDecoder",
    "DraftLayerWorker",
    "Eagle3SpeculativeDecoder",
    "Eagle3Worker",
    "GraphedDraftLayerSpeculativeDecoder",
    "GraphedDraftLayerWorker",
    "GraphedEagle3SpeculativeDecoder",
    "GraphedEagle3Worker",
]


class DraftLayerWorker(LinkedWorker[torch.Tensor]):
    """A draft-layer drafter on its own device, for :class:`DraftLayerSpeculativeDecoder`.

    Args:
        model: A replica of the target with a draft-layer technique attached (``economize.load`` or
            ``economize.apply``), on the drafter's device. Only its embedding, the draft layer and the draft head
            (with EAGLE-1/2, or EAGLE-3 with ``head="shared"``, the model's final norm and LM head) run there.
        fanout: Candidates for the appended token per outcome.

    Raises:
        ConfigError: The model has no draft-layer technique or is split by tensor parallelism.
    """

    def __init__(self, model: nn.Module, fanout: int = 2) -> None:
        from ..techniques._draft_layer import DraftLayerTechnique

        # The abstract base only serves the isinstance check (all concrete subclasses pass it), nothing instantiates it
        technique = attached_drafting_technique(
            model, DraftLayerTechnique, type(self).__name__, "a draft-layer")  # type: ignore[type-abstract]
        super().__init__(technique.device, fanout)
        self.technique: DraftLayerTechnique = technique
        self.footprint = technique.footprint  # the draft layer, as the single-device drafter logs it
        self.source_layers = tuple(technique.source_layers)
        self.observed_layers = tuple(sorted({0, *self.source_layers}))  # 0: the embeddings
        self.cache: DraftKVCache | None = None
        self._first_visible = technique.first_visible()
        self._end = 0  # one past the last slot the current step sees

    # ---- LinkedWorker --------------------------------------------------------------------------------------------
    def _pull(self, end: int) -> None:
        """Rewrite the draft keys and values of positions ``_synced..end-1`` from the target's exact states and keep
        the fused source of ``end - 1``. These positions all lie in the target's last pass."""
        target = self.target
        lo = self._synced
        if target is None or end <= lo:
            return
        assert self.cache is not None
        first, stop = lo - target.start, end - target.start
        if first < 0 or stop > next(iter(target.hidden.values())).shape[0]:
            raise RuntimeError(f"positions {lo}..{end - 1} are not in the target's last pass")
        t = self.technique
        fused = t.fuse([target.hidden[layer][first:stop].to(self.device, non_blocking=True)
                        for layer in self.source_layers])  # [S, d]
        previous = self._states.get(lo - 1)
        if previous is None:
            previous = fused.new_zeros(fused.shape[-1])
        sources = torch.cat([previous[None].to(fused.dtype), fused[:-1]])[None]
        embedding = target.hidden[0][first:stop].to(self.device, non_blocking=True)[None]
        positions = torch.arange(lo, end, device=self.device)[None]
        keys, values = t.layer.keys_values(sources, embedding, t.rotary_table(positions))
        self.cache.scatter(0, keys, values, positions)
        self.cost.record_footprint(CostKind.DRAFT_OBSERVE, end - lo, lo, self.footprint)
        self._states[end - 1] = fused[-1]
        self._synced = end

    def _new_cache(self, capacity: int) -> DraftKVCache:
        layer = self.technique.layer
        return DraftKVCache(1, 1, layer.num_kv_heads, layer.head_dim, capacity, device=self.device,
                            dtype=self.technique.dtype)

    # ---- SpeculationWorker ---------------------------------------------------------------------------------------
    @torch.no_grad()
    def begin(self, prompt: torch.Tensor, capacity: int) -> None:
        if self.target is None:
            raise ConfigError(f"{type(self).__name__} drafts from the target's states; use it through "
                              "DraftLayerSpeculativeDecoder")
        self.cache = self._new_cache(capacity)
        self.cost = CostLog()
        self._states, self._synced = {}, 0
        self._pull(prompt.numel())

    @torch.no_grad()
    def draft(self, context: torch.Tensor, position: int, length: int, choose: Chooser) -> Spine:
        # every committed token but the root was an input of the target's passes: draft from the root alone
        root = position + context.numel() - 1
        self._pull(root)
        self._forget(root - 1)
        current = context.to(self.device).view(-1)[-1:]
        drafts = Drafts()
        for i in range(length):
            p = root + i
            row, self._states[p] = self._chain_step(current, p, self._states[p - 1])
            current, dist = choose(row)
            drafts.add(current, row, dist)
        return drafts.spine()

    def _tree_pass(self, template: TreeTemplate, base: int) -> TreePass[torch.Tensor]:
        def step(nodes: torch.Tensor, valid: torch.Tensor, tokens: torch.Tensor,
                 parents: list[torch.Tensor]) -> tuple[torch.Tensor, list[torch.Tensor]]:
            rows, out = self._tree_step(template, base, nodes, tokens, parents)  # the mask is the template's own
            return rows, list(out)

        return step

    def _move_branch(self, hit: Adoption) -> None:
        assert self.cache is not None
        start, stop, k = hit.source, hit.destination, hit.length
        for buffers in (self.cache.keys[0], self.cache.values[0]):
            buffers[:, :, stop : stop + k] = buffers[:, :, start : start + k].clone()

    # ---- the draft step ------------------------------------------------------------------------------------------
    def _chain_step(self, token: torch.Tensor, position: int,
                    source: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Logits ``[1, V]`` after ``token`` at ``position`` and the step's output (the source of the next one)."""
        assert self.cache is not None
        self.cost.record_footprint(self.kind, 1, position, self.footprint)
        positions = torch.tensor([[position]], device=self.device)
        self._end = position + 1
        rows, out = self._chain_fn(token.view(1, 1), positions, source[None])
        return rows, out[0]

    def _tree_step(self, template: TreeTemplate, base: int, nodes: torch.Tensor, tokens: torch.Tensor,
                   parents: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """Logits ``[S, V]`` after the tree nodes ``nodes`` ``[1, S]`` holding ``tokens`` ``[S]``, each from the
        source in ``parents``, and the nodes' outputs ``[S, d]``."""
        assert self.cache is not None
        self.cost.record_footprint(self.kind, tokens.numel(), base, self.footprint)
        bases = template.zero + base
        self._end = base + template.depth.shape[-1]
        return self._tree_fn(tokens.view(1, -1), nodes, bases, torch.stack(parents), template.depth,
                             template.visible[0])

    def _limit(self) -> int:
        """Slots the current step attends over (its last visible slot is ``_end - 1``)."""
        return self._end

    def _chain_fn(self, tokens: torch.Tensor, positions: torch.Tensor,
                  sources: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        assert self.cache is not None
        limit = self._limit()
        keys = torch.arange(limit, device=self.device)[None]
        query = positions[0][:, None]
        visible = (keys <= query) & ((keys >= self._first_visible) | (keys == query))
        return self._step(tokens, positions, positions, sources, visible, limit)

    def _tree_fn(self, tokens: torch.Tensor, nodes: torch.Tensor, bases: torch.Tensor, sources: torch.Tensor,
                 depth: torch.Tensor, tree: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        assert self.cache is not None
        size = tree.shape[-1]
        limit = self._limit()
        keys = torch.arange(limit, device=self.device)[None]
        relative = keys - bases[:, None]  # [1, N]
        inside = (relative >= 0) & (relative < size)
        own = tree[nodes[0]].gather(1, relative.clamp(0, size - 1).expand(nodes.shape[1], -1))  # [S, N]
        visible = ((relative < 0) & (keys >= self._first_visible)) | (inside & own)
        return self._step(tokens, bases + depth[:, nodes[0]], bases + nodes, sources, visible, limit)

    def _step(self, tokens: torch.Tensor, positions: torch.Tensor, slots: torch.Tensor, sources: torch.Tensor,
              visible: torch.Tensor, limit: int) -> tuple[torch.Tensor, torch.Tensor]:
        """One pass of the draft layer over ``S`` tokens ``[1, S]`` at ``positions``, writing their keys and values to
        ``slots``: float logits ``[S, V]`` and outputs ``[S, d]``."""
        assert self.cache is not None
        t = self.technique
        layer = t.layer
        residual, q, k, v = layer.project(sources[None], t.embed(tokens), t.rotary_table(positions))
        self.cache.scatter(0, k, v, slots)
        attended = layer.attend(q, *self.cache.read(0, limit), visible[None, None])
        out = layer.finish(residual, attended)[0]
        return t.draft_logits(out), out


class GraphedDraftLayerWorker(DraftLayerWorker):
    """An :class:`DraftLayerWorker` that replays its chain steps and tree passes from CUDA graphs.

    Every step attends over the whole cache (masked), so a graph depends only on the number of tokens of the step.
    The cache is kept across generations and rebuilt (dropping the graphs) when a generation needs more positions.

    Args:
        model: See :class:`DraftLayerWorker`.
        fanout: See :class:`DraftLayerWorker`.
        granularity: Cache capacities are rounded up to a multiple of this.
    """

    def __init__(self, model: nn.Module, fanout: int = 2, *, granularity: int = 128) -> None:
        super().__init__(model, fanout)
        self.granularity = granularity
        self.cuda = self.device.type == "cuda"
        self._pool = GraphPool(self.device, enabled=self.cuda)
        self._chain_graph = self._pool.step(super()._chain_fn)
        self._tree_graph = self._pool.step(super()._tree_fn)
        self._bases: torch.Tensor | None = None

    @property
    def num_graphs(self) -> int:
        return self._pool.num_graphs

    def _new_cache(self, capacity: int) -> DraftKVCache:
        cache = self.cache
        if cache is not None and cache.capacity >= capacity:
            return cache
        self._pool.reset()
        return super()._new_cache(-(-capacity // self.granularity) * self.granularity)

    def _limit(self) -> int:
        assert self.cache is not None
        return self.cache.capacity

    def _chain_fn(self, tokens: torch.Tensor, positions: torch.Tensor,
                  sources: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        with self._on_device():
            rows, out = self._chain_graph(tokens, positions, sources)
            return rows.clone(), out.clone()

    def _tree_fn(self, tokens: torch.Tensor, nodes: torch.Tensor, bases: torch.Tensor, sources: torch.Tensor,
                 depth: torch.Tensor, tree: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        with self._on_device():
            rows, out = self._tree_graph(tokens, nodes, bases, sources, depth, tree)
            return rows.clone(), out.clone()

    def _on_device(self) -> contextlib.AbstractContextManager[object]:
        return torch.cuda.device(self.device) if self.cuda else contextlib.nullcontext()


class DraftLayerSpeculativeDecoder(LinkedSpeculativeDecoder):
    """:class:`~economize.distributed.AsyncSpeculativeDecoder` with a :class:`DraftLayerWorker` that receives the
    target's exact states of the source layers for every committed token.

    Args:
        target: The model whose output is reproduced, on its device.
        drafter: The draft-layer worker on another device.
        config: Decoding settings (``fanout`` is the worker's own).

    Raises:
        ConfigError: ``drafter`` is not a :class:`DraftLayerWorker`.
    """

    worker_class = DraftLayerWorker
    drafter: DraftLayerWorker

    def __init__(self, target: nn.Module, drafter: DraftLayerWorker, config: SSDConfig | None = None) -> None:
        super().__init__(target, drafter, config)


class GraphedDraftLayerSpeculativeDecoder(GraphedLinkedSpeculativeDecoder, DraftLayerSpeculativeDecoder):
    """:class:`DraftLayerSpeculativeDecoder` with CUDA graphs on both devices.

    Args:
        target: The model whose output is reproduced, on its device.
        drafter: The graphed draft-layer worker on another device.
        config: Decoding settings.
        max_width: Verification passes over more tokens run eagerly.

    Raises:
        ConfigError: ``drafter`` is not a :class:`GraphedDraftLayerWorker`.
    """

    worker_class = GraphedDraftLayerWorker
    drafter: GraphedDraftLayerWorker


#: The EAGLE-3 names of the classes (they run any draft-layer technique).
Eagle3Worker = DraftLayerWorker
GraphedEagle3Worker = GraphedDraftLayerWorker
Eagle3SpeculativeDecoder = DraftLayerSpeculativeDecoder
GraphedEagle3SpeculativeDecoder = GraphedDraftLayerSpeculativeDecoder

DRAFT_LAYER_WORKERS = DrafterWorkers(
    worker=DraftLayerWorker, decoder=DraftLayerSpeculativeDecoder,
    graphed_worker=GraphedDraftLayerWorker, graphed_decoder=GraphedDraftLayerSpeculativeDecoder,
)
"""The SSD workers of every draft-layer technique; a new one registers them under its name."""

for _name in ("eagle", "eagle3"):
    DRAFTER_WORKERS.register(_name, DRAFT_LAYER_WORKERS)
