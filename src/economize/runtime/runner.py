"""Running any contiguous range of decoder layers on a batch of ragged rows.

The :class:`LayerRunner` is the only code that calls decoder layers. The exact path runs layers ``1..L`` with the
model's own weights. A drafter runs a sub-range (e.g. ``m+1..r``) and may substitute layers through ``overrides``
(trainable copies of those layers). Dynamic-depth techniques pass a :class:`~economize.runtime.LayerPolicy` that skips
layers or routes a subset of tokens through them.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field

import torch

from ..arch import ArchAdapter, StateKind
from ..errors import IncompatibleTechniquesError, UnsupportedModelError
from .cache import CACHES, CacheSpec, KVCache, RecurrentSnapshot, RingSpec
from .context import ContextLayout, LayerContext
from .mask import MASK_FORMATS, TreeLayout
from .policy import Decision, LayerAction, LayerPolicy, Route

__all__ = ["ExactPass", "LayerCall", "LayerRunner"]

#: A decoder layer or a stand-in with the same calling convention.
LayerCall = Callable[..., torch.Tensor]


@dataclass
class ExactPass:
    """Result of one pass of the unmodified model.

    Attributes:
        final: Output of the last layer, ``[B, S, d]``.
        hidden: Captured hidden states ``{layer: [B, S, d]}`` (``0`` is the token embedding).
        context: The pass's :class:`LayerContext` (positions, slots, padding).
    """

    final: torch.Tensor
    hidden: dict[int, torch.Tensor] = field(default_factory=dict)
    context: LayerContext | None = None


class LayerRunner:
    """Executes the decoder layers of the model behind ``adapter`` on a :class:`~economize.runtime.KVCache`.

    Raises:
        UnsupportedModelError: No mask format is registered in :data:`~economize.runtime.MASK_FORMATS` for the model's
            attention implementation.
    """

    def __init__(self, adapter: ArchAdapter) -> None:
        if adapter.attn_implementation not in MASK_FORMATS:
            raise UnsupportedModelError(
                f"economize has no mask format for attention implementation {adapter.attn_implementation!r}; load the "
                f"model with attn_implementation in {MASK_FORMATS.names()} or register one in "
                "economize.runtime.MASK_FORMATS"
            )
        self.adapter = adapter
        self._devices: dict[int, torch.device] = {}
        self._recurrent: frozenset[int] = frozenset()
        self.refresh()

    def refresh(self) -> None:
        """Forget what the runner derived from the model: layer devices and the set of recurrent layers.

        Call it after the layers changed (layers removed, the model moved to other devices).
        """
        adapter = self.adapter
        self._devices = {}
        self._recurrent = frozenset(
            i for i in range(1, adapter.num_layers + 1) if adapter.state_kind(i) is StateKind.RECURRENT
        )

    @property
    def has_recurrent_layers(self) -> bool:
        """Whether some layer keeps a recurrent state (such models need :meth:`replay` after rejected drafts)."""
        return bool(self._recurrent)

    # ---- set-up --------------------------------------------------------------------------------------------------
    def new_cache(
        self, capacity: int, batch_size: int = 1, kind: str | None = None, ring: RingSpec | None = None
    ) -> KVCache:
        """An empty cache of the kind the model needs (``adapter.cache_kind`` unless ``kind`` is given), with the
        window cache ``ring`` for the sliding layers if given."""
        spec = CacheSpec(capacity, batch_size, kind or self.adapter.cache_kind, ring)
        return CACHES.get(spec.kind)(spec)

    def context(
        self,
        cache: KVCache,
        positions: torch.Tensor,
        valid: torch.Tensor | None = None,
        *,
        trusted: bool = False,
        consecutive: bool = False,
        whole_sequences: bool = False,
    ) -> LayerContext:
        """A chain-layout pass at ``positions`` (see :meth:`LayerContext.chain`)."""
        return LayerContext.chain(
            self.adapter, cache, positions, valid, trusted=trusted, consecutive=consecutive,
            whole_sequences=whole_sequences,
        )

    def context_from_layout(
        self, cache: KVCache, layout: ContextLayout, token_inputs: Mapping[str, torch.Tensor] | None = None
    ) -> LayerContext:
        """The context of a pass laid out as ``layout`` (see :meth:`LayerContext.from_layout`).

        The CUDA graphs of the decoding loop rebuild the context of every replayed exact pass with it, from the
        layout of the context :meth:`context` built inside the graph, so graphed and eager passes share one
        construction.
        """
        return LayerContext.from_layout(self.adapter, cache, layout, dict(token_inputs or {}))

    def tree_context(
        self, cache: KVCache, tree: TreeLayout, valid: torch.Tensor | None = None, *, trusted: bool = False
    ) -> LayerContext:
        """A pass over the nodes of ``tree`` (see :meth:`LayerContext.for_tree`).

        Recurrent layers of a hybrid model run every node's root-to-node path as a chain from a branched copy of
        the row's state and leave the cache's states where they were, so the decoder replays the committed path.

        Raises:
            IncompatibleTechniquesError: The model has recurrent layers and the cache cannot branch their states
                (it does not declare ``supports_state_edit``).
        """
        if self._recurrent and not type(cache).supports_state_edit:
            raise IncompatibleTechniquesError(
                f"{type(cache).__name__} cannot branch the recurrent states of {self.adapter.model_type!r} over the "
                "nodes of a draft tree; use chains or a cache that declares supports_state_edit"
            )
        return LayerContext.for_tree(self.adapter, cache, tree, valid, trusted=trusted)

    # ---- passes --------------------------------------------------------------------------------------------------
    def embed(self, tokens: torch.Tensor, ctx: LayerContext) -> torch.Tensor:
        """Input of layer 1 for ``tokens`` ``[B, S]`` (also fills the per-token inputs of ``ctx``)."""
        return self.adapter.embed(tokens, ctx)

    def run(
        self,
        hidden: torch.Tensor,
        ctx: LayerContext,
        first: int,
        last: int,
        *,
        overrides: Mapping[int, LayerCall] | None = None,
        capture: Iterable[int] = (),
        policy: LayerPolicy | None = None,
    ) -> tuple[torch.Tensor, dict[int, torch.Tensor]]:
        """Run layers ``first..last`` (1-indexed, inclusive) on ``hidden`` ``[B, S, d]``.

        Every layer writes its K/V into ``ctx.cache`` at ``ctx.slots``. Returns the output of layer ``last`` and the
        outputs of the layers listed in ``capture``.
        """
        overrides = overrides or {}
        capture = frozenset(capture)
        captured: dict[int, torch.Tensor] = {}
        with ctx.cache.writing_at(ctx.slots):
            for index in range(first, last + 1):
                override = overrides.get(index)
                layer = self.adapter.layer(index) if override is None else override
                device = self._device(index)
                if hidden.device != device:
                    hidden = hidden.to(device)
                decision: Decision = LayerAction.RUN if policy is None else policy.decide(index, hidden, ctx)
                hidden = self._apply(index, layer, hidden, ctx, decision)
                if policy is not None:
                    replaced = policy.after(index, hidden, ctx)
                    if replaced is not None:
                        hidden = replaced
                if index in capture:
                    captured[index] = hidden
        return hidden, captured

    def forward(
        self,
        tokens: torch.Tensor,
        ctx: LayerContext,
        *,
        capture: Iterable[int] = (),
        policy: LayerPolicy | None = None,
    ) -> ExactPass:
        """The model's own pass over ``tokens`` ``[B, S]`` within ``ctx`` (layer ``0`` in ``capture`` is the input)."""
        capture = frozenset(capture)
        hidden = self.embed(tokens, ctx)
        final, captured = self.run(hidden, ctx, 1, self.adapter.num_layers, capture=capture, policy=policy)
        if 0 in capture:
            captured[0] = hidden
        return ExactPass(final, captured, ctx)

    def logits(self, final: torch.Tensor) -> torch.Tensor:
        """Next-token logits from last-layer states (final norm, LM head and any logit transform of the family)."""
        return self.adapter.logits(final)

    def replay(
        self, cache: KVCache, snapshot: RecurrentSnapshot, tokens: torch.Tensor, start: torch.Tensor,
        lengths: torch.Tensor,
    ) -> None:
        """Bring recurrent states in line with the tokens actually committed after a speculative pass.

        A pass over ``tokens`` ``[B, S]`` starting at positions ``start`` ``[B]`` advanced every recurrent state by
        ``S`` tokens, padding included. Rows that committed only their first ``lengths[b] < S`` tokens get the state
        of ``snapshot`` (taken before the pass) back and run those tokens again. K/V slots need no correction: the
        replay rewrites them with the same values.
        """
        if not cache.keeps_recurrent_state:
            return
        width = tokens.shape[1]
        for length in sorted(set(lengths.tolist()) - {width}):
            rows = (lengths == length).nonzero(as_tuple=True)[0]
            cache.restore(snapshot, rows)
            if length == 0:
                continue
            with cache.selecting(rows):
                positions = start[rows, None] + torch.arange(length, device=start.device)
                ctx = self.context(cache, positions, trusted=True)
                self.forward(tokens[rows, :length], ctx)

    # ---- internals -------------------------------------------------------------------------------------------
    def _device(self, index: int) -> torch.device:
        device = self._devices.get(index)
        if device is None:
            device = self._devices[index] = self.adapter.layer_device(index)
        return device

    def _apply(
        self, index: int, layer: LayerCall, hidden: torch.Tensor, ctx: LayerContext, decision: Decision
    ) -> torch.Tensor:
        if isinstance(decision, torch.Tensor):
            decision = Route.from_mask(decision)
        recurrent = index in self._recurrent
        if recurrent and decision is not LayerAction.RUN:
            raise IncompatibleTechniquesError(
                f"layer {index} keeps a recurrent state: it must run on every token (got {decision})"
            )
        if decision is LayerAction.RUN:
            if recurrent and ctx.tree is not None:
                return self._run_tree_paths(index, layer, hidden, ctx)
            return self.adapter.call_layer(index, layer, hidden, ctx)
        if decision is LayerAction.SKIP:
            return hidden
        self.adapter.fill_kv(index, layer, hidden, ctx)
        if decision is LayerAction.SKIP_FILL_KV:
            return hidden
        route: Route = decision
        index_d = route.indices.to(hidden.device)[:, :, None].expand(-1, -1, hidden.shape[-1])
        picked = hidden.gather(1, index_d)
        sub = ctx.select(route.indices, route.valid)
        with ctx.cache.writing_at(sub.slots):
            out = self.adapter.call_layer(index, layer, picked, sub)
        out = torch.where(route.valid.to(hidden.device)[:, :, None], out, picked)
        return hidden.scatter(1, index_d, out)

    def _run_tree_paths(self, index: int, layer: LayerCall, hidden: torch.Tensor, ctx: LayerContext) -> torch.Tensor:
        """Recurrent layer ``index`` over the nodes of a draft tree.

        Node ``q`` gets the layer's output at the end of its root-to-``q`` path, which runs as one chain row from a
        copy of its tree's row state. The cache's states are put back afterwards, so the pass leaves them before the
        tree; the decoder then replays the committed path (:meth:`replay`).
        """
        tree, cache = ctx.tree, ctx.cache
        assert tree is not None
        if cache.active_batch_size != cache.batch_size:
            raise IncompatibleTechniquesError(
                "draft trees on recurrent layers run on the whole batch, not on rows chosen by KVCache.selecting()"
            )
        device = hidden.device
        batch, nodes = hidden.shape[:2]
        visible = tree.visible.to(device)
        if visible.shape[1:] != (nodes, nodes):  # a step over part of a tree (dynamic trees): its paths start earlier
            raise IncompatibleTechniquesError(
                f"layer {index} keeps a recurrent state: a tree pass must hold every node of its trees"
            )
        valid = visible.diagonal(dim1=1, dim2=2) if ctx.valid is None else ctx.valid.to(device)
        base = tree.base.to(device)
        depth = torch.where(valid, tree.positions.to(device) - base[:, None], 0)
        longest = int(depth.max()) + 1  # one host read per recurrent layer of a tree pass
        # path[b, q, d] is the ancestor of node q at depth d; column ``longest`` collects the other nodes
        column = torch.where(visible, depth[:, None, :], longest)
        node = torch.arange(nodes, device=device).expand(batch, nodes, nodes)
        path = torch.zeros(batch, nodes, longest + 1, dtype=torch.long, device=device)
        path = path.scatter(2, column, node)[..., :longest].reshape(batch, nodes * longest)
        steps = torch.arange(longest, device=device)
        on_path = (steps <= depth[..., None]) & valid[..., None]
        positions = torch.where(on_path, base[:, None, None] + steps, 0).reshape(batch * nodes, longest)
        on_path = on_path.reshape(batch * nodes, longest)
        slots = torch.full_like(positions, cache.trash_slot)
        inputs = {
            name: _gather_rows(t, path).reshape(batch * nodes, longest, *t.shape[2:])
            for name, t in ctx.token_inputs.items()
        }
        paths = LayerContext.from_layout(self.adapter, cache, ContextLayout(positions, slots, on_path), inputs)
        chains = _gather_rows(hidden, path).reshape(batch * nodes, longest, hidden.shape[-1])
        key = self._state_key(index)
        conv, state = cache.recurrent_state(key)
        cache.put_recurrent_state(key, *(None if s is None else s.repeat_interleave(nodes, 0) for s in (conv, state)))
        try:
            with cache.writing_at(slots):
                out = self.adapter.call_layer(index, layer, chains, paths)
        finally:
            cache.put_recurrent_state(key, conv, state)
        last = depth.reshape(batch * nodes, 1, 1).expand(-1, 1, out.shape[-1])
        return out.gather(1, last).reshape(batch, nodes, -1)

    def _state_key(self, index: int) -> int:
        """The ``layer_idx`` under which recurrent layer ``index`` keeps its states in the cache."""
        layer = self.adapter.layer(index)
        for module in (self.adapter.mixer(layer), layer):
            key = getattr(module, "layer_idx", None)
            if isinstance(key, int):
                return key
        return index - 1


def _gather_rows(tensor: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    """``tensor[b, indices[b, i]]`` for a tensor ``[B, S, ...]`` and indices ``[B, K]``."""
    index = indices.to(tensor.device).view(*indices.shape, *([1] * (tensor.dim() - 2)))
    return tensor.gather(1, index.expand(-1, -1, *tensor.shape[2:]))
