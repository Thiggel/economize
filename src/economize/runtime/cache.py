"""Caches written at explicit slots, for any batch layout.

Hugging Face attention layers call ``past_key_values.update(key, value, layer_idx)`` and attend over whatever the
cache returns. Draft-and-verify decoding needs more freedom than the append-only caches of ``transformers``: a draft
step writes the K/V of a speculative token, the next exact pass overwrites the same slots, a tree pass writes nodes
that are later compacted onto the accepted path, and finished rows of a batch are dropped.

:class:`PositionalKVCache` provides this. The caller names the slot of every token with :meth:`KVCache.writing_at`,
every layer's ``update`` writes there, and the attention mask (:mod:`economize.runtime.mask`) decides which slots each
query sees. Buffers hold ``capacity + 1`` slots: the last one, :attr:`KVCache.trash_slot`, absorbs the writes of
padding tokens and is never visible.

Buffers are allocated on a layer's first write with the shape, dtype and device of its K/V, so layers with different
head sizes (Gemma-4), compressed latents (DeepSeek MLA) or devices (``device_map``) need no layout description.

:class:`HybridCache` adds the conv and recurrent states of recurrent layers (Qwen3.5's gated delta nets,
Nemotron-H's Mamba-2) with :meth:`HybridCache.snapshot` and :meth:`HybridCache.restore`, since recurrent states
cannot be rewound by masking.
"""

from __future__ import annotations

import inspect
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterator
from contextlib import AbstractContextManager, contextmanager, nullcontext
from dataclasses import dataclass
from typing import Any, ClassVar

import torch
import torch.nn.functional as F
from transformers.cache_utils import DynamicCache

from ..errors import ConfigError, UnsupportedModelError
from ..modules.kv_cache import SlotLayout, write_slots
from ..modules.slot_gather import gather_slots
from ..registry import Registry
from ..validation import check_positive

__all__ = [
    "CACHES",
    "CacheSpec",
    "HybridCache",
    "KVCache",
    "PositionalKVCache",
    "RecurrentSnapshot",
    "RingSpec",
    "StaticKVCache",
]

#: ``transformers`` >= 5.14 keeps several conv/recurrent states per layer, addressed by ``state_idx``.
_INDEXED_STATES = "state_idx" in inspect.signature(DynamicCache.update_conv_state).parameters

#: Key position given to never-written slots and to the trash slot; no query position reaches it.
UNSEEN_POSITION = 2**40


#: Conv and recurrent states of every recurrent layer, as returned by :meth:`HybridCache.snapshot`.
RecurrentSnapshot = dict[int, tuple[torch.Tensor | None, torch.Tensor | None]]

@dataclass(frozen=True)
class RingSpec:
    """The window cache of sliding-window layers: a ring of :attr:`slots` slots instead of ``capacity + 1``.

    A sliding layer never attends further back than its window, so its buffer holds position ``p`` at slot
    ``p mod slots``, followed by :attr:`page` extra slots: the first of them (slot :attr:`slots`) is the layer's trash
    slot, and the others are never written (they make the buffer a whole number of FlashAttention pages). Which
    position a ring slot holds depends on the pass, so ring layers get their key positions from
    :meth:`key_positions` and not from :meth:`KVCache.key_positions`.

    A pass of ``Q`` tokens whose last token sits at ``P`` must find every key its queries see still in the ring, and
    no slot it reads may hold a token newer than ``P`` that the mask would take for an older one: the ring therefore
    needs ``slots - page >= Q + W + window - 2``, with ``W`` the furthest a draft may run ahead of the committed
    tokens (rejected drafts stay in their slots). :meth:`for_pass` sizes it so.

    Attributes:
        layers: The ring layers (0-based ``layer_idx``); every other layer keeps ``capacity + 1`` slots.
        window: The sliding window of the ring layers (one value for all of them).
        slots: Ring slots ``R``, a multiple of :attr:`page`.
        page: FlashAttention page size of the ring (FA2 reads paged buffers in multiples of 256 slots).
    """

    layers: frozenset[int]
    window: int
    slots: int
    page: int = 256

    def __post_init__(self) -> None:
        check_positive("ring window", self.window)
        check_positive("ring page", self.page)
        if self.slots % self.page or self.slots < self.window + self.page:
            raise ConfigError(
                f"a ring needs a multiple of {self.page} slots and at least window + page = "
                f"{self.window + self.page}; got {self.slots}"
            )

    @classmethod
    def for_pass(cls, layers: frozenset[int], window: int, queries: int, ahead: int, page: int = 256) -> RingSpec:
        """The smallest ring for passes of up to ``queries`` tokens with drafts up to ``ahead`` tokens past the
        committed ones."""
        need = window + queries + ahead + page
        return cls(frozenset(layers), window, -(-need // page) * page, page)

    @property
    def buffer_slots(self) -> int:
        """Slots of a ring layer's buffer (the ring, its trash slot and the padding to a whole page)."""
        return self.slots + self.page

    def physical(self, slots: torch.Tensor, trash_slot: int) -> torch.Tensor:
        """The ring slots of the logical ``slots`` (positions, or the cache's ``trash_slot``)."""
        return torch.where(slots == trash_slot, self.slots, slots % self.slots)

    def key_positions(self, positions: torch.Tensor) -> torch.Tensor:
        """The position each ring slot holds for a pass at ``positions`` ``[B, Q]``, as ``[B, buffer_slots]``.

        Slot ``s`` holds the newest position up to the pass's last one, ``P_b``, that maps to it (``P_b - ((P_b - s)
        mod slots)``); slots of negative positions, the trash slot and the padding get :data:`UNSEEN_POSITION`.
        """
        last = positions.amax(-1, keepdim=True)
        slot = torch.arange(self.buffer_slots, device=positions.device)
        held = last - torch.remainder(last - slot, self.slots)
        unseen = (held < 0) | (slot >= self.slots)
        return torch.where(unseen, torch.full_like(held, UNSEEN_POSITION), held)


@dataclass(frozen=True)
class CacheSpec:
    """What :meth:`economize.runtime.LayerRunner.new_cache` builds.

    Attributes:
        capacity: Number of usable positions per row (the buffers hold one more slot, the trash slot).
        batch_size: Number of rows.
        kind: Key of :data:`CACHES`; the model's adapter chooses it (``"hybrid"`` for linear-attention layers).
        ring: The window cache of the sliding layers (only on caches that declare :attr:`KVCache.supports_ring`),
            or ``None``.
    """

    capacity: int
    batch_size: int = 1
    kind: str = "positional"
    ring: RingSpec | None = None

    def __post_init__(self) -> None:
        check_positive("cache capacity", self.capacity)
        check_positive("batch size", self.batch_size)


class KVCache(SlotLayout, ABC):
    """A cache that decoder layers write through ``update`` at slots chosen by the caller.

    Layers are addressed by their 0-based ``layer_idx`` (the attribute Hugging Face attention modules pass).
    """

    #: Whether some layer keeps a state that every token advances (a recurrent layer). Masks cannot rewind such a
    #: state, so decoders :meth:`snapshot` it before a speculative pass and the runner replays the committed tokens
    #: (:meth:`economize.runtime.LayerRunner.replay`). Caches that hold only K/V slots keep the default ``False``.
    keeps_recurrent_state: ClassVar[bool] = False
    #: The cache class CUDA graphs use in place of this one (:func:`economize.decoding.graphs.static_cache_class`): a
    #: subclass that never rebinds its buffers and has ``activate(batch_size)``, like :class:`StaticKVCache`.
    #: ``None`` keeps decoding with this cache eager. A variant fits only the class that names it, so a subclass
    #: that does not name its own starts from ``None`` (it may hold state the parent's variant would drop).
    static_variant: ClassVar[type[KVCache] | None] = None
    #: Whether :meth:`detached` and :meth:`clone` reproduce this cache from :meth:`tensor_stores`. Like
    #: :attr:`static_variant` it holds only for the class that declares it: a subclass may keep state the stores miss
    #: (the per-batch-size buffer sets of :class:`StaticKVCache`, the inner cache of a wrapper), so it starts from
    #: ``False`` and its copies raise until it declares ``True``.
    copyable: ClassVar[bool] = False
    #: Whether :meth:`put_recurrent_state` replaces a recurrent layer's states directly (rollouts that run a
    #: recurrent layer per start position keep the per-start states this way). Subclasses inherit it.
    supports_state_edit: ClassVar[bool] = False
    #: Whether the cache keeps the layers of :attr:`CacheSpec.ring` in a ring of window size. Like :attr:`copyable`
    #: it holds only for the class that declares it (a subclass may address its buffers by position).
    supports_ring: ClassVar[bool] = False

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        if "static_variant" not in cls.__dict__:
            cls.static_variant = None
        if "copyable" not in cls.__dict__:
            cls.copyable = False
        if "supports_ring" not in cls.__dict__:
            cls.supports_ring = False

    def __init__(self, spec: CacheSpec) -> None:
        if spec.ring is not None and not type(self).supports_ring:
            raise ConfigError(f"{type(self).__name__} has no window cache (it does not declare `supports_ring`)")
        self.spec = spec
        self.capacity = spec.capacity
        self.batch_size = spec.batch_size
        self._slots: torch.Tensor | None = None
        self._ring_slots: torch.Tensor | None = None
        self._rows: torch.Tensor | None = None
        self._key_positions: dict[torch.device, torch.Tensor] = {}

    @property
    def ring(self) -> RingSpec | None:
        """The window cache of the sliding layers, or ``None``."""
        return self.spec.ring

    def key_positions(self, device: torch.device) -> torch.Tensor:
        """Position of the token stored in each slot of a chain layout, ``[num_slots]`` (slot ``i`` holds position
        ``i``; the trash slot holds :data:`UNSEEN_POSITION`)."""
        table = self._key_positions.get(device)
        if table is None:
            table = torch.arange(self.num_slots, device=device)
            table[-1] = UNSEEN_POSITION
            self._key_positions[device] = table
        return table

    @contextmanager
    def writing_at(self, slots: torch.Tensor) -> Iterator[KVCache]:
        """Direct every write inside the block to ``slots``: ``[S]`` (same slots in every row) or ``[B, S]``."""
        previous = self._slots, self._ring_slots
        self._slots = slots
        # computed here, once per pass: a CUDA graph records it with the pass
        self._ring_slots = None if self.ring is None else self.ring.physical(slots, self.trash_slot)
        try:
            yield self
        finally:
            self._slots, self._ring_slots = previous

    @contextmanager
    def selecting(self, rows: torch.Tensor) -> Iterator[KVCache]:
        """Restrict reads and writes inside the block to the batch rows ``rows`` (a 1-D LongTensor).

        Layers then see a cache of ``len(rows)`` rows. This is meant for rare corrections (re-running the accepted
        tokens of some rows of a hybrid model); the full-batch path copies nothing.
        """
        previous, self._rows = self._rows, rows
        try:
            yield self
        finally:
            self._rows = previous

    # ---- CUDA graphs -------------------------------------------------------------------------------------------
    def external_pass(self, frontier: list[int | None]) -> AbstractContextManager[Any]:
        """A context around a replayed (CUDA graph) exact pass whose row ``b`` writes from slot ``frontier[b]`` on.

        A graph cannot run the host code a cache may run at the boundary of an exact pass (see
        :class:`~economize.techniques._kv.CacheWrapper`), so the decoding loop runs it here, outside the graph, and
        ``writing_at`` inside the block skips it. A plain cache has none.
        """
        return nullcontext()

    def graph_state(self) -> tuple[Any, ...]:
        """What the kernels a graph records over this cache depend on beyond its buffers' addresses.

        A recorded graph is replayed only while this value is unchanged; a cache returns a new value when, for
        example, it stops handing packed codes to attention. A plain cache returns ``()``.
        """
        return ()

    @property
    def active_batch_size(self) -> int:
        """Rows that layers see: the selected rows inside :meth:`selecting`, else the whole batch."""
        return self.batch_size if self._rows is None else len(self._rows)

    @property
    def active_slots(self) -> torch.Tensor:
        """The slots set by the innermost :meth:`writing_at`."""
        if self._slots is None:
            raise RuntimeError(f"{type(self).__name__} was written outside {type(self).__name__}.writing_at(...)")
        return self._slots

    @abstractmethod
    def update(
        self, key_states: torch.Tensor, value_states: torch.Tensor, layer_idx: int, *args: Any, **kwargs: Any
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Write layer ``layer_idx``'s K/V at the active slots and return its whole buffers (the attention keys)."""

    @abstractmethod
    def read(self, layer_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """The buffers of layer ``layer_idx`` (``[B, heads, num_slots, dim]`` each)."""

    def write(self, layer_idx: int, key_states: torch.Tensor, value_states: torch.Tensor) -> None:
        """Write K/V at the active slots without returning the buffers (used to fill skipped layers)."""
        self.update(key_states, value_states, layer_idx)

    @abstractmethod
    def has_layer(self, layer_idx: int) -> bool:
        """Whether layer ``layer_idx`` has written anything yet."""

    @abstractmethod
    def gather(self, rows: torch.Tensor, src: torch.Tensor, dst: torch.Tensor) -> None:
        """Copy, in every layer, slots ``src[i, j]`` of row ``rows[i]`` to slots ``dst[i, j]`` (tree compaction)."""

    @abstractmethod
    def reorder(self, rows: torch.Tensor) -> None:
        """Keep the batch rows ``rows`` in this order (drops finished rows when ``len(rows)`` is smaller)."""

    def snapshot(self) -> RecurrentSnapshot:
        """Copies of the states that masks cannot rewind (none for a K/V-only cache, see
        :attr:`keeps_recurrent_state`)."""
        return {}

    def restore(self, snapshot: RecurrentSnapshot, rows: torch.Tensor | None = None) -> None:
        """Put the states of ``snapshot`` back, for every row or only for ``rows`` (nothing for a K/V-only cache)."""

    # ---- copies and direct state access ------------------------------------------------------------------------
    def tensor_stores(self) -> tuple[dict[int, torch.Tensor], ...]:
        """Every per-layer tensor store that makes up the cache's state, in a fixed order (none here).

        :meth:`detached` and :meth:`clone` rebuild a cache by mapping these stores; subclasses that add stores
        extend the tuple.
        """
        return ()

    def detached(self) -> KVCache:
        """A cache of the same class over the detached tensors of this one (no copy): a stop-gradient view.

        Raises:
            UnsupportedModelError: The class does not declare :attr:`copyable` (a ``TypeError``).
        """
        return self._mapped(torch.Tensor.detach, "detached")

    def clone(self) -> KVCache:
        """A cache of the same class over copies of this one's tensors (gradients flow back to the originals).

        Raises:
            UnsupportedModelError: The class does not declare :attr:`copyable` (a ``TypeError``).
        """
        return self._mapped(torch.Tensor.clone, "clone")

    def _mapped(self, fn: Callable[[torch.Tensor], torch.Tensor], what: str) -> KVCache:
        """A new cache of this class whose stores hold ``fn`` of this cache's tensors (see :attr:`copyable`)."""
        if not type(self).copyable:
            raise UnsupportedModelError(
                f"{type(self).__name__} does not support {what}() (it does not declare `copyable`)"
            )
        copy = type(self)(self.spec)
        copy.batch_size = self.batch_size
        for source, target in zip(self.tensor_stores(), copy.tensor_stores(), strict=True):
            target.update({layer: fn(tensor) for layer, tensor in source.items()})
        return copy

    def recurrent_state(self, layer_idx: int) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """Conv window and recurrent state of layer ``layer_idx`` (``(None, None)`` for a K/V-only cache)."""
        return None, None

    def put_recurrent_state(self, layer_idx: int, conv: torch.Tensor | None, recurrent: torch.Tensor | None) -> None:
        """Replace layer ``layer_idx``'s conv window and recurrent state for every row (``None`` drops one).

        Raises:
            UnsupportedModelError: The cache does not declare :attr:`supports_state_edit` (a ``TypeError``).
        """
        raise UnsupportedModelError(f"{type(self).__name__} keeps no recurrent states that can be replaced")


class PositionalKVCache(KVCache):
    """Per-layer K/V buffers of shape ``[batch, heads, capacity + 1, dim]``, allocated on first write."""

    copyable: ClassVar[bool] = True
    supports_ring: ClassVar[bool] = True

    def __init__(self, spec: CacheSpec) -> None:
        super().__init__(spec)
        self.keys: dict[int, torch.Tensor] = {}
        self.values: dict[int, torch.Tensor] = {}

    def tensor_stores(self) -> tuple[dict[int, torch.Tensor], ...]:
        """The key and value buffers."""
        return (*super().tensor_stores(), self.keys, self.values)

    def update(
        self, key_states: torch.Tensor, value_states: torch.Tensor, layer_idx: int, *args: Any, **kwargs: Any
    ) -> tuple[torch.Tensor, torch.Tensor]:
        slots = self.active_slots
        ring = self.ring is not None and layer_idx in self.ring.layers
        if ring:
            assert self._ring_slots is not None  # writing_at sets it whenever there is a ring
            slots = self._ring_slots
        if layer_idx not in self.keys:
            self.keys[layer_idx] = self._allocate(key_states, ring)
            self.values[layer_idx] = self._allocate(value_states, ring)
        keys, values = self.keys[layer_idx], self.values[layer_idx]
        if self._rows is None:
            write_slots(keys, slots, key_states)
            write_slots(values, slots, value_states)
            return keys, values
        rows = self._rows.to(keys.device)
        sub_keys, sub_values = keys.index_select(0, rows), values.index_select(0, rows)
        write_slots(sub_keys, slots, key_states)
        write_slots(sub_values, slots, value_states)
        keys.index_copy_(0, rows, sub_keys)
        values.index_copy_(0, rows, sub_values)
        return sub_keys, sub_values

    def read(self, layer_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        if layer_idx not in self.keys:
            raise KeyError(f"layer {layer_idx} has not written to the cache yet")
        keys, values = self.keys[layer_idx], self.values[layer_idx]
        if self._rows is None:
            return keys, values
        rows = self._rows.to(keys.device)
        return keys.index_select(0, rows), values.index_select(0, rows)

    def has_layer(self, layer_idx: int) -> bool:
        return layer_idx in self.keys

    def gather(self, rows: torch.Tensor, src: torch.Tensor, dst: torch.Tensor) -> None:
        if self.ring is not None:
            raise ConfigError("a window cache cannot move slots (tree decoding needs the full cache)")
        gather_slots([*self.keys.values(), *self.values.values()], rows, src, dst)  # one launch on CUDA

    def reorder(self, rows: torch.Tensor) -> None:
        for buffers in (self.keys, self.values):
            for index, buffer in buffers.items():
                selected = buffer.index_select(0, rows.to(buffer.device))
                if len(rows) == buffer.shape[0]:
                    buffer.copy_(selected)
                else:
                    buffers[index] = selected
        self.batch_size = len(rows)

    def _allocate(self, like: torch.Tensor, ring: bool = False) -> torch.Tensor:
        if self._rows is not None:
            raise RuntimeError("a cache layer must be first written for the whole batch, not inside selecting()")
        if like.shape[0] != self.batch_size:
            raise ConfigError(f"the cache holds {self.batch_size} rows but a layer wrote {like.shape[0]}")
        slots = self.ring.buffer_slots if ring and self.ring is not None else self.num_slots
        return like.new_zeros(like.shape[0], like.shape[1], slots, *like.shape[3:])


class StaticKVCache(PositionalKVCache):
    """A :class:`PositionalKVCache` whose buffers keep their storage for the life of the cache (for CUDA graphs).

    A captured CUDA graph reads and writes the buffers it saw at capture. This cache therefore never rebinds a
    buffer: it keeps one set of buffers per batch size, :meth:`reorder` to fewer rows copies the kept rows into the
    set of the smaller size (allocated on first use), and :meth:`activate` switches to the set of a new
    generation's batch size. Memory grows with the number of distinct batch sizes the cache has seen, one full set
    per size.

    A new generation reuses the buffers without clearing them. The stale entries are harmless: a query sees only
    slots below its own position (and its own), and every one of them is written by the current generation before
    it is read.
    """

    supports_ring: ClassVar[bool] = True

    def __init__(self, spec: CacheSpec) -> None:
        super().__init__(spec)
        self._sets: dict[int, tuple[dict[int, torch.Tensor], dict[int, torch.Tensor]]] = {
            spec.batch_size: (self.keys, self.values)
        }

    @property
    def batch_sizes(self) -> tuple[int, ...]:
        """Batch sizes with a buffer set."""
        return tuple(sorted(self._sets))

    def activate(self, batch_size: int) -> None:
        """Use the buffers of ``batch_size`` rows from now on (a new set is allocated on its first write)."""
        check_positive("batch size", batch_size)
        self.keys, self.values = self._sets.setdefault(batch_size, ({}, {}))
        self.batch_size = batch_size

    def reorder(self, rows: torch.Tensor) -> None:
        if len(rows) == self.batch_size:
            super().reorder(rows)  # same size: copies in place
            return
        keys, values = self.keys, self.values
        self.activate(len(rows))
        for source, target in ((keys, self.keys), (values, self.values)):
            for layer, buffer in source.items():
                selected = buffer.index_select(0, rows.to(buffer.device))
                if layer in target and target[layer].shape == selected.shape:
                    target[layer].copy_(selected)
                else:
                    target[layer] = selected


PositionalKVCache.static_variant = StaticKVCache  # named here: the variant is defined after the class
StaticKVCache.static_variant = StaticKVCache


class _LinearLayerView:
    """What linear-attention layers read from ``cache_params.layers[idx]``.

    ``transformers`` >= 5.14 indexes the states by ``state_idx`` (``conv_states[0]``) and checks ``record_past``
    (``True`` keeps the layer on the unfused path that calls :meth:`HybridCache.update_conv_state`); older
    versions read the tensors directly.
    """

    record_past = True

    def __init__(self, conv: torch.Tensor | None, recurrent: torch.Tensor | None) -> None:
        self.conv_states: Any = {0: conv} if _INDEXED_STATES else conv
        self.recurrent_states: Any = {0: recurrent} if _INDEXED_STATES else recurrent


class _LayerList:
    def __init__(self, cache: HybridCache) -> None:
        self._cache = cache

    def __getitem__(self, layer_idx: int) -> _LinearLayerView:
        return _LinearLayerView(*self._cache.recurrent_state(layer_idx))

    def __len__(self) -> int:
        return 1 + max([*self._cache.keys, *self._cache.conv_states, -1])


_CONV, _RECURRENT = 0, 1


class HybridCache(PositionalKVCache):
    """K/V buffers for attention layers plus the states of recurrent layers (gated delta nets, Mamba-2).

    It implements the part of the ``transformers`` cache API that these layers call (``has_previous_state``,
    ``layers[idx]``, ``update_conv_state``, ``update_recurrent_state``) for Qwen3.5, Qwen3-Next and Nemotron-H, on
    ``transformers`` 5.13 and later. Each recurrent layer keeps a conv window ``[B, channels, kernel]`` and a
    recurrent state. States advance with every token they see, so a decoder that processes speculative tokens takes
    a :meth:`snapshot` before and restores it afterwards, then replays the accepted tokens.
    """

    keeps_recurrent_state: ClassVar[bool] = True
    copyable: ClassVar[bool] = True
    supports_state_edit: ClassVar[bool] = True

    def __init__(self, spec: CacheSpec) -> None:
        super().__init__(spec)
        self.conv_states: dict[int, torch.Tensor] = {}
        self.recurrent_states: dict[int, torch.Tensor] = {}
        self._views: dict[tuple[int, int], torch.Tensor] = {}

    def tensor_stores(self) -> tuple[dict[int, torch.Tensor], ...]:
        """The key and value buffers, then the conv windows and recurrent states."""
        return (*super().tensor_stores(), self.conv_states, self.recurrent_states)

    def put_recurrent_state(self, layer_idx: int, conv: torch.Tensor | None, recurrent: torch.Tensor | None) -> None:
        """Store ``conv`` and ``recurrent`` as layer ``layer_idx``'s states as they are (``None`` drops one)."""
        for which, (store, value) in enumerate(zip(self._stores, (conv, recurrent))):
            self._views.pop((layer_idx, which), None)
            if value is None:
                store.pop(layer_idx, None)
            else:
                store[layer_idx] = value

    @property
    def layers(self) -> _LayerList:
        """``cache_params.layers[idx]`` of the ``transformers`` cache API. A fresh view per access: a stored one
        would form a reference cycle with the cache, which only the cyclic collector frees, so during training
        every micro-batch's caches (and the recurrent states they hold) would pile up on the GPU until it ran."""
        return _LayerList(self)

    @contextmanager
    def selecting(self, rows: torch.Tensor) -> Iterator[KVCache]:
        if self._rows is not None:
            raise RuntimeError("HybridCache.selecting() cannot be nested")
        self._views = {}
        try:
            with super().selecting(rows) as cache:
                yield cache
        finally:
            for (layer_idx, which), view in self._views.items():
                store = self._stores[which][layer_idx]
                store.index_copy_(0, rows.to(store.device), view.to(store.dtype))
            self._views = {}

    @property
    def _stores(self) -> tuple[dict[int, torch.Tensor], dict[int, torch.Tensor]]:
        return self.conv_states, self.recurrent_states

    def recurrent_state(self, layer_idx: int) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """Conv and recurrent state of layer ``layer_idx`` (copies of the selected rows inside :meth:`selecting`).

        Single-token steps of some layers update these tensors in place. Outside :meth:`selecting` the stored
        tensors themselves are handed out; inside, the copy handed out (one per layer and state, however often it
        is read) is written back when the block ends unless the layer stored a new state meanwhile.
        """
        return self._view(layer_idx, _CONV), self._view(layer_idx, _RECURRENT)

    def has_previous_state(self, layer_idx: int | None = None, state_idx: int | None = None) -> bool:
        """Whether layer ``layer_idx`` has seen tokens (any layer when ``None``)."""
        if layer_idx is None:
            return bool(self.recurrent_states)
        return layer_idx in self.recurrent_states

    def update_conv_state(
        self, conv_states: torch.Tensor, layer_idx: int, state_idx: int = 0, conv_kernel_size: int | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Store the conv window of layer ``layer_idx`` and return what the layer convolves.

        With ``conv_kernel_size`` (``transformers`` >= 5.14), ``conv_states`` are the new conv inputs: the last
        ``conv_kernel_size`` inputs are stored and the inputs are returned with the stored ones prepended. Without
        it, ``conv_states`` is the new window itself, or, when it is narrower than the stored window (a single-token
        Mamba-2 step), the newest inputs, which are shifted into the window; the window is returned.
        """
        self._views.pop((layer_idx, _CONV), None)
        previous = self._rowwise(self.conv_states.get(layer_idx))
        if conv_kernel_size is None:
            window = conv_states
            if previous is not None and conv_states.shape[-1] < previous.shape[-1]:
                window = torch.cat([previous[..., conv_states.shape[-1]:], conv_states.to(previous.dtype)], dim=-1)
            self._store(self.conv_states, layer_idx, window)
            return window
        full = conv_states if previous is None else torch.cat([previous.to(conv_states.dtype), conv_states], dim=-1)
        window = full[..., -conv_kernel_size:]
        if window.shape[-1] < conv_kernel_size:
            window = F.pad(window, (conv_kernel_size - window.shape[-1], 0))
        self._store(self.conv_states, layer_idx, window)
        return full

    def update_recurrent_state(
        self, recurrent_states: torch.Tensor, layer_idx: int, state_idx: int = 0, **kwargs: Any
    ) -> torch.Tensor:
        """Store the recurrent state after the current tokens."""
        self._views.pop((layer_idx, _RECURRENT), None)
        self._store(self.recurrent_states, layer_idx, recurrent_states)
        return recurrent_states

    def snapshot(self) -> RecurrentSnapshot:
        """Copies of every conv and recurrent state (K/V buffers need none: masks hide stale slots)."""
        layers = sorted({*self.conv_states, *self.recurrent_states})
        return {i: (_clone(self.conv_states.get(i)), _clone(self.recurrent_states.get(i))) for i in layers}

    def restore(self, snapshot: RecurrentSnapshot, rows: torch.Tensor | None = None) -> None:
        """Put the states of ``snapshot`` back, for every row or only for ``rows``.

        A state that did not exist when the snapshot was taken is dropped (``rows=None``) or zeroed in ``rows``,
        which is how a layer that has not seen tokens behaves.

        Raises:
            ConfigError: ``rows`` is given and the snapshot holds a state that the cache no longer has (the states of
                the other rows would be undefined); restore every row instead.
        """
        if rows is not None:
            for index, states in snapshot.items():
                for which, store in enumerate(self._stores):
                    if states[which] is not None and index not in store:
                        raise ConfigError(
                            f"cannot restore some rows of layer {index}: the cache has no state for it, so the "
                            "other rows would have none; restore every row (rows=None)"
                        )
        for which, store in enumerate(self._stores):
            for index in list(store):
                state = snapshot.get(index, (None, None))[which]
                if state is not None:
                    continue
                if rows is None:
                    del store[index]
                else:
                    store[index].index_fill_(0, rows.to(store[index].device), 0)
            for index, states in snapshot.items():
                state = states[which]
                if state is None:
                    continue
                if rows is None:
                    store[index] = state.clone()
                else:
                    r = rows.to(state.device)
                    store[index].index_copy_(0, r, state.index_select(0, r))

    def reorder(self, rows: torch.Tensor) -> None:
        super().reorder(rows)
        for store in self._stores:
            for index, state in store.items():
                store[index] = state.index_select(0, rows.to(state.device))

    def _view(self, layer_idx: int, which: int) -> torch.Tensor | None:
        view = self._views.get((layer_idx, which))
        if view is None:
            view = self._rowwise(self._stores[which].get(layer_idx))
            if self._rows is not None and view is not None:
                self._views[(layer_idx, which)] = view
        return view

    def _rowwise(self, state: torch.Tensor | None) -> torch.Tensor | None:
        if state is None or self._rows is None:
            return state
        return state.index_select(0, self._rows.to(state.device))

    def _store(self, store: dict[int, torch.Tensor], layer_idx: int, state: torch.Tensor) -> None:
        if self._rows is None:
            store[layer_idx] = state
            return
        if layer_idx not in store:
            raise ConfigError(
                f"layer {layer_idx} has no state yet, so a state for selected rows cannot be stored: write the layer "
                "for the whole batch first (run it outside HybridCache.selecting)"
            )
        store[layer_idx].index_copy_(0, self._rows.to(state.device), state.to(store[layer_idx].dtype))


def _clone(tensor: torch.Tensor | None) -> torch.Tensor | None:
    return None if tensor is None else tensor.clone()


#: Cache classes by :attr:`CacheSpec.kind`. Each is constructed as ``cls(spec)``.
CACHES: Registry[type[KVCache]] = Registry("cache", "economize.caches")
CACHES.register("positional", PositionalKVCache)
CACHES.register("hybrid", HybridCache)
