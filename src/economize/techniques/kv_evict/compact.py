"""Physical storage for positional eviction: the sinks and a ring of recent entries per covered layer.

:class:`CompactWindowCache` decodes exactly as the masked :class:`~.technique.WindowEvictionCache` (the same cuts,
the same visible entries for every real query), but the K/V of every covered layer live in buffers of
``P = sinks + ring + 1`` slots that the wrapper owns, so the inner cache never allocates them and evicted entries
free their memory. Logical slot ``s`` (the position) is stored at physical slot

* ``s`` for a sink (``s < sinks``),
* ``sinks + (s - sinks) % ring`` otherwise,
* ``P - 1`` (the physical trash slot) for the logical trash slot.

The ring holds ``recent + headroom`` entries: the ``recent`` entries every cut keeps plus room for the uncommitted
tokens of one cycle (a verify pass, the draft steps before it, a rollback). A ``held`` tensor ``[B, P]`` per layer
records the logical slot each physical slot holds (``-1`` for none). Attention gets the owned buffers as keys, and
the layer's mask is gathered from the logical mask at the held labels, so a query sees the same entries as under
the dense cache: the ones the original mask allows, minus the evicted range. Since the cuts never decrease, an
entry overwritten below the current cut is never visible again; an overwrite at or above the cut (a cycle longer
than the headroom) sets a device flag, which the next pass boundary turns into a :class:`~economize.ConfigError`.

Three paths write a covered layer:

* the first write of a generation (the prompt pass) attends over the new K/V only and keeps the sinks and the last
  ``ring`` entries of every row;
* a pass of at most ``headroom`` queries writes into the ring in place and attends over the owned buffers (the
  path decoding steps and CUDA graphs take);
* a longer pass (a chunk of a chunked prefill, eager only) attends over the owned buffers and the new K/V side by
  side, then keeps what the first path keeps.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import torch

from ...errors import ConfigError
from ...modules.kv_cache import write_slots
from ...modules.slot_gather import gather_slots
from ...runtime import KVCache
from .technique import WindowEvictionCache, _capturing

if TYPE_CHECKING:
    from .technique import KVEvictTechnique

__all__ = ["CompactWindowCache"]

_FRESH, _NARROW, _WIDE = "fresh", "narrow", "wide"


@dataclass
class _Store:
    """The owned buffers of one batch size: per covered layer keys and values ``[B, H_kv, P, D]`` and the logical
    slot ``[B, P]`` every physical slot holds (``-1`` for none)."""

    keys: dict[int, torch.Tensor] = field(default_factory=dict)
    values: dict[int, torch.Tensor] = field(default_factory=dict)
    held: dict[int, torch.Tensor] = field(default_factory=dict)


class CompactWindowCache(WindowEvictionCache):
    """A :class:`~.technique.WindowEvictionCache` that stores only the kept entries (see the module docstring).

    Over the static cache of CUDA graphs the owned buffers, like the cuts, keep one set per batch size at fixed
    addresses, and a new generation refills them in place.

    Raises:
        ConfigError: A pass, a rollback or a chunk needs entries the ring no longer holds (raise ``headroom``,
            or use ``storage="masked"``), a covered layer gets no dense mask, or a recorded pass is longer than
            the headroom.
    """

    def __init__(self, inner: KVCache, owner: KVEvictTechnique) -> None:
        super().__init__(inner, owner)
        slots = self.num_slots
        self.sinks = min(self.method.sinks, slots - 1)
        self.ring_size = max(1, min(owner.config.recent + owner.config.headroom, slots - 1 - self.sinks))
        self.headroom = min(owner.config.headroom, self.ring_size)
        self.physical_slots = self.sinks + self.ring_size + 1
        self._stores: dict[int, _Store] = {}
        self._store = self._stores.setdefault(inner.batch_size, _Store())
        self._overflow: torch.Tensor | None = None  # [1] bool on the device, at a fixed address
        self._overflow_seen: torch.Tensor | None = None  # its pinned host copy (CUDA), read one pass late
        self._written: set[int] = set()
        self._held_from: list[int] = [0] * inner.batch_size  # lowest non-sink slot a pruning write kept, per row
        self._route: dict[int, tuple[str, torch.Tensor]] = {}

    # ---- layout -------------------------------------------------------------------------------------------------
    def phys(self, slots: torch.Tensor) -> torch.Tensor:
        """The physical slot of every logical slot of ``slots``."""
        ring = self.sinks + (slots - self.sinks).remainder(self.ring_size)
        inner = torch.where(slots < self.sinks, slots, ring)
        return torch.where(slots == self.trash_slot, torch.full_like(slots, self.physical_slots - 1), inner)

    def owned_buffers(self) -> dict[int, torch.Tensor]:
        return {
            layer: self.stand_in(keys.shape[0], keys.shape[1], self.num_slots, keys.shape[3], keys.dtype, keys.device)
            for layer, keys in self._store.keys.items()
        }

    def _device(self) -> torch.device | None:
        buffers = self.positional_buffers()
        return next(iter(buffers.values())).device if buffers else None

    def has_layer(self, layer_idx: int) -> bool:
        return layer_idx in self._store.keys or self.inner.has_layer(layer_idx)

    def _owns(self, layer_idx: int) -> bool:
        return self.covers(layer_idx)

    def _query_slots(self, batch: int | None = None) -> torch.Tensor:
        """``[B', Q]`` int64 logical slots of the pass's queries in the active rows."""
        slots = self.active_slots.long()
        if slots.dim() == 1:
            slots = slots[None].expand(self.active_batch_size if batch is None else batch, -1)
        return slots

    def _rows_on(self, device: torch.device) -> torch.Tensor | None:
        rows = self.active_rows
        return None if rows is None else rows.to(device)

    def _cut(self, device: torch.device) -> torch.Tensor:
        """``[B']`` device cuts of the active rows."""
        cut = self._cut_sets.get(len(self._cuts))
        if cut is None or cut.device != device:
            cut = self._sync(device)
            assert cut is not None
        rows = self._rows_on(device)
        return cut if rows is None else cut.index_select(0, rows)

    # ---- masks --------------------------------------------------------------------------------------------------
    def attention_mask(self, layer_idx: int, mask: torch.Tensor | None) -> torch.Tensor | None:
        """The mask of the keys :meth:`update` returns for cache layer ``layer_idx``: ``mask`` (over the logical
        slots) gathered at the logical slot of every returned key, without the evicted range. Layers the cache
        does not own keep the masked reference's behaviour."""
        if not self._owns(layer_idx):
            return super().attention_mask(layer_idx, mask)
        if layer_idx not in self._written:
            if _capturing(mask) if isinstance(mask, torch.Tensor) else False:
                raise ConfigError("a CUDA graph cannot record the first write of a compact eviction cache layer")
            slots = self._query_slots()
            labels = torch.where(slots != self.trash_slot, slots, torch.full_like(slots, -1))
            self._route[layer_idx] = (_FRESH, labels)
            if not isinstance(mask, torch.Tensor):  # causal (None) or structured: the new keys alone, in order
                return mask
            span = self._span(slots)
            if span is not None:  # the usual prompt pass: a view of the mask instead of a [B, h, Q, Q] copy
                return mask[..., span[0] : span[1]]
            return self._visible(mask, labels)
        if not isinstance(mask, torch.Tensor):
            raise ConfigError("compact KV eviction needs dense attention masks after the prompt pass")
        slots = self._query_slots().to(mask.device)
        held = self._store.held[layer_idx]
        rows = self._rows_on(held.device)
        held = held if rows is None else held.index_select(0, rows)
        if slots.shape[1] <= self.headroom:
            real = slots != self.trash_slot
            labels = held.clone().scatter_(1, self.phys(slots), torch.where(real, slots, torch.full_like(slots, -1)))
            self._route[layer_idx] = (_NARROW, labels)
            return self._visible(mask, labels)
        if _capturing(mask):
            raise ConfigError(
                f"a recorded pass of {slots.shape[1]} queries exceeds the compact eviction cache's headroom "
                f"({self.headroom}): increase headroom or use storage='masked'"
            )
        real = slots != self.trash_slot
        first = torch.where(real, slots, torch.full_like(slots, self.num_slots)).amin(1, keepdim=True)
        owned = torch.where((held >= 0) & (held < first), held, torch.full_like(held, -1))
        labels = torch.cat([owned, torch.where(real, slots, torch.full_like(slots, -1))], dim=1)
        self._route[layer_idx] = (_WIDE, labels)
        return self._visible(mask, labels)

    def _span(self, slots: torch.Tensor) -> tuple[int, int] | None:
        """``(start, end)`` when every active row writes the same consecutive real slots and no row has evicted
        yet, so the fresh keys are the mask's columns ``start:end``. Reads the slots back once per layer of a
        prompt pass (never under CUDA graphs, which cannot record a first write)."""
        if any(cut > self.sinks for cut in self._cuts):
            return None
        start = int(slots[0, 0])
        expected = torch.arange(start, start + slots.shape[1], device=slots.device)
        if start == self.trash_slot or not bool((slots == expected).all()):
            return None
        return start, start + slots.shape[1]

    def _visible(self, mask: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """``mask`` ``[B' or 1, h, Q, slots]`` at the keys with logical slots ``labels`` ``[B', K]`` (``-1``: no
        entry), without the evicted range. Padding queries always see key 0 (an empty row would produce NaN); their
        outputs only feed the K/V of the trash slot. Allocates the gathered mask and nothing else of its size."""
        labels = labels.to(mask.device)
        batch, keys = labels.shape
        heads, queries = mask.shape[1], mask.shape[2]
        index = torch.where(labels >= 0, labels, torch.full_like(labels, self.trash_slot))
        index = index[:, None, None, :].expand(batch, heads, queries, keys)
        gathered = mask.expand(batch, -1, -1, -1).gather(-1, index)
        cut = self._cut(mask.device)
        padding = (self._query_slots(batch).to(mask.device) == self.trash_slot)[:, None, :, None]
        hidden = ((labels < 0) | ((labels >= self.sinks) & (labels < cut[:, None])))[:, None, None, :]
        if mask.dtype == torch.bool:
            gathered &= ~hidden
            gathered[..., :1] |= padding
        else:
            gathered.masked_fill_(hidden, torch.finfo(mask.dtype).min)
            gathered[..., :1].masked_fill_(padding, 0.0)
        return gathered

    # ---- writes -------------------------------------------------------------------------------------------------
    def update(
        self, key_states: torch.Tensor, value_states: torch.Tensor, layer_idx: int, *args: Any, **kwargs: Any
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not self._owns(layer_idx):
            return super().update(key_states, value_states, layer_idx, *args, **kwargs)
        route = self._route.pop(layer_idx, None)
        if route is None:
            raise ConfigError(
                f"cache layer {layer_idx} was written without the kv_evict layer hook (compact storage needs it)"
            )
        kind, labels = route
        if kind == _FRESH:
            self._prune_write(layer_idx, key_states, value_states)
            return key_states, value_states
        if kind == _NARROW:
            return self._ring_write(layer_idx, key_states, value_states, labels)
        keys, values = self._owned(layer_idx)
        keys = torch.cat([keys, key_states.to(keys.dtype)], dim=2)
        values = torch.cat([values, value_states.to(values.dtype)], dim=2)
        self._prune_write(layer_idx, key_states, value_states)
        return keys, values

    def write(self, layer_idx: int, key_states: torch.Tensor, value_states: torch.Tensor) -> None:
        if not self._owns(layer_idx):
            super().write(layer_idx, key_states, value_states)
            return
        slots = self._query_slots()
        if layer_idx in self._written and slots.shape[1] <= self.headroom:
            held = self._owned_held(layer_idx)
            real = slots.to(held.device) != self.trash_slot
            slots = slots.to(held.device)
            labels = held.clone().scatter_(1, self.phys(slots), torch.where(real, slots, torch.full_like(slots, -1)))
            self._ring_write(layer_idx, key_states, value_states, labels)
        else:
            self._prune_write(layer_idx, key_states, value_states)

    def read(self, layer_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """The layer's K/V laid out over the logical slots (zeros where nothing is held), for readers that index
        the cache by position. Costs a dense ``[B', H_kv, slots, D]`` copy."""
        if not self._owns(layer_idx):
            return super().read(layer_idx)
        if layer_idx not in self._store.keys:
            raise KeyError(f"layer {layer_idx} has not written to the cache yet")
        keys, values = self._owned(layer_idx)
        held = self._owned_held(layer_idx)
        index = torch.where(held >= 0, held, torch.full_like(held, self.trash_slot))
        out = []
        for buffer in (keys, values):
            dense = buffer.new_zeros(buffer.shape[0], buffer.shape[1], self.num_slots, buffer.shape[3])
            dense.scatter_(2, index[:, None, :, None].expand_as(buffer), buffer)
            out.append(dense)
        return out[0], out[1]

    def _owned(self, layer_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        keys, values = self._store.keys[layer_idx], self._store.values[layer_idx]
        rows = self._rows_on(keys.device)
        if rows is None:
            return keys, values
        return keys.index_select(0, rows), values.index_select(0, rows)

    def _owned_held(self, layer_idx: int) -> torch.Tensor:
        held = self._store.held[layer_idx]
        rows = self._rows_on(held.device)
        return held if rows is None else held.index_select(0, rows)

    def _allocate(self, layer_idx: int, like: torch.Tensor) -> None:
        if layer_idx in self._store.keys:
            return
        if self.active_rows is not None:
            raise RuntimeError("a cache layer must be first written for the whole batch, not inside selecting()")
        if like.shape[0] != self.inner.batch_size:
            raise ConfigError(f"the cache holds {self.inner.batch_size} rows but a layer wrote {like.shape[0]}")
        shape = (like.shape[0], like.shape[1], self.physical_slots, like.shape[3])
        self._store.keys[layer_idx] = like.new_zeros(shape)
        self._store.values[layer_idx] = like.new_zeros(shape)
        self._store.held[layer_idx] = torch.full(
            (like.shape[0], self.physical_slots), -1, dtype=torch.long, device=like.device
        )
        if self._overflow is None:
            self._overflow = torch.zeros(1, dtype=torch.bool, device=like.device)
            if like.is_cuda:
                self._overflow_seen = torch.zeros(1, dtype=torch.bool, pin_memory=True)

    def _ring_write(
        self, layer_idx: int, key_states: torch.Tensor, value_states: torch.Tensor, labels: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Write a short pass into the ring in place and return the owned buffers of the active rows."""
        store = self._store
        keys, values, held = store.keys[layer_idx], store.values[layer_idx], store.held[layer_idx]
        slots = self._query_slots().to(held.device)
        phys = self.phys(slots)
        rows = self._rows_on(keys.device)
        old = (held if rows is None else held.index_select(0, rows)).gather(1, phys)
        cut = self._cut(held.device).clamp(min=self.sinks)[:, None]
        lost = (slots != self.trash_slot) & (old >= cut) & (old < slots)
        assert self._overflow is not None
        self._overflow.logical_or_(lost.any().view(1))
        labels = labels.to(held.device)
        if rows is None:
            write_slots(keys, phys, key_states)
            write_slots(values, phys, value_states)
            held.copy_(labels)
            return keys, values
        sub_keys, sub_values = keys.index_select(0, rows), values.index_select(0, rows)
        write_slots(sub_keys, phys, key_states)
        write_slots(sub_values, phys, value_states)
        keys.index_copy_(0, rows, sub_keys)
        values.index_copy_(0, rows, sub_values)
        held.index_copy_(0, rows, labels)
        return sub_keys, sub_values

    def _prune_write(self, layer_idx: int, key_states: torch.Tensor, value_states: torch.Tensor) -> None:
        """Write a pass but keep only the sinks and the last ``ring`` entries of every row (eager paths)."""
        self._allocate(layer_idx, key_states)
        store = self._store
        keys, values, held = store.keys[layer_idx], store.values[layer_idx], store.held[layer_idx]
        slots = self._query_slots().to(held.device)
        real = slots != self.trash_slot
        end = torch.where(real, slots + 1, torch.zeros_like(slots)).amax(1, keepdim=True)
        keep = real & ((slots < self.sinks) | (slots >= end - self.ring_size))
        trash = torch.full_like(slots, self.physical_slots - 1)
        phys = torch.where(keep, self.phys(slots), trash)
        labels = torch.where(keep, slots, torch.full_like(slots, -1))
        rows = self._rows_on(keys.device)
        if rows is None:
            write_slots(keys, phys, key_states)
            write_slots(values, phys, value_states)
            held.scatter_(1, phys, labels)
        else:
            sub_keys, sub_values = keys.index_select(0, rows), values.index_select(0, rows)
            sub_held = held.index_select(0, rows)
            write_slots(sub_keys, phys, key_states)
            write_slots(sub_values, phys, value_states)
            sub_held.scatter_(1, phys, labels)
            keys.index_copy_(0, rows, sub_keys)
            values.index_copy_(0, rows, sub_values)
            held.index_copy_(0, rows, sub_held)
        held[:, -1] = -1
        self._written.add(layer_idx)
        ids = range(len(self._held_from)) if rows is None else rows.tolist()
        for row, last in zip(ids, (end[:, 0] - self.ring_size).tolist(), strict=True):
            self._held_from[row] = max(self._held_from[row], int(last))

    # ---- pass boundaries ----------------------------------------------------------------------------------------
    def pass_begin(self, frontier: list[int | None]) -> None:
        super().pass_begin(frontier)
        self._check_overflow()
        if not self._written:
            return
        for row, (f, cut, kept) in enumerate(zip(frontier, self._cuts, self._held_from, strict=True)):
            if f is not None and max(cut, self.sinks) < min(kept, f):
                raise ConfigError(
                    f"row {row} needs entries from slot {max(cut, self.sinks)} on, but the compact eviction cache "
                    f"kept only those from slot {kept} on: a rollback or a chunk is longer than the headroom "
                    f"({self.headroom}); increase headroom, use storage='masked' or an unchunked prefill"
                )

    def _check_overflow(self) -> None:
        flag = self._overflow
        if flag is None:
            return
        seen = self._overflow_seen
        if seen is None:
            lost = bool(flag)
        else:  # read the copy made at the previous boundary: no synchronization with the device
            lost = bool(seen)
            seen.copy_(flag, non_blocking=True)
        if lost:
            raise ConfigError(
                "a pass overwrote entries of the compact eviction cache that attention still needed: a chunked "
                f"prompt is longer than the ring ({self.ring_size} = recent + headroom) or the uncommitted tokens of a "
                f"cycle exceed the headroom ({self.headroom}); increase headroom or use storage='masked'"
            )

    # ---- layout changes -----------------------------------------------------------------------------------------
    def on_gather(self, rows: torch.Tensor, src: torch.Tensor, dst: torch.Tensor) -> None:
        store = self._store
        if not store.keys:
            return
        device = next(iter(store.held.values())).device
        rows, src, dst = rows.to(device), src.to(device).long(), dst.to(device).long()
        phys_src, phys_dst = self.phys(src), self.phys(dst)
        gather_slots([*store.keys.values(), *store.values.values()], rows, phys_src, phys_dst)
        cut = torch.tensor(self._cuts, device=device).index_select(0, rows).clamp(min=self.sinks)[:, None]
        assert self._overflow is not None
        r = rows[:, None]
        for held in store.held.values():
            moved = held[r, phys_src]
            old = held[r, phys_dst]
            lost = (old >= cut) & (old < dst) & (moved != dst)
            self._overflow.logical_or_(lost.any().view(1))
            held[r, phys_dst] = torch.where(moved == src, dst, torch.full_like(dst, -1))
            held[:, -1] = -1

    def on_reorder(self, rows: torch.Tensor) -> None:
        source = self._store
        size = len(rows)
        if self._static and size != self.inner.batch_size:
            raise AssertionError("the inner cache reorders first")  # pragma: no cover
        target = self._stores.setdefault(size, _Store()) if self._static else source
        for name in ("keys", "values", "held"):
            src_buffers, dst_buffers = getattr(source, name), getattr(target, name)
            for layer, buffer in list(src_buffers.items()):
                selected = buffer.index_select(0, rows.to(buffer.device))
                current = dst_buffers.get(layer)
                if current is not None and current.shape == selected.shape:
                    current.copy_(selected)
                else:
                    dst_buffers[layer] = selected
        self._store = target
        self._held_from = [self._held_from[r] for r in rows.tolist()]
        super().on_reorder(rows)

    def on_restart(self) -> None:
        super().on_restart()
        self._store = self._stores.setdefault(self.inner.batch_size, _Store())
        for store in self._stores.values():
            for held in store.held.values():
                held.fill_(-1)
        if self._overflow is not None:
            self._overflow.zero_()
        if self._overflow_seen is not None:
            self._overflow_seen.zero_()
        self._written = set()
        self._held_from = [0] * self.inner.batch_size
        self._route = {}
