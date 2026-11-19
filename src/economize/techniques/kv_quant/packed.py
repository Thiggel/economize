"""Packed KIVI storage: low-bit codes and a full-precision ring per covered layer, no dense buffer.

:class:`PackedKIVICache` quantizes exactly what :class:`~.kivi.KIVICache` quantizes, at the same pass boundaries,
but keeps the quantized entries of every covered layer only as packed codes (``bits`` per element, keys per channel
in groups of ``group_size`` tokens, values per token in groups of ``group_size`` channels) with fp16 scales and zero
points. The entries that are not quantized yet live in a ring of ``residual_length + headroom`` slots: slot ``s``
is stored at ``s % ring`` (the logical trash slot at index ``ring``). The ring holds the ``residual_length`` most
recent committed entries plus room for the uncommitted tokens of one cycle (draft steps, a verify pass, a rollback).
A write that would overwrite an entry not yet quantized sets a device flag, which the next pass boundary turns
into a :class:`~economize.ConfigError`.

The inner cache never sees a write of a covered layer, so it allocates no dense buffer for it: the memory the cache
holds per covered layer is that of the codes, the scales, the zero points and the ring
(:meth:`PackedKIVICache.allocated_bytes`).

Three paths read a covered layer:

* attention implementations that read the ``"kivi"`` packed format get zero-memory stand-ins that carry a
  :class:`~economize.attention.kivi.KIVIPacked` with the codes and the rings, for passes of at most ``headroom``
  queries (decoding steps); the kernel dequantizes on the fly and reads the rest from the rings;
* any other attention implementation (``sdpa``, ``eager``), a longer pass or a pass over selected rows gets a
  transient dense copy of the layer: the codes dequantized below the quantized prefix, the ring above it;
* a prompt pass longer than ``headroom`` builds that dense copy with the prompt written into it, packs what KIVI
  quantizes at the end of the prompt straight from it, and keeps only the last ``ring`` entries in the ring.

Dequantized values equal those of the simulated storage bit for bit, so both storages decode the same tokens.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from ...attention.kivi import KIVIPacked, dequantize, pack_codes
from ...attention.packed import attach_packed
from ...errors import ConfigError
from ...modules.kv_cache import write_slots
from ...modules.slot_gather import gather_slots
from ...runtime import KVCache
from .kivi import _FP16_MAX, KIVICache, KIVIConfig, quantize, value_group

__all__ = ["PackedKIVICache"]

_TENSORS = ("key_codes", "key_scale", "key_zero", "value_codes", "value_scale", "value_zero", "key_ring", "value_ring")


@dataclass
class _Store:
    """Codes, scales, zero points and rings of one layer (see the module docstring for the shapes)."""

    value_group: int
    key_codes: torch.Tensor
    key_scale: torch.Tensor
    key_zero: torch.Tensor
    value_codes: torch.Tensor
    value_scale: torch.Tensor
    value_zero: torch.Tensor
    key_ring: torch.Tensor
    value_ring: torch.Tensor

    def tensors(self) -> list[torch.Tensor]:
        return [getattr(self, name) for name in _TENSORS]

    def reorder(self, rows: torch.Tensor) -> None:
        for name in _TENSORS:
            tensor = getattr(self, name)
            setattr(self, name, tensor.index_select(0, rows.to(tensor.device)))


class PackedKIVICache(KIVICache):
    """A :class:`~.kivi.KIVICache` that stores its covered layers packed (see the module docstring).

    Args:
        inner: The cache it wraps (a positional cache).
        config: The KIVI configuration (``headroom`` sizes the ring).
        layers: 0-indexed cache layers it covers (``None``: all).
        reads_packed: Whether the model's attention implementation reads the ``"kivi"`` packed format.

    Raises:
        ConfigError: A cycle writes past the ring (raise ``headroom`` or use ``storage="simulated"``), or a scale or
            zero point overflows fp16 (use ``storage="simulated"``).
    """

    def __init__(
        self, inner: KVCache, config: KIVIConfig, layers: frozenset[int] | None, *, reads_packed: bool = False
    ) -> None:
        super().__init__(inner, config, layers)
        self.reads_packed = reads_packed
        self.headroom = config.headroom
        self.residual_ring = max(1, min(config.residual_length + config.headroom, self.num_slots - 1))
        self._stores: dict[int, _Store] = {}
        self._counts: dict[torch.device, tuple[torch.Tensor, torch.Tensor]] = {}
        self._flags: dict[torch.device, torch.Tensor] = {}  # [2] bool: ring overwrite, fp16 overflow
        self._prequantized: set[int] = set()  # layers whose prompt was packed while it was written

    # ---- layout ----------------------------------------------------------------------------------------------
    def _phys(self, slots: torch.Tensor) -> torch.Tensor:
        """Ring index of every logical slot of ``slots`` (the trash slot maps to index ``ring``)."""
        trash = torch.full_like(slots, self.residual_ring)
        return torch.where(slots == self.trash_slot, trash, slots.remainder(self.residual_ring))

    def allocated_bytes(self) -> int:
        """Bytes of every tensor the cache holds for its covered layers."""
        return sum(t.numel() * t.element_size() for store in self._stores.values() for t in store.tensors())

    def owned_buffers(self) -> dict[int, torch.Tensor]:
        return {
            layer: self.stand_in(
                store.key_ring.shape[0], store.key_ring.shape[1], self.num_slots, store.key_ring.shape[3],
                store.key_ring.dtype, store.key_ring.device,
            )
            for layer, store in self._stores.items()
        }

    def has_layer(self, layer_idx: int) -> bool:
        return layer_idx in self._stores or self.inner.has_layer(layer_idx)

    def _allocate(self, layer_idx: int, keys: torch.Tensor, values: torch.Tensor) -> _Store:
        if self.active_rows is not None:
            raise RuntimeError("a cache layer must be first written for the whole batch, not inside selecting()")
        if keys.shape[0] != self.batch_size:
            raise ConfigError(f"the cache holds {self.batch_size} rows but a layer wrote {keys.shape[0]}")
        batch, heads, _, dim = keys.shape
        dim_v = values.shape[3]
        bits, g, slots = self.config.bits, self.config.group_size, self.num_slots
        per_byte = 8 // bits
        for name, d in (("key", dim), ("value", dim_v)):
            if d % per_byte:
                raise ConfigError(
                    f"{name} head dim {d} does not pack into whole bytes at {bits} bits; use storage='simulated'"
                )
        group = value_group(dim_v, g)
        device = keys.device

        def codes(d: int) -> torch.Tensor:
            return torch.zeros(batch, heads, slots, d // per_byte, dtype=torch.uint8, device=device)

        def half(*shape: int) -> torch.Tensor:
            return torch.zeros(*shape, dtype=torch.float16, device=device)

        store = _Store(
            value_group=group,
            key_codes=codes(dim),
            key_scale=half(batch, heads, -(-slots // g), dim),
            key_zero=half(batch, heads, -(-slots // g), dim),
            value_codes=codes(dim_v),
            value_scale=half(batch, heads, slots, dim_v // group),
            value_zero=half(batch, heads, slots, dim_v // group),
            key_ring=torch.zeros(batch, heads, self.residual_ring + 1, dim, dtype=keys.dtype, device=device),
            value_ring=torch.zeros(batch, heads, self.residual_ring + 1, dim_v, dtype=values.dtype, device=device),
        )
        self._stores[layer_idx] = store
        return store

    # ---- counts and flags ------------------------------------------------------------------------------------
    def _device_counts(self, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        """``int32`` ``[B]`` quantized key and value prefixes on ``device``."""
        counts = self._counts.get(device)
        if counts is None:
            counts = self._counts[device] = (
                torch.tensor(self.keys_quantized, dtype=torch.int32, device=device),
                torch.tensor(self.values_quantized, dtype=torch.int32, device=device),
            )
        return counts

    def _flag(self, device: torch.device) -> torch.Tensor:
        flag = self._flags.get(device)
        if flag is None:
            flag = self._flags[device] = torch.zeros(2, dtype=torch.bool, device=device)
        return flag

    def _check_flags(self) -> None:
        if not self._flags:
            return
        ring, overflow = (bool(x) for x in torch.stack([f.cpu() for f in self._flags.values()]).any(0))
        for flag in self._flags.values():
            flag.zero_()
        if ring:
            raise ConfigError(
                f"a cycle wrote more than the packed KIVI ring holds ({self.residual_ring} slots = residual_length "
                "+ headroom) and overwrote entries not quantized yet: increase headroom or use storage='simulated'"
            )
        if overflow:
            raise ConfigError(
                "a KIVI scale or zero point exceeds the fp16 range of the packed storage; use storage='simulated'"
            )

    # ---- pass boundaries -------------------------------------------------------------------------------------
    def pass_begin(self, frontier: list[int | None]) -> None:
        self._check_flags()
        super().pass_begin(frontier)

    def pass_end(self) -> None:
        super().pass_end()
        self._prequantized.clear()
        self._check_flags()

    def _quantize_committed(self, committed: list[int | None]) -> None:
        super()._quantize_committed(committed)
        self._counts.clear()

    def _apply_jobs(self, jobs: dict[tuple[str, int, int], list[int]]) -> None:
        for layer_idx, store in self._stores.items():
            if layer_idx in self._prequantized:
                continue
            for (kind, start, stop), rows in jobs.items():
                ring = store.key_ring if kind == "k" else store.value_ring
                index = torch.tensor(rows, device=ring.device)
                block = ring.index_select(0, index)[:, :, self._phys(torch.arange(start, stop, device=ring.device))]
                self._pack_block(store, kind, index, start, block)

    def _pack_block(self, store: _Store, kind: str, rows: torch.Tensor, start: int, block: torch.Tensor) -> None:
        """Pack ``block`` ``[rows, H, T, D]`` (slots ``start:start+T`` of the ``rows``) into ``store``."""
        n, heads, tokens, dim = block.shape
        g, bits = self.config.group_size, self.config.bits
        stop = start + tokens
        if kind == "k":  # per channel, groups of g tokens (start and tokens are multiples of R, hence of g)
            codes, scale, low = quantize(block.reshape(n, heads, tokens // g, g, dim), bits, dim=3)
            store.key_codes[rows, :, start:stop] = pack_codes(codes.reshape(block.shape), bits)
            store.key_scale[rows, :, start // g : stop // g] = scale.squeeze(3).half()
            store.key_zero[rows, :, start // g : stop // g] = low.squeeze(3).half()
        else:  # per token, groups of channels
            group = store.value_group
            codes, scale, low = quantize(block.reshape(n, heads, tokens, dim // group, group), bits, dim=4)
            store.value_codes[rows, :, start:stop] = pack_codes(codes.reshape(block.shape), bits)
            store.value_scale[rows, :, start:stop] = scale.squeeze(4).half()
            store.value_zero[rows, :, start:stop] = low.squeeze(4).half()
        flag = self._flag(block.device)
        flag[1] |= (scale.abs() > _FP16_MAX).any() | (low.abs() > _FP16_MAX).any()

    # ---- K/V traffic -----------------------------------------------------------------------------------------
    def update(
        self, key_states: torch.Tensor, value_states: torch.Tensor, layer_idx: int, *args: object, **kwargs: object
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.covers(layer_idx):
            return super().update(key_states, value_states, layer_idx, *args, **kwargs)
        store = self._stores.get(layer_idx) or self._allocate(layer_idx, key_states, value_states)
        rows = self.active_rows
        slots = self.active_slots.to(store.key_ring.device).long()
        queries = slots.shape[-1]
        if self._prompt_ends is not None and rows is None and queries > self.headroom:
            return self._write_prompt(layer_idx, store, slots, key_states, value_states)
        self._write_ring(store, slots, key_states, value_states, rows)
        if self.reads_packed and rows is None and queries <= self.headroom:
            return self._packed_view(store)
        return self._dense(store, rows)

    def write(self, layer_idx: int, key_states: torch.Tensor, value_states: torch.Tensor) -> None:
        self.update(key_states, value_states, layer_idx)

    def read(self, layer_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        store = self._stores.get(layer_idx)
        if store is None:
            return super().read(layer_idx)
        return self._dense(store, self.active_rows)

    def _write_ring(
        self, store: _Store, slots: torch.Tensor, keys: torch.Tensor, values: torch.Tensor, rows: torch.Tensor | None
    ) -> None:
        device = store.key_ring.device
        _, vq = self._device_counts(device)
        if rows is not None:
            rows = rows.to(device)
            vq = vq.index_select(0, rows)
        grid = slots if slots.dim() == 2 else slots[None]
        real = grid != self.trash_slot
        flag = self._flag(device)
        flag[0] |= (real & (grid - vq[:, None].long() >= self.residual_ring)).any()
        phys = self._phys(slots)
        for ring, states in ((store.key_ring, keys), (store.value_ring, values)):
            if rows is None:
                write_slots(ring, phys, states)
            else:
                sub = ring.index_select(0, rows)
                write_slots(sub, phys, states)
                ring.index_copy_(0, rows, sub)

    def _write_prompt(
        self, layer_idx: int, store: _Store, slots: torch.Tensor, keys: torch.Tensor, values: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """A prompt pass longer than the headroom: write it into a dense copy, pack what the end of the prompt
        quantizes from that copy, keep the last ``ring`` entries of every row in the ring."""
        assert self._prompt_ends is not None
        device = store.key_ring.device
        dense_keys, dense_values = self._dense(store)
        write_slots(dense_keys, slots, keys)
        write_slots(dense_values, slots, values)
        for (kind, start, stop), rows in self._plan(self._prompt_ends).items():
            index = torch.tensor(rows, device=device)
            source = dense_keys if kind == "k" else dense_values
            self._pack_block(store, kind, index, start, source.index_select(0, index)[:, :, start:stop])
        self._prequantized.add(layer_idx)
        ends = torch.tensor([e or 0 for e in self._prompt_ends], device=device)
        grid = slots if slots.dim() == 2 else slots[None].expand(keys.shape[0], -1)
        keep = (grid != self.trash_slot) & (grid >= ends[:, None] - self.residual_ring)
        target = torch.where(keep, self._phys(grid), torch.full_like(grid, self.residual_ring))
        write_slots(store.key_ring, target, keys)
        write_slots(store.value_ring, target, values)
        return dense_keys, dense_values

    def _dense(self, store: _Store, rows: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """Fresh dense keys and values ``[B', H, slots, D]`` of a layer: the codes dequantized below the quantized
        prefixes, the ring above them."""
        device = store.key_ring.device
        kq, vq = self._device_counts(device)
        tensors = store.tensors()
        if rows is not None:
            rows = rows.to(device)
            kq, vq = kq.index_select(0, rows), vq.index_select(0, rows)
            tensors = [t.index_select(0, rows) for t in tensors]
        key_codes, key_scale, key_zero, value_codes, value_scale, value_zero, key_ring, value_ring = tensors
        bits, g = self.config.bits, self.config.group_size
        position = torch.arange(self.num_slots, device=device)
        phys = self._phys(position)
        keys = dequantize(key_codes, key_scale, key_zero, bits, "key", g, key_ring.dtype)
        values = dequantize(value_codes, value_scale, value_zero, bits, "value", store.value_group, value_ring.dtype)
        keys = torch.where((position < kq[:, None])[:, None, :, None], keys, key_ring[:, :, phys])
        values = torch.where((position < vq[:, None])[:, None, :, None], values, value_ring[:, :, phys])
        return keys, values

    def _packed_view(self, store: _Store) -> tuple[torch.Tensor, torch.Tensor]:
        """Zero-memory stand-ins; the key carries the packed layer for the attention kernel."""
        kq, vq = self._device_counts(store.key_ring.device)
        batch, heads, _, dim = store.key_ring.shape
        ring = store.key_ring
        packed = KIVIPacked(
            bits=self.config.bits,
            key_group=self.config.group_size,
            value_group=store.value_group,
            key_codes=store.key_codes,
            key_scale=store.key_scale,
            key_zero=store.key_zero,
            value_codes=store.value_codes,
            value_scale=store.value_scale,
            value_zero=store.value_zero,
            keys_quantized=kq,
            values_quantized=vq,
            key_ring=store.key_ring,
            value_ring=store.value_ring,
            ring=self.residual_ring,
            materialize=lambda: self._dense(store),
        )
        key = attach_packed(self.stand_in(batch, heads, self.num_slots, dim, ring.dtype, ring.device), packed)
        value_ring = store.value_ring
        value = self.stand_in(batch, heads, self.num_slots, value_ring.shape[3], value_ring.dtype, value_ring.device)
        return key, value

    # ---- layout changes --------------------------------------------------------------------------------------
    def on_gather(self, rows: torch.Tensor, src: torch.Tensor, dst: torch.Tensor) -> None:
        # Tree compaction moves uncommitted entries only, which all live in the rings.
        for store in self._stores.values():
            device = store.key_ring.device
            gather_slots(
                [store.key_ring, store.value_ring], rows.to(device), self._phys(src.to(device)),
                self._phys(dst.to(device)),
            )

    def on_reorder(self, rows: torch.Tensor) -> None:
        super().on_reorder(rows)
        for store in self._stores.values():
            store.reorder(rows)
        self._counts.clear()

    def on_restart(self) -> None:
        super().on_restart()
        self._stores.clear()
        self._counts.clear()
        self._flags.clear()
        self._prequantized.clear()
