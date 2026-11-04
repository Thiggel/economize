"""A key/value cache for the attention layers that a drafter adds to a model, and the slot layout it shares with
the model's own caches (:mod:`economize.runtime.cache`)."""

from __future__ import annotations

import torch

from ..validation import check_positive

__all__ = ["DraftKVCache", "SlotLayout", "write_slots"]


class SlotLayout:
    """Buffers of ``capacity + 1`` slots per row: slot ``p`` holds position ``p`` and the last one, the trash slot,
    absorbs the writes of padding and of discarded tokens. Subclasses set :attr:`capacity`."""

    capacity: int

    @property
    def trash_slot(self) -> int:
        """Index of the slot that absorbs discarded writes (``capacity``); no attention ever sees it."""
        return self.capacity

    @property
    def num_slots(self) -> int:
        """Slots per row including the trash slot (``capacity + 1``)."""
        return self.capacity + 1


def write_slots(buffer: torch.Tensor, slots: torch.Tensor, states: torch.Tensor) -> None:
    """Write ``states`` ``[B, heads, S, ...]`` into ``buffer`` ``[B, heads, num_slots, ...]`` at ``slots`` along
    dim 2, cast to the buffer's dtype. ``slots`` is ``[S]`` (the same slots in every row) or ``[B, S]``; entries sent
    to the same slot, other than the trash slot, must not collide. The trailing dimensions are free: K/V buffers
    have one (``head_dim``), per-slot side tensors (quantization scales, keep masks) may have none or several."""
    states = states.to(buffer.dtype)
    slots = slots.to(buffer.device)
    if slots.dim() == 1:
        buffer.index_copy_(2, slots, states)
        return
    shape = (slots.shape[0], 1, slots.shape[1]) + (1,) * (states.dim() - 3)
    buffer.scatter_(2, slots.reshape(shape).expand_as(states), states)


class DraftKVCache(SlotLayout):
    """Keys and values of one or more draft attention layers, laid out like the model's cache.

    Every layer holds two buffers ``[B, heads, capacity + 1, head_dim]``. Slot ``p`` holds position ``p``, and the
    last slot (:attr:`trash_slot`) absorbs the writes of padding, of uncommitted tokens and of stopped rows. Reads
    (:meth:`read`) stop before the trash slot, so whatever lands there never reaches an attention: a NaN or an
    unbounded value in the trash slot would otherwise poison the attention output even under a zero weight.

    Args:
        num_layers: Number of draft attention layers.
        batch_size: Rows of the batch.
        num_heads: Key/value heads per layer.
        head_dim: Size of every head.
        capacity: Number of usable positions per row.
        device: Device of the buffers.
        dtype: Storage dtype of the buffers. Writes are cast to it.
    """

    def __init__(
        self,
        num_layers: int,
        batch_size: int,
        num_heads: int,
        head_dim: int,
        capacity: int,
        *,
        device: torch.device | str,
        dtype: torch.dtype,
    ) -> None:
        sizes = dict(num_layers=num_layers, batch_size=batch_size, num_heads=num_heads, head_dim=head_dim,
                     capacity=capacity)
        for name, size in sizes.items():
            check_positive(name, size)
        self.capacity = capacity
        shape = (batch_size, num_heads, capacity + 1, head_dim)
        self.keys = [torch.zeros(shape, device=device, dtype=dtype) for _ in range(num_layers)]
        self.values = [torch.zeros(shape, device=device, dtype=dtype) for _ in range(num_layers)]

    # ---- layout -------------------------------------------------------------------------------------------------
    @property
    def num_layers(self) -> int:
        return len(self.keys)

    @property
    def batch_size(self) -> int:
        return self.keys[0].shape[0]

    @property
    def device(self) -> torch.device:
        return self.keys[0].device

    @property
    def dtype(self) -> torch.dtype:
        return self.keys[0].dtype

    def slots(self, positions: torch.Tensor, keep: torch.Tensor | None = None) -> torch.Tensor:
        """The slot of every position: the position itself where ``keep`` holds, else the trash slot."""
        positions = positions.to(self.device)
        return positions if keep is None else torch.where(keep.to(self.device), positions, self.trash_slot)

    # ---- writes -------------------------------------------------------------------------------------------------
    def scatter(self, layer: int, keys: torch.Tensor, values: torch.Tensor, slots: torch.Tensor) -> None:
        """Write the entries of several tokens per row: ``keys``/``values`` ``[B, heads, S, head_dim]`` go to the
        slots ``[B, S]``. Entries sent to the same slot, other than the trash slot, must not collide."""
        write_slots(self.keys[layer], slots, keys)
        write_slots(self.values[layer], slots, values)

    def write(self, layer: int, keys: torch.Tensor, values: torch.Tensor, slots: torch.Tensor) -> None:
        """Write one entry per row: ``keys``/``values`` ``[B, heads, head_dim]`` go to the slots ``[B]``."""
        rows = torch.arange(slots.shape[0], device=self.device)
        slots = slots.to(self.device)
        self.keys[layer][rows, :, slots] = keys.to(self.dtype)
        self.values[layer][rows, :, slots] = values.to(self.dtype)

    # ---- reads --------------------------------------------------------------------------------------------------
    def read(self, layer: int, limit: int | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """Keys and values of the first ``limit`` slots (default: all usable slots), ``[B, heads, limit, dim]``.

        The trash slot is never part of the result.
        """
        limit = self.capacity if limit is None else min(limit, self.capacity)
        return self.keys[layer][:, :, :limit], self.values[layer][:, :, :limit]

    def visible(self, positions: torch.Tensor, limit: int | None = None, *, inclusive: bool = True) -> torch.Tensor:
        """Causal mask ``[B, Q, limit]`` of the slots that queries at ``positions`` ``[B, Q]`` may attend to.

        A query sees the slots before its position, and its own slot when ``inclusive``. Positions at or past
        ``limit`` (stopped rows, padding) are clamped to the last slot read, and slot ``0`` stays visible to every
        query, so no attention row is empty (an empty row would produce NaN).
        """
        limit = self.capacity if limit is None else min(limit, self.capacity)
        query = positions.to(self.device).clamp(min=0, max=limit - 1)[..., None]
        slots = torch.arange(limit, device=self.device)
        seen = slots <= query if inclusive else slots < query
        return seen | (slots == 0)

    # ---- batch --------------------------------------------------------------------------------------------------
    def reorder(self, rows: torch.Tensor) -> None:
        """Keep and reorder the batch rows ``rows`` (as :meth:`~economize.decoding.Drafter.reorder`)."""
        rows = rows.to(self.device)
        self.keys = [k.index_select(0, rows) for k in self.keys]
        self.values = [v.index_select(0, rows) for v in self.values]
