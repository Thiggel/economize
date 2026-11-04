"""Cache wrappers: KV caches that forward to an inner cache and act at the boundaries of exact passes.

KV-cache techniques (:meth:`economize.core.Technique.wrap_cache`) wrap the cache of every generation. A wrapper sees
every write (``update``), every read, the slots of every pass (``writing_at``), tree compaction (``gather``) and
dropped rows (``reorder``), which is all a compressed or sparse cache needs.

**Frontier.** The decoding loop never rewrites a slot below the first slot of an exact pass: every slot of row
``b`` below that pass's smallest non-trash slot ``c_b`` holds the K/V of a committed token computed by an exact
pass (draft K/V are written at or above ``c_b`` and rewritten by the next exact pass). :class:`CacheWrapper` calls
:meth:`CacheWrapper.pass_begin` with these frontiers when the outermost ``writing_at`` block of a pass opens,
except during drafting (phase ``"draft"``), so that subclasses compress or evict only committed entries. The
frontier of a row never decreases during a generation. The tokens of an exact pass that were accepted are the ones
below the next exact pass's frontier: a verifying pass also computes draft tokens that are then rejected.
:meth:`CacheWrapper.pass_scope` is a context around the pass's forward, and :meth:`CacheWrapper.pass_end` closes it.

**CUDA graphs.** A replayed pass cannot run host code, so the decoding loop announces a graphed exact pass with
:meth:`CacheWrapper.external_pass`, which runs the boundaries outside the graph, and ``writing_at`` inside it skips
them. A wrapper of a ``graph_safe`` technique hands layers tensors at fixed addresses only, reports what recorded
kernels depend on beyond those addresses in :meth:`CacheWrapper.graph_state`, and lives as long as the static cache it
wraps: :meth:`CacheWrapper.restart` begins the next generation on it.

:class:`SlotStore` keeps per-slot side tensors (quantization scales, keep masks, attention scores) in step with
the cache: written at the active slots, moved by ``gather`` and dropped with rows by ``reorder``.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from contextlib import AbstractContextManager, contextmanager, nullcontext
from dataclasses import dataclass, field
from typing import Any, TypeVar

import torch

from ...errors import IncompatibleTechniquesError
from ...modules.kv_cache import write_slots  # re-exported: KV techniques import it from the package
from ...modules.slot_gather import gather_slots
from ...runtime import KVCache, StaticKVCache, current_phase

#: ``(layer_idx, key buffer, committed [B]) -> (stored bytes [B], uncompressed bytes [B])`` of one covered layer.
LayerBytes = Callable[[int, torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]]

__all__ = [
    "CacheWrapper", "KVStats", "LayerBytes", "RowStats", "SlotStore", "find_wrapper", "frontiers", "innermost",
    "write_slots",
]

W = TypeVar("W", bound="CacheWrapper")


@dataclass
class RowStats:
    """Memory accounting of one row of a generation, in bytes.

    Attributes:
        kv_bytes: Size of the row's committed cache entries (all layers) at its last exact pass.
        kv_bytes_full: The same entries in the cache's own dtype, without compression or eviction.
        kv_bytes_read: Sum over the row's exact passes of the bytes of committed entries attention read.
        kv_bytes_read_full: The same sum without compression or eviction.
        passes: Exact passes counted.
    """

    kv_bytes: float = 0.0
    kv_bytes_full: float = 0.0
    kv_bytes_read: float = 0.0
    kv_bytes_read_full: float = 0.0
    passes: int = 0

    def as_dict(self) -> dict[str, float]:
        """The fields plus ``kv_compression`` (full bytes over stored bytes of the last pass)."""
        return {
            "kv_bytes": self.kv_bytes,
            "kv_bytes_full": self.kv_bytes_full,
            "kv_bytes_read": self.kv_bytes_read,
            "kv_bytes_read_full": self.kv_bytes_read_full,
            "kv_passes": float(self.passes),
            "kv_compression": self.kv_bytes_full / self.kv_bytes if self.kv_bytes > 0 else 1.0,
        }


@dataclass
class KVStats:
    """Memory accounting of one generation, per original batch row (the row order of the decode call)."""

    rows: dict[int, RowStats] = field(default_factory=dict)

    def row(self, index: int) -> RowStats:
        """Statistics of original row ``index`` (zeros if the row made no exact pass)."""
        return self.rows.get(index, RowStats())

    def record(self, index: int, stored: float, full: float) -> None:
        """Account one exact pass of row ``index`` whose committed prefix takes ``stored`` (``full``) bytes."""
        stats = self.rows.setdefault(index, RowStats())
        stats.kv_bytes, stats.kv_bytes_full = stored, full
        stats.kv_bytes_read += stored
        stats.kv_bytes_read_full += full
        stats.passes += 1


class CacheWrapper(KVCache):
    """A :class:`~economize.runtime.KVCache` that forwards to ``inner`` and adds behaviour around it.

    The wrapper owns no K/V of its own: shape, slots, row selection and buffers live in the inner cache, which may
    itself be a wrapper. Subclasses override :meth:`transform_update`, :meth:`transform_read`,
    :meth:`pass_begin`, :meth:`pass_end`, :meth:`on_gather` and :meth:`on_reorder`.

    Recurrent (hybrid) caches are rejected: their states cannot be compressed or evicted per slot.

    Over the window cache of :attr:`DecodeConfig.window_cache <economize.decoding.DecodeConfig.window_cache>` the
    sliding layers keep only a ring of slightly more than their window (:class:`~economize.runtime.RingSpec`), whose
    slots are reused as the sequence grows. Their cache is already bounded by the window, so a wrapper never covers
    them: it acts on the full-length layers only.

    Attributes:
        layers: The cache layers the wrapper acts on (``None``: all of them; see :meth:`covers`).
    """

    layers: frozenset[int] | None = None

    def __init__(self, inner: KVCache) -> None:  # no super().__init__: every field lives in the inner cache
        if innermost(inner).keeps_recurrent_state:
            raise IncompatibleTechniquesError(
                f"{type(self).__name__} needs a positional KV cache; models with recurrent layers keep a hybrid cache"
            )
        self.inner = inner
        self.stats = KVStats()
        self._depth = 0
        self._in_pass = False
        self._external = False  # inside external_pass: writing_at leaves the boundaries to it
        self._selected: torch.Tensor | None = None
        self._ids = list(range(inner.batch_size))  # original row index of every current row
        ring = getattr(innermost(inner), "ring", None)
        self._ring_layers: frozenset[int] = frozenset() if ring is None else ring.layers

    @property
    def wraps_static_cache(self) -> bool:
        """Whether the innermost cache keeps its buffers at fixed addresses (the cache of CUDA graphs)."""
        return isinstance(innermost(self.inner), StaticKVCache)

    def restart(self) -> None:
        """Begin a new generation on the same inner cache, which the caller has already activated for the new batch
        size: fresh statistics and row ids, then :meth:`on_restart`. Only wrappers of static caches are reused, and each
        technique restarts its own wrapper (inner wrappers are not restarted here)."""
        if self._depth or self._external:
            raise RuntimeError("a cache wrapper cannot restart inside a pass")
        self.stats = KVStats()
        self._in_pass = False
        self._selected = None
        self._ids = list(range(self.inner.batch_size))
        self.on_restart()

    def on_restart(self) -> None:
        """Reset per-generation side state for :meth:`restart` (the new batch size is :attr:`batch_size`)."""

    # ---- delegated state ---------------------------------------------------------------------------------------
    @property
    def spec(self) -> Any:  # type: ignore[override]
        return self.inner.spec

    @property
    def capacity(self) -> int:  # type: ignore[override]
        return self.inner.capacity

    @property
    def batch_size(self) -> int:  # type: ignore[override]
        return self.inner.batch_size

    @property
    def trash_slot(self) -> int:
        return self.inner.trash_slot

    @property
    def num_slots(self) -> int:
        return self.inner.num_slots

    @property
    def keys(self) -> dict[int, torch.Tensor]:
        """The wrapped cache's per-layer key buffers (the stored, possibly compressed K, as layers read them)."""
        return self.inner.keys  # type: ignore[attr-defined]

    @keys.setter
    def keys(self, buffers: dict[int, torch.Tensor]) -> None:
        self.inner.keys = buffers  # type: ignore[attr-defined]

    @property
    def values(self) -> dict[int, torch.Tensor]:
        """The wrapped cache's per-layer value buffers."""
        return self.inner.values  # type: ignore[attr-defined]

    @values.setter
    def values(self, buffers: dict[int, torch.Tensor]) -> None:
        self.inner.values = buffers  # type: ignore[attr-defined]

    @property
    def get_seq_length(self) -> Callable[..., int]:
        """The wrapped cache's ``get_seq_length`` (Hugging Face's cache API). Like an attribute lookup on the wrapped
        cache, it raises :class:`AttributeError` when that cache has none, so ``hasattr`` answers for the wrapped
        cache."""
        return self.inner.get_seq_length  # type: ignore[attr-defined]

    def key_positions(self, device: torch.device) -> torch.Tensor:
        return self.inner.key_positions(device)

    @property
    def active_batch_size(self) -> int:
        return self.inner.active_batch_size

    @property
    def active_slots(self) -> torch.Tensor:
        return self.inner.active_slots

    @property
    def row_ids(self) -> list[int]:
        """Original batch index of every current row (rows are dropped as they finish)."""
        return list(self._ids)

    @property
    def active_rows(self) -> torch.Tensor | None:
        """The rows selected with :meth:`selecting`, or ``None`` for the whole batch."""
        return self._selected

    # ---- pass boundaries ---------------------------------------------------------------------------------------
    @contextmanager
    def writing_at(self, slots: torch.Tensor) -> Iterator[KVCache]:
        with self.inner.writing_at(slots):
            outer = self._depth == 0 and not self._external
            self._depth += 1
            try:
                if outer:
                    self._in_pass = self._selected is None and current_phase() != "draft"
                    if self._in_pass:
                        self.pass_begin(frontiers(slots, self.trash_slot, self.inner.active_batch_size))
                with self.pass_scope() if outer and self._in_pass else nullcontext():
                    yield self
            finally:
                self._depth -= 1
                if outer and self._in_pass:
                    self._in_pass = False
                    self.pass_end()

    @contextmanager
    def external_pass(self, frontier: list[int | None]) -> Iterator[None]:
        """The boundaries of an exact pass that runs inside the block as a CUDA graph replay: :meth:`pass_begin`
        with ``frontier``, :meth:`pass_scope` around the block and :meth:`pass_end`, inner wrappers first as in an
        eager pass. ``writing_at`` inside the block leaves the boundaries alone."""
        if self._depth or self._external:
            raise RuntimeError("external_pass cannot be nested in another pass")
        with self.inner.external_pass(frontier):
            self._external = self._in_pass = True
            try:
                self.pass_begin(frontier)
                with self.pass_scope():
                    yield
            finally:
                self._external = self._in_pass = False
                self.pass_end()

    def graph_state(self) -> tuple[Any, ...]:
        """The inner cache's state followed by :meth:`own_graph_state`."""
        return (*self.inner.graph_state(), self.own_graph_state())

    def own_graph_state(self) -> Any:
        """What the tensors this wrapper hands to layers depend on beyond fixed addresses (default: nothing). It may
        change only in :meth:`pass_begin` and :meth:`restart`, so a graph recorded under another state is dropped."""
        return None

    @contextmanager
    def selecting(self, rows: torch.Tensor) -> Iterator[KVCache]:
        with self.inner.selecting(rows):
            previous, self._selected = self._selected, rows
            try:
                yield self
            finally:
                self._selected = previous

    @property
    def in_exact_pass(self) -> bool:
        """Whether an exact pass (prefill, decode or refresh, not a draft step) is writing."""
        return self._in_pass

    def pass_begin(self, frontier: list[int | None]) -> None:
        """An exact pass starts; ``frontier[b]`` is row ``b``'s first slot written by the pass (``None`` when the
        row only writes padding). Every slot below it holds committed exact K/V."""

    def pass_scope(self) -> AbstractContextManager[Any]:
        """A context entered after :meth:`pass_begin` and exited before :meth:`pass_end`, around the exact pass's
        forward (default: none). It sees an exception raised by the pass."""
        return nullcontext()

    def pass_end(self) -> None:
        """The exact pass that :meth:`pass_begin` announced has written every layer."""

    # ---- K/V traffic -------------------------------------------------------------------------------------------
    def update(
        self, key_states: torch.Tensor, value_states: torch.Tensor, layer_idx: int, *args: Any, **kwargs: Any
    ) -> tuple[torch.Tensor, torch.Tensor]:
        key_states, value_states = self.transform_update(key_states, value_states, layer_idx)
        keys, values = self.inner.update(key_states, value_states, layer_idx, *args, **kwargs)
        return self.transform_read(keys, values, layer_idx)

    def read(self, layer_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        keys, values = self.inner.read(layer_idx)
        return self.transform_read(keys, values, layer_idx)

    def write(self, layer_idx: int, key_states: torch.Tensor, value_states: torch.Tensor) -> None:
        key_states, value_states = self.transform_update(key_states, value_states, layer_idx)
        self.inner.write(layer_idx, key_states, value_states)

    def has_layer(self, layer_idx: int) -> bool:
        return self.inner.has_layer(layer_idx)

    def transform_update(
        self, key_states: torch.Tensor, value_states: torch.Tensor, layer_idx: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """What is written to the inner cache for new K/V (default: unchanged)."""
        return key_states, value_states

    def transform_read(
        self, keys: torch.Tensor, values: torch.Tensor, layer_idx: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """What attention sees for the inner cache's buffers (default: unchanged)."""
        return keys, values

    # ---- layout changes ----------------------------------------------------------------------------------------
    def gather(self, rows: torch.Tensor, src: torch.Tensor, dst: torch.Tensor) -> None:
        self.inner.gather(rows, src, dst)
        self.on_gather(rows, src, dst)

    def reorder(self, rows: torch.Tensor) -> None:
        self.inner.reorder(rows)
        self._ids = [self._ids[r] for r in rows.tolist()]
        self.on_reorder(rows)

    def on_gather(self, rows: torch.Tensor, src: torch.Tensor, dst: torch.Tensor) -> None:
        """Move side state with the cache's tree compaction (see :meth:`KVCache.gather`)."""

    def on_reorder(self, rows: torch.Tensor) -> None:
        """Keep the side state of ``rows`` in this order."""

    # ---- memory accounting -------------------------------------------------------------------------------------
    def covers(self, layer_idx: int) -> bool:
        """Whether the wrapper acts on cache layer ``layer_idx``: one of :attr:`layers` (every layer when it is
        ``None``) that is not a ring layer of the window cache."""
        return (self.layers is None or layer_idx in self.layers) and layer_idx not in self._ring_layers

    def positional_buffers(self) -> dict[int, torch.Tensor]:
        """The innermost cache's per-layer key buffers ``[B, H_kv, slots, D]``, in layer order."""
        base = innermost(self.inner)
        keys = getattr(base, "keys", None)
        if not isinstance(keys, dict):
            name = type(self).__name__
            raise IncompatibleTechniquesError(f"{name} needs a positional cache, got {type(base).__name__}")
        return dict(sorted(keys.items()))

    @staticmethod
    def dense_bytes(keys: torch.Tensor, committed: torch.Tensor, dtype: torch.dtype | None = None) -> torch.Tensor:
        """Bytes of ``committed`` (``[B]``) uncompressed K and V entries per row of a layer with key buffer ``keys``,
        stored in ``dtype`` (default: the buffer's)."""
        element = keys.element_size() if dtype is None else torch.empty((), dtype=dtype).element_size()
        return 2.0 * committed * keys.shape[1] * keys.shape[3] * element

    def account(self, frontier: list[int | None], covered_bytes: LayerBytes) -> None:
        """Account one exact pass whose committed prefix of row ``b`` ends at ``frontier[b]``.

        Layers the wrapper does not :meth:`cover <covers>` count as dense on both sides. For a covered layer,
        ``covered_bytes(layer_idx, keys, committed)`` returns the stored and the uncompressed bytes per row
        (``[B]`` float64 tensors; ``committed`` is the frontier as a float64 tensor, 0 for padding rows).
        """
        committed = torch.tensor([c or 0 for c in frontier], dtype=torch.float64)
        stored = torch.zeros_like(committed)
        full = torch.zeros_like(committed)
        for layer_idx, keys in self.positional_buffers().items():
            if self.covers(layer_idx):
                s, f = covered_bytes(layer_idx, keys, committed)
            else:
                s = f = self.dense_bytes(keys, committed)
            stored += s
            full += f
        padding = [c is None for c in frontier]  # rows that only write padding are not accounted
        self.record_bytes(
            [None if pad else v for pad, v in zip(padding, stored.tolist())],
            [None if pad else v for pad, v in zip(padding, full.tolist())],
        )

    def record_bytes(self, stored: Sequence[float | None], full: Sequence[float | None]) -> None:
        """Account one exact pass: ``stored[b]`` and ``full[b]`` bytes of committed entries of current row ``b``.

        ``None`` marks a row that only writes padding in this pass (its frontier is ``None``): the row is not
        accounted, so its last byte counts and its pass count stay as they were.
        """
        for b, (s, f) in enumerate(zip(stored, full)):
            if s is None or f is None:
                continue
            self.stats.record(self._ids[b], s, f)


class SlotStore:
    """Per-layer side tensors ``[B, H, num_slots, ...]`` that follow the cache's slots and rows."""

    def __init__(self) -> None:
        self.tensors: dict[int, torch.Tensor] = {}

    def __contains__(self, layer: int) -> bool:
        return layer in self.tensors

    def __getitem__(self, layer: int) -> torch.Tensor:
        return self.tensors[layer]

    def ensure(self, layer: int, shape: tuple[int, ...], *, dtype: torch.dtype, device: torch.device,
               fill: float | bool = 0) -> torch.Tensor:
        """The tensor of ``layer``, allocated with ``fill`` on first use."""
        tensor = self.tensors.get(layer)
        if tensor is None:
            tensor = self.tensors[layer] = torch.full(shape, fill, dtype=dtype, device=device)
        return tensor

    def write(self, layer: int, slots: torch.Tensor, values: torch.Tensor, rows: torch.Tensor | None = None) -> None:
        """Write ``values`` ``[B', H, S, ...]`` at ``slots`` (``[S]`` or ``[B', S]``) of the ``rows`` (all rows if
        ``None``)."""
        tensor = self.tensors[layer]
        if rows is None:
            write_slots(tensor, slots, values)
            return
        rows = rows.to(tensor.device)
        sub = tensor.index_select(0, rows)
        write_slots(sub, slots, values)
        tensor.index_copy_(0, rows, sub)

    def view(self, layer: int, rows: torch.Tensor | None = None) -> torch.Tensor:
        """The tensor of ``layer`` (a copy of the ``rows`` if given)."""
        tensor = self.tensors[layer]
        return tensor if rows is None else tensor.index_select(0, rows.to(tensor.device))

    def gather(self, rows: torch.Tensor, src: torch.Tensor, dst: torch.Tensor) -> None:
        gather_slots(list(self.tensors.values()), rows, src, dst)

    def reorder(self, rows: torch.Tensor) -> None:
        for layer, tensor in self.tensors.items():
            self.tensors[layer] = tensor.index_select(0, rows.to(tensor.device))


def frontiers(slots: torch.Tensor, trash: int, batch_size: int) -> list[int | None]:
    """Smallest non-trash slot of every row (``None`` for rows that only write padding)."""
    if slots.dim() == 1:
        real = slots[slots != trash]
        value = int(real.min()) if real.numel() else None
        return [value] * batch_size
    smallest = slots.amin(dim=1).tolist()  # the trash slot is above every real slot
    return [None if s >= trash else int(s) for s in smallest]


def innermost(cache: KVCache) -> KVCache:
    """The cache at the bottom of a chain of wrappers."""
    while isinstance(cache, CacheWrapper):
        cache = cache.inner
    return cache


def find_wrapper(cache: object, cls: type[W], owner: object | None = None) -> W | None:
    """The first wrapper of type ``cls`` (and ``owner``, when given) in the chain starting at ``cache``.

    Returns ``None`` for any other cache, for example the stand-in cache that captures the K/V of a skipped layer.
    """
    while isinstance(cache, CacheWrapper):
        if isinstance(cache, cls) and (owner is None or getattr(cache, "owner", None) is owner):
            return cache
        cache = cache.inner
    return None
