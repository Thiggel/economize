"""The ``quest`` technique: every decoding query reads only the cache pages whose keys can score high (Quest).

Quest (Tang et al., ICML 2024) keeps the channel-wise smallest and largest key of every page of ``page_size``
committed slots. A decoding step bounds the query's score against every page from these two vectors and reads the
``token_budget / page_size - 1`` pages with the largest bounds, plus the page of the running token. The cache keeps
every entry: Quest saves the time attention spends reading the cache, never its memory.

:class:`QuestCache` maintains the bounds of every covered layer (recomputed at pass boundaries for the pages that
became committed) and attaches a :class:`~economize.attention.QuestPages` to the keys of every step whose query has
at most ``max_queries`` tokens. The ``"economize"`` attention kernel reads the attachment and loads only the selected
pages and the dense tail (:func:`~economize.attention.quest_decode_attention`); the fallbacks of that implementation
hide the other pages in the mask, so CPU runs and prefills compute the same attention. Prompts and longer queries
(verification passes of speculative decoding) attend densely, as in the reference, which keeps prefill dense.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar

import torch

from ...attention.packed import attach_packed, packed_of
from ...attention.quest import QuestPages, bounds_dtype, page_bounds
from ...attention.registry import ATTENTION_KERNELS, accepts_packed_kv
from ...core.config import TechniqueConfig
from ...core.metadata import Family, Saving, TechniqueInfo
from ...core.technique import Technique, register_technique
from ...errors import ConfigError, IncompatibleTechniquesError
from ...runtime import KVCache
from ...validation import check_layers
from .._kv import CacheWrapper, KVCacheTechnique, RowStats, check_positional, layer_indices

if TYPE_CHECKING:
    from ...arch import ArchAdapter

__all__ = ["QuestCache", "QuestConfig", "QuestTechnique", "quest_layers"]

FORMAT = "quest"


@dataclass(frozen=True)
class QuestConfig(TechniqueConfig):
    """Quest: query-aware selection of cache pages at decode time.

    Attributes:
        page_size: Slots per page (a power of two, at least 16; 16 as in the reference).
        token_budget: Cache entries every decoding query reads, the running page included: it reads
            ``token_budget // page_size - 1`` committed pages and the page of its own token. The paper evaluates
            budgets from 256 to 4096 on 32k-token contexts.
        dense_layers: The first cache layers that attend densely (2 as in the reference, whose first two layers
            have no page structure).
        max_queries: Longest query (tokens per row) that is restricted to the pages; longer ones, prompts among
            them, attend densely. 1 as in the reference, which selects for single decoding queries.
        layers: 1-indexed layers Quest acts on (empty: every layer); ``dense_layers``, sliding-window and
            KV-sharing layers are left out in any case.
        include_sliding: Also restrict sliding-window and chunked layers (the reference has none; their window is
            usually shorter than the budget).
    """

    technique: ClassVar[str] = "quest"
    page_size: int = 16
    token_budget: int = 2048
    dense_layers: int = 2
    max_queries: int = 1
    layers: tuple[int, ...] = ()
    include_sliding: bool = False

    def __post_init__(self) -> None:
        if self.page_size < 16 or self.page_size & (self.page_size - 1):
            raise ConfigError(f"quest page_size must be a power of two of at least 16, got {self.page_size}")
        if self.token_budget < 2 * self.page_size:
            raise ConfigError(
                f"quest token_budget {self.token_budget} must hold at least two pages of {self.page_size} slots"
            )
        if self.dense_layers < 0:
            raise ConfigError(f"quest dense_layers must be non-negative, got {self.dense_layers}")
        if self.max_queries < 1:
            raise ConfigError(f"quest max_queries must be positive, got {self.max_queries}")

    @property
    def pages(self) -> int:
        """Committed pages a decoding query reads (the reference's page budget minus the running page)."""
        return self.token_budget // self.page_size - 1

    def validate(self, adapter: ArchAdapter) -> None:
        check_layers(self.layers, adapter)
        check_positional(adapter, self.technique)
        implementation = str(getattr(adapter.config, "_attn_implementation", None) or "eager")
        if not accepts_packed_kv(implementation, FORMAT):
            readers = [name for name in ATTENTION_KERNELS.names() if accepts_packed_kv(name, FORMAT)]
            raise IncompatibleTechniquesError(
                f"quest needs an attention implementation that reads its pages ({', '.join(readers) or 'none'}; "
                f"after economize.attention.install()), got {implementation!r}"
            )
        if not quest_layers(self, adapter):
            raise ConfigError("quest covers no layer (dense_layers, layers and sliding-window layers leave none)")


def quest_layers(config: QuestConfig, adapter: ArchAdapter) -> frozenset[int]:
    """The cache layers (0-indexed) Quest restricts: in ``layers``, at or above ``dense_layers``, writing their own
    cache, and full attention unless ``include_sliding``."""
    chosen = layer_indices(config.layers, adapter)
    covered = set()
    for index in range(config.dense_layers + 1, adapter.num_layers + 1):
        if chosen is not None and index - 1 not in chosen:
            continue
        if adapter.kv_source(index) != index:
            continue
        if not config.include_sliding and adapter.attention_spec(index).kind != "full":
            continue
        covered.add(index - 1)
    return frozenset(covered)


class QuestCache(CacheWrapper):
    """A cache wrapper that attaches Quest's page bounds to the keys of every decoding step.

    ``done[b]`` is row ``b``'s number of committed pages, the full pages below its frontier at the last exact pass.
    The bounds of a layer are computed from the key buffer the first time a step reads the layer after a pass
    boundary moved ``done``: pages below the smallest ``done`` of the batch stay valid for good (committed slots are
    never rewritten), the pages above it are recomputed. Steps of a :meth:`selecting` pass (a subset of the rows)
    get no attachment and attend densely.

    In an exact pass every attachment must be read by an attention function that knows the format; a layer whose
    attachment was dropped (an attention implementation other than ``"economize"``, or a technique that copies the
    keys after this wrapper) raises :class:`~economize.IncompatibleTechniquesError` when the pass ends, so Quest
    never silently decodes densely.

    Attributes:
        config: The technique's configuration.
        layers: The cache layers Quest restricts (0-indexed).
    """

    def __init__(self, inner: KVCache, config: QuestConfig, layers: frozenset[int]) -> None:
        super().__init__(inner)
        self.config = config
        self.layers = layers
        self.on_restart()

    def on_restart(self) -> None:
        self._done = [0] * self.batch_size
        self._done_device: torch.Tensor | None = None
        self._bounds: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        self._valid: dict[int, int] = {}  # pages of every row whose bounds are final
        self._last: tuple[int, int] | None = None  # (layer, query width) of the last update
        self._frontier: list[int | None] = []
        self._attached: set[int] = set()
        self._used: set[int] = set()
        self._key_bytes: dict[int, tuple[int, int]] = {}  # layer -> (key element size, bound element size)

    @property
    def done(self) -> list[int]:
        """Committed pages of every current row."""
        return list(self._done)

    def bounds(self, layer_idx: int) -> tuple[torch.Tensor, torch.Tensor] | None:
        """``(kmin, kmax)`` ``[B, H_kv, P, D]`` of cache layer ``layer_idx`` (``None`` before its first sparse
        step); pages below each row's ``done`` are valid."""
        return self._bounds.get(layer_idx)

    # ---- pass boundaries ---------------------------------------------------------------------------------------
    def pass_begin(self, frontier: list[int | None]) -> None:
        page = self.config.page_size
        self._done = [d if f is None else max(d, f // page) for d, f in zip(self._done, frontier)]
        self._done_device = None
        self._frontier = list(frontier)
        self._attached.clear()
        self._used.clear()

    @contextmanager
    def pass_scope(self) -> Iterator[None]:
        yield  # an exception of the pass propagates before the check
        unread = sorted(self._attached - self._used)
        if unread:
            raise IncompatibleTechniquesError(
                f"quest attached pages to cache layers {unread} that no attention function read: run the model with "
                "attn_implementation='economize' (economize.attention.install()) and apply no KV technique that "
                "copies the keys after quest"
            )

    def pass_end(self) -> None:
        self._account()

    # ---- K/V traffic -------------------------------------------------------------------------------------------
    def transform_update(
        self, key_states: torch.Tensor, value_states: torch.Tensor, layer_idx: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self._last = (layer_idx, key_states.shape[2])
        return key_states, value_states

    def transform_read(
        self, keys: torch.Tensor, values: torch.Tensor, layer_idx: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        last, self._last = self._last, None
        if (
            last is None
            or last[0] != layer_idx
            or last[1] > self.config.max_queries
            or not self.covers(layer_idx)
            or self.active_rows is not None
            or keys.shape[0] != len(self._done)
            or packed_of(keys) is not None
            or max(self._done, default=0) <= self.config.pages
        ):
            return keys, values
        pages = QuestPages(
            *self._layer_bounds(layer_idx, keys),
            done=self._done_on(keys.device),
            min_done=min(self._done),
            max_done=max(self._done),
            page=self.config.page_size,
            pages=self.config.pages,
            max_queries=self.config.max_queries,
            on_use=self._used.add,
            layer=layer_idx,
        )
        if self.in_exact_pass:
            self._attached.add(layer_idx)
        return attach_packed(keys, pages), values

    def _done_on(self, device: torch.device) -> torch.Tensor:
        if self._done_device is None or self._done_device.device != device:
            self._done_device = torch.tensor(self._done, dtype=torch.int32, device=device)
        return self._done_device

    def _layer_bounds(self, layer_idx: int, keys: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """The layer's bounds with every committed page up to date."""
        need = max(self._done)
        first = self._valid.get(layer_idx, 0)
        bounds = self._bounds.get(layer_idx)
        if bounds is None or bounds[0].shape[2] < need:
            capacity = max(need, 2 * bounds[0].shape[2] if bounds is not None else need)
            shape = (keys.shape[0], keys.shape[1], capacity, keys.shape[3])
            grown = tuple(torch.zeros(shape, dtype=bounds_dtype(keys.dtype), device=keys.device) for _ in range(2))
            if bounds is not None:
                for old, new in zip(bounds, grown):
                    new[:, :, : old.shape[2]] = old
            bounds = self._bounds[layer_idx] = (grown[0], grown[1])
            element = torch.empty((), dtype=bounds[0].dtype).element_size()
            self._key_bytes[layer_idx] = (keys.element_size(), element)
        if first < need:
            low, high = page_bounds(keys, first, need, self.config.page_size)
            bounds[0][:, :, first:need] = low
            bounds[1][:, :, first:need] = high
        self._valid[layer_idx] = min(self._done)
        return bounds

    # ---- layout changes ----------------------------------------------------------------------------------------
    def on_reorder(self, rows: torch.Tensor) -> None:
        index = rows.tolist()
        self._done = [self._done[r] for r in index]
        self._done_device = None
        for layer, (low, high) in self._bounds.items():
            selected = rows.to(low.device)
            self._bounds[layer] = (low.index_select(0, selected), high.index_select(0, selected))

    # ---- accounting --------------------------------------------------------------------------------------------
    def _account(self) -> None:
        """Account the exact pass that just ended: every row stores its committed entries and the bounds of its
        committed pages, and reads ``pages`` pages, the partial page and the bounds in the layers that were sparse
        (dense layers and rows with at most ``pages`` committed pages read every committed entry)."""
        page, budget = self.config.page_size, self.config.pages
        for b, frontier in enumerate(self._frontier):
            if frontier is None or b >= len(self._done):
                continue
            done = self._done[b]
            stored = full = read = 0.0
            for layer_idx, keys in self.positional_buffers().items():
                entry = 2.0 * keys.shape[1] * keys.shape[3]  # K and V of one slot, in elements
                element, bound = self._key_bytes.get(layer_idx, (keys.element_size(), keys.element_size()))
                dense = entry * frontier * keys.element_size()
                full += dense
                if not self.covers(layer_idx):
                    stored += dense
                    read += dense
                    continue
                metadata = entry * done * bound
                stored += dense + metadata
                if layer_idx in self._attached and done > budget:
                    read += entry * (budget * page + frontier - done * page) * element + metadata
                else:
                    read += dense
            stats = self.stats.rows.setdefault(self._ids[b], RowStats())
            stats.kv_bytes, stats.kv_bytes_full = stored, full
            stats.kv_bytes_read += read
            stats.kv_bytes_read_full += full
            stats.passes += 1


@register_technique
class QuestTechnique(KVCacheTechnique):
    """Quest: every decoding query reads only the KV-cache pages whose keys can score high.

    See :class:`QuestConfig`. Training-free; no parameters.

    It attaches outside every quantizing or evicting cache wrapper (``order`` 40), so that its pages reach the
    attention kernel, and refuses techniques that attach their own packed K/V (``kivi_fused``,
    ``kv_evict_fused``): a key tensor carries one attachment.
    """

    name: ClassVar[str] = QuestConfig.technique
    config_class: ClassVar[type[QuestConfig]] = QuestConfig
    info: ClassVar[TechniqueInfo] = TechniqueInfo(
        family=Family.SPARSE_DECODE,
        saves=Saving.LATENCY,
        lossless=False,
    )
    graph_safe: ClassVar[bool] = False  # the selection and the attachment are host-side per step
    order: ClassVar[int] = 40
    config: QuestConfig

    def __init__(self, config: QuestConfig, adapter: Any) -> None:
        super().__init__(config, adapter)
        self.covered = quest_layers(config, adapter)

    @classmethod
    def conflict(
        cls, config: TechniqueConfig | None, other: type[Technique], other_config: TechniqueConfig | None
    ) -> str | None:
        reason = super().conflict(config, other, other_config)
        packed = getattr(other, "packed_format", None)
        if reason is None and packed is not None:
            reason = (
                f"quest attaches its pages to the keys the attention kernel reads, and {other.name} attaches its own "
                f"packed K/V ({packed!r}) there"
            )
        return reason

    def make_wrapper(self, cache: KVCache) -> CacheWrapper:
        return QuestCache(cache, self.config, self.covered)

