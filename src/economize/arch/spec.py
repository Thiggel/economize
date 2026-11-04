"""Per-layer attention pattern and state kind of a decoder layer, and the kinds of layers economize can run.

A layer kind is registered once in :data:`LAYER_KINDS`; the adapters and the runtime read everything they need about
a kind (its state, its window field, the ``config.layer_types`` names that denote it) from that entry::

    from economize.arch.spec import LayerKind, StateKind, register_layer_kind

    register_layer_kind(LayerKind("dilated", StateKind.KV, window_field="dilation", layer_types=("dilated_attention",)))

A new positional-K/V kind also needs its visibility rule in :data:`economize.runtime.MASKS` under the same name.
"""

from __future__ import annotations

import enum
from collections.abc import Callable, Iterator, Mapping
from collections.abc import Set as AbstractSet
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol

import torch

from ..errors import ConfigError
from ..registry import Registry

if TYPE_CHECKING:
    from ..attention.banded import StructuredMask

__all__ = [
    "LAYER_KINDS",
    "LAYER_TYPE_KINDS",
    "RECURRENT_KINDS",
    "STATELESS_KINDS",
    "WINDOWED_KINDS",
    "AttentionSpec",
    "HeadGeometry",
    "LayerKind",
    "PassContext",
    "StateKind",
    "layer_kind_of",
    "register_layer_kind",
]


class StateKind(enum.Enum):
    """What a decoder layer remembers about earlier tokens."""

    #: K/V per position: any slot can be overwritten, and masking hides stale ones (attention layers).
    KV = "kv"
    #: One state advanced token by token (linear attention, state-space layers). Rejected drafts need a restore.
    RECURRENT = "recurrent"
    #: Nothing: the layer transforms each token on its own (MLP or MoE blocks of hybrid stacks).
    NONE = "none"


@dataclass(frozen=True)
class LayerKind:
    """One kind of decoder layer.

    Attributes:
        name: The kind, :attr:`AttentionSpec.kind`. For :attr:`StateKind.KV` kinds also the key of the visibility rule
            in :data:`economize.runtime.MASKS`.
        state: What layers of this kind remember about earlier tokens.
        window_field: Config field holding :attr:`AttentionSpec.window` (``sliding_window``,
            ``attention_chunk_size``), ``None`` for kinds without a window.
        layer_types: ``config.layer_types`` entries of Hugging Face models that denote this kind.
    """

    name: str
    state: StateKind
    window_field: str | None = None
    layer_types: tuple[str, ...] = ()


#: Every layer kind economize can run, by :attr:`LayerKind.name`.
LAYER_KINDS: Registry[LayerKind] = Registry("layer kind", "economize.layer_kinds")


def register_layer_kind(kind: LayerKind, *, override: bool = False) -> LayerKind:
    """Register ``kind`` under its name (``override`` replaces an existing entry)."""
    return LAYER_KINDS.register(kind.name, kind, override=override)


for _kind in (
    LayerKind("full", StateKind.KV, layer_types=("full_attention",)),
    LayerKind("sliding", StateKind.KV, "sliding_window", ("sliding_attention",)),
    LayerKind("chunked", StateKind.KV, "attention_chunk_size", ("chunked_attention",)),
    LayerKind("linear", StateKind.RECURRENT, layer_types=("linear_attention",)),
    LayerKind("mamba", StateKind.RECURRENT, layer_types=("mamba",)),
    # Nemotron-H's MLP and MoE blocks mix no information across tokens: no cache, no mask.
    LayerKind("none", StateKind.NONE, layer_types=("mlp", "moe")),
):
    register_layer_kind(_kind)


def layer_kind_of(layer_type: str) -> LayerKind | None:
    """The registered kind whose :attr:`LayerKind.layer_types` contain ``layer_type``, else ``None``."""
    for name in LAYER_KINDS.names():
        kind = LAYER_KINDS.get(name)
        if layer_type in kind.layer_types:
            return kind
    return None


class _LayerTypeKinds(Mapping[str, str]):
    """Read-only view: ``config.layer_types`` entry -> kind name, over the current :data:`LAYER_KINDS`."""

    def _table(self) -> dict[str, str]:
        return {t: name for name in LAYER_KINDS.names() for t in LAYER_KINDS.get(name).layer_types}

    def __getitem__(self, layer_type: str) -> str:
        return self._table()[layer_type]

    def __iter__(self) -> Iterator[str]:
        return iter(self._table())

    def __len__(self) -> int:
        return len(self._table())


class _KindNames(AbstractSet[str]):
    """Read-only view: the names of the registered kinds that satisfy a predicate."""

    def __init__(self, predicate: Callable[[LayerKind], bool]) -> None:
        self._predicate = predicate

    def _names(self) -> set[str]:
        return {name for name in LAYER_KINDS.names() if self._predicate(LAYER_KINDS.get(name))}

    def __contains__(self, name: object) -> bool:
        return isinstance(name, str) and name in LAYER_KINDS and self._predicate(LAYER_KINDS.get(name))

    def __iter__(self) -> Iterator[str]:
        return iter(sorted(self._names()))

    def __len__(self) -> int:
        return len(self._names())


#: ``config.layer_types`` entries and the kind each one denotes (a view of :data:`LAYER_KINDS`).
LAYER_TYPE_KINDS: Mapping[str, str] = _LayerTypeKinds()
#: Kinds whose layers keep a recurrent state instead of a positional K/V cache (a view of :data:`LAYER_KINDS`).
RECURRENT_KINDS: AbstractSet[str] = _KindNames(lambda kind: kind.state is StateKind.RECURRENT)
#: Kinds whose layers mix no information across tokens (a view of :data:`LAYER_KINDS`).
STATELESS_KINDS: AbstractSet[str] = _KindNames(lambda kind: kind.state is StateKind.NONE)
#: Kinds that need :attr:`AttentionSpec.window` (a view of :data:`LAYER_KINDS`).
WINDOWED_KINDS: AbstractSet[str] = _KindNames(lambda kind: kind.window_field is not None)


@dataclass(frozen=True)
class AttentionSpec:
    """Which cached positions one decoder layer attends to, and which rotary table it uses.

    Attributes:
        kind: Layer kind, a key of :data:`LAYER_KINDS`: a mask kind of :data:`economize.runtime.MASKS` (``"full"``,
            ``"sliding"``, ``"chunked"``), a recurrent kind (``"linear"``, ``"mamba"``) whose layers take a padding
            mask instead, or ``"none"`` for layers that mix no tokens.
        window: Sliding-window length (a query at position ``p`` sees positions ``p - window + 1 .. p``) or chunk
            size (positions in the same chunk of ``window`` tokens). ``None`` for the other kinds.
        rope: Key of the rotary table passed to :meth:`ArchAdapter.position_embeddings`. Models with one table use
            ``"default"``; Gemma-3/4 and OLMo-3 use the layer type.
    """

    kind: str = "full"
    window: int | None = None
    rope: str = "default"

    def __post_init__(self) -> None:
        if self.kind in WINDOWED_KINDS and (self.window is None or self.window < 1):
            raise ConfigError(f"attention kind {self.kind!r} needs a positive window, got {self.window!r}")

    @property
    def state_kind(self) -> StateKind:
        """The :attr:`LayerKind.state` of :attr:`kind`; :attr:`StateKind.KV` for a kind that only has a mask rule."""
        return LAYER_KINDS.get(self.kind).state if self.kind in LAYER_KINDS else StateKind.KV


@dataclass(frozen=True)
class HeadGeometry:
    """Head counts and head width of one attention layer (Gemma-4's full-attention layers differ from its sliding
    layers).

    Attributes:
        heads: Query heads.
        kv_heads: Key/value heads.
        head_dim: Width of one head.
    """

    heads: int
    kv_heads: int
    head_dim: int


class PassContext(Protocol):
    """What an adapter reads from the context of a pass through some decoder layers.

    :class:`economize.runtime.LayerContext` implements it; the protocol keeps :mod:`economize.arch` free of runtime
    imports.

    Attributes:
        cache: The cache every layer reads (``read(layer_idx)``) and writes (``write(layer_idx, key, value)``).
        slots: Cache slot every token writes, ``[B, S]``.
        token_inputs: Extra per-token inputs that some families feed to every layer.
    """

    cache: Any
    slots: torch.Tensor
    token_inputs: dict[str, Any]

    def mask(self, spec: AttentionSpec, device: torch.device) -> torch.Tensor | StructuredMask | None:
        """The attention mask of a layer with pattern ``spec`` on ``device``."""
        ...

    def rotary(self, rope: str, like: torch.Tensor) -> object:
        """Rotary table ``rope`` for the pass's positions, in the dtype and on the device of ``like``."""
        ...

    def position_ids(self, device: torch.device) -> torch.Tensor:
        """The pass's positions on ``device``."""
        ...
