"""Everything a decoder layer needs besides its input: cache, positions, slots, masks and rotary tables."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch

from ..arch.spec import AttentionSpec, StateKind
from ..attention.banded import StructuredMask
from ..attention.flash import declare_prefix, declare_ring, decode_hints_wanted, ring_hints_wanted
from ..errors import ConfigError
from .cache import KVCache
from .mask import MASK_FORMATS, STRUCTURED_MASKS, TreeLayout, visibility

if TYPE_CHECKING:
    from ..arch.base import ArchAdapter

__all__ = ["ContextLayout", "LayerContext"]


@dataclass(frozen=True, eq=False)
class ContextLayout:
    """Where the tokens of a pass sit: the tensors that define a :class:`LayerContext` besides its model and cache.

    A layout outlives the context it came from (e.g. as the output of a CUDA graph, whose context is rebuilt after
    every replay by :meth:`LayerContext.from_layout`).

    Attributes:
        positions: Sequence position of every token ``[B, S]`` (padding tokens use ``0``).
        slots: Cache slot every token writes ``[B, S]`` (padding tokens use the trash slot).
        valid: Which tokens are real ``[B, S]``; ``None`` when all are.
        tree: Tree nodes among the keys, ``None`` for a chain.
        consecutive: The real tokens of every row sit at consecutive positions starting at ``positions[:, 0]``, with
            padding only after them, in a chain layout (slot = position). Masks of such a pass are declared causal
            over a prefix of the cache (:func:`economize.attention.flash.declare_prefix`), which lets the attention
            run on FlashAttention under CUDA graphs. Set by the decoder's chain passes; single-token passes have it
            by construction.
    """

    positions: torch.Tensor
    slots: torch.Tensor
    valid: torch.Tensor | None = None
    tree: TreeLayout | None = None
    consecutive: bool = False


@dataclass(eq=False)
class LayerContext:
    """One pass of ``S`` tokens per row through some decoder layers.

    Masks and rotary tables are built on first use and memoized per attention pattern and device, so a pass over
    ``L`` layers builds each distinct mask once.

    Attributes:
        adapter: The model's architecture adapter.
        cache: The cache every layer reads and writes.
        positions: Sequence position of every token, ``[B, S]`` (padding tokens use ``0``).
        slots: Cache slot every token writes, ``[B, S]`` (padding tokens use the trash slot).
        valid: Which tokens are real, ``[B, S]``; ``None`` when all are.
        tree: Tree nodes among the keys, for tree drafting and verification.
        token_inputs: Extra per-token inputs ``[B, S, ...]`` that some families feed to every layer (Gemma-4's
            per-layer embeddings), filled by :meth:`ArchAdapter.embed`.
        consecutive: See :attr:`ContextLayout.consecutive`.
        whole_sequences: The pass runs every row over positions ``0 .. S - 1`` with no padding and no tree, and its
            layers attend over exactly those ``S`` keys (e.g. a :class:`~economize.techniques.duo_attention.gates.
            SinglePassCache`). When the model's attention implementation takes structured masks
            (:attr:`~economize.arch.ArchAdapter.takes_structured_masks`), layers whose kind has an entry in
            :data:`~economize.runtime.mask.STRUCTURED_MASKS` get that mask (a band, attended without a dense
            ``[S, S]`` mask) instead of a dense one.
    """

    adapter: ArchAdapter
    cache: KVCache
    positions: torch.Tensor
    slots: torch.Tensor
    valid: torch.Tensor | None = None
    tree: TreeLayout | None = None
    token_inputs: dict[str, torch.Tensor] = field(default_factory=dict)
    consecutive: bool = False
    whole_sequences: bool = False
    _masks: dict[tuple[AttentionSpec, torch.device], torch.Tensor | StructuredMask | None] = field(
        default_factory=dict, repr=False
    )
    _rotary: dict[tuple[str, torch.device, torch.dtype], object] = field(default_factory=dict, repr=False)
    _position_ids: dict[torch.device, torch.Tensor] = field(default_factory=dict, repr=False)

    # ---- construction ------------------------------------------------------------------------------------------
    @classmethod
    def from_layout(
        cls,
        adapter: ArchAdapter,
        cache: KVCache,
        layout: ContextLayout,
        token_inputs: dict[str, torch.Tensor] | None = None,
        *,
        trusted: bool = True,
    ) -> LayerContext:
        """The context of a pass laid out as ``layout`` (every constructor goes through here).

        Args:
            token_inputs: Per-token inputs already computed for this pass (e.g. by the pass that produced the
                layout); empty by default, filled by :meth:`ArchAdapter.embed`.
            trusted: Skip the bounds check (it synchronizes with the GPU).
        """
        ctx = cls(
            adapter,
            cache,
            layout.positions,
            layout.slots,
            layout.valid,
            layout.tree,
            dict(token_inputs or {}),
            consecutive=layout.consecutive and layout.tree is None,
        )
        if not trusted:
            ctx.check()
        return ctx

    @property
    def layout(self) -> ContextLayout:
        """The positions, slots, padding and tree of this pass."""
        return ContextLayout(self.positions, self.slots, self.valid, self.tree, self.consecutive)

    @classmethod
    def chain(
        cls,
        adapter: ArchAdapter,
        cache: KVCache,
        positions: torch.Tensor,
        valid: torch.Tensor | None = None,
        *,
        trusted: bool = False,
        consecutive: bool = False,
        whole_sequences: bool = False,
    ) -> LayerContext:
        """Tokens stored at the slot equal to their position (the layout of ordinary decoding).

        Args:
            positions: ``[B, S]``, or ``[S]`` for the same positions in every row of the cache.
            valid: ``[B, S]`` padding mask; padding tokens get position ``0`` and the trash slot.
            trusted: Skip the bounds check (it synchronizes with the GPU; decoding loops pass ``True``).
            consecutive: The caller guarantees :attr:`ContextLayout.consecutive` (implied for ``S = 1``).
            whole_sequences: The caller guarantees :attr:`whole_sequences` (it needs ``valid=None``).

        Raises:
            ConfigError: ``whole_sequences`` with a padding mask.
        """
        if whole_sequences and valid is not None:
            raise ConfigError("a pass over whole sequences has no padding (valid must be None)")
        if positions.dim() == 1:
            positions = positions[None].expand(cache.active_batch_size, -1)
        consecutive = consecutive or positions.shape[1] == 1
        slots = positions
        if valid is not None:
            slots = torch.where(valid, positions, torch.full_like(positions, cache.trash_slot))
            positions = torch.where(valid, positions, torch.zeros_like(positions))
        layout = ContextLayout(positions, slots, valid, consecutive=consecutive)
        ctx = cls.from_layout(adapter, cache, layout, trusted=trusted)
        ctx.whole_sequences = whole_sequences
        return ctx

    @classmethod
    def for_tree(
        cls,
        adapter: ArchAdapter,
        cache: KVCache,
        tree: TreeLayout,
        valid: torch.Tensor | None = None,
        *,
        trusted: bool = False,
    ) -> LayerContext:
        """The nodes of ``tree`` as queries (``valid`` marks real nodes)."""
        return cls.from_layout(adapter, cache, ContextLayout(tree.positions, tree.slots, valid, tree), trusted=trusted)

    def select(self, indices: torch.Tensor, valid: torch.Tensor) -> LayerContext:
        """The context of the tokens ``indices`` ``[B, K]`` (``valid`` false for padding entries)."""
        positions = self.positions.gather(1, indices)
        slots = torch.where(valid, self.slots.gather(1, indices), torch.full_like(indices, self.cache.trash_slot))
        if self.valid is not None:
            valid = valid & self.valid.gather(1, indices)
        tree = self.tree.select(indices) if self.tree is not None else None
        inputs = {name: _gather_tokens(t, indices) for name, t in self.token_inputs.items()}
        layout = ContextLayout(torch.where(valid, positions, 0), slots, valid, tree)
        return LayerContext.from_layout(self.adapter, self.cache, layout, inputs)

    def check(self) -> None:
        """Raise :class:`~economize.errors.ConfigError` if a position or slot lies outside the cache."""
        if self.positions.numel() == 0:
            return
        if int(self.slots.min()) < 0 or int(self.slots.max()) > self.cache.trash_slot:
            raise ConfigError(f"cache slots must lie in 0..{self.cache.trash_slot}")
        if int(self.positions.max()) >= self.cache.capacity:
            raise ConfigError(
                f"position {int(self.positions.max())} exceeds the cache capacity {self.cache.capacity}; "
                "build a larger cache"
            )

    # ---- per-layer inputs ----------------------------------------------------------------------------------------
    @property
    def batch_size(self) -> int:
        return self.positions.shape[0]

    @property
    def length(self) -> int:
        """Tokens per row ``S``."""
        return self.positions.shape[1]

    def mask(self, spec: AttentionSpec, device: torch.device) -> torch.Tensor | StructuredMask | None:
        """The attention mask for a layer with pattern ``spec`` on ``device``, in the model's mask format.

        Linear-attention and state-space layers get the ``[B, S]`` padding mask (``None`` without padding), and
        token-wise layers ``None``. A pass over :attr:`whole_sequences` may give attention layers a
        :class:`~economize.attention.StructuredMask` instead of a tensor.
        """
        key = (spec, device)
        if key not in self._masks:
            state = spec.state_kind
            if state is StateKind.NONE:
                self._masks[key] = None
            elif state is StateKind.RECURRENT:
                self._masks[key] = None if self.valid is None else self.valid.to(device=device, dtype=torch.long)
            elif (structured := self._structured_mask(spec)) is not None:
                self._masks[key] = structured
            else:
                tree = None if self.tree is None else self.tree.to(device)
                ring = self.cache.ring if spec.kind == "sliding" else None
                if ring is not None and spec.window != ring.window:
                    raise ConfigError(f"a sliding layer of window {spec.window} in a window cache of {ring.window}")
                positions = self.positions.to(device)
                keys = self.cache.key_positions(device) if ring is None else ring.key_positions(positions)
                visible = visibility(spec, positions, keys, tree)
                fmt = MASK_FORMATS.get(self.adapter.attn_implementation)
                mask = fmt(visible, self.adapter.dtype)
                if ring is not None:
                    if self._ring_mask(mask):
                        declare_ring(mask, positions[:, 0], ring.window, ring.slots, ring.page)
                elif self._prefix_mask(spec, mask):
                    window = spec.window if spec.kind == "sliding" else None
                    declare_prefix(mask, positions[:, 0], window)
                self._masks[key] = mask
        return self._masks[key]

    def _structured_mask(self, spec: AttentionSpec) -> StructuredMask | None:
        """The structured mask of a layer with pattern ``spec``, or ``None`` for a dense one."""
        if not self.whole_sequences or self.tree is not None or self.valid is not None:
            return None
        if self.cache.ring is not None and spec.kind == "sliding":  # ring slots are not in position order
            return None
        if spec.kind not in STRUCTURED_MASKS:
            return None
        if not self.adapter.takes_structured_masks:
            return None
        return STRUCTURED_MASKS.get(spec.kind)(spec, self.length)

    def _prefix_mask(self, spec: AttentionSpec, mask: torch.Tensor) -> bool:
        """Whether ``mask`` is causal over a prefix of the cache and worth declaring so: a consecutive chain pass,
        a full or sliding-window layer, a boolean mask for the ``"economize"`` kernel on a GPU where the hinted
        FlashAttention path exists (:func:`~economize.attention.flash.decode_hints_wanted`)."""
        return (
            self.consecutive
            and self.tree is None
            and spec.kind in ("full", "sliding")
            and isinstance(mask, torch.Tensor)  # flex attention builds a BlockMask
            and mask.dtype == torch.bool
            and mask.dim() == 4
            and getattr(self.adapter.config, "_attn_implementation", None) == "economize"
            and decode_hints_wanted(mask.device, mask.shape[0])
        )

    def _ring_mask(self, mask: torch.Tensor) -> bool:
        """Whether a window-cache mask is worth declaring to FlashAttention (:func:`~economize.attention.flash.
        declare_ring`): a consecutive chain pass with a boolean mask for the ``"economize"`` kernel on a GPU whose
        FlashAttention reads the ring. Every batch size qualifies; short queries still apply the decode shape policy
        when they run."""
        return (
            self.consecutive
            and self.tree is None
            and isinstance(mask, torch.Tensor)
            and mask.dtype == torch.bool
            and mask.dim() == 4
            and getattr(self.adapter.config, "_attn_implementation", None) == "economize"
            and ring_hints_wanted(mask.device)
        )

    def rotary(self, rope: str, like: torch.Tensor) -> object:
        """Rotary table ``rope`` for this pass's positions, in the dtype and on the device of ``like``."""
        key = (rope, like.device, like.dtype)
        if key not in self._rotary:
            home = self.adapter.rotary_device
            if home == like.device:
                self._rotary[key] = self.adapter.position_embeddings(like, self.position_ids(home), rope)
            else:  # computed where the rotary module lives, then moved to the layer (``device_map``)
                table = self.adapter.position_embeddings(like.to(home), self.position_ids(home), rope)
                self._rotary[key] = _to_device(table, like.device)
        return self._rotary[key]

    def position_ids(self, device: torch.device) -> torch.Tensor:
        """``positions`` on ``device``."""
        if device not in self._position_ids:
            self._position_ids[device] = self.positions.to(device)
        return self._position_ids[device]


def _gather_tokens(tensor: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    index = indices.to(tensor.device).view(*indices.shape, *([1] * (tensor.dim() - 2)))
    return tensor.gather(1, index.expand(-1, -1, *tensor.shape[2:]))


def _to_device(table: object, device: torch.device) -> object:
    """``table`` (a tensor or nested tuples / lists of them) on ``device``."""
    if isinstance(table, torch.Tensor):
        return table.to(device)
    if isinstance(table, (tuple, list)):
        return type(table)(_to_device(t, device) for t in table)
    return table
