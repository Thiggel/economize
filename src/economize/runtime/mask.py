"""Attention masks over a whole cache buffer.

Two registries split the job. :data:`MASKS` holds one visibility rule per mask kind (``"full"``, ``"sliding"``,
``"chunked"``): given query and key positions, it says which keys a query may see. :data:`MASK_FORMATS` turns the
resulting boolean matrix into what one attention implementation expects (a boolean mask for ``sdpa``, an additive
one for ``eager``). A new kind or implementation is supported by registering one function::

    @MASKS.register("dilated")
    def dilated(query_positions, key_positions, spec):
        return (key_positions <= query_positions) & ((query_positions - key_positions) % spec.window == 0)

A pass over whole sequences from position 0 (no padding, no tree, one key per query position) can skip the dense
mask: :data:`STRUCTURED_MASKS` gives, per mask kind, a :class:`~economize.attention.StructuredMask` that the
attention function attends by its shape (:mod:`economize.attention.banded`); kinds without an entry keep the dense
mask. :class:`~economize.runtime.LayerContext` uses it for passes built with ``whole_sequences=True``.

Draft trees are described by a :class:`TreeLayout`; :func:`visibility` lets a tree node see the committed prefix
and its own ancestors, and nothing else.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, cast

import torch

from ..arch.spec import AttentionSpec
from ..attention.banded import BandMask, StructuredMask
from ..errors import ConfigError
from ..registry import Registry

__all__ = [
    "MASKS",
    "MASK_FORMATS",
    "STRUCTURED_MASKS",
    "MaskFormat",
    "MaskRule",
    "StructuredRule",
    "TreeLayout",
    "additive_mask",
    "block_mask",
    "boolean_mask",
    "causal_band",
    "sliding_band",
    "tree_ancestors",
    "visibility",
]


class MaskRule(Protocol):
    def __call__(self, query_positions: torch.Tensor, key_positions: torch.Tensor, spec: AttentionSpec) -> torch.Tensor:
        """Broadcastable boolean tensor: ``True`` where a query at ``query_positions`` may see a key."""
        ...


class MaskFormat(Protocol):
    def __call__(self, visible: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        """The mask an attention implementation takes, from ``visible`` ``[B, Q, K]`` (typed as a tensor; flex
        attention gets a ``BlockMask`` in the same place)."""
        ...


class StructuredRule(Protocol):
    def __call__(self, spec: AttentionSpec, length: int) -> StructuredMask | None:
        """The mask of a layer with pattern ``spec`` in a pass over whole sequences of ``length`` tokens from position
        0, or ``None`` to use the dense mask."""
        ...


#: Visibility rules by :attr:`AttentionSpec.kind`.
MASKS: Registry[MaskRule] = Registry("attention mask", "economize.masks")
#: Mask encodings by attention implementation (``config._attn_implementation``).
MASK_FORMATS: Registry[MaskFormat] = Registry("mask format", "economize.mask_formats")
#: Masks of passes over whole sequences by :attr:`AttentionSpec.kind`, for implementations that take them.
STRUCTURED_MASKS: Registry[StructuredRule] = Registry("structured mask", "economize.structured_masks")


@MASKS.register("full")
def causal(query_positions: torch.Tensor, key_positions: torch.Tensor, spec: AttentionSpec) -> torch.Tensor:
    """Every earlier position and the query's own."""
    return key_positions <= query_positions


@MASKS.register("sliding")
def sliding(query_positions: torch.Tensor, key_positions: torch.Tensor, spec: AttentionSpec) -> torch.Tensor:
    """The last ``spec.window`` positions up to the query's own."""
    window = cast(int, spec.window)  # AttentionSpec guarantees a window for the windowed kinds
    return (key_positions <= query_positions) & (key_positions > query_positions - window)


@MASKS.register("chunked")
def chunked(query_positions: torch.Tensor, key_positions: torch.Tensor, spec: AttentionSpec) -> torch.Tensor:
    """Earlier positions in the query's chunk of ``spec.window`` tokens (Llama-4)."""
    window = spec.window
    if window is None:
        raise ConfigError("chunked attention needs a window (the chunk size)")
    same_chunk = torch.div(key_positions, window, rounding_mode="floor") == torch.div(
        query_positions, window, rounding_mode="floor"
    )
    return (key_positions <= query_positions) & same_chunk


@STRUCTURED_MASKS.register("full")
def causal_band(spec: AttentionSpec, length: int) -> StructuredMask:
    """Causal attention, run as one ``is_causal`` call."""
    return BandMask(length)


@STRUCTURED_MASKS.register("sliding")
def sliding_band(spec: AttentionSpec, length: int) -> StructuredMask:
    """The last ``spec.window`` positions, attended in blocks of queries over the keys each block sees."""
    return BandMask(length, window=cast(int, spec.window))


@MASK_FORMATS.register("sdpa")
def boolean_mask(visible: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """``[B, 1, Q, K]``, ``True`` where attention is allowed (``scaled_dot_product_attention`` convention)."""
    return visible[:, None]


@MASK_FORMATS.register("eager")
def additive_mask(visible: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """``[B, 1, Q, K]``, ``0`` where attention is allowed and the most negative finite value elsewhere."""
    mask = torch.zeros(visible.shape, dtype=dtype, device=visible.device)
    return mask.masked_fill_(~visible, torch.finfo(dtype).min)[:, None]


@MASK_FORMATS.register("flex_attention")
def block_mask(visible: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """A flex-attention ``BlockMask`` whose ``mask_mod`` reads ``visible`` ``[B, Q, K]`` (any pattern, trees too).

    The block structure is computed from ``visible`` itself, so blocks with no visible key are skipped by the kernel.
    """
    from torch.nn.attention.flex_attention import create_block_mask

    batch, queries, keys = visible.shape

    def mask_mod(b: torch.Tensor, h: torch.Tensor, q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        return visible[b, q, k]

    # a BlockMask, which flex-attention layers take where the other implementations take a mask tensor
    return cast(torch.Tensor, create_block_mask(mask_mod, batch, None, queries, keys, device=visible.device))


@dataclass(frozen=True, eq=False)
class TreeLayout:
    """Where the nodes of a batch of draft trees live in the cache and which of them each query sees.

    Slots below ``base`` hold the committed prefix of each row. Tree nodes occupy ``slots``; every other slot at or
    above ``base`` is invisible.

    Attributes:
        base: First slot not part of the committed prefix, ``[B]``.
        slots: Cache slot of every node, ``[B, T]`` (padding nodes use the trash slot).
        positions: Sequence position of every node (``base`` plus its depth), ``[B, T]``.
        visible: ``visible[b, q, t]``: query ``q`` sees node ``t`` (its ancestors and itself), ``[B, Q, T]``.
    """

    base: torch.Tensor
    slots: torch.Tensor
    positions: torch.Tensor
    visible: torch.Tensor

    @classmethod
    def build(
        cls, parents: torch.Tensor, base: torch.Tensor, valid: torch.Tensor | None = None, trash_slot: int = -1
    ) -> TreeLayout:
        """Layout of trees given by ``parents`` ``[B, T]`` (index of each node's parent, ``-1`` for the root, parents
        before children), stored from slot ``base`` ``[B]`` on in node order. Padding nodes (``valid`` false) go to
        ``trash_slot`` and are seen by nobody."""
        reach = tree_ancestors(parents)
        depth = reach.sum(-1) - 1
        offsets = torch.arange(parents.shape[1], device=parents.device)
        slots = base[:, None] + offsets[None]
        positions = base[:, None] + depth
        if valid is not None:
            reach = reach & valid[:, None, :] & valid[:, :, None]
            slots = torch.where(valid, slots, torch.full_like(slots, trash_slot))
            positions = torch.where(valid, positions, torch.zeros_like(positions))
        return cls(base, slots, positions, reach)

    def to(self, device: torch.device) -> TreeLayout:
        """The same layout with every tensor on ``device``."""
        return TreeLayout(*(t.to(device) for t in (self.base, self.slots, self.positions, self.visible)))

    def select(self, indices: torch.Tensor) -> TreeLayout:
        """The layout for the queries ``indices`` ``[B, K]`` (keys unchanged)."""
        rows = indices[:, :, None].expand(-1, -1, self.visible.shape[2])
        return TreeLayout(self.base, self.slots, self.positions, self.visible.gather(1, rows))


def tree_ancestors(parents: torch.Tensor) -> torch.Tensor:
    """``reach[b, i, j]``: node ``j`` is node ``i`` or one of its ancestors, ``[B, T, T]`` bool.

    Repeated squaring of the parent relation takes ``ceil(log2 T)`` batched matrix products instead of a loop over
    tree depth. No host synchronization (CUDA graphs record it).
    """
    batch, size = parents.shape
    eye = torch.eye(size, device=parents.device)
    has_parent = (parents >= 0).float()[..., None]
    edges = torch.zeros(batch, size, size, device=parents.device)
    edges = edges.scatter(2, parents.clamp_min(0)[..., None], has_parent)
    step = torch.maximum(eye, edges)
    reach = step
    for _ in range(max(size - 1, 1).bit_length()):
        reach = (torch.bmm(reach, reach) > 0).float()
    return reach > 0


def visibility(
    spec: AttentionSpec,
    query_positions: torch.Tensor,
    key_positions: torch.Tensor,
    tree: TreeLayout | None = None,
) -> torch.Tensor:
    """Which cache slots each query sees, ``[B, Q, num_slots]`` bool.

    Args:
        spec: The layer's attention pattern; its kind selects the rule in :data:`MASKS`.
        query_positions: Positions of the queries, ``[B, Q]``.
        key_positions: Position held by every slot in chain layout, ``[num_slots]`` (see
            :meth:`economize.runtime.KVCache.key_positions`), or per row, ``[B, num_slots]`` (a window cache, see
            :meth:`economize.runtime.RingSpec.key_positions`).
        tree: Tree nodes among the keys, if the pass verifies or drafts a tree.

    Raises:
        ConfigError: A tree over per-row key positions.
    """
    rule = MASKS.get(spec.kind)
    queries = query_positions[:, :, None]
    keys = key_positions[None, None, :] if key_positions.dim() == 1 else key_positions[:, None, :]
    visible = rule(queries, keys, spec)
    if tree is None:
        return visible
    if key_positions.dim() != 1:
        raise ConfigError("tree passes need the chain layout of the full cache, not a window cache")
    slot_ids = torch.arange(key_positions.shape[0], device=key_positions.device)
    visible = visible & (slot_ids[None, None, :] < tree.base[:, None, None])
    nodes = tree.visible & rule(queries, tree.positions[:, None, :], spec)
    return visible.scatter(2, tree.slots[:, None, :].expand_as(nodes), nodes)
