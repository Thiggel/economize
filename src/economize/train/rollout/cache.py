"""The keys, values and masks of a rollout: :class:`RolloutCache`, :class:`RolloutContext` and the visibility rules
of the attention layers and of a sequence predictor's memory, plus :class:`CheckpointedLayer`, which replays a
layer against the cache during backward."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import AbstractContextManager, contextmanager, nullcontext
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch
import torch.utils.checkpoint

from ...arch.spec import RECURRENT_KINDS, STATELESS_KINDS, AttentionSpec
from ...runtime import KVCache, LayerCall, LayerContext
from ...runtime.mask import MASK_FORMATS, MASKS

if TYPE_CHECKING:
    from ...attention.banded import StructuredMask

__all__ = ["CheckpointedLayer", "RolloutCache", "RolloutContext", "memory_visibility", "rollout_visibility"]


class RolloutCache(KVCache):
    """The teacher's cache of a batch followed by the keys and values written during a rollout.

    ``update`` appends the step's keys and values (one block of ``S`` per step) and returns the teacher's buffers
    followed by every block written so far; the teacher's buffers are never modified. Inside :meth:`replaying`,
    ``update`` appends nothing and returns the blocks a layer saw when it first ran (for activation checkpointing,
    which runs a layer again during backward).
    """

    def __init__(self, prefix: KVCache) -> None:
        super().__init__(prefix.spec)
        self.prefix = prefix
        self.blocks: dict[int, tuple[list[torch.Tensor], list[torch.Tensor]]] = {}
        self._replay: dict[int, int] | None = None

    def update(
        self, key_states: torch.Tensor, value_states: torch.Tensor, layer_idx: int, *args: Any, **kwargs: Any
    ) -> tuple[torch.Tensor, torch.Tensor]:
        prefix_keys, prefix_values = self.prefix.read(layer_idx)
        keys, values = self.blocks.setdefault(layer_idx, ([], []))
        key_states, value_states = key_states.to(prefix_keys.dtype), value_states.to(prefix_values.dtype)
        if self._replay is not None:
            n = self._replay.get(layer_idx, 0)
            return (torch.cat([prefix_keys, *keys[:n], key_states], 2),
                    torch.cat([prefix_values, *values[:n], value_states], 2))
        keys.append(key_states)
        values.append(value_states)
        return torch.cat([prefix_keys, *keys], 2), torch.cat([prefix_values, *values], 2)

    def block_counts(self) -> dict[int, int]:
        """The number of blocks written so far, by layer."""
        return {layer: len(keys) for layer, (keys, _) in self.blocks.items()}

    @contextmanager
    def replaying(self, counts: dict[int, int]) -> Iterator[RolloutCache]:
        """Inside the block, ``update`` of layer ``i`` returns the prefix, the first ``counts[i]`` blocks and the
        given keys and values, and stores nothing (``counts`` from :meth:`block_counts` before the first run)."""
        previous, self._replay = self._replay, counts
        try:
            yield self
        finally:
            self._replay = previous

    def read(self, layer_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        prefix_keys, prefix_values = self.prefix.read(layer_idx)
        keys, values = self.blocks.get(layer_idx, ([], []))
        return torch.cat([prefix_keys, *keys], 2), torch.cat([prefix_values, *values], 2)

    def has_layer(self, layer_idx: int) -> bool:
        return self.prefix.has_layer(layer_idx)

    def gather(self, rows: torch.Tensor, src: torch.Tensor, dst: torch.Tensor) -> None:
        raise NotImplementedError("a rollout cache is append-only")

    def reorder(self, rows: torch.Tensor) -> None:
        raise NotImplementedError("a rollout cache is append-only")


def rollout_visibility(
    spec: AttentionSpec, starts: torch.Tensor, step: int, prefix_positions: torch.Tensor
) -> torch.Tensor:
    """Which keys the ``S`` queries of rollout step ``step`` see, ``[S, P + step * S]`` (bool).

    Query ``i`` sits at position ``starts[i] + step``. It sees the prefix keys at positions ``<= starts[i]`` that
    the layer's mask rule allows, and, in each of the ``step`` blocks of jump keys (block ``j`` at positions
    ``starts + j``), only the key of its own start.

    Args:
        spec: The layer's attention pattern (its mask rule comes from :data:`~economize.runtime.MASKS`).
        starts: Start positions ``[S]``.
        step: The rollout step ``k >= 1``.
        prefix_positions: Position of every slot of the teacher's buffers ``[P]``.
    """
    rule = MASKS.get(spec.kind)
    query = (starts + step)[:, None]
    prefix = rule(query, prefix_positions[None], spec) & (prefix_positions[None] <= starts[:, None])
    offsets = torch.arange(1, step + 1, device=starts.device)
    allowed = rule(query, starts[:, None] + offsets[None], spec)  # [S, k]: the rule for the own start's keys
    own = torch.eye(len(starts), dtype=torch.bool, device=starts.device)
    jump = (own[:, None, :] & allowed[:, :, None]).reshape(len(starts), step * len(starts))
    return torch.cat([prefix, jump], 1)


def memory_visibility(starts: torch.Tensor, step: int, length: int) -> torch.Tensor:
    """Which memory entries of a :class:`~economize.techniques.jump.SequencePredictor` the ``S`` predictions of
    rollout step ``step`` see, ``[S, length + (step - 1) * S]`` (bool).

    The memory holds one entry per position of the sequence (from the teacher's states) followed by one block of
    ``S`` entries per earlier step (block ``j`` at positions ``starts + j``, from the drafter's states). Prediction
    ``i`` sees the teacher's entries of positions ``<= starts[i]`` and, in each block, the entry of its own start,
    so it sees exactly the positions before ``starts[i] + step``.
    """
    prefix = torch.arange(length, device=starts.device)[None] <= starts[:, None]
    own = torch.eye(len(starts), dtype=torch.bool, device=starts.device).repeat(1, step - 1)
    return torch.cat([prefix, own], 1)


@dataclass(eq=False)
class RolloutContext(LayerContext):
    """The context of one rollout step: queries at ``starts + step``, keys from a :class:`RolloutCache`."""

    starts: torch.Tensor | None = None
    step: int = 1

    def mask(self, spec: AttentionSpec, device: torch.device) -> torch.Tensor | StructuredMask | None:
        key = (spec, device)
        if key not in self._masks:
            # stateless kinds need no mask; recurrent kinds see one-token sequences per start (see PerStartLayer)
            if spec.kind in STATELESS_KINDS or spec.kind in RECURRENT_KINDS:
                self._masks[key] = None
            else:
                cache = self.cache
                prefix = cache.prefix if isinstance(cache, RolloutCache) else cache
                assert self.starts is not None
                visible = rollout_visibility(spec, self.starts.to(device), self.step, prefix.key_positions(device))
                visible = visible[None].expand(self.batch_size, -1, -1)
                self._masks[key] = MASK_FORMATS.get(self.adapter.attn_implementation)(visible, self.adapter.dtype)
        return self._masks[key]


class CheckpointedLayer:
    """An attention or token-wise layer of the rollout that keeps only its input for backward.

    With gradients enabled the layer runs under activation checkpointing. Its saved activations (the keys and values
    of the prefix and of every earlier step, repeated over the query heads, and the MLP's hidden units) grow with
    the horizon and the number of starts, and dominate the rollout's memory at long horizons or dense starts.
    Backward runs the layer again inside ``context`` (the jump path's) with the cache in
    :meth:`RolloutCache.replaying` mode, so that the recomputation sees the keys and values of the forward run and
    writes nothing. The keys and values a step writes stay in the cache for later steps, and their gradients reach
    this step's projections through the recorded graph.
    """

    def __init__(
        self, layer: LayerCall, cache: RolloutCache, context: Callable[[], AbstractContextManager[None]] = nullcontext
    ) -> None:
        self.layer = layer
        self.cache = cache
        self.context = context

    def __call__(self, hidden: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        if not torch.is_grad_enabled():
            out = self.layer(hidden, *args, **kwargs)
            return out[0] if isinstance(out, tuple) else out
        counts = self.cache.block_counts()
        first = [True]

        def run(x: torch.Tensor) -> torch.Tensor:
            replay = nullcontext() if first[0] else self.cache.replaying(counts)
            first[0] = False
            with replay, self.context():
                out = self.layer(x, *args, **kwargs)
            return out[0] if isinstance(out, tuple) else out

        return torch.utils.checkpoint.checkpoint(run, hidden, use_reentrant=False)
