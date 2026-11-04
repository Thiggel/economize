"""The differentiable forward pass of DuoAttention's gate training.

The reference (mit-han-lab/duo-attention, ``duo_attn/patch/llama.py``, the ``two_way`` attention used by
``duo_attn/train.py``) gives every KV head of every layer a gate ``alpha`` in ``[0, 1]`` and replaces the head's
attention output by ``alpha * full + (1 - alpha) * streaming``. ``full`` is the output of the unmodified model's
attention at that layer (computed from the full model's own states, without gradients), ``streaming`` the head's
attention restricted to the first ``sinks`` positions and the last ``recent`` ones, computed from the mixed model's
states. Gates are shared by the query heads of a KV head.

:class:`GatedPass` reproduces this with two passes through :class:`~economize.runtime.LayerRunner`:

1. :meth:`GatedPass.teacher` runs the model unchanged (no gradients) and keeps, per gated layer, the input of the
   attention's output projection (the concatenated head outputs) and the final normed states.
2. :meth:`GatedPass.mixed` runs the model with the streaming mask in every gated layer; a forward pre-hook on the
   output projection mixes the teacher's head outputs in with the gates. Every layer runs under activation
   checkpointing (``torch.utils.checkpoint``), so only the layer inputs stay in memory.

Both passes run whole sequences from position 0, so with economize's ``sdpa`` function or the ``"economize"``
kernel no layer gets a dense ``[T, T]`` mask: full-attention layers run as one causal call, and sliding-window layers
and the streaming heads attend in blocks of queries over the keys each block sees
(:class:`~economize.attention.BandMask`). The dense masks made the masked SDPA kernel score every query against every
key in every layer, which dominated a step at long contexts.

Gates exist for softmax layers that attend over the whole context and own their K/V (:func:`gated_layers`).
Sliding-window and chunked layers of hybrid models (Gemma-2/3/4, OLMo-3, Ministral) are left out: their heads see a
bounded window already, and :class:`~economize.techniques.duo_attention.DuoAttentionConfig` keeps them in full. A
KV-sharing layer (Gemma-4's last layers) attends over the cache of an earlier layer and follows that layer's gates,
as it follows its row of the pattern at deployment.
"""

from __future__ import annotations

from collections.abc import Callable, Collection, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING, Any

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from ...attention.banded import BandMask
from ...errors import ConfigError, UnsupportedModelError
from ...runtime import KVCache
from ...runtime.cache import CacheSpec
from ...runtime.mask import MASK_FORMATS

if TYPE_CHECKING:
    from torch.utils.hooks import RemovableHandle

    from ...arch import ArchAdapter
    from ...runtime import LayerRunner

__all__ = [
    "GatedPass",
    "HeadGates",
    "SinglePassCache",
    "TeacherStates",
    "gated_layers",
    "kv_followers",
    "streaming_visibility",
]


def gated_layers(adapter: ArchAdapter, layers: Sequence[int] = ()) -> tuple[int, ...]:
    """The 1-indexed layers that get one gate per KV head: softmax layers over the whole context (attention kind
    ``"full"``) that write their own K/V, optionally restricted to ``layers``.

    Raises:
        ConfigError: ``layers`` names a layer that cannot be gated, or no layer can be gated.
    """
    eligible = [
        index
        for index in range(1, adapter.num_layers + 1)
        if adapter.attention_module(index) is not None
        and adapter.attention_spec(index).kind == "full"
        and adapter.kv_source(index) == index
    ]
    if layers:
        wrong = sorted(set(layers) - set(eligible))
        if wrong:
            raise ConfigError(
                f"layers {wrong} cannot be gated: DuoAttention gates full-attention layers that own their K/V "
                f"({eligible})"
            )
        eligible = sorted(set(layers))
    if not eligible:
        raise ConfigError("the model has no full-attention layer to gate")
    return tuple(eligible)


def kv_followers(adapter: ArchAdapter, gated: Collection[int]) -> dict[int, int]:
    """Full-attention layers that attend over the cache of a gated layer (KV sharing), mapped to that layer."""
    return {
        index: adapter.kv_source(index)
        for index in range(1, adapter.num_layers + 1)
        if adapter.kv_source(index) != index
        and adapter.kv_source(index) in gated
        and adapter.attention_spec(index).kind == "full"
    }


def streaming_visibility(
    query_positions: torch.Tensor, key_positions: torch.Tensor, sinks: int, recent: int
) -> torch.Tensor:
    """``[B, Q, K]`` bool: a streaming head at position ``q`` sees key ``k <= q`` when ``k < sinks`` or
    ``q - k < recent`` (the mask of the reference's ``streaming_attn_torch``)."""
    q = query_positions[..., None]
    return (key_positions <= q) & ((key_positions < sinks) | (q - key_positions < recent))


class SinglePassCache(KVCache):
    """The cache of one pass over whole sequences that start at position 0 (slot ``p`` holds position ``p``).

    ``update`` returns the pass's own K/V without copying them into a buffer, so a layer that is recomputed under
    activation checkpointing writes nothing in place, and the graph holds no second copy of the keys. Only the
    layers in ``keep`` (0-based, the sources of KV-sharing layers) are remembered for :meth:`read`.

    Args:
        length: Tokens per row.
        batch_size: Rows.
        keep: 0-based layers whose K/V later layers read.
    """

    def __init__(self, length: int, batch_size: int, keep: Collection[int] = ()) -> None:
        super().__init__(CacheSpec(length, batch_size))
        self._keep = frozenset(keep)
        self._stored: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}

    @property
    def num_slots(self) -> int:
        """One slot per token, no trash slot (the pass has no padding writes to absorb)."""
        return self.capacity

    def key_positions(self, device: torch.device) -> torch.Tensor:
        table = self._key_positions.get(device)
        if table is None:
            table = self._key_positions[device] = torch.arange(self.capacity, device=device)
        return table

    def update(
        self, key_states: torch.Tensor, value_states: torch.Tensor, layer_idx: int, *args: Any, **kwargs: Any
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if key_states.shape[-2] != self.capacity:
            raise ConfigError(f"a single-pass cache of {self.capacity} tokens got {key_states.shape[-2]} keys")
        if layer_idx in self._keep:
            self._stored[layer_idx] = (key_states, value_states)
        return key_states, value_states

    def read(self, layer_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        if layer_idx not in self._stored:
            raise KeyError(f"layer {layer_idx} has not written to the cache yet")
        return self._stored[layer_idx]

    def has_layer(self, layer_idx: int) -> bool:
        return layer_idx in self._stored

    def gather(self, rows: torch.Tensor, src: torch.Tensor, dst: torch.Tensor) -> None:
        raise NotImplementedError("a single-pass cache holds no slots to compact")

    def reorder(self, rows: torch.Tensor) -> None:
        raise NotImplementedError("a single-pass cache cannot drop rows")


class HeadGates(nn.Module):
    """One trainable gate per KV head of every gated layer (``float32``, on the layer's device).

    Args:
        adapter: The model's adapter.
        layers: 1-indexed gated layers (:func:`gated_layers`).
        initial_value: Starting value of every gate (1: every head starts as a retrieval head, as in the reference).
    """

    def __init__(self, adapter: ArchAdapter, layers: Sequence[int], initial_value: float = 1.0) -> None:
        super().__init__()
        self.layers = tuple(layers)
        self.num_layers = adapter.num_layers
        self.kv_heads = {index: adapter.head_geometry(index).kv_heads for index in range(1, adapter.num_layers + 1)}
        self.gates = nn.ParameterDict({
            str(index): nn.Parameter(
                torch.full((self.kv_heads[index],), float(initial_value), dtype=torch.float32,
                           device=adapter.layer_device(index))
            )
            for index in self.layers
        })

    def __getitem__(self, index: int) -> nn.Parameter:
        return self.gates[str(index)]

    @torch.no_grad()
    def clamp_(self) -> None:
        """Clip every gate to ``[0, 1]`` in place (the reference does this before each optimizer step)."""
        for gate in self.gates.values():
            gate.clamp_(0, 1)

    def flat(self) -> torch.Tensor:
        """All gates in one vector (on the first gate's device), in layer order."""
        values = [self[index] for index in self.layers]
        return torch.cat([v.to(values[0].device) for v in values])

    def rows(self) -> list[list[float]]:
        """The gate values per decoder layer and KV head, as the reference's ``full_attention_heads.tsv`` holds
        them; layers without gates get ``1.0`` (full attention) for every KV head."""
        return [
            self[index].detach().float().cpu().tolist() if str(index) in self.gates else [1.0] * self.kv_heads[index]
            for index in range(1, self.num_layers + 1)
        ]


@dataclass
class TeacherStates:
    """What the unmodified model's pass leaves for the mixed pass.

    Attributes:
        attention: Per gated or following layer, the input of the attention's output projection ``[B, T, H * D]``
            (the full-attention head outputs).
        final: Final normed states ``[B, T, d]``.
    """

    attention: dict[int, torch.Tensor]
    final: torch.Tensor


class GatedPass:
    """The teacher pass and the gated mixed pass of DuoAttention's gate training on one model.

    Construction installs a forward pre-hook on the output projection of every gated and following layer; the hooks
    act only inside :meth:`teacher` and :meth:`mixed`. Call :meth:`remove` to take them off.

    Args:
        runner: The model's layer runner.
        gates: The gates.
        sinks: Sink positions of a streaming head.
        recent: Recent positions of a streaming head.
        output_projection: Attribute name of the output projection inside the attention module.
        checkpoint_layers: Recompute every layer in the backward pass instead of keeping its activations.
        structured_masks: Give the attention band masks it attends by their structure, when the model's attention
            function takes them (:attr:`~economize.arch.ArchAdapter.takes_structured_masks`); ``False`` builds dense
            masks in every layer. Both attend the same keys, so the results agree up to rounding.
    """

    def __init__(
        self,
        runner: LayerRunner,
        gates: HeadGates,
        *,
        sinks: int,
        recent: int,
        output_projection: str = "o_proj",
        checkpoint_layers: bool = True,
        structured_masks: bool = True,
    ) -> None:
        adapter = runner.adapter
        if adapter.cache_kind != "positional":
            raise UnsupportedModelError("DuoAttention gate training supports models of softmax-attention layers only")
        self.runner, self.adapter, self.gates = runner, adapter, gates
        self.sinks, self.recent = int(sinks), int(recent)
        self.checkpoint_layers = checkpoint_layers
        self.structured_masks = structured_masks and adapter.takes_structured_masks
        self.followers = kv_followers(adapter, gates.layers)
        #: gated or following layer -> the layer whose gates it uses
        self.gate_of = {**{index: index for index in gates.layers}, **self.followers}
        #: 0-based layers whose K/V a KV-sharing layer reads
        self._keep = {
            adapter.kv_source(index) - 1
            for index in range(1, adapter.num_layers + 1)
            if adapter.kv_source(index) != index
        }
        self._length = 0
        self._masks: dict[torch.device, Any] = {}
        self._actions: dict[int, Callable[[torch.Tensor], torch.Tensor | None]] = {}
        self._handles: list[RemovableHandle] = []
        for index in self.gate_of:
            module = adapter.attention_module(index)
            projection = getattr(module, output_projection, None)
            if not isinstance(projection, nn.Module):
                raise UnsupportedModelError(
                    f"layer {index}'s attention has no output projection {output_projection!r} "
                    "(set output_projection to its attribute name)"
                )
            self._handles.append(projection.register_forward_pre_hook(partial(self._pre_hook, index)))

    def remove(self) -> None:
        """Take the hooks off the model."""
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    # ---- passes ------------------------------------------------------------------------------------------------
    @torch.no_grad()
    def teacher(self, input_ids: torch.Tensor) -> TeacherStates:
        """The unmodified model's head outputs at every gated layer and its final normed states."""
        attention: dict[int, torch.Tensor] = {}

        def keep(index: int, x: torch.Tensor) -> None:
            attention[index] = x.detach()

        overrides = {index: self._acting(index, partial(keep, index)) for index in self.gate_of}
        final = self._run(input_ids, overrides)
        return TeacherStates(attention, final)

    def mixed(self, input_ids: torch.Tensor, teacher: TeacherStates) -> torch.Tensor:
        """Final normed states of the model whose gated heads output ``g * full + (1 - g) * streaming``."""
        overrides: dict[int, Callable[..., torch.Tensor]] = {}
        for index in range(1, self.adapter.num_layers + 1):
            call: Callable[..., torch.Tensor] = self.adapter.layer(index)
            gated = index in self.gate_of
            if gated:
                call = self._acting(index, partial(self._mix, index, teacher.attention[index]))
            if self.checkpoint_layers and torch.is_grad_enabled():
                call = partial(_checkpointed, call)
            if gated:
                call = self._streaming(call)
            overrides[index] = call
        return self._run(input_ids, overrides)

    def mix(self, index: int, streaming: torch.Tensor, full: torch.Tensor) -> torch.Tensor:
        """``g * full + (1 - g) * streaming`` for the heads of layer ``index`` (``[B, T, H * D]`` each), with the
        gates repeated over the query heads of each KV head (exact at ``g`` = 0 and 1).

        The gates are used as they are: :meth:`HeadGates.clamp_` keeps them in ``[0, 1]`` before every step, so the
        reference's forward ``clamp(0, 1)`` changes no value, and in recent ``torch`` it has a zero gradient at the
        bounds, which would freeze gates initialized at exactly 1."""
        geometry = self.adapter.head_geometry(index)
        gate = self.gates[self.gate_of[index]].to(streaming.device)
        gate = gate.repeat_interleave(geometry.heads // geometry.kv_heads).view(1, 1, geometry.heads, 1)
        shape = (*streaming.shape[:-1], geometry.heads, -1)
        out = torch.lerp(streaming.reshape(shape).float(), full.reshape(shape).float(), gate)
        return out.reshape(streaming.shape).to(streaming.dtype)

    # ---- internals ---------------------------------------------------------------------------------------------
    def _run(self, input_ids: torch.Tensor, overrides: dict[int, Callable[..., torch.Tensor]]) -> torch.Tensor:
        rows, length = input_ids.shape
        if length != self._length:
            self._length, self._masks = length, {}
        cache = SinglePassCache(length, rows, self._keep)
        ctx = self.runner.context(
            cache, torch.arange(length, device=input_ids.device), trusted=True,
            whole_sequences=self.structured_masks,
        )
        hidden = self.runner.embed(input_ids, ctx)
        final, _ = self.runner.run(hidden, ctx, 1, self.adapter.num_layers, overrides=overrides)
        return self.adapter.final_norm(final.to(self.adapter.head_device))

    def _streaming(self, call: Callable[..., torch.Tensor]) -> Callable[..., torch.Tensor]:
        """``call`` with the streaming mask in place of the layer's causal mask (built outside any checkpointed
        region, so that the recomputation in the backward pass reuses it)."""

        def run(hidden: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
            kwargs["attention_mask"] = self.streaming_mask(hidden.device)
            return call(hidden, *args, **kwargs)

        return run

    def streaming_mask(self, device: torch.device) -> Any:
        """The streaming heads' mask over the current pass's length, in the model's attention implementation's
        format (cached per device; every row of a batch shares it), or a :class:`~economize.attention.BandMask`
        with :attr:`structured_masks`."""
        mask = self._masks.get(device)
        if mask is None and self.structured_masks:
            mask = self._masks[device] = BandMask(self._length, window=self.recent, sinks=self.sinks)
        elif mask is None:
            positions = torch.arange(self._length, device=device)
            visible = streaming_visibility(positions[None], positions, self.sinks, self.recent)
            mask = self._masks[device] = MASK_FORMATS.get(self.adapter.attn_implementation)(visible, self.adapter.dtype)
        return mask

    def _acting(self, index: int, action: Callable[[torch.Tensor], torch.Tensor | None]) -> Callable[..., torch.Tensor]:
        """Layer ``index``'s call with ``action`` applied to its output projection's input.

        The action is installed inside the call, so that a recomputation under activation checkpointing (which
        reruns the call during the backward pass) applies it again.
        """
        layer = self.adapter.layer(index)

        def run(hidden: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
            with self._active(index, action):
                return layer(hidden, *args, **kwargs)

        return run

    @contextmanager
    def _active(self, index: int, action: Callable[[torch.Tensor], torch.Tensor | None]) -> Iterator[None]:
        previous = self._actions.get(index)
        self._actions[index] = action
        try:
            yield
        finally:
            if previous is None:
                self._actions.pop(index, None)
            else:
                self._actions[index] = previous

    def _mix(self, index: int, full: torch.Tensor, streaming: torch.Tensor) -> torch.Tensor:
        return self.mix(index, streaming, full)

    def _pre_hook(self, index: int, module: nn.Module, args: tuple[Any, ...]) -> tuple[Any, ...] | None:
        action = self._actions.get(index)
        if action is None:
            return None
        out = action(args[0])
        return None if out is None else (out, *args[1:])


def _checkpointed(call: Callable[..., torch.Tensor], hidden: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
    out = checkpoint(call, hidden, *args, use_reentrant=False, **kwargs)
    return out[0] if isinstance(out, tuple) else out
