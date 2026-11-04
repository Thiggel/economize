"""Layers of a rollout span: which ones a rollout can run, and how layers with a recurrent state run per start
(:func:`recurrent_start_states`, :class:`PerStartLayer`)."""

from __future__ import annotations

import sys
from collections.abc import Callable, Iterator
from contextlib import AbstractContextManager, contextmanager, nullcontext
from typing import TYPE_CHECKING, Any, cast

import torch
import torch.utils.checkpoint

from ...arch.spec import RECURRENT_KINDS, STATELESS_KINDS, StateKind
from ...errors import IncompatibleTechniquesError
from ...runtime import CacheSpec, KVCache, LayerCall

if TYPE_CHECKING:
    from ...core.model import EfficientModel

__all__ = [
    "PerStartLayer",
    "check_rollout_layers",
    "recurrent_input_layers",
    "recurrent_layers",
    "recurrent_start_states",
]


def check_rollout_layers(model: EfficientModel, first: int, last: int, *, recurrent: bool = False) -> None:
    """Raise :class:`IncompatibleTechniquesError` unless layers ``first..last`` can run in a rollout (attention
    layers with their own keys and values, token-wise layers, and with ``recurrent`` layers that keep a recurrent
    state, which a rollout runs per start through
    :class:`PerStartLayer`)."""
    adapter = model.adapter
    for index in range(first, last + 1):
        spec = adapter.attention_spec(index)
        if spec.kind in RECURRENT_KINDS:
            if recurrent:
                continue
            raise IncompatibleTechniquesError(f"layer {index} keeps a recurrent state; rollouts cannot run it")
        if spec.kind not in STATELESS_KINDS and adapter.kv_source(index) != index:
            raise IncompatibleTechniquesError(f"layer {index} reads the keys of layer {adapter.kv_source(index)}; "
                                              "rollouts do not support shared key/value layers")


def recurrent_layers(model: EfficientModel, first: int, last: int) -> list[int]:
    """The layers of ``first..last`` that keep a recurrent state."""
    adapter = model.adapter
    return [i for i in range(first, last + 1) if adapter.state_kind(i) is StateKind.RECURRENT]


def recurrent_input_layers(model: EfficientModel, first: int, last: int) -> tuple[int, ...]:
    """The teacher states a rollout over layers ``first..last`` needs besides the drafter's own: the input (the
    output of layer ``i - 1``) of every layer ``i`` of the span that keeps a recurrent state."""
    return tuple(i - 1 for i in recurrent_layers(model, first, last))


@torch.no_grad()
def recurrent_start_states(
    model: EfficientModel, layers: list[int], hidden: dict[int, torch.Tensor], starts: torch.Tensor
) -> KVCache:
    """The states of the recurrent ``layers`` after each start, as an exact pass that ended there leaves them.

    Each layer runs (with the model's own weights) over its teacher input ``hidden[i - 1]`` ``[B, T, d]`` in
    segments that end at the starts, and its conv window and recurrent state are recorded after every segment.

    Returns:
        A cache of the model's own kind with ``B * S`` rows (row ``b * S + j`` holds row ``b``'s state after
        position ``starts[j]``).

    Raises:
        IncompatibleTechniquesError: The model's cache does not declare
            :attr:`~economize.runtime.KVCache.supports_state_edit`.
    """
    runner = model.runner
    ends = (starts + 1).tolist()
    states: KVCache | None = None
    for layer in layers:
        inputs = hidden[layer - 1]
        rows = inputs.shape[0]
        cache = runner.new_cache(ends[-1], rows)  # the model's own kind, which keeps the recurrent states
        if not (cache.keeps_recurrent_state and cache.supports_state_edit):  # PerStartLayer replaces its states
            raise IncompatibleTechniquesError(f"rollouts run recurrent layers on a cache whose states can be "
                                              f"replaced, not on the model's {type(cache).__name__}")
        if states is None:
            states = type(cache)(CacheSpec(1, rows * len(ends), cache.spec.kind))
        convs, recurrents = [], []
        begin = 0
        for end in ends:
            ctx = runner.context(cache, torch.arange(begin, end, device=inputs.device), trusted=True)
            runner.run(inputs[:, begin:end], ctx, layer, layer)
            conv, state = cache.recurrent_state(layer - 1)  # caches index layers from 0 (``layer_idx``)
            if conv is not None:
                convs.append(conv.clone())
            if state is not None:
                recurrents.append(state.clone())
            begin = end
        # no views (layers update states in place with gradients enabled)
        states.put_recurrent_state(
            layer - 1,
            torch.stack(convs, 1).flatten(0, 1).clone() if convs else None,
            torch.stack(recurrents, 1).flatten(0, 1).clone() if recurrents else None,
        )
    assert states is not None
    return states


_ONE_TOKEN_RULES = ("torch_recurrent_gated_delta_rule",)  # one-token updates that a fast kernel may replace


def _torch_rule(fn: Any) -> Callable[..., Any] | None:
    """The torch implementation behind a ``transformers`` kernel dispatcher ``fn`` (``None``: there is none).

    ``use_kernel_func_from_hub_with_fallback`` wraps the torch function with ``functools.wraps`` (``__wrapped__``),
    and with the ``kernels`` package installed that wrapper sits in the forward closure of a ``Func`` module.
    """
    for _ in range(4):
        inner = getattr(fn, "__wrapped__", None)
        if inner is not None:
            return cast(Callable[..., Any], inner)
        cells = getattr(getattr(type(fn), "forward", None), "__closure__", None) or ()
        fn = next((c.cell_contents for c in cells if callable(c.cell_contents)), None)
        if fn is None:
            return None
    return None


@contextmanager
def _differentiable_one_token_rules() -> Iterator[None]:
    """Run the one-token state updates of ``transformers`` models on their own torch implementation.

    ``transformers`` replaces the one-token gated delta rule of Qwen3-Next / Qwen3.5 / OLMo-Hybrid with the fused
    recurrent kernel of ``fla`` when it is installed, and that kernel has no backward. The torch implementation it
    wraps computes the same update and is differentiable.
    """
    swapped = []
    for key, module in list(sys.modules.items()):
        if not key.startswith("transformers.models.") or module is None:
            continue
        for name in _ONE_TOKEN_RULES:
            fn = vars(module).get(name)  # not getattr: lazy modules would import their submodules
            rule = None if fn is None else _torch_rule(fn)
            if rule is not None:
                swapped.append((module, name, fn))
                setattr(module, name, rule)
    try:
        yield
    finally:
        for module, name, fn in swapped:
            setattr(module, name, fn)


def _per_start(value: Any, rows: int, starts: int) -> Any:
    """``value`` with every ``[B, S, ...]`` tensor reshaped to ``[B * S, 1, ...]``."""
    if isinstance(value, torch.Tensor) and value.dim() >= 2 and tuple(value.shape[:2]) == (rows, starts):
        return value.reshape(rows * starts, 1, *value.shape[2:])
    if isinstance(value, tuple):
        return tuple(_per_start(v, rows, starts) for v in value)
    return value


class PerStartLayer:
    """A recurrent layer that runs the ``S`` queries of every row as ``B * S`` one-token sequences, each advancing
    its own start's state in ``states``.

    With gradients enabled the step runs under activation checkpointing: the one-token update of a gated delta net
    saves several ``[B * S, heads, d_k, d_v]`` fp32 tensors for backward, which over ``K`` steps and several layers
    exceeds the memory of everything else in the rollout. Only the states before and after the step are kept, and
    backward recomputes the step from the state it started from, inside ``context`` (the draft path's context, so
    that LoRA updates active during the forward step are active again in the recomputation). The step works on
    copies of the states it starts from, which also lets layers that update a state in place after reading it (the
    one-token Mamba-2 step of ``transformers``) train: the recomputation ends before that update.
    """

    checkpoint: bool = True

    def __init__(
        self, layer: LayerCall, states: KVCache, layer_idx: int,
        context: Callable[[], AbstractContextManager[None]] = nullcontext,
    ) -> None:
        self.layer = layer
        self.states = states
        self.layer_idx = layer_idx  # 0-based, as the caches index layers
        self.context = context

    def _put(self, conv: torch.Tensor | None, state: torch.Tensor | None) -> None:
        self.states.put_recurrent_state(self.layer_idx, conv, state)

    def __call__(self, hidden: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        rows, starts, width = hidden.shape
        args = tuple(_per_start(a, rows, starts) for a in args)
        kwargs = {key: _per_start(value, rows, starts) for key, value in kwargs.items()}
        for key in ("past_key_values", "cache_params"):
            if key in kwargs:
                kwargs[key] = self.states
        kwargs["attention_mask"] = None

        def run(
            flat: torch.Tensor, conv: torch.Tensor | None, state: torch.Tensor | None
        ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
            # copies: older transformers update the conv window (and some layers the state) in place, which would
            # change the checkpointed inputs before the recomputation reads them
            self._put(None if conv is None else conv.clone(), None if state is None else state.clone())
            rules = _differentiable_one_token_rules() if torch.is_grad_enabled() else nullcontext()
            with self.context(), rules:
                out = self.layer(flat, *args, **kwargs)
            out = out[0] if isinstance(out, tuple) else out
            return (out, *self.states.recurrent_state(self.layer_idx))

        flat = hidden.reshape(rows * starts, 1, width)
        conv, state = self.states.recurrent_state(self.layer_idx)
        if self.checkpoint and torch.is_grad_enabled():
            out, conv, state = torch.utils.checkpoint.checkpoint(run, flat, conv, state, use_reentrant=False)
        else:
            out, conv, state = run(flat, conv, state)
        self._put(conv, state)
        return out.reshape(rows, starts, width)
