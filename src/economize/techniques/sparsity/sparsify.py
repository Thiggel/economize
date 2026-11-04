"""Functions of TEAL activation sparsity (Liu et al., "Training-Free Activation Sparsity in Large Language
Models", ICLR 2025, arXiv:2408.14690; reference code github.com/FasterDecoding/TEAL).

TEAL zeroes the low-magnitude entries of the input of every linear map in a decoder layer,
``s_t(x) = x * 1[|x| > t]``, so that a sparse GEMV kernel can skip the matching weight columns during
single-token decoding. Each map gets its own threshold ``t``, a quantile of the dense model's input magnitudes on
generic text, and its own sparsity level, chosen per layer by a greedy search (Algorithm 1) under a
parameter-weighted layer-level sparsity budget.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Mapping
from contextvars import ContextVar
from dataclasses import dataclass, fields
from typing import Any

import torch
from torch import Tensor, nn
from torch.utils.hooks import RemovableHandle

from ...errors import ConfigError
from ...runtime.phase import Phase, current_phase
from ...validation import check_choice

__all__ = [
    "PREFILL_MODES",
    "GreedyStep",
    "PhasePolicy",
    "block_sparsity",
    "greedy_allocation",
    "greedy_error",
    "layer_tokens",
    "magnitude_quantile",
    "sparsify",
    "sparsify_tokens",
    "track_layer_tokens",
    "uniform_allocation",
]

#: How TEAL treats the tokens of one forward call. ``"dense"``: unchanged; ``"second_half"``: the last half of the
#: positions of every sequence is sparsified (a single position is its own last half), the protocol of TEAL's
#: perplexity and log-likelihood evaluations (it spares the attention sinks); ``"all"``: every position.
PREFILL_MODES = ("dense", "second_half", "all")


@dataclass(frozen=True)
class PhasePolicy:
    """Which tokens TEAL sparsifies in each kind of forward call, by decoding phase (:mod:`economize.runtime.phase`).

    Every field is a mode of :data:`PREFILL_MODES`. The default is TEAL's: its sparse GEMV kernels serve the
    decoding steps, and the exact passes stay dense, so that verification stays lossless also when a drafter
    proposes no tokens. A call is classified by its phase, never by the shape of a map's input (a MoE router or
    shared expert sees ``[B * S, D]``, which says nothing about the phase).

    Attributes:
        prefill: The exact pass over the prompt (phase ``"prefill"``).
        refresh: Later exact passes with a drafter, which verify the drafts (phase ``"refresh"``).
        decode: Later exact passes without a drafter, one token per row (phase ``"decode"``).
        draft: Drafter steps (phase ``"draft"``).
        outside: Calls outside the decoding loop (no phase: training, plain ``forward``) with several tokens per
            sequence. Such a call over a single token per sequence is a decoding step and follows ``decode``.
    """

    prefill: str = "dense"
    refresh: str = "dense"
    decode: str = "all"
    draft: str = "all"
    outside: str = "dense"

    def __post_init__(self) -> None:
        for f in fields(self):
            check_choice(f"PhasePolicy.{f.name}", getattr(self, f.name), PREFILL_MODES)

    @classmethod
    def from_prefill(cls, mode: str) -> PhasePolicy:
        """The policy of the config string ``prefill=mode``: calls over several tokens per sequence outside the
        decoding steps (the exact passes and plain forwards) follow ``mode``; decoding and drafting steps are
        sparsified."""
        return cls(prefill=mode, refresh=mode, outside=mode)

    @classmethod
    def coerce(cls, value: str | Mapping[str, str] | PhasePolicy) -> PhasePolicy:
        """A policy from a config value: a policy, a ``prefill`` mode string, or a mapping of fields (its JSON
        form)."""
        if isinstance(value, PhasePolicy):
            return value
        if isinstance(value, str):
            if value not in PREFILL_MODES:
                raise ConfigError(f"prefill must be one of {PREFILL_MODES} or a PhasePolicy, got {value!r}")
            return cls.from_prefill(value)
        if isinstance(value, Mapping):
            unknown = set(value) - {f.name for f in fields(cls)}
            if unknown:
                raise ConfigError(f"unknown PhasePolicy fields {sorted(unknown)}")
            return cls(**value)
        raise ConfigError(f"prefill must be a mode string, a mapping or a PhasePolicy, got {value!r}")

    def to_dict(self) -> dict[str, str]:
        """The JSON form, read back by :meth:`coerce`."""
        return {f.name: getattr(self, f.name) for f in fields(self)}

    def mode(self, phase: Phase | None, tokens: int | None) -> str:
        """The mode of a call in ``phase`` over ``tokens`` positions per sequence (``None``: unknown)."""
        if phase is None:
            return self.decode if tokens == 1 else self.outside
        return str(getattr(self, phase))


#: ``(batch, positions)`` of the decoder layer call in progress, set by :func:`track_layer_tokens`.
_LAYER_TOKENS: ContextVar[tuple[int, int] | None] = ContextVar("economize_teal_layer_tokens", default=None)


def layer_tokens() -> tuple[int, int] | None:
    """``(batch, positions)`` of the hidden states of the tracked decoder layer that is running, else ``None``."""
    return _LAYER_TOKENS.get()


def track_layer_tokens(layers: Iterable[nn.Module]) -> list[RemovableHandle]:
    """Record the token layout of every call of ``layers`` for :func:`layer_tokens` (hooks; remove the handles to
    stop). The maps inside a layer see it also when their input is flattened to ``[B * S, D]``."""
    handles: list[RemovableHandle] = []
    for layer in layers:
        tokens: list[Any] = []

        def enter(_module: nn.Module, args: tuple[Any, ...], kwargs: dict[str, Any], stack: list[Any] = tokens) -> None:
            hidden = args[0] if args else kwargs.get("hidden_states")
            layout = tuple(hidden.shape[:2]) if isinstance(hidden, Tensor) and hidden.dim() == 3 else None
            stack.append(_LAYER_TOKENS.set(layout))  # type: ignore[arg-type]

        def leave(_module: nn.Module, _args: Any, _kwargs: Any, _output: Any, stack: list[Any] = tokens) -> None:
            if stack:
                _LAYER_TOKENS.reset(stack.pop())

        handles.append(layer.register_forward_pre_hook(enter, with_kwargs=True))
        handles.append(layer.register_forward_hook(leave, with_kwargs=True, always_call=True))
    return handles


def sparsify(x: Tensor, threshold: Tensor | float) -> Tensor:
    """TEAL's magnitude sparsification ``x * 1[|x| > threshold]``."""
    return x * (x.abs() > threshold)


def sparsify_tokens(x: Tensor, threshold: Tensor | float, policy: str | PhasePolicy = "dense") -> Tensor:
    """Sparsify the tokens of a map's input ``x`` (``[..., S, D]`` or flattened ``[B * S, D]``) that ``policy``
    sparsifies in the current phase.

    The positions per sequence come from the tracked decoder layer (:func:`track_layer_tokens`) when one runs, so
    a flattened input is classified like the layer's hidden states; otherwise from ``x.shape[-2]``.
    """
    policy = PhasePolicy.coerce(policy)
    layout = layer_tokens()
    if layout is None and x.dim() < 2:
        return sparsify(x, threshold)
    tokens = layout[1] if layout is not None else x.shape[-2]
    mode = policy.mode(current_phase(), tokens)
    if mode == "dense":
        return x
    if mode == "all":
        return sparsify(x, threshold)
    return _second_half(x, threshold, layout)


def _second_half(x: Tensor, threshold: Tensor | float, layout: tuple[int, int] | None) -> Tensor:
    """Sparsify the last half of every sequence's positions (``max(1, S // 2)`` of them)."""
    shape = x.shape
    if layout is not None and x.shape[:-1].numel() == layout[0] * layout[1]:
        x = x.reshape(*layout, shape[-1])
    elif x.dim() < 2:
        return sparsify(x, threshold)
    half = max(1, x.shape[-2] // 2)
    out = torch.cat((x[..., :-half, :], sparsify(x[..., -half:, :], threshold)), dim=-2)
    return out.reshape(shape)


def magnitude_quantile(sorted_magnitudes: Tensor, q: float) -> float:
    """The ``q``-quantile of ascending magnitudes (lower interpolation), the threshold that zeroes a fraction ``q``.

    TEAL inverts the CDF of the signed activations at ``0.5 + q / 2``, which equals this quantile of ``|x|`` for the
    zero-mean symmetric distributions it assumes; the magnitude quantile hits the target sparsity also when they
    are skewed. ``q = 0`` gives threshold 0 (only exact zeros are dropped).
    """
    n = sorted_magnitudes.numel()
    if q <= 0 or n == 0:
        return 0.0
    k = min(n, max(1, round(q * n)))
    return float(sorted_magnitudes[k - 1])


def block_sparsity(sparsities: Mapping[str, float], sizes: Mapping[str, int]) -> float:
    """Parameter-weighted mean sparsity of the maps of one layer (TEAL's block-level sparsity ``P``).

    A map with ``f`` parameters and input sparsity ``p`` skips ``p * f`` weight reads, so this is the fraction of
    the layer's weight traffic that a sparse kernel saves.
    """
    total = sum(sizes.values())
    return sum(sparsities[name] * size for name, size in sizes.items()) / max(total, 1)


def uniform_allocation(sizes: Mapping[str, int], target: float) -> dict[str, float]:
    """The same sparsity for every map ("TEAL uniform" in the paper)."""
    return dict.fromkeys(sizes, target)


@dataclass(frozen=True)
class GreedyStep:
    """One accepted move of the greedy search: the map whose sparsity grew, the new ``P`` and the layer error."""

    name: str
    block_sparsity: float
    error: float


def greedy_allocation(
    error: Callable[[Mapping[str, float]], float],
    sizes: Mapping[str, int],
    target: float,
    *,
    step: float = 0.05,
    unit: int | None = None,
) -> tuple[dict[str, float], list[GreedyStep]]:
    """TEAL's block-wise greedy optimization (Algorithm 1 of the paper, ``teal/greedyopt.py`` of the reference).

    Starting from zero, each iteration tries raising the sparsity of every map by its step and keeps the move with
    the lowest layer-output error, until the layer-level sparsity :func:`block_sparsity` reaches ``target``. The
    step of a map with ``f`` parameters is ``step * unit / f`` (the reference uses ``unit = hidden_size**2``, so
    ``q_proj`` moves by ``step`` and the larger MLP maps by proportionally less), which makes every move add the
    same amount to ``P``.

    Args:
        error: Layer-output error for given per-map sparsity levels.
        sizes: Parameter count of every map.
        target: Layer-level sparsity to reach.
        step: Base step size (the reference's ``base_step_size``; 0.05 in the paper).
        unit: Parameter count that moves by exactly ``step`` (default: the largest map).

    Returns:
        The per-map sparsity levels and the accepted moves.
    """
    unit = unit or max(sizes.values())
    steps = {name: step * unit / size for name, size in sizes.items()}

    def raised(levels: Mapping[str, float], name: str) -> dict[str, float]:
        return {**levels, name: min(1.0, levels[name] + steps[name])}

    levels = dict.fromkeys(sizes, 0.0)
    path: list[GreedyStep] = []
    while block_sparsity(levels, sizes) < target - 1e-9:
        best, best_error = None, math.inf
        for name in sizes:
            if levels[name] < 1.0 and (trial_error := error(raised(levels, name))) < best_error:
                best, best_error = name, trial_error
        if best is None:
            break
        levels = raised(levels, best)
        path.append(GreedyStep(best, block_sparsity(levels, sizes), best_error))
    return levels, path


def greedy_error(reference: Tensor, output: Tensor, fraction: float = 0.25) -> Tensor:
    """The layer-output error of TEAL's greedy search, as in the reference implementation.

    The L2 norm over the last ``fraction`` of the sequence positions of ``reference - output`` (``[B, S, D]``),
    averaged over the batch and hidden dimensions.
    """
    start = int(output.shape[1] * (1 - fraction))
    return torch.linalg.vector_norm((reference[:, start:] - output[:, start:]).float(), dim=1).mean()
