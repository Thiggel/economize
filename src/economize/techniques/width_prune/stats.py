"""Calibration statistics of the width-pruning criteria: shapes of the prunable units and activation statistics
collected with hooks during full forward passes of the dense model.

All criteria score the unmodified model: every layer sees the outputs of the dense layers before it. FLAP's
reference code runs the layers one at a time but propagates the outputs of the dense layer (it prunes after the
outputs are computed), and Minitron and LLM-Pruner score the dense model in one pass, so a hooked forward of the
whole model gives the same statistics, also on a model split over devices with ``device_map``.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from typing import Any

import torch
from torch import Tensor, nn

from ...errors import UnsupportedModelError
from .._structured.layout import DecoderLayout, LayerShape

__all__ = ["ChannelMoments", "LayerShape", "SampleNorm", "hooked", "layer_shape", "run_dense"]


def layer_shape(layout: DecoderLayout, layer: nn.Module) -> LayerShape:
    """The head and channel counts of ``layer``, read from its modules (so they are right after earlier pruning).

    Raises:
        UnsupportedModelError: The attention module has no ``head_dim``, or the head counts do not divide.
    """
    head_dim = getattr(layer.get_submodule(layout.attention), "head_dim", None)
    if not isinstance(head_dim, int):
        raise UnsupportedModelError(f"{layout.attention} of {type(layer).__name__} has no integer head_dim")
    q, k = (layout.linear(layer, name) for name in layout.attn_inputs[:2])
    heads, kv_heads = q.out_features // head_dim, k.out_features // head_dim
    if heads * head_dim != q.out_features or kv_heads * head_dim != k.out_features or heads % kv_heads:
        raise UnsupportedModelError(f"query / key widths {q.out_features} / {k.out_features} are not whole "
                                    f"groups of heads of width {head_dim}")
    return LayerShape(heads, kv_heads, head_dim, layout.linear(layer, layout.mlp_inputs[0]).out_features)


class ChannelMoments:
    """Running mean and variance of every channel over all tokens (Chan et al.'s parallel update, float64).

    Args:
        width: Number of channels.
    """

    def __init__(self, width: int) -> None:
        self.width, self.count = width, 0
        self._mean: Tensor | None = None
        self._m2: Tensor | None = None

    def update(self, x: Tensor) -> None:
        """Add the tokens of ``x`` (``[..., width]``)."""
        x = x.detach().reshape(-1, self.width).double()
        n = x.shape[0]
        if n == 0:
            return
        mean = x.mean(0)
        m2 = (x - mean).pow(2).sum(0)
        if self._mean is None or self._m2 is None:
            self._mean, self._m2, self.count = mean, m2, n
            return
        delta = mean - self._mean
        total = self.count + n
        self._mean = self._mean + delta * (n / total)
        self._m2 = self._m2 + m2 + delta.pow(2) * (self.count * n / total)
        self.count = total

    @property
    def mean(self) -> Tensor:
        """Per-channel mean (float64, CPU)."""
        return torch.zeros(self.width, dtype=torch.float64) if self._mean is None else self._mean.cpu()

    @property
    def variance(self) -> Tensor:
        """Per-channel unbiased variance (float64, CPU)."""
        if self._m2 is None or self.count < 2:
            return torch.zeros(self.width, dtype=torch.float64)
        return (self._m2 / (self.count - 1)).cpu()


class SampleNorm:
    """Minitron's best aggregation (Muralidharan et al., 2024, Table 11): a per-token score, averaged over the
    sequence of each sample, then the L2 norm over samples.

    Args:
        width: Number of scored units.
        token_score: Maps activations ``[B, T, C]`` to per-token scores ``[B, T, width]`` (default: ``|x|``).
    """

    def __init__(self, width: int, token_score: Callable[[Tensor], Tensor] | None = None) -> None:
        self.width = width
        self.token_score = token_score or torch.abs
        self._squares: Tensor | None = None

    def update(self, x: Tensor) -> None:
        """Add the samples of ``x`` (``[B, T, C]``)."""
        x = x.detach().double()
        per_sample = self.token_score(x.reshape(-1, x.shape[-2], x.shape[-1])).mean(1)
        squares = per_sample.pow(2).sum(0)
        self._squares = squares if self._squares is None else self._squares + squares

    @property
    def value(self) -> Tensor:
        """The scores (float64, CPU)."""
        return torch.zeros(self.width, dtype=torch.float64) if self._squares is None else self._squares.sqrt().cpu()


#: ``(module, callback, side)``: the callback receives the module's first input (``side="input"``) or its output.
Hook = tuple[nn.Module, Callable[[Tensor], None], str]


@contextmanager
def hooked(hooks: Sequence[Hook]) -> Iterator[None]:
    """Register forward hooks that pass a module's input or output tensor to a callback."""
    handles = []
    for module, callback, side in hooks:
        if side == "input":
            def pre(_m: nn.Module, args: tuple[Any, ...], cb: Callable[[Tensor], None] = callback) -> None:
                cb(args[0])

            handles.append(module.register_forward_pre_hook(pre))
        else:
            def post(_m: nn.Module, _a: Any, out: Any, cb: Callable[[Tensor], None] = callback) -> None:
                cb(out[0] if isinstance(out, tuple) else out)

            handles.append(module.register_forward_hook(post))
    try:
        yield
    finally:
        for handle in handles:
            handle.remove()


@torch.no_grad()
def run_dense(adapter: Any, batches: Sequence[Tensor], hooks: Sequence[Hook]) -> None:
    """One forward of the whole model over every calibration batch, with ``hooks`` registered."""
    with hooked(hooks):
        for batch in batches:
            adapter.model(input_ids=batch.to(adapter.device), use_cache=False)
