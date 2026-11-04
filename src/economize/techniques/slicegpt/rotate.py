"""SliceGPT's rotate-and-slice of a pre-norm decoder (Ashkboos et al., "SliceGPT: Compress Large Language Models by
Deleting Rows and Columns", ICLR 2024, arXiv:2401.15024; https://github.com/microsoft/TransformerCompression).

Every RMSNorm weight is folded into the maps that read the normalized stream, which makes the network invariant to
an orthogonal change of basis ``Q`` of the residual stream: the maps that write the stream are multiplied by ``Q^T``
on their outputs, the maps that read it by ``Q`` on their inputs, and a norm without weight commutes with ``Q``.
With ``Q`` the principal components of calibration activations, the trailing columns carry little signal and are
deleted, which leaves a model of a smaller hidden width.

:func:`rotate_per_block` follows ``rotate_and_slice_sequential`` of the official code (one basis per residual
point, fitted on the outputs of the already sliced layers, and a shortcut matrix on every residual connection).
:func:`rotate_global` uses one basis for all points of the dense model, which needs no shortcut. Both read the bases
from a :class:`RotationStore`: during calibration the store computes and records them, and a reloaded technique
replays the recorded ones without calibration data, which gives the same model bit for bit.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Hashable, Sequence
from typing import Any, cast

import torch
from torch import Tensor, nn

from ...core.calibration import CalibrationStep, LayerCall, capture_inputs, layer_device, layer_types
from .._structured.layout import DecoderLayout
from .._structured.surgery import fold_rotate_inputs, rotate_embedding, rotate_outputs, unit_norm
from .pca import CovarianceAccumulator
from .shortcut import install_shortcuts, set_shortcuts

__all__ = ["RotationStore", "rotate_global", "rotate_per_block"]

#: Computes the full ``[D, D]`` basis of one residual point from calibration activations.
Fit = Callable[[], Tensor]


class RotationStore:
    """The recorded bases of a SliceGPT rotation (float32 buffers owned by the technique).

    Args:
        rotations: ``[points, D, d]``: the leading ``d`` columns of the basis of each residual point.
        head: ``[D, D]``: the full basis of the last point when the head input keeps all ``D`` channels, else empty.
        fitting: Compute the bases (``True``) or replay the recorded ones (``False``).
    """

    def __init__(self, rotations: Tensor, head: Tensor, *, fitting: bool) -> None:
        self.rotations, self.head, self.fitting = rotations, head, fitting

    def basis(self, point: int, fit: Fit, *, full: bool = False) -> Tensor:
        """The basis of ``point`` (``[D, d]``, or ``[D, D]`` with ``full``), as float64 on the CPU.

        When fitting, ``fit`` runs and the result is recorded first; the returned basis is always read back from
        the float32 record, so that the model a fit produces equals the model a replay produces.
        """
        if self.fitting:
            q = fit()
            self.rotations[point].copy_(q[:, : self.rotations.shape[2]].to(self.rotations))
            if full:
                self.head.copy_(q.to(self.head))
        return (self.head if full else self.rotations[point]).detach().double().cpu()


def _fold(layer: nn.Module, layout: DecoderLayout, norm: str, inputs: Sequence[str], q: Tensor, original: int,
          *, scale_eps: bool) -> None:
    """Fold norm ``norm`` into the maps ``inputs`` with rotation ``q`` and make it a plain RMS normalization."""
    module = layout.norm(layer, norm)
    weight = cast(Tensor, module.weight)
    gamma = weight.detach()
    width = int(q.shape[1])
    for name in inputs:
        fold_rotate_inputs(layout.linear(layer, name), gamma, q, math.sqrt(original / width))
    unit_norm(module, width, original, scale_eps=scale_eps)


def _fold_head(adapter: Any, q: Tensor, original: int, *, scale_eps: bool) -> None:
    """Fold the final norm into the LM head with rotation ``q`` and untie the head from the embedding."""
    norm, head = adapter.final_norm, adapter.lm_head
    width = int(q.shape[1])
    fold_rotate_inputs(head, norm.weight.detach(), q, math.sqrt(original / width))
    unit_norm(norm, width, original, scale_eps=scale_eps)
    for config in _configs(adapter):
        config.tie_word_embeddings = False


def _configs(adapter: Any) -> list[Any]:
    """The adapter's config and the model's config, each once (they are usually the same object)."""
    found = {id(c): c for c in (adapter.config, getattr(adapter.model, "config", None)) if c is not None}
    return list(found.values())


class _Layers:
    """Runs single decoder layers on calibration states, each on its own device (``device_map``)."""

    def __init__(self, adapter: Any, batches: Sequence[Tensor], offload: bool) -> None:
        self.adapter, self.offload = adapter, offload
        self.inputs, self._calls = capture_inputs(adapter, batches, offload=offload)
        self._types = layer_types(adapter)
        self._moved: dict[tuple[Hashable, torch.device], list[LayerCall]] = {}

    def step(self, index: int, states: list[Tensor]) -> CalibrationStep:
        device = layer_device(self.adapter, index)
        key = (self._types[index - 1], device)
        if key not in self._moved:
            self._moved[key] = [c[key[0]].to(device) for c in self._calls]
        return CalibrationStep(index, self.adapter.layer(index), states, self._moved[key], device, self.offload)


def _covariance(states: Sequence[Tensor], width: int) -> Tensor:
    accumulator = CovarianceAccumulator(width)
    for x in states:
        accumulator.update(x)
    return accumulator.basis()


def _rotated(states: Sequence[Tensor], q: Tensor) -> list[Tensor]:
    return [x @ q.to(device=x.device, dtype=x.dtype) for x in states]


@torch.no_grad()
def rotate_per_block(adapter: Any, layout: DecoderLayout, store: RotationStore, width: int, *, slice_head: bool,
                     batches: Sequence[Tensor] | None = None, offload: bool = False) -> None:
    """Rotate and slice every residual point with its own basis (the paper's method).

    Point ``0`` is the input of layer 1, point ``2l - 1`` the stream between the attention and the MLP of layer
    ``l``, and point ``2l`` the output of layer ``l``. Each basis is fitted on the activations of the model whose
    earlier points are already sliced (two layer forwards per layer).

    Args:
        adapter: The model.
        layout: Its decoder-layer layout.
        store: Where the bases are recorded (fitting) or read from (replay).
        width: The kept width ``d``.
        slice_head: Whether the LM head reads ``d`` channels (else the last layer writes all ``D``).
        batches: Calibration token batches (fitting only).
        offload: Keep the calibration states on the CPU between layers.
    """
    original, num_layers = adapter.hidden_size, adapter.num_layers
    layers = _Layers(adapter, batches, offload) if store.fitting and batches is not None else None
    states: list[Tensor] = layers.inputs if layers is not None else []
    q_in = store.basis(0, lambda: _covariance(states, original))
    rotate_embedding(adapter.embed_tokens, q_in)
    states = _rotated(states, q_in)
    for index in range(1, num_layers + 1):
        layer = adapter.layer(index)
        install_shortcuts(layer, layout)
        _fold(layer, layout, layout.attn_norm, layout.attn_inputs, q_in, original, scale_eps=True)
        set_shortcuts(layer, q_in.T, None)  # the attention output and the MLP still use the original basis
        step = layers.step(index, states) if layers is not None else None

        def fit_mid(step: CalibrationStep | None = step) -> Tensor:
            assert step is not None
            accumulator = CovarianceAccumulator(original)
            handle = layout.norm(step.layer, layout.mlp_norm).register_forward_pre_hook(
                lambda _m, args: accumulator.update(args[0])
            )
            try:
                step.forward()
            finally:
                handle.remove()
            return accumulator.basis()

        q_mid = store.basis(2 * index - 1, fit_mid)
        rotate_outputs(layout.linear(layer, layout.attn_output), q_mid)
        _fold(layer, layout, layout.mlp_norm, layout.mlp_inputs, q_mid, original, scale_eps=True)
        set_shortcuts(layer, q_in.T @ q_mid, q_mid.T)
        outputs: list[Tensor] = []

        def fit_out(step: CalibrationStep | None = step, outputs: list[Tensor] = outputs) -> Tensor:
            assert step is not None
            outputs.extend(step.forward())
            return _covariance(outputs, original)

        last_full = index == num_layers and not slice_head
        q_out = store.basis(2 * index, fit_out, full=last_full)
        rotate_outputs(layout.linear(layer, layout.mlp_output), q_out)
        set_shortcuts(layer, q_in.T @ q_mid, q_mid.T @ q_out)
        states, q_in = _rotated(outputs, q_out), q_out
    _fold_head(adapter, q_in, original, scale_eps=True)


@torch.no_grad()
def rotate_global(adapter: Any, layout: DecoderLayout, store: RotationStore, *,
                  batches: Sequence[Tensor] | None = None) -> None:
    """Rotate and slice the whole residual stream with one basis (no shortcut matrices).

    The basis is the principal components of all ``2L + 1`` residual points of the dense model, each point's
    covariance normalized to unit trace per batch so that the deep layers' larger activations do not dominate it.
    The norms keep the configured epsilon (the rewritten model is a plain checkpoint), which leaves an epsilon that
    is effectively ``d / D`` times the original one.

    Args:
        adapter: The model; its config is rewritten to the new ``hidden_size`` (with an explicit ``head_dim``).
        layout: Its decoder-layer layout.
        store: Where the basis is recorded (fitting) or read from (replay), point ``0``.
        batches: Calibration token batches (fitting only).
    """
    original = adapter.hidden_size
    head_dim = int(layout.linear(adapter.layer(1), layout.attn_inputs[0]).out_features
                   // adapter.config.num_attention_heads)

    def fit() -> Tensor:
        assert batches is not None
        accumulator = CovarianceAccumulator(original, normalize=True)

        def hook(_module: nn.Module, args: tuple[Any, ...], kwargs: dict[str, Any]) -> None:
            accumulator.update(args[0] if args else kwargs["hidden_states"])

        modules = [adapter.final_norm]
        for layer in adapter.layers:
            modules += [layer, layout.norm(layer, layout.mlp_norm)]
        handles = [m.register_forward_pre_hook(hook, with_kwargs=True) for m in modules]
        try:
            for batch in batches:
                adapter.model(input_ids=batch.to(adapter.device), use_cache=False)
        finally:
            for handle in handles:
                handle.remove()
        return accumulator.basis()

    q = store.basis(0, fit)
    rotate_embedding(adapter.embed_tokens, q)
    for layer in adapter.layers:
        _fold(layer, layout, layout.attn_norm, layout.attn_inputs, q, original, scale_eps=False)
        rotate_outputs(layout.linear(layer, layout.attn_output), q)
        _fold(layer, layout, layout.mlp_norm, layout.mlp_inputs, q, original, scale_eps=False)
        rotate_outputs(layout.linear(layer, layout.mlp_output), q)
    _fold_head(adapter, q, original, scale_eps=False)
    for config in _configs(adapter):
        config.head_dim = head_dim
        config.hidden_size = int(q.shape[1])
