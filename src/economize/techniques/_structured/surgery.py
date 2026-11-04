"""In-place changes of the shape of linear maps, embeddings and norms (the building blocks of slicing and width
pruning). Every function replaces the parameters of the module it is given, so module names and state-dict keys
stay those of the architecture and the rewritten model saves as a plain checkpoint."""

from __future__ import annotations

from typing import cast

import torch
from torch import Tensor, nn

__all__ = [
    "fold_rotate_inputs",
    "keep_input_channels",
    "keep_output_channels",
    "rotate_embedding",
    "rotate_outputs",
    "set_linear",
    "unit_norm",
]


def _parameter(value: Tensor, like: Tensor) -> nn.Parameter:
    return nn.Parameter(value.to(dtype=like.dtype, device=like.device).contiguous(), requires_grad=like.requires_grad)


def set_linear(linear: nn.Linear, weight: Tensor, bias: Tensor | None = None) -> None:
    """Give ``linear`` a new weight (``[out, in]``) and bias, cast to its dtype and device, and update its sizes.

    Args:
        linear: The map to change.
        weight: The new weight.
        bias: The new bias; ``None`` keeps none if the map has none (a map with a bias must be given one).
    """
    old: Tensor = linear.weight
    linear.weight = _parameter(weight, old)
    linear.out_features, linear.in_features = int(weight.shape[0]), int(weight.shape[1])
    if linear.bias is not None:
        assert bias is not None, "a map with a bias needs a new bias"
        linear.bias = _parameter(bias, old)
    elif bias is not None:
        linear.bias = _parameter(bias, old)


@torch.no_grad()
def fold_rotate_inputs(linear: nn.Linear, gamma: Tensor | None, q: Tensor, factor: float = 1.0) -> None:
    """``W <- factor * W diag(gamma) q``: fold a norm's weight into the map, then rotate (and slice) its inputs.

    Args:
        linear: A map that reads the (normalized) residual stream.
        gamma: The norm weight to fold in (``None``: the inputs are not normalized).
        q: Rotation ``[in, new_in]`` (the leading columns of an orthogonal matrix).
        factor: Scalar on the result (SliceGPT's ``sqrt(D / d)`` keeps the norm's mean over the original width).
    """
    weight = linear.weight.double()
    if gamma is not None:
        weight = weight * gamma.double().to(weight.device)
    set_linear(linear, factor * weight @ q.double().to(weight.device), linear.bias)


@torch.no_grad()
def rotate_outputs(linear: nn.Linear, q: Tensor) -> None:
    """``W <- q^T W`` and ``b <- q^T b``: rotate (and slice) the outputs of a map that writes the residual stream."""
    q = q.double().to(linear.weight.device)
    bias = None if linear.bias is None else q.T @ linear.bias.double()
    set_linear(linear, q.T @ linear.weight.double(), bias)


@torch.no_grad()
def rotate_embedding(embedding: nn.Embedding, q: Tensor) -> None:
    """``E <- E q``: rotate (and slice) the token embeddings."""
    weight = embedding.weight
    embedding.weight = _parameter(weight.double() @ q.double().to(weight.device), weight)
    embedding.embedding_dim = int(q.shape[1])


@torch.no_grad()
def unit_norm(norm: nn.Module, width: int, original: int, *, scale_eps: bool = True) -> None:
    """Make an RMSNorm (already folded into its consumers) a plain ``x / rms(x)`` over ``width`` channels.

    Its weight becomes ones. With ``scale_eps`` its epsilon is scaled by ``original / width`` so that, together with
    the ``sqrt(original / width)`` the consumers carry, the norm divides by the mean square over the original width
    (SliceGPT's ``RMSN``) with the original epsilon. Without it the norm keeps the epsilon the model's config names,
    as a plain checkpoint of the smaller width does.
    """
    weight = cast(Tensor, norm.weight)
    norm.weight = _parameter(torch.ones(width, dtype=torch.float64), weight)
    if not scale_eps:
        return
    for name in ("variance_epsilon", "eps"):
        value = getattr(norm, name, None)
        if isinstance(value, float):
            setattr(norm, name, value * original / width)


@torch.no_grad()
def keep_output_channels(linear: nn.Linear, index: Tensor) -> None:
    """Keep the output channels ``index`` (rows of the weight, entries of the bias)."""
    index = index.to(linear.weight.device)
    bias = None if linear.bias is None else linear.bias[index]
    set_linear(linear, linear.weight[index], bias)


@torch.no_grad()
def keep_input_channels(linear: nn.Linear, index: Tensor, bias: Tensor | None = None) -> None:
    """Keep the input channels ``index`` (columns of the weight); ``bias`` replaces the map's bias if given."""
    index = index.to(linear.weight.device)
    set_linear(linear, linear.weight[:, index], linear.bias if bias is None else bias)
