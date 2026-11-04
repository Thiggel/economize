"""The surgery of width pruning: delete the rows and columns of the removed heads, channels and residual channels,
fold FLAP's compensation into biases, and rewrite the config when it can describe the result.

Every map keeps its module name, so the pruned model has the state-dict keys of the original (plus the biases
the compensation adds). Whether a plain checkpoint of the result loads back as the same model is decided by
building each decoder layer anew from the rewritten config on the meta device and comparing its parameter shapes
(:func:`describes`), so the check holds for any architecture with the decoder layout and needs no table of
model types.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch
from torch import Tensor, nn

from .._structured.layout import DecoderLayout
from .._structured.surgery import keep_input_channels, keep_output_channels, set_linear
from .stats import LayerShape, layer_shape

__all__ = [
    "add_config_biases",
    "compensation_bias",
    "describes",
    "group_channels",
    "prune_hidden",
    "prune_layer",
    "realized_shapes",
    "rewrite_config",
]


def group_channels(groups: Tensor, width: int) -> Tensor:
    """The channels of the kept groups when each group spans ``width`` consecutive channels."""
    return (groups.long()[:, None] * width + torch.arange(width)[None, :]).flatten()


def _complement(index: Tensor, total: int) -> Tensor:
    mask = torch.ones(total, dtype=torch.bool)
    mask[index] = False
    return torch.nonzero(mask).flatten()


def compensation_bias(linear: nn.Linear, kept: Tensor, mean: Tensor) -> Tensor:
    """FLAP's bias after removing the input channels outside ``kept``: the old bias plus ``W[:, removed] E[x_removed]``,
    the expected contribution of the removed channels (float64)."""
    removed = _complement(kept, linear.in_features).to(linear.weight.device)
    weight = linear.weight.detach().double()
    bias = weight[:, removed] @ mean.double().to(weight.device)[removed]
    return bias if linear.bias is None else bias + linear.bias.detach().double()


@torch.no_grad()
def prune_layer(
    layout: DecoderLayout, layer: nn.Module, groups: Tensor, channels: Tensor,
    attn_mean: Tensor | None = None, mlp_mean: Tensor | None = None,
) -> None:
    """Keep the key/value groups ``groups`` and the FFN channels ``channels`` of one decoder layer.

    Args:
        layout: The decoder layout.
        layer: The layer, changed in place.
        groups: Kept key/value groups (sorted, 0-based).
        channels: Kept FFN channels (sorted, 0-based).
        attn_mean: Mean input of the attention output map, for FLAP's compensation (``None``: none).
        mlp_mean: Mean input of the down map, for FLAP's compensation (``None``: none).
    """
    shape = layer_shape(layout, layer)
    if groups.numel() < shape.kv_heads:
        q, *kv = layout.maps(layer, layout.attn_inputs).values()  # no value map where values are the keys
        o = layout.linear(layer, layout.attn_output)
        query, key = group_channels(groups, shape.group_width), group_channels(groups, shape.head_dim)
        bias = None if attn_mean is None else compensation_bias(o, query, attn_mean)
        keep_output_channels(q, query)
        for linear in kv:
            keep_output_channels(linear, key)
        keep_input_channels(o, query, bias)
    if channels.numel() < shape.intermediate:
        down = layout.linear(layer, layout.mlp_output)
        bias = None if mlp_mean is None else compensation_bias(down, channels, mlp_mean)
        for name in layout.mlp_inputs:
            keep_output_channels(layout.linear(layer, name), channels)
        keep_input_channels(down, channels, bias)


@torch.no_grad()
def _keep_norm(norm: nn.Module, index: Tensor) -> None:
    for name in ("weight", "bias"):
        value = getattr(norm, name, None)
        if isinstance(value, Tensor) and value.ndim == 1:
            setattr(norm, name, nn.Parameter(value[index.to(value.device)].contiguous(),
                                             requires_grad=value.requires_grad))
    if isinstance(getattr(norm, "normalized_shape", None), tuple):
        norm.normalized_shape = (int(index.numel()),)  # type: ignore[assignment]


@torch.no_grad()
def prune_hidden(adapter: Any, layout: DecoderLayout, index: Tensor) -> None:
    """Keep the residual channels ``index`` everywhere: the embedding columns, every norm, the input columns of the
    maps that read the stream, the output rows of the maps that write it and the input columns of the head (a tied
    head stays tied)."""
    embed, head = adapter.embed_tokens, adapter.lm_head
    tied = head is not None and head.weight is embed.weight
    weight = embed.weight
    embed.weight = nn.Parameter(weight[:, index.to(weight.device)].contiguous(), requires_grad=weight.requires_grad)
    embed.embedding_dim = int(index.numel())
    for i in range(1, adapter.num_layers + 1):
        layer = adapter.layer(i)
        for name in (layout.attn_norm, layout.mlp_norm):
            _keep_norm(layout.norm(layer, name), index)
        for linear in layout.maps(layer, (*layout.attn_inputs, *layout.mlp_inputs)).values():
            keep_input_channels(linear, index)
        for name in (layout.attn_output, layout.mlp_output):
            keep_output_channels(layout.linear(layer, name), index)
    _keep_norm(adapter.final_norm, index)
    if tied:
        head.weight = embed.weight
        head.in_features = int(index.numel())
    elif head is not None:
        keep_input_channels(head, index)


def realized_shapes(adapter: Any, layout: DecoderLayout) -> list[LayerShape]:
    """The shape of every decoder layer, in order."""
    return [layer_shape(layout, adapter.layer(i)) for i in range(1, adapter.num_layers + 1)]


def _configs(adapter: Any) -> list[Any]:
    found = {id(c): c for c in (adapter.config, getattr(adapter.model, "config", None)) if c is not None}
    return list(found.values())


def rewrite_config(adapter: Any, layout: DecoderLayout, compensated: tuple[bool, bool]) -> bool:
    """Write the realized widths into the config(s) of the model.

    ``hidden_size`` is always written, the head and FFN widths by the layout (:meth:`DecoderLayout.write_widths`:
    for the standard layout an explicit ``head_dim`` always and the counts when every layer has the same ones). With
    compensation the bias flags (``attention_bias``, ``mlp_bias``) are set where the config has them.

    Args:
        adapter: The pruned model's adapter.
        layout: Its decoder layout.
        compensated: Whether the attention output maps and the down maps received compensation biases.

    Returns:
        Whether the config gives every layer its realized widths.
    """
    shapes = realized_shapes(adapter, layout)
    hidden = int(adapter.embed_tokens.weight.shape[1])
    uniform = True
    for config in _configs(adapter):
        config.hidden_size = hidden
        uniform = layout.write_widths(config, shapes) and uniform
        for flag, on in zip(("attention_bias", "mlp_bias"), compensated):
            if on and hasattr(config, flag):
                setattr(config, flag, True)
    return uniform


def _fresh_layer(layer: nn.Module, config: Any, index: int) -> nn.Module | None:
    """A layer of the same class built from ``config`` on the meta device (``None`` if it cannot be built)."""
    try:
        with torch.device("meta"):
            return type(layer)(config, index)
    except Exception:
        return None


@torch.no_grad()
def add_config_biases(adapter: Any, copies: Mapping[int, Sequence[nn.Module]] | None = None) -> None:
    """Give every map that a layer built from the config has a bias for, but the model lacks, a zero bias (the
    query, key and value maps once ``attention_bias`` is set for the output map's compensation). ``copies`` are
    other techniques' copies of base layers (by 1-indexed layer), which are rebuilt from the same config."""
    config = adapter.config
    for i in range(1, adapter.num_layers + 1):
        layer = adapter.layer(i)
        fresh = _fresh_layer(layer, config, i - 1)
        if fresh is None:
            return
        for name, module in fresh.named_modules():
            if not isinstance(module, nn.Linear) or module.bias is None:
                continue
            for target in (layer, *(copies or {}).get(i, ())):
                real = target.get_submodule(name)
                if isinstance(real, nn.Linear) and real.bias is None:
                    set_linear(real, real.weight, torch.zeros(real.out_features, dtype=real.weight.dtype))


def describes(adapter: Any) -> bool:
    """Whether a model built from the config has exactly the parameter shapes of the pruned model (so it saves
    and loads, and serves, as a plain checkpoint)."""
    config = adapter.config
    if int(adapter.embed_tokens.weight.shape[1]) != int(config.hidden_size):
        return False
    for i in range(1, adapter.num_layers + 1):
        layer = adapter.layer(i)
        fresh = _fresh_layer(layer, config, i - 1)
        if fresh is None:
            return False
        want = {n: tuple(p.shape) for n, p in fresh.named_parameters()}
        have = {n: tuple(p.shape) for n, p in layer.named_parameters()}
        if want != have:
            return False
    return True
