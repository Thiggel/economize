"""Importance criteria of structured width pruning.

* FLAP: An et al., "Fluctuation-based Adaptive Structured Pruning for Large Language Models", AAAI 2024
  (https://github.com/CASIA-IVA-Lab/FLAP). The importance of an input channel ``c`` of the attention output map
  and of the down map is ``Var[x_c] * ||W[:, c]||^2`` (``WIFV``); the attention metric is squared, as in the
  reference code. ``UL-UM`` (``structure="uniform"``) sums it over each head and keeps the top heads and channels of
  every layer; ``AL-AM`` (``"adaptive"``) standardizes it over the channels of each layer, averages it over each
  head and keeps everything above one global threshold chosen so that the removed parameters match the sparsity.
* LLM-Pruner: Ma et al., "LLM-Pruner: On the Structural Pruning of Large Language Models", NeurIPS 2023
  (https://github.com/horseee/LLM-Pruner). First-order Taylor importance of the weights (``param_first``):
  ``|W * dL/dW|`` of the calibration cross-entropy, summed over every weight of a coupled group (the query, key and
  value rows of a head group and the matching columns of the output map; the gate and up rows and the down column
  of an FFN channel).
* Minitron: Muralidharan et al., "Compact Language Models via Pruning and Knowledge Distillation", NeurIPS 2024.
  Activation-based importance with the paper's best aggregation (mean over the sequence, L2 over the batch): the
  L2 norm over ``head_dim`` of every head's attention output, the magnitude of every FFN channel's input to the down
  map, and for the residual stream the magnitudes of the outputs of every norm (both norms of every layer and the
  final norm), summed.

A criterion returns per-layer :class:`Scores` in which larger means more important. A new criterion is a
:class:`WidthCriterion` subclass registered in :data:`WIDTH_CRITERIA`; no other module changes.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, ClassVar

import torch
import torch.nn.functional as F
from torch import Tensor

from ...registry import Registry
from .._structured.layout import DecoderLayout
from .stats import ChannelMoments, Hook, LayerShape, SampleNorm, layer_shape, run_dense

if TYPE_CHECKING:
    from .config import WidthPruneConfig

__all__ = ["FLAP", "WIDTH_CRITERIA", "LLMPruner", "Minitron", "Scores", "WidthCriterion", "standardize"]


@dataclass
class Scores:
    """Importance of the prunable units (float64, CPU; larger is more important).

    Attributes:
        groups: Per pruned layer (1-indexed), one score per key/value group.
        channels: Per pruned layer, one score per FFN channel.
        hidden: One score per residual-stream channel (``None``: the criterion did not score them).
        attn_mean: Per pruned layer, the mean input of the attention output map (FLAP's compensation).
        mlp_mean: Per pruned layer, the mean input of the down map (FLAP's compensation).
        comparable: Whether the scores of different layers and of groups and channels are on one scale (FLAP's
            standardized scores), so that one global threshold may select them.
    """

    groups: dict[int, Tensor]
    channels: dict[int, Tensor]
    hidden: Tensor | None = None
    attn_mean: dict[int, Tensor] = field(default_factory=dict)
    mlp_mean: dict[int, Tensor] = field(default_factory=dict)
    comparable: bool = False


class WidthCriterion(ABC):
    """Scores attention head groups, FFN channels and (optionally) residual channels.

    Attributes:
        structure: Default :attr:`WidthPruneConfig.structure`.
        keep_first: Default leading layers left unpruned.
        keep_last: Default trailing layers left unpruned.
        compensates: Whether the criterion records the mean inputs for FLAP's bias compensation (the default of
            :attr:`WidthPruneConfig.compensate`).
        adaptive: Whether the scores support ``structure="adaptive"`` (comparable across layers).
        scores_hidden: Whether the criterion scores residual channels (``hidden_sparsity > 0``).
        uses_gradients: Whether the criterion backpropagates the calibration loss into the weights, so that
            :attr:`WidthPruneConfig.gradient_chunk` bounds its gradient memory.
    """

    structure: ClassVar[str] = "uniform"
    keep_first: ClassVar[int] = 0
    keep_last: ClassVar[int] = 0
    compensates: ClassVar[bool] = False
    adaptive: ClassVar[bool] = False
    scores_hidden: ClassVar[bool] = False
    uses_gradients: ClassVar[bool] = False

    @abstractmethod
    def score(
        self, adapter: Any, layout: DecoderLayout, batches: Sequence[Tensor], layers: Sequence[int],
        config: WidthPruneConfig,
    ) -> Scores:
        """Score the units of ``layers`` (1-indexed) of the dense model on the calibration ``batches``.

        Args:
            adapter: The model's :class:`~economize.arch.ArchAdapter`.
            layout: Its decoder layout.
            batches: Calibration token ids, one ``[B, T]`` tensor per batch.
            layers: The layers whose heads and channels are pruned.
            config: The technique's configuration (``structure``, ``hidden_sparsity``, ``compensate``).
        """


#: Width-pruning criteria by name (the values of :attr:`WidthPruneConfig.method`).
WIDTH_CRITERIA: Registry[type[WidthCriterion]] = Registry("width-pruning criterion", "economize.width_criteria")


def standardize(x: Tensor) -> Tensor:
    """``(x - mean) / std`` over the last dimension (unbiased std, as ``torch.std`` in FLAP's reference); a constant
    row becomes zeros."""
    std = x.std(-1, keepdim=True)
    return (x - x.mean(-1, keepdim=True)) / torch.where(std > 0, std, torch.ones_like(std))


def _shapes(adapter: Any, layout: DecoderLayout, layers: Sequence[int]) -> dict[int, LayerShape]:
    return {i: layer_shape(layout, adapter.layer(i)) for i in layers}


def _column_norms(linear: torch.nn.Linear) -> Tensor:
    """``||W[:, c]||^2`` of every input channel (float64, CPU)."""
    return linear.weight.detach().double().pow(2).sum(0).cpu()


@WIDTH_CRITERIA.register("flap")
class FLAP(WidthCriterion):
    """Fluctuation of the inputs times the squared column norms (``WIFV``), with bias compensation; adaptive by
    default (``AL-AM``)."""

    structure: ClassVar[str] = "adaptive"
    compensates: ClassVar[bool] = True
    adaptive: ClassVar[bool] = True

    def score(
        self, adapter: Any, layout: DecoderLayout, batches: Sequence[Tensor], layers: Sequence[int],
        config: WidthPruneConfig,
    ) -> Scores:
        shapes = _shapes(adapter, layout, layers)
        attn = {i: ChannelMoments(shapes[i].heads * shapes[i].head_dim) for i in layers}
        mlp = {i: ChannelMoments(shapes[i].intermediate) for i in layers}
        hooks: list[Hook] = []
        for i in layers:
            layer = adapter.layer(i)
            hooks.append((layout.linear(layer, layout.attn_output), attn[i].update, "input"))
            hooks.append((layout.linear(layer, layout.mlp_output), mlp[i].update, "input"))
        run_dense(adapter, batches, hooks)
        adaptive = config.resolved_structure == "adaptive"
        scores = Scores({}, {}, comparable=adaptive)
        for i in layers:
            shape, layer = shapes[i], adapter.layer(i)
            head_metric = (attn[i].variance * _column_norms(layout.linear(layer, layout.attn_output))).pow(2)
            channel_metric = mlp[i].variance * _column_norms(layout.linear(layer, layout.mlp_output))
            if adaptive:
                head_metric, channel_metric = standardize(head_metric), standardize(channel_metric)
                groups = head_metric.view(shape.kv_heads, shape.group_width).mean(1)
            else:
                groups = head_metric.view(shape.kv_heads, shape.group_width).sum(1)
            scores.groups[i], scores.channels[i] = groups, channel_metric
            scores.attn_mean[i], scores.mlp_mean[i] = attn[i].mean, mlp[i].mean
        return scores


_ATTN_MAPS = ("q", "k", "v", "o")


@contextmanager
def _gradients_on(model: Any, params: Sequence[torch.Tensor]) -> Iterator[None]:
    """Only ``params`` require gradients inside the block; every flag and ``.grad`` is restored afterwards."""
    flags = [(p, p.requires_grad, p.grad) for p in model.parameters()]
    try:
        for p in model.parameters():
            p.requires_grad_(False)
        for p in params:
            p.requires_grad_(True)
            p.grad = None
        yield
    finally:
        for p, flag, grad in flags:
            p.requires_grad_(flag)
            p.grad = grad


@WIDTH_CRITERIA.register("llm_pruner")
class LLMPruner(WidthCriterion):
    """First-order Taylor importance ``|W * dL/dW|`` summed over each coupled group. Its Llama scripts prune the
    blocks of layers 5 to ``L - 2`` (``--block_attention_layer_start 4 --block_attention_layer_end 30`` on 32
    layers), the defaults of ``keep_first`` and ``keep_last`` here.

    The gradients of the calibration loss are accumulated over all batches, as in the reference (one backward per
    batch, the importance taken from the summed gradient). They are taken for ``config.gradient_chunk`` pruned
    layers at a time (all at once by default), and every weight's salience is reduced on its device to one value per
    output row (query, key, value, gate and up maps) or input column (output and down maps) before the gradients
    are freed, so the memory beyond the model is the gradients of one chunk."""

    keep_first: ClassVar[int] = 4
    keep_last: ClassVar[int] = 2
    uses_gradients: ClassVar[bool] = True

    def score(
        self, adapter: Any, layout: DecoderLayout, batches: Sequence[Tensor], layers: Sequence[int],
        config: WidthPruneConfig,
    ) -> Scores:
        shapes = _shapes(adapter, layout, layers)
        outputs = (layout.attn_output, layout.mlp_output)  # salience summed over rows: one value per input column
        names = (*layout.attn_inputs, layout.attn_output, *layout.mlp_inputs, layout.mlp_output)
        salience: dict[int, dict[str, Tensor]] = {}
        for chunk in _chunks(layers, config.gradient_chunk):
            maps = {i: layout.maps(adapter.layer(i), names) for i in chunk}
            weights = [m.weight for i in chunk for m in maps[i].values()]
            with _gradients_on(adapter.model, weights), torch.enable_grad():
                for batch in batches:
                    _calibration_loss(adapter, batch).backward()
                for i in chunk:
                    salience[i] = {n: _taylor(m.weight, 0 if n in outputs else 1) for n, m in maps[i].items()}
        scores = Scores({}, {})
        for i in layers:
            shape, layer_salience = shapes[i], salience[i]
            groups = torch.stack([layer_salience[n].view(shape.kv_heads, -1).sum(1)  # no value map: K = V
                                  for n in layout.attn_inputs if n in layer_salience]).sum(0)
            scores.groups[i] = groups + layer_salience[layout.attn_output].view(shape.kv_heads, -1).sum(1)
            gate, up = (layer_salience[n] for n in layout.mlp_inputs)
            scores.channels[i] = gate + up + layer_salience[layout.mlp_output]
        return scores


def _chunks(layers: Sequence[int], size: int) -> list[Sequence[int]]:
    """``layers`` in consecutive chunks of ``size`` (one chunk when ``size`` is 0)."""
    size = size or max(len(layers), 1)
    return [layers[k:k + size] for k in range(0, len(layers), size)]


def _calibration_loss(adapter: Any, batch: Tensor) -> Tensor:
    """Mean next-token cross-entropy of the model on one calibration batch."""
    ids = batch.to(adapter.device)
    logits = adapter.model(input_ids=ids, use_cache=False).logits[:, :-1].float()
    target = ids[:, 1:].to(logits.device)
    return F.cross_entropy(logits.reshape(-1, logits.shape[-1]), target.reshape(-1))


def _taylor(weight: Tensor, dim: int) -> Tensor:
    """``|W * dL/dW|`` summed over ``dim`` on the weight's device (float64, returned on the CPU)."""
    assert weight.grad is not None, "the calibration loss does not depend on this weight"
    return (weight.detach().double() * weight.grad.double()).abs().sum(dim).cpu()


@WIDTH_CRITERIA.register("minitron")
class Minitron(WidthCriterion):
    """Activation magnitudes, mean over the sequence and L2 over the batch; also scores the residual stream. A head
    group's score is the L2 norm of the scores of its query heads."""

    scores_hidden: ClassVar[bool] = True

    def score(
        self, adapter: Any, layout: DecoderLayout, batches: Sequence[Tensor], layers: Sequence[int],
        config: WidthPruneConfig,
    ) -> Scores:
        shapes = _shapes(adapter, layout, layers)
        heads: dict[int, SampleNorm] = {}
        channels = {i: SampleNorm(shapes[i].intermediate) for i in layers}
        hooks: list[Hook] = []
        for i in layers:
            shape, layer = shapes[i], adapter.layer(i)

            def per_head(x: Tensor, s: LayerShape = shape) -> Tensor:
                return x.view(*x.shape[:-1], s.heads, s.head_dim).norm(dim=-1)

            heads[i] = SampleNorm(shape.heads, per_head)
            hooks.append((layout.linear(layer, layout.attn_output), heads[i].update, "input"))
            hooks.append((layout.linear(layer, layout.mlp_output), channels[i].update, "input"))
        norms: list[SampleNorm] = []
        if config.hidden_sparsity > 0:
            modules = [layout.norm(adapter.layer(i), name) for i in range(1, adapter.num_layers + 1)
                       for name in (layout.attn_norm, layout.mlp_norm)]
            for module in (*modules, adapter.final_norm):
                norms.append(SampleNorm(adapter.hidden_size))
                hooks.append((module, norms[-1].update, "output"))
        run_dense(adapter, batches, hooks)
        scores = Scores({}, {})
        for i in layers:
            shape = shapes[i]
            scores.groups[i] = heads[i].value.view(shape.kv_heads, shape.group_size).norm(dim=1)
            scores.channels[i] = channels[i].value
        if norms:
            scores.hidden = torch.stack([n.value for n in norms]).sum(0)
        return scores
