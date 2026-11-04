"""Deja Vu's contextual MLP sparsity (Liu et al., "Deja Vu: Contextual Sparsity for Efficient LLMs at Inference
Time", ICML 2023, arXiv:2310.17157; reference code github.com/FMInference/DejaVu).

A small predictor per layer reads the hidden state of a token and scores every FFN channel; only the
``1 - sparsity`` fraction of channels with the highest scores is computed, in the gate, up and down maps alike
(Section 4.1 of the paper, "contextual sparsity prediction"). The predictor is the reference's two-layer network
``Linear(d, rank) -> Linear(rank, ffn)``. Its labels are the channels that matter for the token: the reference
labels a channel of the ReLU MLPs of OPT active when its activation is positive. The gated SwiGLU/GeGLU MLPs of
current decoders have no exact zeros, so the labels here are the top ``1 - sparsity`` fraction of
``|act(gate(x)) * up(x)|``, the channels with the largest contribution to the down map's input. This adaptation
to gated MLPs is ours; the reference has none. The loss is the binary cross-entropy of the scores against these
labels, as in the reference.

The paper's lookahead predictor (Section 4.3, "asynchronous lookahead") reads the input of the layer before the
attention block, so the prediction can overlap the attention; ``predictor_input="lookahead"`` reads the output of
the layer's first norm. The attention-head sparsity of the paper is not implemented (heads are a separate
technique family here).

Training: the predictors are fitted layer by layer at attach time on the calibration text (``fit_steps`` Adam
steps per layer on the dense model's activations), and the objective ``dejavu`` trains them end to end on any
dataset of :mod:`economize.train` (:class:`DejaVuObjective`).
"""

from __future__ import annotations

import math
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from functools import partial
from typing import Any, ClassVar, cast

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from ...core.calibration import LayerwiseCalibration, calibration_source, token_batches
from ...core.metadata import Family, KernelChoice, Saving, TechniqueInfo
from ...core.technique import Capability, Technique, register_technique
from ...errors import UnsupportedModelError
from ...train.objective import (
    Batch,
    ObjectiveOutput,
    OptimDefaults,
    TeacherOutputs,
    TechniqueObjective,
    register_objective,
)
from ..sparsity.sparsify import track_layer_tokens
from .backends import MLP_SPARSITY_BACKENDS, MlpOps
from .config import DejaVuConfig
from .mlp import GatedMlp, MlpPatch, dense_first_half, ffn_width, gated_mlps, masked, topk_mask

__all__ = [
    "DejaVuObjective",
    "DejaVuTechnique",
    "dejavu_labels",
    "dejavu_mlp",
    "fit_predictor",
    "kept_channels",
    "kept_mlp",
    "predictor_recall",
]


def kept_channels(ffn: int, sparsity: float) -> int:
    """The number of FFN channels computed per token at ``sparsity`` (at least one)."""
    return max(1, round((1.0 - sparsity) * ffn))


@torch.no_grad()
def dejavu_labels(mlp: GatedMlp, x: Tensor, k: int, chunk: int = 4096) -> Tensor:
    """The label of every channel of ``mlp`` for inputs ``x``: the ``k`` channels with the largest
    ``|act(gate(x)) * up(x)|`` (boolean, ``[..., ffn]``).

    The contributions are computed ``chunk`` tokens at a time, so only the boolean labels exist for all tokens at
    once (a 31B model's fit over 64K tokens would otherwise hold several ``[tokens, ffn]`` float tensors)."""
    flat = x.reshape(-1, x.shape[-1])
    labels = [topk_mask((mlp.act(mlp.gate(part)) * mlp.up(part)).abs().float(), k) for part in flat.split(chunk)]
    return torch.cat(labels).reshape(*x.shape[:-1], -1)


def predictor_recall(scores: Tensor, labels: Tensor, k: int) -> Tensor:
    """The fraction of labelled channels among the ``k`` highest ``scores``, averaged over tokens."""
    hits = (topk_mask(scores.detach(), k) & labels).sum(-1).float()
    return (hits / labels.sum(-1).clamp(min=1).float()).mean()


#: Tokens per piece of a long prefill in :meth:`DejaVuTechnique._rule`. Scores, masks and the MLP are per token, so
#: the pieces give the same output while the ``[tokens, ffn]`` scores and top-k buffers of only one piece exist at a
#: time (a 31B model's prefill of a long LongBench prompt ran out of an 80 GB GPU without them).
PREFILL_CHUNK = 4096


def kept_mlp(mlp: GatedMlp, x: Tensor, ops: MlpOps, keep: Tensor) -> Tensor:
    """The gated MLP of ``x`` computed on the channels in ``keep`` (boolean, ``[..., ffn]``) only."""
    h = masked(mlp.act(ops.rows(x, mlp.gate, keep)), keep) * ops.rows(x, mlp.up, keep)
    return ops.cols(h, mlp.down)


def dejavu_mlp(mlp: GatedMlp, x: Tensor, ops: MlpOps, mode: str, scores: Tensor, k: int) -> Tensor:
    """The Deja Vu forward of one gated MLP: the ``k`` channels with the highest predictor ``scores``."""
    keep = topk_mask(scores, k)
    if mode == "second_half":
        keep = dense_first_half(keep)
    return kept_mlp(mlp, x, ops, keep)


def fit_predictor(
    predictor: nn.Module,
    sources: Tensor,
    labels: Tensor,
    *,
    steps: int,
    lr: float,
    batch: int,
    seed: int,
) -> float:
    """Fit ``predictor`` to boolean ``labels`` ``[n, ffn]`` from ``sources`` ``[n, d]`` with Adam on the binary
    cross-entropy; returns the loss of the last step. The predictor is trained in float32 and returned in its
    own dtype."""
    dtype = next(predictor.parameters()).dtype
    predictor.float()
    optimizer = torch.optim.Adam(predictor.parameters(), lr=lr)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    n, last = sources.shape[0], math.nan
    with torch.enable_grad():
        for _ in range(steps):
            index = torch.randint(n, (min(batch, n),), generator=generator).to(sources.device)
            scores = predictor(sources[index].float())
            loss = F.binary_cross_entropy_with_logits(scores, labels[index].to(scores.dtype))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            last = float(loss.detach())
    predictor.to(dtype)
    return last


def _first(output: Any) -> Tensor:
    return cast(Tensor, output[0] if isinstance(output, tuple) else output)


@dataclass
class _Record:
    """Scores (with gradient) and labels of one MLP call while :meth:`DejaVuTechnique.recording` is active."""

    layer: int
    scores: Tensor
    labels: Tensor


@register_technique
class DejaVuTechnique(Technique):
    """Contextual FFN-channel sparsity with a trained predictor per layer (Deja Vu, see :class:`DejaVuConfig`).

    The predictors are the submodules ``predictors[str(layer)]``, saved and loaded with the technique's state dict
    together with the buffer ``fitted``; a loaded technique skips the attach-time fit. The base weights are never
    changed by the dense backend, the Triton backend stores the down weights transposed while attached, and
    :meth:`detach` restores both.

    Attributes:
        predictors: One ``Linear(d, rank) -> Linear(rank, ffn)`` per selected layer, keyed by the layer.
        fitted: Whether the predictors were fitted (at attach time or by training) or loaded.
    """

    name: ClassVar[str] = DejaVuConfig.technique
    config_class: ClassVar[type[DejaVuConfig]] = DejaVuConfig
    info = TechniqueInfo(
        family=Family.ACTIVATION_SPARSITY,
        saves=Saving.COMPUTE | Saving.LATENCY,
        lossless=False,
        trainable=("predictors",),
        kernel=KernelChoice("backend", MLP_SPARSITY_BACKENDS),
    )
    capabilities: ClassVar[Capability] = Capability.ACTIVATIONS
    order: ClassVar[int] = 1
    config: DejaVuConfig
    fitted: torch.Tensor

    def __init__(self, config: DejaVuConfig, adapter: Any) -> None:
        super().__init__(config, adapter)
        mlps = gated_mlps(adapter, config)
        self.layers = list(mlps)
        self._k = {i: kept_channels(ffn_width(m), config.sparsity) for i, m in mlps.items()}
        dtype = getattr(torch, config.predictor_dtype)
        predictors = nn.ModuleDict()
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(config.seed)
            for index, mlp in mlps.items():
                width = getattr(mlp.gate, "in_features", None)
                d = width if isinstance(width, int) else adapter.hidden_size
                predictor = nn.Sequential(
                    nn.Linear(d, config.rank, bias=False), nn.Linear(config.rank, ffn_width(mlp), bias=False)
                )
                predictors[str(index)] = predictor.to(device=adapter.layer_device(index), dtype=dtype)
        self.predictors = predictors
        self.register_buffer("fitted", torch.tensor(False))
        self._patches: list[Any] = []
        self._sources: dict[int, Tensor] = {}  # the lookahead norm outputs of the running layer call
        self._records: list[_Record] | None = None

    # ---- attach / detach -----------------------------------------------------------------------------------------
    def attach(self, model: nn.Module, adapter: Any, context: Any = None) -> None:
        self.config.validate(adapter)
        self.detach()  # the fit reads the dense MLPs
        if self.config.fit_steps > 0 and not bool(self.fitted):
            self.fit(token_batches(calibration_source(context, self.name)))
        self.install()

    def install(self) -> None:
        """(Re)install the patched MLPs and the hooks the predictors read."""
        self.detach()
        backend = MLP_SPARSITY_BACKENDS.get(self.config.backend)()
        policy = self.config.policy
        for index in self.layers:
            layer = self.adapter.layer(index)
            mlp = GatedMlp(getattr(layer, self.config.mlp), self.config)
            if self.config.predictor_input == "lookahead":
                self._patches.append(self._norm(layer, index).register_forward_hook(partial(self._keep_source, index)))
            self._patches.append(MlpPatch(mlp, partial(self._rule, index), backend.ops(mlp), policy))
            self._patches.append(mlp.module.register_forward_pre_hook(partial(self._record, index, mlp)))
        self._patches += track_layer_tokens(self.adapter.layers)

    def detach(self) -> None:
        for patch in self._patches:
            patch.remove()
        self._patches = []
        self._sources.clear()

    @property
    def summary(self) -> dict[str, Any]:
        """Target sparsity, number of sparsified MLPs, whether the predictors are fitted and installed."""
        return {
            "target": self.config.sparsity,
            "layers": len(self.layers),
            "fitted": bool(self.fitted),
            "installed": bool(self._patches),
        }

    # ---- prediction ----------------------------------------------------------------------------------------------
    def predictor(self, layer: int) -> nn.Module:
        """The predictor of a 1-indexed layer."""
        return self.predictors[str(layer)]

    def kept(self, layer: int) -> int:
        """Channels computed per token in ``layer``."""
        return self._k[layer]

    def scores(self, layer: int, source: Tensor) -> Tensor:
        """The predictor's channel scores for ``source`` (the MLP input or the lookahead norm output)."""
        predictor = self.predictor(layer)
        weight = cast(Tensor, predictor[0].weight)  # type: ignore[index]
        return cast(Tensor, predictor(source.to(weight.dtype)))

    def _norm(self, layer: nn.Module, index: int) -> nn.Module:
        norm = getattr(layer, self.config.lookahead_norm, None)
        if not isinstance(norm, nn.Module):
            raise UnsupportedModelError(f"layer {index} has no module {self.config.lookahead_norm!r} to look ahead")
        return norm

    def _keep_source(self, index: int, _module: nn.Module, _args: Any, output: Any) -> None:
        self._sources[index] = _first(output)

    def _source(self, index: int, x: Tensor) -> Tensor:
        if self.config.predictor_input == "mlp":
            return x
        source = self._sources.get(index)
        if source is None or source.shape[:-1] != x.shape[:-1]:
            raise UnsupportedModelError(
                f"layer {index}: the lookahead input {self.config.lookahead_norm!r} did not run before the MLP"
            )
        return source

    def _rule(self, index: int, mlp: GatedMlp, x: Tensor, ops: MlpOps, mode: str) -> Tensor:
        source, k = self._source(index, x), self._k[index]
        if x.shape[:-1].numel() <= PREFILL_CHUNK:  # decoding and short prefills: one piece
            return dejavu_mlp(mlp, x, ops, mode, self.scores(index, source), k)
        flat, flat_source = x.reshape(-1, x.shape[-1]), source.reshape(-1, source.shape[-1])
        keep = torch.cat([topk_mask(self.scores(index, part), k) for part in flat_source.split(PREFILL_CHUNK)])
        if mode == "second_half":  # positional: built on the mask of all tokens
            keep = dense_first_half(keep.reshape(*x.shape[:-1], -1)).reshape(flat.shape[0], -1)
        pieces = zip(flat.split(PREFILL_CHUNK), keep.split(PREFILL_CHUNK))
        return torch.cat([kept_mlp(mlp, part, ops, mask) for part, mask in pieces]).reshape(*x.shape[:-1], -1)

    def _record(self, index: int, mlp: GatedMlp, _module: nn.Module, args: tuple[Any, ...]) -> None:
        if self._records is None or not args:
            return
        x = cast(Tensor, args[0])
        labels = dejavu_labels(mlp, x.detach(), self._k[index])
        with torch.enable_grad():
            scores = self.scores(index, self._source(index, x).detach())
        self._records.append(_Record(index, scores, labels))

    @contextmanager
    def recording(self) -> Iterator[list[_Record]]:
        """Collect the predictor scores (with gradient) and labels of every MLP call inside the block; the MLPs
        themselves run unchanged."""
        records: list[_Record] = []
        self._records = records
        try:
            yield records
        finally:
            self._records = None

    # ---- attach-time fit -----------------------------------------------------------------------------------------
    @torch.no_grad()
    def fit(self, batches: list[Tensor]) -> dict[int, float]:
        """Fit every predictor on the dense model's activations of ``batches``, layer by layer; returns the final
        loss per layer and sets :attr:`fitted`."""
        config = self.config
        mlps = gated_mlps(self.adapter, config)
        losses: dict[int, float] = {}
        calibration = LayerwiseCalibration(
            self.adapter, batches, propagate="original", layers=self.layers, offload=config.offload
        )
        for step in calibration:
            mlp = mlps[step.index]
            inputs: list[Tensor] = []
            sources: list[Tensor] = []
            store = torch.device("cpu") if config.offload else None
            handles = [mlp.module.register_forward_pre_hook(partial(_collect_input, inputs, store))]
            if config.predictor_input == "lookahead":
                norm = self._norm(step.layer, step.index)
                handles.append(norm.register_forward_hook(partial(_collect_output, sources, store)))
            try:
                step.forward()
            finally:
                for handle in handles:
                    handle.remove()
            device = next(mlp.module.parameters()).device
            x, source = _sample(inputs, sources, config.max_tokens, config.seed + step.index, device)
            labels = dejavu_labels(mlp, x, self._k[step.index])
            losses[step.index] = fit_predictor(
                self.predictor(step.index),
                source,
                labels,
                steps=config.fit_steps,
                lr=config.fit_lr,
                batch=config.fit_batch,
                seed=config.seed + step.index,
            )
        self.fitted.fill_(True)
        return losses


def _collect_input(store: list[Tensor], device: torch.device | None, _module: nn.Module, args: tuple[Any, ...]) -> None:
    store.append(_flat(args[0], device))


def _collect_output(
    store: list[Tensor], device: torch.device | None, _module: nn.Module, _args: Any, output: Any
) -> None:
    store.append(_flat(_first(output), device))


def _flat(x: Tensor, device: torch.device | None) -> Tensor:
    """``x`` as ``[tokens, dim]``, moved to ``device`` (``None`` keeps it where it is)."""
    x = x.detach().reshape(-1, x.shape[-1])
    return x if device is None else x.to(device)


def _sample(
    inputs: list[Tensor], sources: list[Tensor], max_tokens: int, seed: int, device: torch.device
) -> tuple[Tensor, Tensor]:
    """A seeded random subset of at most ``max_tokens`` of the collected tokens, as ``(inputs, sources)`` on
    ``device`` (the sources are the inputs when none were collected). Empties both lists, so that only the subset
    outlives the call."""
    x = torch.cat(inputs)
    source = torch.cat(sources) if sources else None
    inputs.clear()
    sources.clear()
    generator = torch.Generator(device="cpu").manual_seed(seed)
    keep = torch.randperm(x.shape[0], generator=generator)[:max_tokens].to(x.device)
    x = x[keep].to(device)
    return x, (x if source is None else source[keep].to(device))


@register_objective("dejavu", default_for=(DejaVuTechnique.name,))
@dataclass
class DejaVuObjective(TechniqueObjective[DejaVuTechnique]):
    """Train the Deja Vu predictors on the binary cross-entropy against the dense model's channel labels.

    Every step runs the base model once with the technique attached; each MLP call contributes the cross-entropy
    of its predictor's scores (from the detached hidden state) against :func:`dejavu_labels`, averaged over the
    calls. Only the predictors train. The metric ``recall`` is :func:`predictor_recall` averaged over the calls.
    Padding positions are dropped.

    Attributes:
        technique: Name of the attached Deja Vu technique.
    """

    defaults: ClassVar[OptimDefaults] = OptimDefaults(lr=1e-3, warmup_steps=20)
    technique: str = DejaVuTechnique.name

    def teacher_request(self, model: Any) -> None:
        return None

    def loss(self, model: Any, batch: Batch, teacher: TeacherOutputs | None) -> ObjectiveOutput:
        dejavu = self.trained(model)
        with dejavu.recording() as records, torch.no_grad():
            model.base_model(input_ids=batch.input_ids, attention_mask=batch.attention_mask, use_cache=False)
        if not records:
            raise UnsupportedModelError("no MLP call was recorded: is the Deja Vu technique installed?")
        valid = None if batch.attention_mask is None else batch.attention_mask.reshape(-1).bool()
        losses, recalls = [], []
        for record in records:
            scores = record.scores.reshape(-1, record.scores.shape[-1])
            labels = record.labels.reshape(-1, record.labels.shape[-1])
            if valid is not None and valid.numel() == scores.shape[0]:
                scores, labels = scores[valid.to(scores.device)], labels[valid.to(labels.device)]
            losses.append(F.binary_cross_entropy_with_logits(scores.float(), labels.float()))
            recalls.append(predictor_recall(scores, labels, dejavu.kept(record.layer)))
        device = losses[0].device
        loss = torch.stack([value.to(device) for value in losses]).mean()
        recall = torch.stack([value.to(device) for value in recalls]).mean()
        dejavu.fitted.fill_(True)
        return ObjectiveOutput(loss, {"recall": recall.detach()})

    def evaluate(self, model: Any, batch: Batch, teacher: TeacherOutputs | None) -> dict[str, Any]:
        output = self.loss(model, batch, teacher)
        return {"loss": output.loss.detach(), **output.metrics}
