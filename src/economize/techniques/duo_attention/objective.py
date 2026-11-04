"""``duo_attention_gates``: train DuoAttention's head pattern, and :func:`train_pattern`, which writes it.

The objective follows the reference's ``duo_attn/train.py``. Every KV head of every gated layer gets a gate
``alpha`` (initialized to 1, clipped to ``[0, 1]`` before each step) and outputs ``alpha * full + (1 - alpha) *
streaming`` (see :mod:`.gates`). The base model is frozen. The loss compares the final normed hidden states of the
gated model with those of the unmodified model at the supervised positions (the batch's ``loss_mask``; for
:class:`~economize.train.data.PasskeyRetrieval` the questions and answers) and adds an L1 penalty on the gates::

    loss = sum_{supervised t} mean_d (h_gated[t, d] - h_full[t, d])**2 / #supervised + reg_weight * mean |alpha|

Defaults and their sources (the reference's ``scripts/train.sh``, which produced its released patterns, unless
noted): ``sinks`` 128 and ``recent`` 256 (``--sink_size``, ``--recent_size``), ``reg_weight`` 0.05, ``initial_value``
1.0 (``train.py`` argparse default), AdamW at ``lr`` 0.02 without weight decay and with ``torch``'s default betas
(``train.py``: ``AdamW(..., lr=args.lr, weight_decay=0)``), no gradient clipping, the ``"trapezoid"`` schedule with
floor 0.1 (``train.py``'s ``LambdaLR``), 2000 steps, batch size 1 and seed 42 (:func:`train_pattern`). Contexts of 1000
to 32000 tokens come from the data (:class:`~economize.train.data.PasskeyRetrieval`).

Differences from the reference, all documented where they arise: the streaming mask is token-exact (the reference's
default kernel masks 128-token blocks), the gates are ``float32`` and mixed in ``float32``, the forward pass and the
penalty use the gates without a further clamp (the clip before each step already keeps them in ``[0, 1]``, and in
recent ``torch`` a clamp has zero gradient at its bounds, which would freeze gates that start at exactly 1), the two
models run as two sequential passes (the reference duplicates the batch), sliding-window and chunked layers get no
gates, and every layer runs under activation checkpointing without splitting the sequence.

Cost: with ``sdpa`` (or the ``"economize"`` kernel) both passes hand the attention band masks
(:class:`~economize.attention.BandMask`) in place of dense ``[Q, K]`` masks: full-attention layers run one
``is_causal`` call, sliding-window layers and the streaming heads attend block by block over the keys they can see.
The outputs are those of the dense masks up to floating-point rounding, and the attention cost of the sliding and
streaming layers grows linearly with the context instead of quadratically. Other implementations get dense masks.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
from torch import nn

from ...errors import ConfigError
from ...train.objective import (
    Batch,
    Objective,
    ObjectiveOutput,
    OptimDefaults,
    ParamGroup,
    TeacherOutputs,
    TeacherRequest,
    register_objective,
)
from ...validation import check_number, check_positive
from .gates import GatedPass, HeadGates, gated_layers
from .pattern import write_pattern
from .technique import DuoAttentionConfig

if TYPE_CHECKING:
    from ...core.model import EfficientModel
    from ...train.loop import TrainConfig, TrainResult

__all__ = ["DuoAttentionGates", "TrainedPattern", "train_pattern"]


@register_objective("duo_attention_gates")
@dataclass
class DuoAttentionGates(Objective):
    """DuoAttention's gate training (``duo_attn/train.py``): distill the full model's final states into a model
    whose heads mix full and streaming attention, with an L1 penalty pushing the gates to 0.

    Attributes:
        sinks: Positions every streaming head sees from the start (reference ``--sink_size 128``).
        recent: Most recent positions a streaming head sees (``--recent_size 256``).
        reg_weight: Weight of the L1 penalty, the mean absolute gate value (``--reg_weight 0.05``).
        initial_value: Starting value of every gate (``--initial_value 1.0``).
        layers: 1-indexed layers to gate (empty: every full-attention layer that owns its K/V, which leaves out the
            sliding-window layers of hybrid models).
        output_projection: Attribute name of the attention's output projection, whose input is mixed.
        checkpoint: Recompute every layer in the backward pass (activation checkpointing, as the reference).

    After training, :attr:`gates` holds the trained gates (also after :meth:`teardown`).
    """

    sinks: int = 128
    recent: int = 256
    reg_weight: float = 0.05
    initial_value: float = 1.0
    layers: tuple[int, ...] = ()
    output_projection: str = "o_proj"
    checkpoint: bool = True
    gates: HeadGates | None = field(default=None, init=False, repr=False, compare=False)
    _pass: GatedPass | None = field(default=None, init=False, repr=False, compare=False)

    defaults = OptimDefaults(
        lr=0.02, weight_decay=0.0, betas=(0.9, 0.999), warmup_steps=0, schedule="trapezoid", min_lr_ratio=0.1,
        grad_clip=None,
    )

    def __post_init__(self) -> None:
        check_positive("sinks", self.sinks)
        check_positive("recent", self.recent)
        check_number("reg_weight", self.reg_weight, low=0)
        check_number("initial_value", self.initial_value, low=0, high=1)
        self.layers = tuple(int(layer) for layer in self.layers)

    # ---- lifecycle ---------------------------------------------------------------------------------------------
    def setup(self, model: EfficientModel) -> None:
        attached = [name for name, _ in model.techniques.items()]
        if attached:
            raise ConfigError(
                "duo_attention_gates trains on the bare model: wrap it with economize.apply(hf_model) and attach "
                f"no technique (attached: {attached})"
            )
        self.gates = HeadGates(model.adapter, gated_layers(model.adapter, self.layers), self.initial_value)
        self._pass = GatedPass(
            model.runner, self.gates, sinks=self.sinks, recent=self.recent,
            output_projection=self.output_projection, checkpoint_layers=self.checkpoint,
        )

    def teardown(self, model: EfficientModel) -> None:
        if self._pass is not None:
            self._pass.remove()
            self._pass = None

    def modules(self, model: EfficientModel) -> list[nn.Module]:
        return [self._trained()]

    def parameter_groups(self, model: EfficientModel) -> list[ParamGroup]:
        return [ParamGroup("gates", list(self._trained().parameters()), "head", weight_decay=0.0)]

    def begin_step(self, step: int, total_steps: int) -> None:
        self._trained().clamp_()

    def teacher_request(self, model: EfficientModel) -> TeacherRequest | None:
        return None  # the objective runs the full model itself, keeping its head outputs

    # ---- loss ----------------------------------------------------------------------------------------------------
    def loss(self, model: EfficientModel, batch: Batch, teacher: TeacherOutputs | None) -> ObjectiveOutput:
        if self._pass is None:
            raise ConfigError("duo_attention_gates.loss called before setup")
        gates = self._trained()
        supervised = _supervised(batch)
        full = self._pass.teacher(batch.input_ids)
        mixed = self._pass.mixed(batch.input_ids, full)
        supervised = supervised.to(mixed.device)
        error = (mixed.float() - full.final.float()).pow(2).mean(-1)
        distill = (error * supervised).sum() / supervised.sum().clamp_min(1)
        values = gates.flat()
        reg = values.abs().mean().to(distill.device)  # duo_attn/loss.py l1_loss on the raw gates
        loss = distill + self.reg_weight * reg
        with torch.no_grad():
            metrics = {
                "distill": distill.detach(),
                "reg": reg.detach(),
                "gate_mean": values.detach().mean(),
                "gate_low": (values.detach() < 0.5).float().mean(),
            }
        return ObjectiveOutput(loss, metrics)

    def _trained(self) -> HeadGates:
        if self.gates is None:
            raise ConfigError("duo_attention_gates has no gates before setup")
        return self.gates


def _supervised(batch: Batch) -> torch.Tensor:
    """``[B, T]`` float mask of the compared positions: the loss mask within the attention mask. The pass runs every
    row from position 0, so padding must be on the right."""
    ids = batch.input_ids
    mask = torch.ones_like(ids, dtype=torch.float32)
    if batch.attention_mask is not None:
        valid = batch.attention_mask.to(torch.long)
        if bool((valid[:, 1:] > valid[:, :-1]).any()):
            raise ConfigError("duo_attention_gates needs right-padded batches (a 0 before a 1 in attention_mask)")
        mask = mask * valid
    if batch.loss_mask is not None:
        mask = mask * batch.loss_mask.to(torch.float32)
    return mask


@dataclass
class TrainedPattern:
    """What :func:`train_pattern` returns.

    Attributes:
        scores: Trained gate values per decoder layer and KV head, clipped to ``[0, 1]`` (``1.0`` for layers without
            gates), as the reference's ``full_attention_heads.tsv`` holds them.
        layers: The gated layers (1-indexed).
        sinks: Sinks the streaming heads saw in training.
        recent: Recent window the streaming heads saw in training.
        path: The written pattern directory (``None`` when none was asked for).
        train: The training run's result.
    """

    scores: list[list[float]]
    layers: tuple[int, ...]
    sinks: int
    recent: int
    path: Path | None
    train: TrainResult

    def config(self, **fields: Any) -> DuoAttentionConfig:
        """A :class:`DuoAttentionConfig` with these gates inline and the training's sinks, window and layers
        (``fields`` override them, e.g. ``sparsity``)."""
        settings: dict[str, Any] = {"sinks": self.sinks, "recent": self.recent, "layers": self.layers}
        return DuoAttentionConfig(pattern=tuple(tuple(row) for row in self.scores), **{**settings, **fields})


def train_pattern(
    model: EfficientModel | nn.Module,
    data: Any,
    out_dir: str | Path | None = None,
    *,
    objective: DuoAttentionGates | None = None,
    config: TrainConfig | None = None,
    settings: Mapping[str, Any] | None = None,
    **overrides: Any,
) -> TrainedPattern:
    """Train DuoAttention's gates on ``model`` and write the pattern directory the reference writes.

    ``out_dir`` receives ``full_attention_heads.tsv`` (one row per decoder layer, one gate value per KV head) and a
    ``config.json`` with ``sink_size``, ``recent_size``, the gated ``layers`` and the training settings, so that
    ``DuoAttentionConfig.from_pattern(out_dir)`` (or ``DuoAttentionConfig(pattern=out_dir)``) deploys it.

    Args:
        model: The base model, bare or wrapped by ``economize.apply`` without techniques. It is left unchanged.
        data: Batches with a loss mask on the compared positions, normally a
            :class:`~economize.train.data.PasskeyRetrieval`; anything :func:`~economize.train.fit` accepts.
        out_dir: The pattern directory to write (``None``: return the scores only).
        objective: The objective (default ``DuoAttentionGates()``, the reference's settings).
        config: Run settings (default: the reference's 2000 steps, batch size 1 and seed 42, and the objective's
            optimizer defaults).
        settings: Further ``config.json`` entries (e.g. the data's settings).
        overrides: :class:`~economize.train.TrainConfig` fields (``steps``, ``lr``, ``output_dir``, ...).

    Returns:
        The trained scores, the written directory and the training result.
    """
    from ...core.model import EfficientModel, apply
    from ...train.loop import TrainConfig, fit

    wrapped = model if isinstance(model, EfficientModel) else apply(model)
    objective = DuoAttentionGates() if objective is None else objective.with_options()  # a fresh copy
    run = replace(config or TrainConfig(steps=2000, batch_size=1, seed=42), **overrides)
    result = fit(wrapped, data, objective, config=run)
    gates = objective._trained()
    scores = [[min(max(v, 0.0), 1.0) for v in row] for row in gates.rows()]
    path = None
    if out_dir is not None:
        options = run.optimizer_settings(objective.defaults)
        record = {
            "model_name": getattr(getattr(wrapped.base_model, "config", None), "_name_or_path", None),
            "lr": options.lr,
            "num_steps": run.steps,
            "batch_size": run.batch_size,
            "reg_weight": objective.reg_weight,
            "initial_value": objective.initial_value,
            "seed": run.seed,
            "supervision": "distill",
            "trainer": "economize.duo_attention_gates",
            **dict(settings or {}),
        }
        path = write_pattern(out_dir, scores, sinks=objective.sinks, recent=objective.recent, layers=gates.layers,
                             settings=record)
    return TrainedPattern(scores, gates.layers, objective.sinks, objective.recent, path, result)
