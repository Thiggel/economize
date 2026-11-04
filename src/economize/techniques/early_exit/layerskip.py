"""``layerskip``: train the model so that its intermediate layers can exit early (LayerSkip, Elhoushi et al., 2024).

Two ingredients, both following the paper's formulas (blocks are 1-indexed here, the paper's layer ``l`` is block
``l + 1``):

- Layer dropout. Block ``l`` of each sequence is skipped (its output replaced by its input) with probability
  ``p_l = dropout_max * (exp((l - 1) ln 2 / (L - 1)) - 1)``, from 0 for the first block to ``dropout_max`` for the
  last.
- Early-exit loss. The cross-entropy of every enabled exit (the model's own final norm and LM head applied after
  block ``l``) is weighted by ``e(l) = exit_scale * sum_{i < l} i`` for ``l < L`` and
  ``e(L) = L - 1 + exit_scale * sum_{i < L - 1} i``, normalized over the enabled exits. With the rotational
  curriculum, step ``t`` enables the exits of blocks ``l`` with ``(l - 1 - t) mod rotation == 0`` and the last block.

The model's own weights (or its PEFT adapters) are trained; the decoding-time side is a self-speculative early-exit
drafter that reads the same LM head (:class:`~economize.techniques.early_exit.EarlyExitConfig` with ``head="shared"``).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, ClassVar, Literal

import torch
from torch.utils.checkpoint import checkpoint

from ...train.objective import (
    Batch,
    Objective,
    ObjectiveOutput,
    OptimDefaults,
    ParamGroup,
    TeacherOutputs,
    register_objective,
)
from ...train.objectives.base import BaseParameters, base_parameter_groups
from ...train.objectives.ce import next_token_nll
from ...validation import check_choice, check_positive

if TYPE_CHECKING:
    from torch.utils.hooks import RemovableHandle

    from ...core.model import EfficientModel

__all__ = ["LayerSkip", "exit_weights", "layer_dropout_rates"]


def layer_dropout_rates(num_layers: int, dropout_max: float) -> list[float]:
    """Skip probability of blocks ``1..L`` (exponential in depth, 0 for the first block)."""
    if num_layers == 1:
        return [0.0]
    return [dropout_max * (math.exp((l - 1) * math.log(2) / (num_layers - 1)) - 1) for l in range(1, num_layers + 1)]


def exit_weights(num_layers: int, exit_scale: float) -> list[float]:
    """Unnormalized loss weight of the exit after blocks ``1..L``."""
    weights = [exit_scale * (l - 1) * l / 2 for l in range(1, num_layers)]
    weights.append(num_layers - 1 + exit_scale * (num_layers - 2) * (num_layers - 1) / 2)
    return weights


@register_objective("layerskip")
@dataclass
class LayerSkip(Objective):
    """Early-exit loss with layer dropout.

    Attributes:
        exit_layers: Blocks with an exit (``None``: every block). The last block always has one.
        exit_scale: ``e_scale`` of the exit weights.
        dropout_max: Skip probability of the last block.
        curriculum: ``"rotational"`` (a rotating subset of exits per step) or ``"none"`` (every exit, every step).
        rotation: Period ``R`` of the rotational curriculum.
        parameters: Which base parameters are trained (see :data:`~economize.train.objectives.base.BaseParameters`).
    """

    defaults: ClassVar[OptimDefaults] = OptimDefaults(lr=2e-5, warmup_steps=100)
    exit_layers: tuple[int, ...] | None = None
    exit_scale: float = 0.1
    dropout_max: float = 0.1
    curriculum: Literal["rotational", "none"] = "rotational"
    rotation: int = 4
    parameters: BaseParameters = "trainable"
    _step: int = field(default=0, init=False, repr=False)
    _dropping: bool = field(default=False, init=False, repr=False)
    _handles: list[RemovableHandle] = field(default_factory=list, init=False, repr=False)
    _captured: dict[int, torch.Tensor] = field(default_factory=dict, init=False, repr=False)
    _was_training: bool = field(default=False, init=False, repr=False)

    def __post_init__(self) -> None:
        check_choice("curriculum", self.curriculum, ("rotational", "none"))
        check_positive("rotation", self.rotation)

    # ---- objective interface -------------------------------------------------------------------------------------
    def parameter_groups(self, model: EfficientModel) -> list[ParamGroup]:
        return base_parameter_groups(model, self.parameters)

    def teacher_request(self, model: EfficientModel) -> None:
        return None

    def setup(self, model: EfficientModel) -> None:
        self._install(model)
        self._was_training = model.base_model.training
        model.base_model.train()

    def teardown(self, model: EfficientModel) -> None:
        self._remove()
        model.base_model.train(self._was_training)

    def begin_step(self, step: int, total_steps: int) -> None:
        self._step = step

    def loss(self, model: EfficientModel, batch: Batch, teacher: TeacherOutputs | None) -> ObjectiveOutput:
        num_layers = model.adapter.num_layers
        if not self._handles and (self.dropout_max > 0 or len(self._exits(num_layers)) > 1):
            raise RuntimeError("layerskip.loss called before setup()")
        return self._loss(model, batch, dropout=True, enabled=self._enabled(num_layers))

    @torch.no_grad()
    def evaluate(self, model: EfficientModel, batch: Batch, teacher: TeacherOutputs | None) -> dict[str, Any]:
        """Next-token NLL of every exit, without layer dropout (the exits' hooks are installed for the call when
        :meth:`setup` has not run)."""
        temporary = not self._handles
        if temporary:
            self._install(model)
        try:
            output = self._loss(model, batch, dropout=False, enabled=self._exits(model.adapter.num_layers))
        finally:
            if temporary:
                self._remove()
        return {"loss": output.loss, **output.metrics}

    # ---- internals ---------------------------------------------------------------------------------------------
    def _install(self, model: EfficientModel) -> None:
        """Forward hooks on the blocks that drop out or have an exit."""
        adapter = model.adapter
        rates = layer_dropout_rates(adapter.num_layers, self.dropout_max)
        exits = self._exits(adapter.num_layers)
        for layer in range(1, adapter.num_layers + 1):
            capture = layer in exits and layer < adapter.num_layers
            if rates[layer - 1] > 0 or capture:
                hook = self._hook(layer, rates[layer - 1], capture)
                self._handles.append(adapter.layer(layer).register_forward_hook(hook, with_kwargs=True))

    def _remove(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        self._captured.clear()

    def _exits(self, num_layers: int) -> tuple[int, ...]:
        layers = range(1, num_layers + 1) if self.exit_layers is None else self.exit_layers
        return tuple(sorted({*layers, num_layers}))

    def _enabled(self, num_layers: int) -> tuple[int, ...]:
        exits = self._exits(num_layers)
        if self.curriculum == "none":
            return exits
        return tuple(l for l in exits if l == num_layers or (l - 1 - self._step) % self.rotation == 0)

    def _hook(self, layer: int, rate: float, capture: bool) -> Any:
        def hook(module: torch.nn.Module, args: tuple, kwargs: dict, output: Any) -> Any:
            hidden_in = args[0] if args else kwargs["hidden_states"]
            out = output[0] if isinstance(output, tuple) else output
            if self._dropping and rate > 0:
                keep = torch.rand(out.shape[0], device=out.device) >= rate
                out = torch.where(keep[:, None, None], out, hidden_in)
            if capture:
                self._captured[layer] = out
            return (out, *output[1:]) if isinstance(output, tuple) else out

        return hook

    def _loss(self, model: EfficientModel, batch: Batch, *, dropout: bool, enabled: tuple[int, ...]) -> ObjectiveOutput:
        adapter = model.adapter
        num_layers = adapter.num_layers
        self._captured.clear()
        self._dropping = dropout
        try:
            out = model.base_model(input_ids=batch.input_ids, attention_mask=batch.attention_mask, use_cache=False)
        finally:
            self._dropping = False
        scale = exit_weights(num_layers, self.exit_scale)
        total = sum(scale[l - 1] for l in enabled)
        metrics: dict[str, torch.Tensor | float] = {}
        loss = torch.zeros((), device=out.logits.device)
        for layer in enabled:
            if layer == num_layers:
                nll = next_token_nll(out.logits, batch)
            else:
                hidden = self._captured[layer]
                nll = checkpoint(lambda h: next_token_nll(adapter.logits(h), batch), hidden, use_reentrant=False)
            metrics[f"nll{layer}"] = nll.detach()
            if scale[layer - 1] > 0:
                loss = loss + scale[layer - 1] / total * nll
        self._captured.clear()
        return ObjectiveOutput(loss, metrics)
