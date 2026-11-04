"""Hidden Transfer (Wu et al., 2024): pseudo hidden states of future tokens at intermediate layers.

After a forward pass over the sequence up to position ``j``, a linear map ``W_i`` (``d x d``, one per future step
``i = 1..k``) turns the hidden state of ``j`` after layer ``t_i`` into a pseudo hidden state of the not yet known
token at position ``j + i``. The pseudo state continues through layers ``t_i + 1..L`` and the model's LM head, which
draft token ``j + i + 1``; it attends to the states of the positions before it (the paper's no-mask variant, which
beats masking the other pseudo states out). The model verifies the drafts, so greedy decoding stays lossless. The
maps are trained with the base model frozen, by ``sum_q KL(P~_q, P_{q+i})`` against the model's distributions,
where the pseudo state at ``q`` sees the real states of the positions before it. The paper's 13B example transfers
the next three tokens at layers 25, 30 and 35 of 40, the default :attr:`HiddenTransferConfig.transfer_fractions`.

In economize, Hidden Transfer is a drafter (:class:`HiddenTransferTechnique`): after an exact pass that ended at
``n``, draft step ``i`` runs layers ``t_i + 1..L`` on ``W_i h^{t_i}_n`` at position ``n + i`` against the cache,
which holds the exact K/V up to ``n`` and the K/V the earlier draft steps wrote. The objective
``hidden_transfer_kl`` (:class:`HiddenTransferDistill`) is :class:`~economize.train.objectives.StepDistill` over the
``k`` steps.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from itertools import pairwise
from typing import TYPE_CHECKING, Any, ClassVar, cast

import torch
from torch import nn

from ...core.config import TechniqueConfig
from ...core.metadata import Family, Saving, TechniqueInfo
from ...core.technique import Capability, Technique, register_technique
from ...decoding.config import DecodeConfig
from ...decoding.cost import LayerFootprint
from ...decoding.protocols import Draft, Drafter
from ...errors import ConfigError
from ...modules.dtype import COPY_DTYPES, copy_dtype_of
from ...train.objective import register_objective
from ...train.objectives.steps import StepDistill
from ...validation import check_choice, check_number
from .._drafting import StepDrafterDefaults, take_rows

if TYPE_CHECKING:
    from ...arch import ArchAdapter
    from ...core.model import EfficientModel
    from ...decoding.request import DraftRequest
    from ...runtime import ExactPass, KVCache, LayerContext, LayerRunner
    from ...train.objective import Batch, ParamGroup, TeacherOutputs, TeacherRequest

__all__ = [
    "TRAIN_CONTEXTS",
    "HiddenTransferConfig",
    "HiddenTransferDistill",
    "HiddenTransferDrafter",
    "HiddenTransferTechnique",
]

#: What a pseudo state attends to in training: the real states of every earlier position (``"teacher"``, the
#: paper's) or, as in decoding, the exact states up to the source position and the earlier pseudo states (``"draft"``).
TRAIN_CONTEXTS = ("teacher", "draft")


@dataclass(frozen=True)
class HiddenTransferConfig(TechniqueConfig):
    """Hyperparameters of Hidden Transfer.

    Attributes:
        transfer_layers: ``t_1..t_k`` (1-indexed, nondecreasing, below ``L``): step ``i`` maps the state after layer
            ``t_i`` and runs layers ``t_i + 1..L``. ``None``: ``round(f * L)`` for every ``f`` of
            :attr:`transfer_fractions`.
        transfer_fractions: Relative depths of the transfer layers when :attr:`transfer_layers` is ``None`` (the
            paper's 25, 30 and 35 of 40 layers).
        train_context: An entry of :data:`TRAIN_CONTEXTS`.
        param_dtype: Storage and compute dtype of the maps (``"float32"`` or ``"model"``).
    """

    technique: ClassVar[str] = "hidden_transfer"

    transfer_layers: tuple[int, ...] | None = None
    transfer_fractions: tuple[float, ...] = (0.625, 0.75, 0.875)
    train_context: str = "teacher"
    param_dtype: str = "float32"

    def __post_init__(self) -> None:
        if self.transfer_layers is not None:
            layers = tuple(self.transfer_layers)
            if not layers or not all(isinstance(t, int) and not isinstance(t, bool) for t in layers):
                raise ConfigError(f"transfer_layers must be a non-empty tuple of layer numbers, got {layers!r}")
            object.__setattr__(self, "transfer_layers", layers)
        fractions = tuple(self.transfer_fractions)
        if not fractions:
            raise ConfigError("transfer_fractions must not be empty")
        for f in fractions:
            check_number("transfer_fractions", f, low=0, high=1, closed_low=False, closed_high=False)
        object.__setattr__(self, "transfer_fractions", fractions)
        check_choice("train_context", self.train_context, TRAIN_CONTEXTS)
        check_choice("param_dtype", self.param_dtype, COPY_DTYPES)

    def resolved_layers(self, num_layers: int) -> tuple[int, ...]:
        """``t_1..t_k`` for a model of ``num_layers`` layers.

        Raises:
            ConfigError: A layer outside ``1..L-1`` or a decreasing sequence.
        """
        if self.transfer_layers is not None:
            layers = self.transfer_layers
        else:
            layers = tuple(min(num_layers - 1, max(1, round(f * num_layers))) for f in self.transfer_fractions)
        if not all(1 <= t < num_layers for t in layers):
            raise ConfigError(f"transfer_layers must lie in 1..{num_layers - 1}, got {layers}")
        if any(b < a for a, b in pairwise(layers)):
            raise ConfigError(f"transfer_layers must be nondecreasing, got {layers}")
        return layers

    def validate(self, adapter: ArchAdapter) -> None:
        self.resolved_layers(adapter.num_layers)


@register_technique
class HiddenTransferTechnique(StepDrafterDefaults, Technique):
    """The ``k`` transfer maps of Hidden Transfer and their drafter (see the module docstring).

    Parameters (float32 by default, on the device of the LM head): ``transfers.{i}.weight``, initialized to the
    identity (the paper gives no initialization; the identity makes step ``i`` start as the early exit of layer
    ``t_i``'s state through the remaining layers).
    """

    name = HiddenTransferConfig.technique
    config_class = HiddenTransferConfig
    info = TechniqueInfo(family=Family.DRAFTING, saves=Saving.LATENCY, lossless=True, trainable=("transfers",))
    capabilities = Capability.DRAFTER
    config: HiddenTransferConfig

    def __init__(self, config: HiddenTransferConfig, adapter: ArchAdapter) -> None:
        super().__init__(config, adapter)
        self.transfer_layers = config.resolved_layers(adapter.num_layers)
        d = adapter.hidden_size
        self.transfers = nn.ModuleList(nn.Linear(d, d, bias=False) for _ in self.transfer_layers)
        with torch.no_grad():
            for linear in self.transfers:
                cast(nn.Linear, linear).weight.copy_(torch.eye(d))
        self.to(device=adapter.head_device, dtype=copy_dtype_of(config.param_dtype, adapter.dtype))

    @property
    def num_steps(self) -> int:
        """``k``, the number of transfer maps (most draft tokens per cycle)."""
        return len(self.transfer_layers)

    def pseudo(self, step: int, hidden: torch.Tensor) -> torch.Tensor:
        """The pseudo states ``W_step h`` (``step`` is 1-based) of states ``[..., d]`` after layer ``t_step``, in
        the model's dtype and on the device of ``hidden``."""
        linear = cast(nn.Linear, self.transfers[step - 1])
        out = linear(hidden.to(linear.weight.device, linear.weight.dtype))
        return out.to(hidden.device, self.adapter.dtype)

    def run(self, step: int, runner: LayerRunner, pseudo: torch.Tensor, ctx: LayerContext) -> torch.Tensor:
        """Layers ``t_step + 1..L`` on pseudo states ``[B, S, d]``, writing their K/V; returns the last layer's
        output."""
        hidden, _ = runner.run(pseudo, ctx, self.transfer_layers[step - 1] + 1, self.adapter.num_layers)
        return hidden

    def logits(self, hidden: torch.Tensor) -> torch.Tensor:
        """Float logits of the model's final norm and LM head."""
        return self.adapter.logits(hidden).float()

    def footprint(self, step: int) -> LayerFootprint:
        """The layers draft step ``step`` (1-based) runs."""
        return LayerFootprint(self.transfer_layers[step - 1] + 1, self.adapter.num_layers)

    # ---- technique interface ----------------------------------------------------------------------------------
    def drafter(self, runner: LayerRunner) -> HiddenTransferDrafter:
        return HiddenTransferDrafter(self, runner)

    def decode_config(self, cycle_length: int | None = None, lossless: bool = True, **options: Any) -> DecodeConfig:
        """Cycles of ``cycle_length`` tokens (default ``k + 1``: one exact token and ``k`` drafts), verified
        (``lossless=True``, the paper's) or kept. ``options`` are further :class:`~economize.DecodeConfig`
        fields."""
        n = self.num_steps + 1 if cycle_length is None else cycle_length
        return DecodeConfig(cycle_length=n, lossless=lossless, **options)

    def extra_repr(self) -> str:
        return f"transfer_layers={self.transfer_layers}, train_context={self.config.train_context!r}"

    # ---- training (the TrainableDrafter protocol of economize.train) ----------------------------------------------
    def train_request(self, model: EfficientModel) -> TeacherRequest:
        """The outputs of the transfer layers and the teacher's cache."""
        from ...train.objective import TeacherRequest

        return TeacherRequest(hidden_layers=tuple(sorted(set(self.transfer_layers))), cache=True)

    def train_logits(self, model: EfficientModel, batch: Batch, teacher: TeacherOutputs) -> Mapping[int, torch.Tensor]:
        """Draft logits ``{i + 1: [B, T, V]}`` of steps ``i = 1..k`` in the paper's teacher context: position ``j``
        of key ``i + 1`` drafts token ``j + i + 1`` from the pseudo state ``W_i h^{t_i}_j`` at ``j + i``, which sees
        the real states of the positions before ``j + i``. The last ``i`` positions, whose pseudo state would lie
        beyond the sequence, repeat the position before them (the objective ignores them)."""
        length = batch.input_ids.shape[1]
        out: dict[int, torch.Tensor] = {}
        for i in range(1, self.num_steps + 1):
            if i >= length:
                break
            sources = torch.arange(0, length - i, device=batch.input_ids.device)
            hidden = self._teacher_step(model, teacher, sources, i)
            logits = self.logits(hidden)
            out[i + 1] = torch.cat([logits, logits[:, -1:].expand(-1, i, -1)], 1)
        return out

    # ---- training (the StepDrafter protocol of economize.train, objective step_distill) --------------------------
    def step_states(self, model: EfficientModel, batch: Batch, teacher: TeacherOutputs, starts: torch.Tensor,
                    steps: int) -> list[torch.Tensor]:
        """Last-layer outputs of draft steps ``1..steps``: step ``k`` runs layers ``t_k + 1..L`` on
        ``W_k h^{t_k}_s`` at ``s + k``. With ``train_context="teacher"`` it sees the real states of the positions
        before ``s + k`` (the paper's training); with ``"draft"`` the exact states up to ``s`` and the earlier steps'
        pseudo states, as in decoding."""
        if self.config.train_context == "teacher":
            return [self._teacher_step(model, teacher, starts, k) for k in range(1, steps + 1)]
        return self._draft_steps(model, teacher, starts, steps)

    def step_logits(self, states: torch.Tensor, step: int) -> torch.Tensor:
        """:meth:`logits` (every step reads the model's LM head)."""
        return self.logits(states)

    def step_limit(self) -> int | None:
        """One draft step per transfer map."""
        return self.num_steps

    def step_parameter_groups(self) -> list[ParamGroup]:
        """The transfer maps, as new modules."""
        from ...train.objective import ParamGroup

        return [ParamGroup(self.name, list(self.parameters()), "head")]

    def _source(self, teacher: TeacherOutputs, step: int, positions: torch.Tensor) -> torch.Tensor:
        """``W_step h^{t_step}`` at ``positions`` ``[S]``: ``[B, S, d]``."""
        hidden = teacher.hidden[self.transfer_layers[step - 1]]
        return self.pseudo(step, hidden[:, positions.to(hidden.device)])

    def _teacher_step(self, model: EfficientModel, teacher: TeacherOutputs, sources: torch.Tensor,
                      step: int) -> torch.Tensor:
        """Step ``step`` from sources ``[S]`` at ``sources + step``, attending to the teacher's K/V up to
        ``sources + step - 1`` (a one-step rollout from that start)."""
        from ...train.objective import teacher_cache
        from ...train.rollout import RolloutCache, RolloutContext

        cache = RolloutCache(teacher_cache(teacher, "HiddenTransferTechnique"))
        starts = sources + step - 1
        grid = (starts + 1)[None].expand(teacher.hidden[self.transfer_layers[0]].shape[0], -1)
        ctx = RolloutContext(self.adapter, cache, grid, grid, starts=starts, step=1)
        return self.run(step, model.runner, self._source(teacher, step, sources), ctx)

    def _draft_steps(self, model: EfficientModel, teacher: TeacherOutputs, starts: torch.Tensor,
                     steps: int) -> list[torch.Tensor]:
        """Steps ``1..steps`` as one rollout from ``starts``. Layer ``l`` runs at the steps ``k`` with
        ``t_k < l``, which are the first ones because the transfer layers are nondecreasing, so every layer's
        blocks of rollout K/V are its earlier steps' (the layout :class:`~economize.train.rollout.RolloutContext`
        masks)."""
        from ...train.objective import teacher_cache
        from ...train.rollout import RolloutCache, RolloutContext

        cache = RolloutCache(teacher_cache(teacher, "HiddenTransferTechnique"))
        rows = teacher.hidden[self.transfer_layers[0]].shape[0]
        outputs = []
        for k in range(1, steps + 1):
            grid = (starts + k)[None].expand(rows, -1)
            ctx = RolloutContext(self.adapter, cache, grid, grid, starts=starts, step=k)
            outputs.append(self.run(k, model.runner, self._source(teacher, k, starts), ctx))
        return outputs


class HiddenTransferDrafter(Drafter):
    """Drafts with the transfer maps of a :class:`HiddenTransferTechnique`.

    After every exact pass the drafter keeps every row's states after the transfer layers at its last processed
    token ``n``. Draft ``i`` (1-based) runs layers ``t_i + 1..L`` on ``W_i h^{t_i}_n`` at position ``n + i`` and is
    passed to :meth:`DraftRequest.extend`, which costs it by :meth:`HiddenTransferTechnique.footprint`. A cycle
    makes at most ``k`` drafts, so cycles longer than ``k + 1`` end early. The steps run eagerly (their input is
    not the drafted token, so the step-wise ``step`` interface does not apply).
    """

    def __init__(self, technique: HiddenTransferTechnique, runner: LayerRunner) -> None:
        self.technique = technique
        self.runner = runner
        self.source_layers = tuple(sorted(set(technique.transfer_layers)))
        self.layer_span = (min(technique.transfer_layers) + 1, technique.adapter.num_layers)
        self._cache: KVCache | None = None
        self._states: dict[int, torch.Tensor] | None = None

    def begin(self, runner: LayerRunner, cache: KVCache, batch_size: int) -> None:
        self.runner, self._cache, self._states = runner, cache, None

    def observe(self, exact: ExactPass, columns: torch.Tensor) -> None:
        states = {}
        for layer in self.source_layers:
            hidden = exact.hidden[layer]
            rows = torch.arange(hidden.shape[0], device=hidden.device)
            states[layer] = hidden[rows, columns.to(hidden.device)]
        self._states = states

    def reorder(self, rows: torch.Tensor) -> None:
        if self._states is not None:
            self._states = {layer: cast(torch.Tensor, take_rows(h, rows)) for layer, h in self._states.items()}

    def draft_capacity(self, cycle_length: int) -> int:
        return max(0, min(cycle_length - 1, self.technique.num_steps))

    def step(self, tokens: torch.Tensor, positions: torch.Tensor, valid: torch.Tensor | None) -> torch.Tensor:
        raise NotImplementedError("HiddenTransferDrafter drafts in propose(); its steps do not read the tokens")

    def propose(self, request: DraftRequest) -> Draft:
        if self._states is None or self._cache is None:
            raise RuntimeError("HiddenTransferDrafter.propose needs begin() and observe() first")
        technique = self.technique
        depth = min(max((n for n, a in zip(request.budgets, request.active) if a), default=0), technique.num_steps)
        footprint = request.footprint
        try:
            for i in range(1, depth + 1):
                if not request.drafting:
                    break
                valid = request.valid
                ctx = self.runner.context(self._cache, request.positions[:, None],
                                          None if valid is None else valid[:, None], trusted=True)
                source = self._states[technique.transfer_layers[i - 1]]
                hidden = technique.run(i, self.runner, technique.pseudo(i, source)[:, None], ctx)
                request.footprint = technique.footprint(i)
                request.extend(technique.logits(hidden)[:, 0])
        finally:
            request.footprint = footprint
        return request.draft()


@register_objective("hidden_transfer_kl", default_for=(HiddenTransferTechnique.name,))
@dataclass
class HiddenTransferDistill(StepDistill):
    """:class:`~economize.train.objectives.StepDistill` of the transfer maps over their ``k`` steps
    (``KL(model || draft)``, the paper's distillation loss; the base model stays frozen)."""

    technique: str | None = HiddenTransferTechnique.name
