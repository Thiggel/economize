"""FREE: fast and robust early exiting with a shallow-deep module and synchronized parallel decoding (Bae et al., 2023).

The model's first ``L_S`` layers form a shallow model with an exit, all ``L`` layers the deep model. A token whose
shallow confidence (the gap between the two largest probabilities of the exit's distribution) reaches the threshold
``lambda`` is emitted by the shallow model; tokens that exited stack up until one does not, and then the deep layers
run once, in parallel, over the stacked tokens and that token (synchronized parallel decoding). The stacked tokens get
their deep-layer K/V from this pass, so no hidden state is copied.

economize decodes FREE through its drafting cycle: the shallow model is an early-exit drafter
(:class:`FreeTechnique`, a subclass of :class:`~economize.techniques.early_exit.EarlyExitTechnique`), every draft is
kept (``lossless=False``), and the draft-length policy ``softmax_gap`` (:class:`ShallowExitStop`) ends the cycle at
the first token whose confidence falls below ``lambda``. The exact pass that follows is the synchronized pass and
emits that token from the deep model. :meth:`FreeTechnique.decode_config` builds the decoding options. The exact pass
also recomputes the shallow layers of its tokens, which FREE reuses (the layers are the same), so
:meth:`FreeTechnique.cost_summary` reports FREE's own cost (see :mod:`.drafted`).

The threshold follows the paper's adaptive estimator: a Beta mixture over the confidences of tokens whose shallow and
deep predictions agree and disagree, fitted by moments, gives the smallest ``lambda`` whose posterior probability of
agreement reaches ``zeta`` (:func:`bmm_threshold`; :func:`free_calibration_pairs` collects the pairs offline).

Training (objective ``free``, :class:`FreeObjective`) fine-tunes the model with the weighted cross-entropies of both
exits, ``alpha_S = L_S / (L + L_S)`` and ``alpha_L = L / (L + L_S)``, plus layerwise distillation
``(1 / L_S) sum_i MSE(H^i, H^{m(i)})`` from deep layers into the shallow ones, with the paper's dynamic monotone
mapping ``m`` (KD-dyna) or the fixed mappings KD-last and KD-unif. The deep side is detached.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, ClassVar, Literal

import torch

from ...core.metadata import Family, Saving, TechniqueInfo
from ...core.technique import register_technique
from ...decoding.config import DecodeConfig
from ...decoding.policies import POLICIES
from ...decoding.protocols import DraftPolicy
from ...errors import ConfigError
from ...train.objective import (
    Batch,
    ObjectiveOutput,
    OptimDefaults,
    ParamGroup,
    TeacherOutputs,
    TechniqueObjective,
    register_objective,
)
from ...train.objectives.base import BaseParameters, base_parameter_groups
from ...train.objectives.ce import next_token_nll
from ...validation import check_choice, check_positive, check_unit_interval
from ..early_exit.config import EarlyExitConfig
from ..early_exit.technique import EarlyExitTechnique
from .drafted import drafted_cost_summary

if TYPE_CHECKING:
    from torch.utils.hooks import RemovableHandle

    from ...core.model import EfficientModel

__all__ = [
    "KD_MAPPINGS",
    "FreeConfig",
    "FreeObjective",
    "FreeTechnique",
    "ShallowExitStop",
    "beta_moments",
    "bmm_threshold",
    "free_calibration_pairs",
    "kd_layer_map",
    "monotone_layer_map",
    "softmax_gap",
]

#: Layer mappings of the distillation term: ``dynamic`` (KD-dyna), ``last`` (KD-last), ``uniform`` (KD-unif) or none.
KD_MAPPINGS = ("dynamic", "last", "uniform", "none")


def softmax_gap(logits: torch.Tensor) -> torch.Tensor:
    """The gap between the two largest probabilities of ``softmax(logits)`` along the last dimension, in fp32."""
    top = torch.softmax(logits.float(), -1).topk(2, dim=-1).values
    return top[..., 0] - top[..., 1]


@dataclass(frozen=True)
class FreeConfig(EarlyExitConfig):
    """Hyperparameters of FREE.

    Attributes:
        exit_layer: ``L_S``, the depth of the shallow model (``1 <= L_S < L``).
        head: ``"shared"`` (the paper: the model's final norm and LM head after layer ``L_S``) or ``"readout"`` (a
            trained readout adapter first, for a frozen backbone).
        adaptation: Must be ``"frozen"``: the shallow model is the deep model's first layers, which the exact pass
            reuses. Fine-tune the model itself with the objective ``free``.
        threshold: ``lambda``, the softmax-gap confidence at which a token exits (the paper's initial value 0.9).
        posterior: ``zeta`` of the adaptive estimator (:func:`bmm_threshold`).
        max_exits: Most consecutive shallow exits before a deep pass runs anyway (a bound of economize's cycle; the
            paper has none).
    """

    technique: ClassVar[str] = "free"

    head: str = "shared"
    threshold: float = 0.9
    posterior: float = 0.4
    max_exits: int = 31

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.adaptation != "frozen":
            raise ConfigError("FREE's shallow model is the model's first layers: adaptation must be 'frozen' "
                              "(train the model itself with the objective 'free')")
        check_unit_interval("threshold", self.threshold, closed_low=False, closed_high=True)
        check_unit_interval("posterior", self.posterior, closed_low=False, closed_high=True)
        check_positive("max_exits", self.max_exits)


@register_technique
class FreeTechnique(EarlyExitTechnique):
    """The shallow model of FREE as a drafter whose drafts are kept (see the module docstring).

    Parameters: ``readout`` with ``head="readout"``, none otherwise (the objective ``free`` trains the model's own
    weights).
    """

    name = FreeConfig.technique
    config_class = FreeConfig
    info = TechniqueInfo(family=Family.DRAFTING, saves=Saving.LATENCY, lossless=False, trainable=("readout",))
    config: FreeConfig

    def decode_config(self, threshold: float | None = None, **options: Any) -> DecodeConfig:
        """FREE's decoding: up to :attr:`FreeConfig.max_exits` kept shallow exits per deep pass, ending at the first
        token below ``threshold`` (default :attr:`FreeConfig.threshold`). ``options`` are further
        :class:`~economize.DecodeConfig` fields."""
        threshold = self.config.threshold if threshold is None else threshold
        return DecodeConfig(cycle_length=self.config.max_exits + 1, lossless=False, min_confidence=threshold,
                            draft_policy="softmax_gap", **options)

    def cost_summary(self, result: Any) -> dict[str, Any]:
        """:func:`~.drafted.drafted_cost_summary` of one generation: the deep pass reuses the shallow layers of its
        tokens, and a token that does not exit continues from its shallow state."""
        return drafted_cost_summary(result, continue_wasted=True)

    def extra_repr(self) -> str:
        return f"{super().extra_repr()}, threshold={self.config.threshold}"


@POLICIES.register("softmax_gap")
class ShallowExitStop(DraftPolicy):
    """Stop a row at the first draft whose softmax gap (:func:`softmax_gap`) is below ``min_confidence``, the first
    draft of a cycle included (FREE: that token comes from the deep model)."""

    def __init__(self, config: DecodeConfig) -> None:
        super().__init__(config)
        if config.min_confidence is None:
            raise ConfigError("draft_policy 'softmax_gap' needs min_confidence")
        self.threshold = config.min_confidence

    def stops(self, index: int, logits: torch.Tensor) -> torch.Tensor | None:
        return softmax_gap(logits) < self.threshold


# ---- adaptive threshold --------------------------------------------------------------------------------------------
def beta_moments(values: torch.Tensor, eps: float = 1e-4) -> tuple[float, float]:
    """Method-of-moments Beta parameters of samples in ``(0, 1)``: ``alpha = m (m (1 - m) / s^2 - 1)`` and
    ``beta = alpha (1 - m) / m`` for mean ``m`` and variance ``s^2`` (clamped to a valid Beta)."""
    values = values.double().clamp(eps, 1 - eps)
    mean = float(values.mean())
    var = float(values.var(unbiased=False)) if values.numel() > 1 else 0.0
    var = min(max(var, eps * eps), mean * (1 - mean) * (1 - eps))
    alpha = mean * (mean * (1 - mean) / var - 1)
    return alpha, alpha * (1 - mean) / mean


def bmm_threshold(confidence: torch.Tensor, agree: torch.Tensor, posterior: float = 0.4,
                  grid: Sequence[float] | None = None, default: float = 0.9) -> float:
    """FREE's adaptive threshold: the smallest ``lambda`` of ``grid`` with ``p(agree | lambda) >= posterior`` under a
    two-component Beta mixture with equal priors, one component fitted to the confidences of agreeing tokens and one to
    the others.

    Args:
        confidence: Shallow confidences ``[n]`` in ``[0, 1]``.
        agree: Whether the shallow and the deep prediction of each token agree ``[n]`` (bool).
        posterior: ``zeta``.
        grid: Candidate thresholds in increasing order (default ``0.00, 0.01, ..., 1.00``).
        default: The threshold while one of the classes has fewer than two samples (the paper's ``lambda_0``).
    """
    confidence, agree = confidence.flatten().double(), agree.flatten().bool()
    if int(agree.sum()) < 2 or int((~agree).sum()) < 2:
        return default
    components = [beta_moments(confidence[agree]), beta_moments(confidence[~agree])]
    candidates = torch.tensor(list(grid) if grid is not None else [i / 100 for i in range(101)], dtype=torch.float64)
    x = candidates.clamp(1e-4, 1 - 1e-4)
    agree_pdf, disagree_pdf = (torch.distributions.Beta(torch.tensor(a), torch.tensor(b)).log_prob(x).exp()
                               for a, b in components)
    p_agree = agree_pdf / (agree_pdf + disagree_pdf).clamp_min(1e-300)
    passing = (p_agree >= posterior).nonzero()
    return float(candidates[passing[0, 0]]) if passing.numel() else float(candidates[-1])


@torch.no_grad()
def free_calibration_pairs(model: EfficientModel, input_ids: torch.Tensor, technique: str = FreeTechnique.name,
                           attention_mask: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """Shallow confidences and shallow-deep agreements at every position of ``input_ids`` ``[B, T]`` (one teacher-forced
    pass), the inputs of :func:`bmm_threshold`. Feed sequences the model generated itself to match the paper, which
    collects the pairs while decoding."""
    free = model.techniques[technique]
    captured: dict[str, torch.Tensor] = {}

    def hook(module: torch.nn.Module, args: tuple, output: Any) -> None:
        captured["h"] = output[0] if isinstance(output, tuple) else output

    handle = model.adapter.layer(free.exit_layer).register_forward_hook(hook)
    try:
        out = model.base_model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
    finally:
        handle.remove()
    shallow = free.logits(captured["h"]).float()
    keep = torch.ones_like(input_ids, dtype=torch.bool) if attention_mask is None else attention_mask.bool()
    agree = shallow.argmax(-1) == out.logits.argmax(-1).to(shallow.device)
    return softmax_gap(shallow)[keep].cpu(), agree[keep].cpu()


# ---- training ------------------------------------------------------------------------------------------------------
def monotone_layer_map(costs: torch.Tensor) -> list[int]:
    """The non-decreasing assignment ``m(1) <= ... <= m(S)`` of rows to columns that minimizes the summed cost
    ``costs[i, m(i)]`` (``[S, J]``, ``inf`` marks forbidden pairs), as column indices."""
    rows, cols = costs.shape
    best = torch.empty(rows, cols, dtype=torch.float64)
    arg = torch.zeros(rows, cols, dtype=torch.long)
    best[0] = costs[0].double()
    for i in range(1, rows):
        prefix, index = torch.cummin(best[i - 1], 0)  # best previous column <= j
        best[i] = prefix + costs[i].double()
        arg[i] = index
    j = int(best[-1].argmin())
    if not math.isfinite(float(best[-1, j])):
        raise ConfigError("no monotone layer map with finite cost")
    mapping = [j]
    for i in range(rows - 1, 0, -1):
        j = int(arg[i, j])
        mapping.append(j)
    return mapping[::-1]


def kd_layer_map(mapping: str, shallow: int, layers: int, costs: torch.Tensor | None = None) -> list[int]:
    """Deep layer ``m(i)`` for shallow layers ``i = 1..L_S`` (1-indexed): ``last`` maps them onto the deep model's last
    ``L_S`` layers, ``uniform`` onto every ``L / L_S``-th layer, ``dynamic`` onto the monotone map of least summed MSE
    ``costs[i - 1, j - 1]`` over deep layers ``j > i`` (the shallow layers are the deep model's own, so ``j = i``
    would distill a state into itself)."""
    if mapping == "last":
        return [layers - shallow + i for i in range(1, shallow + 1)]
    if mapping == "uniform":
        return [max(i + 1, min(layers, round(i * layers / shallow))) for i in range(1, shallow + 1)]
    if mapping != "dynamic":
        raise ConfigError(f"unknown KD mapping {mapping!r}; choose one of {list(KD_MAPPINGS)}")
    if costs is None:
        raise ConfigError("the dynamic KD mapping needs the layer costs")
    allowed = torch.arange(1, layers + 1)[None, :] > torch.arange(1, shallow + 1)[:, None]
    return [j + 1 for j in monotone_layer_map(costs.double().masked_fill(~allowed, math.inf))]


@register_objective("free", default_for=(FreeTechnique.name,))
@dataclass
class FreeObjective(TechniqueObjective[FreeTechnique]):
    """FREE's fine-tuning loss ``alpha_S CE_S + alpha_L CE_L + kd_weight L_KD`` (see the module docstring).

    Attributes:
        technique: Name of the attached FREE technique.
        kd_mapping: A name in :data:`KD_MAPPINGS` (the paper uses ``"dynamic"``).
        kd_weight: Weight of the distillation term (the paper adds it with weight 1).
        parameters: Which base parameters are trained (see :data:`~economize.train.objectives.base.BaseParameters`);
            the technique's readout, if any, trains as well.
    """

    defaults: ClassVar[OptimDefaults] = OptimDefaults(lr=1e-4, warmup_steps=100)
    technique: str = FreeTechnique.name
    kd_mapping: Literal["dynamic", "last", "uniform", "none"] = "dynamic"
    kd_weight: float = 1.0
    parameters: BaseParameters = "trainable"
    _handles: list[RemovableHandle] = field(default_factory=list, init=False, repr=False)
    _captured: dict[int, torch.Tensor] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        check_choice("kd_mapping", self.kd_mapping, KD_MAPPINGS)

    def parameter_groups(self, model: EfficientModel) -> list[ParamGroup]:
        return base_parameter_groups(model, self.parameters)

    def teacher_request(self, model: EfficientModel) -> None:
        return None

    def setup(self, model: EfficientModel) -> None:
        super().setup(model)
        self._install(model)

    def teardown(self, model: EfficientModel) -> None:
        self._remove()

    def loss(self, model: EfficientModel, batch: Batch, teacher: TeacherOutputs | None) -> ObjectiveOutput:
        temporary = not self._handles
        if temporary:
            self._install(model)
        try:
            return self._loss(model, batch)
        finally:
            if temporary:
                self._remove()

    @torch.no_grad()
    def evaluate(self, model: EfficientModel, batch: Batch, teacher: TeacherOutputs | None) -> dict[str, Any]:
        output = self.loss(model, batch, teacher)
        return {"loss": output.loss, **output.metrics}

    # ---- internals ---------------------------------------------------------------------------------------------
    def _install(self, model: EfficientModel) -> None:
        """Forward hooks that keep the output of every block the loss reads."""
        free = self.trained(model)
        layers = model.adapter.num_layers
        wanted = {free.exit_layer}
        if self.kd_mapping != "none":
            wanted.update(range(1, layers + 1))
        for layer in sorted(wanted):
            self._handles.append(model.adapter.layer(layer).register_forward_hook(self._hook(layer)))

    def _remove(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        self._captured.clear()

    def _hook(self, layer: int) -> Any:
        def hook(module: torch.nn.Module, args: tuple, output: Any) -> None:
            self._captured[layer] = output[0] if isinstance(output, tuple) else output

        return hook

    def _loss(self, model: EfficientModel, batch: Batch) -> ObjectiveOutput:
        free = self.trained(model)
        shallow, layers = free.exit_layer, model.adapter.num_layers
        self._captured.clear()
        out = model.base_model(input_ids=batch.input_ids, attention_mask=batch.attention_mask, use_cache=False)
        deep_nll = next_token_nll(out.logits, batch)
        shallow_nll = next_token_nll(free.logits(self._captured[shallow]), batch)
        loss = (shallow * shallow_nll + layers * deep_nll) / (layers + shallow)
        metrics: dict[str, torch.Tensor | float] = {"nll_shallow": shallow_nll.detach(), "nll_deep": deep_nll.detach()}
        if self.kd_mapping != "none" and self.kd_weight > 0:
            kd = self._distillation(model, batch, shallow, layers)
            loss = loss + self.kd_weight * kd
            metrics["kd"] = kd.detach()
        self._captured.clear()
        return ObjectiveOutput(loss, metrics)

    def _distillation(self, model: EfficientModel, batch: Batch, shallow: int, layers: int) -> torch.Tensor:
        """``(1 / L_S) sum_i MSE(H^i, sg(H^{m(i)}))`` over the batch's real tokens."""
        mask = (batch.attention_mask if batch.attention_mask is not None
                else torch.ones_like(batch.input_ids)).bool()

        def mse(i: int, j: int) -> torch.Tensor:
            student = self._captured[i].float()
            target = self._captured[j].detach().float().to(student.device)
            per_token = (student - target).pow(2).mean(-1)
            return per_token[mask.to(per_token.device)].mean()

        costs = None
        if self.kd_mapping == "dynamic":
            with torch.no_grad():
                costs = torch.full((shallow, layers), math.inf, dtype=torch.float64)
                for i in range(1, shallow + 1):
                    for j in range(i + 1, layers + 1):
                        costs[i - 1, j - 1] = float(mse(i, j))
        mapping = kd_layer_map(self.kd_mapping, shallow, layers, costs)
        terms = [mse(i, j) for i, j in zip(range(1, shallow + 1), mapping)]
        return torch.stack([t.to(terms[0].device) for t in terms]).mean()

