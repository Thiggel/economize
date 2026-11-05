"""CALM: confident adaptive language modeling, a token exits once a confidence measure exceeds a threshold.

After every layer ``l`` from ``min_layer`` to ``L - 1`` the tokens of a generation step that have not exited yet
compute a confidence ``c_l``; a token with ``c_l > threshold(t)`` exits, keeps its hidden state for the remaining
layers, and is read out by the model's own final norm and head. Its K/V at the skipped layers are computed from the
exit state, which is the paper's state propagation. Three confidence measures are registered in
:data:`CONFIDENCE_MEASURES`:

* ``softmax``: the gap between the two largest probabilities of the head's distribution at layer ``l``;
* ``state``: the cosine similarity of the layer's output and input;
* ``classifier``: ``sigmoid(w . rms(h_l) + b)``, trained by the objective ``calm``.

The threshold follows the paper's decay ``clip(0.9 lambda + 0.1 exp(-tau t / N))`` over the generated position ``t``
(``tau = 0``: the constant ``lambda``). The prompt runs every layer.

:func:`calibrate_threshold` chooses ``lambda`` as in the paper with Learn then Test: it decodes calibration prompts
at every candidate threshold, measures the per-prompt inconsistency with the full model (``threshold = 1``), and
returns the smallest threshold whose risk is certified below ``epsilon`` with probability ``1 - delta``
(Hoeffding-Bentkus p-values, fixed-sequence testing from the most conservative candidate down, :func:`ltt_threshold`).

A measure that needs parameters declares them with :meth:`ConfidenceMeasure.modules` (registered on the technique
under the returned names) and exposes a trainable logit through :meth:`ConfidenceMeasure.logit`, which the objective
``calm`` trains; a new measure is a registered :class:`ConfidenceMeasure` subclass and needs no change here.
"""

from __future__ import annotations

import dataclasses
import math
from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar

import torch
import torch.nn.functional as F
from torch import nn

from ...core.config import TechniqueConfig
from ...core.metadata import Family, Saving, TechniqueInfo
from ...core.technique import register_technique
from ...errors import ConfigError
from ...registry import Registry
from ...validation import check_number, check_positive, check_registered, check_unit_interval
from .base import GENERATION_PHASES, DepthPolicy, DepthTechnique, ScalarScore, valid_mask

if TYPE_CHECKING:
    from ...runtime import LayerContext

__all__ = [
    "CONFIDENCE_MEASURES",
    "CalmConfig",
    "CalmPolicy",
    "CalmTechnique",
    "ClassifierConfidence",
    "ConfidenceMeasure",
    "ExitClassifier",
    "SoftmaxConfidence",
    "StateConfidence",
    "calibrate_threshold",
    "hoeffding_bentkus_pvalue",
    "ltt_threshold",
    "token_mismatch",
]


@dataclass(frozen=True)
class CalmConfig(TechniqueConfig):
    """CALM.

    Attributes:
        confidence: Name in :data:`CONFIDENCE_MEASURES`.
        threshold: ``lambda``; a token exits when its confidence is strictly above the (decayed) threshold, so ``1``
            never exits and reproduces the model.
        min_layer: First layer after which a token may exit.
        decay_temperature: ``tau`` of the threshold decay (``0``: constant threshold).
        decay_horizon: ``N`` of the threshold decay (the generation length the schedule is laid out for).
    """

    technique: ClassVar[str] = "calm"
    confidence: str = "softmax"
    threshold: float = 0.9
    min_layer: int = 1
    decay_temperature: float = 0.0
    decay_horizon: int = 256

    def __post_init__(self) -> None:
        check_registered("confidence measure", self.confidence, CONFIDENCE_MEASURES)
        check_unit_interval("threshold", self.threshold, closed_high=True)
        check_positive("min_layer", self.min_layer)
        check_number("decay_temperature", self.decay_temperature, low=0)
        check_positive("decay_horizon", self.decay_horizon)

    def validate(self, adapter: Any) -> None:
        if self.min_layer >= adapter.num_layers:
            raise ConfigError(f"min_layer must be below the number of layers ({adapter.num_layers})")

    def threshold_at(self, generated: torch.Tensor) -> torch.Tensor:
        """Threshold for tokens at generated positions ``generated`` (fp32, same shape)."""
        if self.decay_temperature == 0:
            return torch.full(generated.shape, self.threshold, device=generated.device)
        decay = torch.exp(-self.decay_temperature * generated.float().clamp_min(0) / self.decay_horizon)
        return (0.9 * self.threshold + 0.1 * decay).clamp(0.0, 1.0)


class ExitClassifier(ScalarScore):
    """The ``classifier`` confidence: ``sigmoid(w . rms(h) + b)``, one classifier shared by all exit layers, starting
    at the constant ``0.5``."""

    def __init__(self, hidden_size: int) -> None:
        super().__init__(hidden_size, 0.0, 0.0, activation="sigmoid")


class ConfidenceMeasure(ABC):
    """A CALM confidence measure. Subclasses are registered in :data:`CONFIDENCE_MEASURES`.

    Attributes:
        head_calls: Whether the measure applies the LM head (counted as ``confidence_head_tokens``).
        trainable: Whether :meth:`logit` is implemented (the objective ``calm`` trains it).
    """

    head_calls: ClassVar[bool] = False
    trainable: ClassVar[bool] = False

    def __init__(self, technique: CalmTechnique) -> None:
        self.technique = technique

    @classmethod
    def modules(cls, technique: CalmTechnique) -> dict[str, nn.Module]:
        """Modules the measure needs, registered on the technique under these names (so they are saved with it).
        ``technique.config`` and ``technique.adapter`` are set; no parameters by default."""
        return {}

    @abstractmethod
    def __call__(self, index: int, x: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
        """``[B, S]`` confidences in ``[0, 1]`` (fp32) after layer ``index`` from its input ``x`` and output
        ``out``."""

    def logit(self, index: int, hidden: torch.Tensor) -> torch.Tensor:
        """``[B, S]`` trainable pre-sigmoid confidence (fp32) of the state ``hidden`` after layer ``index``; only
        for :attr:`trainable` measures."""
        raise NotImplementedError(f"{type(self).__name__} has no trainable logit")


#: CALM confidence measures by :attr:`CalmConfig.confidence`.
CONFIDENCE_MEASURES: Registry[type[ConfidenceMeasure]] = Registry(
    "calm confidence measure", "economize.calm_confidence"
)


@CONFIDENCE_MEASURES.register("softmax")
class SoftmaxConfidence(ConfidenceMeasure):
    """Top-1 minus top-2 probability of the head applied to the intermediate state."""

    head_calls: ClassVar[bool] = True

    def __call__(self, index: int, x: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
        probs = F.softmax(self.technique.adapter.logits(out).float(), dim=-1)
        top = probs.topk(2, dim=-1).values
        return (top[..., 0] - top[..., 1]).to(out.device)


@CONFIDENCE_MEASURES.register("state")
class StateConfidence(ConfidenceMeasure):
    """Cosine similarity of the hidden states before and after the layer (saturation of the residual stream)."""

    def __call__(self, index: int, x: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
        return F.cosine_similarity(out.float(), x.float(), dim=-1).clamp(-1.0, 1.0)


@CONFIDENCE_MEASURES.register("classifier")
class ClassifierConfidence(ConfidenceMeasure):
    """The trained :class:`ExitClassifier` of the technique (``technique.classifier``)."""

    trainable: ClassVar[bool] = True

    @classmethod
    def modules(cls, technique: CalmTechnique) -> dict[str, nn.Module]:
        adapter = technique.adapter
        device = adapter.layer_device(adapter.num_layers)
        return {"classifier": ExitClassifier(adapter.hidden_size).to(device=device, dtype=torch.float32)}

    def logit(self, index: int, hidden: torch.Tensor) -> torch.Tensor:
        return self.technique.classifier.logit(hidden)

    def __call__(self, index: int, x: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
        return self.technique.classifier(out).to(out.device)


class CalmPolicy(DepthPolicy):
    """Early exit per token with K/V computed from the exit state (see the module docstring)."""

    def __init__(self, technique: CalmTechnique) -> None:
        super().__init__(technique.adapter.num_layers, GENERATION_PHASES)
        self.technique = technique
        self.measure = technique.measure
        self.exited: torch.Tensor | None = None
        self._threshold: torch.Tensor | None = None

    def begin_pass(self, hidden: torch.Tensor, ctx: LayerContext) -> None:
        self.exited = torch.zeros(hidden.shape[:2], dtype=torch.bool, device=hidden.device)
        self._threshold = self.technique.config.threshold_at(self.generated_positions(hidden, ctx))

    def choose(self, index: int, hidden: torch.Tensor, ctx: LayerContext) -> torch.Tensor | None:
        assert self.exited is not None
        exited = self.exited.to(hidden.device)
        return ~exited if bool(exited.any()) else None

    def transform(self, index: int, x: torch.Tensor, out: torch.Tensor, ran: torch.Tensor | None,
                  ctx: LayerContext) -> torch.Tensor | None:
        config = self.technique.config
        if self.training or not config.min_layer <= index < self.num_layers:
            return None
        assert self.exited is not None and self._threshold is not None
        exited = self.exited.to(out.device)
        live = valid_mask(out, ctx) & ~exited
        if not bool(live.any()):
            return None
        if self.measure.head_calls:
            self.meter.add("confidence_head_tokens", live.sum())
        confidence = self.measure(index, x, out)
        leaving = live & (confidence > self._threshold.to(out.device))
        self.exited = exited | leaving
        self.meter.add("exits", leaving.sum())
        return None


@register_technique
class CalmTechnique(DepthTechnique):
    """CALM: a token exits at the first layer where a confidence measure clears a threshold.

    See :class:`CalmConfig`. The ``classifier`` measure is trained by the objective ``calm``."""

    name: ClassVar[str] = CalmConfig.technique
    config_class: ClassVar[type[TechniqueConfig]] = CalmConfig
    info = TechniqueInfo(
        family=Family.DYNAMIC_DEPTH,
        saves=Saving.COMPUTE | Saving.LATENCY,
        lossless=False,
        trainable=("classifier",),
    )
    config: CalmConfig
    order: ClassVar[int] = 34
    classifier: ExitClassifier  # only with confidence="classifier" (added by ClassifierConfidence.modules)

    def __init__(self, config: CalmConfig, adapter: Any) -> None:
        super().__init__(config, adapter)
        measure = CONFIDENCE_MEASURES.get(config.confidence)
        for name, module in measure.modules(self).items():
            self.add_module(name, module)
        self.measure = measure(self)

    def make_policy(self) -> CalmPolicy:
        return CalmPolicy(self)


# ---- Learn then Test calibration of the threshold ----------------------------------------------------------------
def _binomial_cdf(k: int, n: int, p: float) -> float:
    """``P(Binomial(n, p) <= k)``."""
    if k < 0:
        return 0.0
    if k >= n:
        return 1.0
    log_p, log_q = math.log(p), math.log1p(-p)
    terms = [math.lgamma(n + 1) - math.lgamma(i + 1) - math.lgamma(n - i + 1) + i * log_p + (n - i) * log_q
             for i in range(k + 1)]
    top = max(terms)
    return min(1.0, math.exp(top) * sum(math.exp(t - top) for t in terms))


def hoeffding_bentkus_pvalue(mean_loss: float, n: int, epsilon: float) -> float:
    """Hoeffding-Bentkus p-value of the null hypothesis ``risk > epsilon`` given the mean of ``n`` losses in
    ``[0, 1]`` (Bates et al. 2021; Angelopoulos et al. 2021, used by CALM):
    ``min(exp(-n h(min(mean, eps), eps)), e P(Binom(n, eps) <= ceil(n mean)))`` with
    ``h(a, b) = a log(a / b) + (1 - a) log((1 - a) / (1 - b))``."""
    check_unit_interval("epsilon", epsilon, closed_low=False)
    if n < 1:
        raise ConfigError("Learn then Test needs at least one calibration example")
    a = min(max(mean_loss, 0.0), epsilon)
    h = (a * math.log(a / epsilon) if a > 0 else 0.0) + (1 - a) * math.log((1 - a) / (1 - epsilon))
    hoeffding = math.exp(-n * h)
    bentkus = math.e * _binomial_cdf(math.ceil(n * mean_loss - 1e-9), n, epsilon)
    return min(hoeffding, bentkus, 1.0)


def ltt_threshold(candidates: Sequence[float], losses: torch.Tensor | Sequence[Sequence[float]], *, delta: float,
                  epsilon: float) -> float:
    """The smallest candidate threshold certified by Learn then Test with fixed-sequence testing.

    Args:
        candidates: Thresholds; tested from the largest (most conservative) down, stopping at the first that fails.
        losses: ``[len(candidates), n]`` per-example losses in ``[0, 1]`` at every candidate.
        delta: Allowed probability that the certified risk bound is wrong.
        epsilon: Risk tolerance.

    Returns:
        The smallest certified threshold, or ``1.0`` (never exit: the full model, always valid) if none is.
    """
    check_unit_interval("delta", delta, closed_low=False)
    table = torch.as_tensor(losses, dtype=torch.float64)
    if table.dim() != 2 or table.shape[0] != len(candidates):
        raise ConfigError(f"losses must be [len(candidates), n], got {tuple(table.shape)}")
    if bool(((table < 0) | (table > 1)).any()):
        raise ConfigError("losses must lie in [0, 1]")
    chosen = 1.0
    for index in sorted(range(len(candidates)), key=lambda i: -candidates[i]):
        if hoeffding_bentkus_pvalue(float(table[index].mean()), table.shape[1], epsilon) > delta:
            break
        chosen = float(candidates[index])
    return chosen


def token_mismatch(tokens: torch.Tensor, reference: torch.Tensor) -> float:
    """Fraction of the positions of the longer of two generations at which they differ (a missing token counts as a
    difference): CALM's token-level textual consistency loss, in ``[0, 1]``."""
    tokens, reference = tokens.reshape(-1).cpu(), reference.reshape(-1).cpu()
    length = max(len(tokens), len(reference))
    if length == 0:
        return 0.0
    shared = min(len(tokens), len(reference))
    same = int((tokens[:shared] == reference[:shared]).sum())
    return 1.0 - same / length


#: The default thresholds :func:`calibrate_threshold` tries: 0.05, 0.10, ..., 0.95.
_CANDIDATES = tuple(i / 20 for i in range(1, 20))


def calibrate_threshold(
    model: Any,
    prompts: Sequence[torch.Tensor],
    max_new_tokens: int,
    *,
    delta: float = 0.1,
    epsilon: float = 0.1,
    candidates: Sequence[float] = _CANDIDATES,
    loss: Callable[[torch.Tensor, torch.Tensor], float] = token_mismatch,
    technique: str = "calm",
) -> float:
    """Calibrate CALM's ``lambda`` on ``prompts`` with Learn then Test (see the module docstring) and return it.

    Every prompt is decoded greedily once by the full model and once per candidate threshold (with the technique's
    decay temperature); ``loss(early, full)`` in ``[0, 1]`` scores each generation. Apply the result with
    ``dataclasses.replace(config, threshold=...)``. The technique's configuration is restored afterwards.
    """
    calm = model.techniques[technique]
    original = calm.config
    rows: list[list[float]] = []
    try:
        calm.config = dataclasses.replace(original, threshold=1.0, decay_temperature=0.0)
        full = [model.decode(prompt, max_new_tokens).tokens for prompt in prompts]
        for candidate in candidates:
            calm.config = dataclasses.replace(original, threshold=float(candidate))
            rows.append([loss(model.decode(prompt, max_new_tokens).tokens, reference)
                         for prompt, reference in zip(prompts, full)])
    finally:
        calm.config = original
    return ltt_threshold(candidates, rows, delta=delta, epsilon=epsilon)
