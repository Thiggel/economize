"""Machinery shared by the dynamic-depth techniques: a policy base that counts layer applications, the technique
base that exposes those counts, and the helpers every gated layer uses.

A :class:`DepthPolicy` is the :class:`~economize.runtime.LayerPolicy` of one generation. Subclasses implement
:meth:`DepthPolicy.choose` (which tokens run a layer) and optionally :meth:`DepthPolicy.transform` (how the layer's
output enters the residual stream). The base class

* turns boolean token masks into the cheapest equivalent decision (``RUN``, ``SKIP``, ``SKIP_FILL_KV`` or a
  :class:`~economize.runtime.Route`), so that a policy whose every token runs takes the unmodified code path;
* keeps the layer's input for :meth:`DepthPolicy.transform`;
* counts, per decoding phase and layer, the tokens that entered the layer, ran it, or only had their K/V filled
  (:class:`DepthMeter`); :meth:`DepthTechnique.cost_summary` adds the counts to a result's cost summary;
* records where each row's prompt ends (:meth:`DepthPolicy.generated_positions`), for schedules over the generated
  position (SkipDecode, CALM's threshold decay).

:class:`ScoredDepthPolicy` is the template of the techniques that score every token with a learned module (GateSkip,
Mixture-of-Depths, FlexiDepth): a subclass supplies the score, the rule that turns scores into a token mask, and how
the layer's output is combined with its input; :class:`RoutedDepthPolicy` is its routed variant (Mixture-of-Depths,
FlexiDepth), where a token that ran takes the gated residual and a skipped token a state of its own.
:class:`LayeredDepthConfig` holds the fields of the techniques that
act on a set of layers, and :class:`ScalarScore` is the per-token linear scorer of GateSkip, MoD and CALM.

In training mode (:attr:`DepthPolicy.training`, set by the objectives) every layer runs on every token and the
policy applies its soft or hard gates in :meth:`DepthPolicy.transform`, which keeps the computation differentiable.
"""

from __future__ import annotations

import dataclasses
from abc import ABC, abstractmethod
from collections.abc import Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar

import torch
from torch import nn

from ...core.config import TechniqueConfig
from ...core.technique import Capability, Technique
from ...modules.norm import rms_normalize
from ...runtime import current_phase
from ...runtime.policy import Decision, LayerAction, LayerPolicy, Route
from ...validation import check_choice, check_layers

if TYPE_CHECKING:
    from ...runtime import LayerContext

__all__ = [
    "ALL_PHASES",
    "GENERATION_PHASES",
    "DepthMeter",
    "DepthPolicy",
    "DepthTechnique",
    "LayeredDepthConfig",
    "RoutedDepthPolicy",
    "ScalarScore",
    "ScoredDepthPolicy",
    "gated_residual",
    "layer_modules",
    "module_at",
    "real_tokens",
    "runtime_supports_routed_kv",
    "second_half",
    "top_fraction",
    "unit_rms",
    "valid_mask",
]

#: Phases in which a policy acts by default: every exact pass, and calls outside decoding (``None``).
ALL_PHASES: tuple[str | None, ...] = ("prefill", "decode", "refresh", None)
#: The prompt runs the full model; later exact passes and calls outside decoding are subject to the policy.
GENERATION_PHASES: tuple[str | None, ...] = ("decode", "refresh", None)

_ENTERED, _RAN, _FILLED = range(3)


class DepthMeter:
    """Counts of layer applications of one generation, per phase.

    ``counts[phase]`` is a ``[L + 1, 3]`` tensor whose row ``l`` holds the tokens that entered layer ``l``, ran it,
    and skipped it with their K/V filled (padding excluded). ``extra`` holds named counters of a technique (e.g. the
    LM-head calls of CALM's confidence).
    """

    def __init__(self, num_layers: int) -> None:
        self.num_layers = num_layers
        self.counts: dict[str, torch.Tensor] = {}
        self.extra: dict[str, torch.Tensor | float] = {}

    def record(self, index: int, entered: torch.Tensor | int, ran: torch.Tensor | int, filled: torch.Tensor | int,
               phase: str | None = None) -> None:
        """Add the counts of one layer call (tensors stay on their device; nothing synchronizes)."""
        key = str(phase if phase is not None else current_phase())
        table = self.counts.get(key)
        device = entered.device if isinstance(entered, torch.Tensor) else None
        if table is None:
            table = self.counts[key] = torch.zeros(self.num_layers + 1, 3, dtype=torch.long, device=device)
        for column, value in ((_ENTERED, entered), (_RAN, ran), (_FILLED, filled)):
            table[index, column] += value

    def add(self, name: str, value: torch.Tensor | float) -> None:
        """Add ``value`` to the counter ``name``."""
        self.extra[name] = self.extra.get(name, 0) + (value.detach() if isinstance(value, torch.Tensor) else value)

    def per_layer(self, phase: str | None = None) -> torch.Tensor:
        """``[L, 3]`` counts of layers ``1..L`` (entered, ran, K/V filled) for one phase or all phases."""
        tables = [t.cpu() for key, t in self.counts.items() if phase is None or key == phase]
        if not tables:
            return torch.zeros(self.num_layers, 3, dtype=torch.long)
        return torch.stack(tables).sum(0)[1:]

    def summary(self) -> dict[str, Any]:
        """Totals over all phases: layer applications entered, executed and skipped, and per-layer executed
        fractions."""
        table = self.per_layer()
        entered, ran, filled = (int(table[:, c].sum()) for c in (_ENTERED, _RAN, _FILLED))
        per_layer = (table[:, _RAN].double() / table[:, _ENTERED].clamp_min(1).double()).tolist()
        out: dict[str, Any] = {
            "policy_layer_tokens": entered,
            "executed_layer_tokens": ran,
            "skipped_layer_tokens": entered - ran,
            "kv_fill_layer_tokens": filled,
            "executed_fraction": ran / entered if entered else 1.0,
            "executed_fraction_per_layer": per_layer,
            "phases": {key: _phase_counts(t) for key, t in self.counts.items()},
        }
        for name, value in self.extra.items():
            out[name] = float(value) if isinstance(value, torch.Tensor) else value
        return out


def _phase_counts(table: torch.Tensor) -> dict[str, int]:
    entered, ran, filled = (int(v) for v in table[1:].sum(0).tolist())
    return {"entered": entered, "ran": ran, "kv_filled": filled}


def real_tokens(mask: torch.Tensor | None, like: torch.Tensor) -> torch.Tensor:
    """``[B, S]`` booleans on the device of ``like`` (``[B, S, ...]``): ``mask`` (an attention or validity mask),
    or all ``True`` when it is ``None``."""
    if mask is not None:
        return mask.to(like.device).bool()
    return torch.ones(like.shape[:2], dtype=torch.bool, device=like.device)


def valid_mask(hidden: torch.Tensor, ctx: LayerContext) -> torch.Tensor:
    """``[B, S]`` real tokens of a pass (all ``True`` without padding)."""
    return real_tokens(ctx.valid, hidden)


def gated_residual(x: torch.Tensor, out: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
    """``x + gate * (out - x)`` written as ``out + (gate - 1) * (out - x)``: a gate of exactly ``1`` returns ``out``
    bit for bit, so the trivial setting of a gated technique reproduces the base model exactly.

    ``gate`` is ``[B, S]`` or ``[B, S, 1]`` in any dtype; the result has the dtype of ``out``.
    """
    if gate.dim() == out.dim() - 1:
        gate = gate[..., None]
    return out + ((gate.to(out.dtype) - 1) * (out - x))


def top_fraction(scores: torch.Tensor, valid: torch.Tensor, fraction: float) -> torch.Tensor:
    """``[B, S]`` booleans marking, in every row, the ``ceil(fraction * n)`` of its ``n`` valid tokens with the highest
    ``scores``. The rows of a batch are independent sequences and do not compete for one budget. Ties go to the
    earlier token, so the choice does not depend on padding."""
    if fraction >= 1.0:
        return valid.clone()
    ranked = scores.float().masked_fill(~valid, float("-inf"))
    counts = (valid.sum(-1).float() * fraction).ceil().long()  # [B]
    order = ranked.sort(dim=-1, descending=True, stable=True).indices.argsort(-1)  # rank of each token in its row
    return (order < counts[:, None]) & valid


@dataclass(frozen=True)
class LayeredDepthConfig(TechniqueConfig):
    """Fields shared by the techniques that act on a set of layers (GateSkip, MoD, FlexiDepth, LoRA-Drop).

    Attributes:
        layers: The layers the technique acts on (empty: :meth:`default_layers`).
        skip_prompt: Act on the prompt too (``False``: the prompt runs every layer).
    """

    layers: tuple[int, ...] = ()
    skip_prompt: bool = True

    def __post_init__(self) -> None:
        check_layers(self.layers)

    def validate(self, adapter: Any) -> None:
        check_layers(self.layers, adapter)

    def default_layers(self, num_layers: int) -> tuple[int, ...]:
        """The layers used when :attr:`layers` is empty (every layer; subclasses override)."""
        return tuple(range(1, num_layers + 1))

    def layer_set(self, num_layers: int) -> tuple[int, ...]:
        """:attr:`layers`, or :meth:`default_layers` when it is empty."""
        return self.layers or self.default_layers(num_layers)

    @property
    def phases(self) -> tuple[str | None, ...]:
        """Decoding phases the technique acts in (:data:`ALL_PHASES`, without the prompt unless
        :attr:`skip_prompt`)."""
        return ALL_PHASES if self.skip_prompt else GENERATION_PHASES


def second_half(num_layers: int) -> tuple[int, ...]:
    """Layers ``L // 2 + 1 .. L``."""
    return tuple(range(num_layers // 2 + 1, num_layers + 1))


class ScalarScore(nn.Module):
    """One score per token from a linear map of the token's state: ``act(w . n(x) + b)``, in fp32.

    ``n`` is :func:`unit_rms` (``normalize=True``, independent of the residual stream's growing scale) or the
    identity. The weights start at zero (``std=0``) or normal with standard deviation ``std``, and the bias at
    ``bias``, so that an untrained score is the constant ``act(bias)``.

    Attributes:
        proj: The ``Linear(hidden_size, 1)``.
        activation: ``"sigmoid"`` or ``"identity"`` (a logit).
        normalize: Whether the input is scaled to unit RMS first.
    """

    def __init__(self, hidden_size: int, bias: float = 0.0, std: float = 0.0, *, activation: str = "sigmoid",
                 normalize: bool = True) -> None:
        super().__init__()
        check_choice("activation", activation, ("sigmoid", "identity"))
        self.activation = activation
        self.normalize = normalize
        self.proj = nn.Linear(hidden_size, 1)
        with torch.no_grad():
            self.proj.weight.normal_(0.0, std) if std > 0 else self.proj.weight.zero_()
            self.proj.bias.fill_(bias)

    def logit(self, x: torch.Tensor) -> torch.Tensor:
        """``[B, S]`` pre-activation scores in fp32."""
        x = unit_rms(x) if self.normalize else x.float()
        return self.proj(x.to(self.proj.weight.device))[..., 0]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``[B, S]`` scores in fp32."""
        logit = self.logit(x)
        return torch.sigmoid(logit) if self.activation == "sigmoid" else logit


def runtime_supports_routed_kv() -> bool:
    """Whether the runtime can skip a layer for some tokens of a pass without writing their K/V.

    That needs a ``Route.fill_kv`` field (``False``: the routed-around tokens write no K/V at the layer) and a cache
    that remembers, per layer, which slots hold no K/V so that later queries do not attend to them.
    """
    return "fill_kv" in {f.name for f in dataclasses.fields(Route)}


def module_at(modules: nn.ModuleDict, index: int) -> nn.Module | None:
    """The module of layer ``index`` in a :func:`layer_modules` dict, or ``None`` if the layer has none."""
    key = str(index)
    return modules[key] if key in modules else None  # noqa: SIM401  (nn.ModuleDict has no get)


class DepthPolicy(LayerPolicy):
    """Base of the dynamic-depth policies (see the module docstring).

    Attributes:
        meter: Counts of the layer applications decided by this policy.
        training: Run every layer on every token and gate the outputs differentiably (objectives set it).
        phases: Decoding phases in which the policy acts; other passes run the full model.
        fill_kv: Whether skipped tokens get their K/V written: ``SKIP_FILL_KV`` rather than ``SKIP`` when the whole
            pass skips a layer, and a :class:`~economize.runtime.Route` that fills the K/V of the tokens it routes
            around. ``False`` with a ``Route`` needs :func:`runtime_supports_routed_kv` (Mixture-of-Depths'
            ``kv="routed"``); without it a partial ``Route`` fills the K/V anyway (and is counted so).
    """

    #: Whether skipped tokens get K/V (``SKIP_FILL_KV``) when the whole pass skips a layer.
    fill_kv: bool = True

    def __init__(self, num_layers: int, phases: Iterable[str | None] = ALL_PHASES) -> None:
        self.num_layers = num_layers
        self.phases = tuple(phases)
        self.meter = DepthMeter(num_layers)
        self.training = False
        self._inputs: dict[int, torch.Tensor] = {}
        self._masks: dict[int, torch.Tensor | None] = {}
        self._last = 0
        self._active = True
        self._prompt_end: torch.Tensor | None = None  # [B], set by the prefill passes, kept by reorder

    # ---- the LayerPolicy interface -------------------------------------------------------------------------
    def decide(self, index: int, hidden: torch.Tensor, ctx: LayerContext) -> Decision:
        if self._starts_pass(index):
            if current_phase() == "prefill":  # chunked prefill: the last chunk sets the end
                positions = torch.where(valid_mask(hidden, ctx), ctx.positions.to(hidden.device), -1)
                self._prompt_end = positions.max(-1).values + 1
                self.begin_prompt()
            self._inputs.clear()
            self._masks.clear()
            self._active = self.training or current_phase() in self.phases
            self.begin_pass(hidden, ctx)
        self._last = index
        valid = valid_mask(hidden, ctx)
        if not self._active:
            self._masks[index] = None
            self.meter.record(index, valid.sum(), valid.sum(), 0)
            return LayerAction.RUN
        self._inputs[index] = hidden
        mask = None if self.training else self.choose(index, hidden, ctx)
        self._masks[index] = mask
        return self._decision(index, mask, valid)

    def after(self, index: int, hidden: torch.Tensor, ctx: LayerContext) -> torch.Tensor | None:
        if not self._active:
            return None
        x = self._inputs.pop(index, None)
        if x is None:
            return None
        return self.transform(index, x, hidden, self._masks.get(index), ctx)

    def reorder(self, rows: torch.Tensor) -> None:
        """Keep the prompt ends of the rows ``rows``, so that a row that stays in the batch keeps its own schedule."""
        if self._prompt_end is not None:
            self._prompt_end = self._prompt_end.index_select(0, rows.to(self._prompt_end.device))

    def generated_positions(self, hidden: torch.Tensor, ctx: LayerContext) -> torch.Tensor:
        """``[B, S]`` positions of the tokens of a pass counted from the end of their row's prompt (``0``: the first
        generated token). Outside decoding (training, no prefill seen, or another batch size) the absolute
        positions."""
        positions = ctx.positions.to(hidden.device)
        start = self._prompt_end
        if self.training or start is None or len(start) != len(positions):
            return positions
        return positions - start.to(hidden.device)[:, None]

    # ---- hooks for subclasses ------------------------------------------------------------------------------
    def begin_prompt(self) -> None:
        """Called at the start of every prefill pass (a new prompt, or the next chunk of one), whether or not the
        policy acts in the prefill phase: reset state that lives across the passes of one generation here."""

    def begin_pass(self, hidden: torch.Tensor, ctx: LayerContext) -> None:
        """Called before layer 1 of every pass (reset per-pass state here)."""

    def choose(self, index: int, hidden: torch.Tensor, ctx: LayerContext) -> torch.Tensor | None:
        """``[B, S]`` booleans: the tokens that run layer ``index`` (``None``: all of them)."""
        return None

    def transform(
        self, index: int, x: torch.Tensor, out: torch.Tensor, ran: torch.Tensor | None, ctx: LayerContext
    ) -> torch.Tensor | None:
        """The hidden state passed on after layer ``index``, from its input ``x`` and output ``out`` (equal to
        ``x`` for tokens that skipped), given the mask ``ran`` from :meth:`choose` (``None``: every token ran, or
        training mode). ``None`` keeps ``out``."""
        return None

    # ---- internals -----------------------------------------------------------------------------------------
    def _starts_pass(self, index: int) -> bool:
        """Whether a decision for layer ``index`` begins a new pass: the first decision of the policy, or one for a
        layer at or below the previous decision's layer."""
        return self._last == 0 or index <= self._last

    def _decision(self, index: int, mask: torch.Tensor | None, valid: torch.Tensor) -> Decision:
        entered = valid.sum()
        if mask is None:
            self.meter.record(index, entered, entered, 0)
            return LayerAction.RUN
        mask = mask.to(valid.device)
        ran = (mask & valid).sum()
        filled = entered - ran if self.fill_kv else 0
        routed_kv = not self.fill_kv and runtime_supports_routed_kv()
        run_all = mask | ~valid  # padding follows the cheapest decision
        if bool(run_all.all()):
            self.meter.record(index, entered, entered, 0)
            return LayerAction.RUN
        if not bool((mask & valid).any()):
            self.meter.record(index, entered, 0, filled)
            return LayerAction.SKIP_FILL_KV if self.fill_kv else LayerAction.SKIP
        # A Route fills the K/V of the tokens it routes around unless the runtime can leave them empty.
        self.meter.record(index, entered, ran, 0 if routed_kv else entered - ran)
        route = Route.from_mask(mask & valid)
        return dataclasses.replace(route, fill_kv=False) if routed_kv else route  # type: ignore[call-arg]


class ScoredDepthPolicy(DepthPolicy):
    """Template of the policies that score every token of a layer with a learned module.

    Subclasses implement :meth:`score` (``None``: the layer is not scored), :meth:`select` (inference: which tokens
    run) and :meth:`combine` (how the output enters the residual stream), and may override :meth:`train_select`
    (training: which tokens keep their update; default :meth:`select`). The scores of the current pass are kept in
    :attr:`scores`, where the objectives read them for their regularizers.

    At inference :meth:`choose` scores the layer input and :meth:`transform` combines with the stored score; in
    training mode every token runs, and :meth:`transform` scores the input, selects with :meth:`train_select` and
    combines, which keeps the computation differentiable.
    """

    def __init__(self, num_layers: int, phases: Iterable[str | None] = ALL_PHASES) -> None:
        super().__init__(num_layers, phases)
        self.scores: dict[int, torch.Tensor] = {}

    @abstractmethod
    def score(self, index: int, x: torch.Tensor) -> torch.Tensor | None:
        """``[B, S]`` scores (fp32, differentiable) of the input ``x`` of layer ``index``, or ``None``."""

    @abstractmethod
    def select(self, index: int, score: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        """``[B, S]`` booleans: the tokens that run layer ``index`` at inference."""

    def train_select(self, index: int, score: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        """``[B, S]`` booleans: the tokens that keep their update in training (``score`` still has its gradient)."""
        return self.select(index, score.detach(), valid)

    @abstractmethod
    def combine(self, index: int, x: torch.Tensor, out: torch.Tensor, score: torch.Tensor,
                ran: torch.Tensor | None) -> torch.Tensor:
        """The state passed on from the input ``x``, the output ``out``, the score and the tokens that ran
        (``None``: all of them)."""

    def begin_pass(self, hidden: torch.Tensor, ctx: LayerContext) -> None:
        self.scores = {}

    def choose(self, index: int, hidden: torch.Tensor, ctx: LayerContext) -> torch.Tensor | None:
        score = self.score(index, hidden)
        if score is None:
            return None
        self.scores[index] = score
        return self.select(index, score, valid_mask(hidden, ctx))

    def transform(self, index: int, x: torch.Tensor, out: torch.Tensor, ran: torch.Tensor | None,
                  ctx: LayerContext) -> torch.Tensor | None:
        if self.training:
            score = self.score(index, x)
            if score is None:
                return None
            self.scores[index] = score
            ran = self.train_select(index, score, valid_mask(x, ctx))
        else:
            score = self.scores.get(index)
            if score is None:
                return None
        return self.combine(index, x, out, score, None if ran is None else ran.to(out.device))


class RoutedDepthPolicy(ScoredDepthPolicy):
    """A :class:`ScoredDepthPolicy` whose tokens that ran take the gated residual ``x + score * (out - x)`` and whose
    skipped tokens take :meth:`skip_state` (default: ``x`` unchanged)."""

    def combine(self, index: int, x: torch.Tensor, out: torch.Tensor, score: torch.Tensor,
                ran: torch.Tensor | None) -> torch.Tensor:
        routed = gated_residual(x, out, score)
        if ran is None or self.all_routed(ran):
            return routed
        return torch.where(ran[..., None], routed, self.skip_state(index, x, score))

    def all_routed(self, ran: torch.Tensor) -> bool:
        """Whether the gated residual is already right for every token, so :meth:`skip_state` is not needed.
        Default: at inference, where a skipped token's ``out`` is its input ``x``."""
        return not self.training

    def skip_state(self, index: int, x: torch.Tensor, score: torch.Tensor) -> torch.Tensor:
        """The state passed on by the tokens that skipped layer ``index``."""
        return x


class DepthTechnique(Technique, ABC):
    """Base of the dynamic-depth techniques: a fresh :class:`DepthPolicy` per generation, and its counts in the
    cost summary. Subclasses implement :meth:`make_policy`."""

    capabilities: ClassVar[Capability] = Capability.DEPTH
    graph_safe: ClassVar[bool] = False  # decisions read token masks on the host

    def __init__(self, config: Any, adapter: Any) -> None:
        super().__init__(config, adapter)
        self.last_policy: DepthPolicy | None = None

    @abstractmethod
    def make_policy(self) -> DepthPolicy:
        """A new policy (inference mode)."""

    def layer_policy(self) -> DepthPolicy:
        self.last_policy = self.make_policy()
        return self.last_policy

    def depth_stats(self) -> dict[str, Any]:
        """Layer applications of the last generation (see :meth:`DepthMeter.summary`)."""
        if self.last_policy is None:
            return DepthMeter(self.adapter.num_layers).summary()
        return self.last_policy.meter.summary()

    def cost_summary(self, result: Any) -> dict[str, Any]:
        """``result.cost_summary()`` of the last generation (one prompt) with the skipped layer applications
        subtracted: ``effective_layer_tokens`` counts every decoder-layer application that ran, and
        ``relative_effective_layer_tokens`` relates it to plain decoding of the same output. Layers skipped with
        their K/V filled count in ``kv_fill_layer_tokens`` (K/V projections only)."""
        summary = dict(result.cost_summary())
        stats = self.depth_stats()
        effective = summary["layer_tokens"] - stats["skipped_layer_tokens"]
        plain = summary["layer_tokens"] / summary["relative_layer_tokens"] if summary["relative_layer_tokens"] else 0
        summary.update(stats)
        summary["effective_layer_tokens"] = effective
        summary["relative_effective_layer_tokens"] = effective / plain if plain else 0.0
        return summary


def unit_rms(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """``x`` in fp32 scaled to unit root mean square along the last dimension (:func:`economize.modules.rms_normalize`
    in fp32): router and gate inputs, independent of the residual stream's growing scale."""
    return rms_normalize(x.float(), eps)


def layer_modules(adapter: Any, layers: Iterable[int], factory: Any) -> nn.ModuleDict:
    """``{str(layer): factory(layer)}`` with every module on its layer's device, in fp32."""
    modules = nn.ModuleDict()
    for layer in layers:
        modules[str(layer)] = factory(layer).to(device=adapter.layer_device(layer), dtype=torch.float32)
    return modules
