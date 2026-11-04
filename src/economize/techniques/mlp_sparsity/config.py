"""Configurations of the gated-MLP activation sparsity techniques ``cats`` and ``dejavu``."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar

from ...core.config import TechniqueConfig
from ...errors import ConfigError
from ...validation import (
    check_choice,
    check_layers,
    check_non_negative,
    check_number,
    check_positive,
    check_registered,
    check_unit_interval,
)
from ..sparsity.sparsify import PhasePolicy

if TYPE_CHECKING:
    from ...arch import ArchAdapter

__all__ = ["PREDICTOR_DTYPES", "PREDICTOR_INPUTS", "CatsConfig", "DejaVuConfig", "MlpSparsityConfig"]

#: What a Deja Vu predictor reads: the MLP's own input, or the output of the layer's first norm (the lookahead).
PREDICTOR_INPUTS = ("mlp", "lookahead")
#: Storage dtypes of the Deja Vu predictors.
PREDICTOR_DTYPES = ("float32", "bfloat16", "float16")


def _check_name(name: str, value: object) -> None:
    if not isinstance(value, str) or not value:
        raise ConfigError(f"{name} must be a non-empty attribute name, got {value!r}")


@dataclass(frozen=True)
class MlpSparsityConfig(TechniqueConfig):
    """Fields shared by the techniques that skip FFN channels of a gated MLP ``down(act(gate(x)) * up(x))``.

    Args:
        sparsity: Fraction of FFN channels skipped per token (both papers report 50%).
        prefill: Which tokens each kind of call sparsifies, as for TEAL (:class:`~..sparsity.sparsify.PhasePolicy`, the
            mapping of its fields, or ``"dense"``, ``"second_half"`` or ``"all"`` for the exact passes and the
            multi-token calls outside decoding; decoding and drafting steps are always sparsified with a string).
        backend: Registered kernel backend (``"dense"``: masks around the unchanged dense maps, the numerical
            reference; ``"triton"``: row- and column-sparse GEMV kernels for calls of up to 16 rows).
        layers: 1-indexed layers to sparsify; empty means all.
        mlp: Attribute of a decoder layer that holds the gated MLP.
        gate: Attribute of the MLP holding the gate map.
        up: Attribute of the MLP holding the up map.
        down: Attribute of the MLP holding the down map.
        act: Attribute of the MLP holding the activation function.
    """

    technique: ClassVar[str] = ""

    sparsity: float = 0.5
    prefill: str | PhasePolicy = "dense"
    backend: str = "dense"
    layers: tuple[int, ...] = ()
    mlp: str = "mlp"
    gate: str = "gate_proj"
    up: str = "up_proj"
    down: str = "down_proj"
    act: str = "act_fn"

    def __post_init__(self) -> None:
        from .backends import MLP_SPARSITY_BACKENDS

        check_unit_interval("sparsity", self.sparsity)
        policy = PhasePolicy.coerce(self.prefill)
        if not isinstance(self.prefill, str):  # a mapping (JSON) becomes the policy, so the config stays hashable
            object.__setattr__(self, "prefill", policy)
        check_registered("MLP sparsity backend", self.backend, MLP_SPARSITY_BACKENDS)
        check_layers(self.layers)
        for name in ("mlp", "gate", "up", "down", "act"):
            _check_name(name, getattr(self, name))

    @property
    def policy(self) -> PhasePolicy:
        """The :class:`~..sparsity.sparsify.PhasePolicy` that ``prefill`` describes."""
        return PhasePolicy.coerce(self.prefill)

    def validate(self, adapter: ArchAdapter) -> None:
        check_layers(self.layers, adapter)

    def to_dict(self) -> dict[str, Any]:
        data = super().to_dict()
        if not isinstance(self.prefill, str):
            data["prefill"] = self.policy.to_dict()
        return data


@dataclass(frozen=True)
class CatsConfig(MlpSparsityConfig):
    """CATS: threshold the gate activations of every gated MLP (Lee et al., "CATS: Contextually-Aware
    Thresholding for Sparsity in Large Language Models", COLM 2024).

    Each layer's threshold ``t`` is the ``sparsity`` quantile of ``|act(gate(x))|`` on calibration text; the MLP
    then computes ``down(g * 1[|g| > t] * up(x))`` with ``g = act(gate(x))``, so the up rows and the down columns
    of the dropped channels are never needed. The gate map stays dense.

    Args:
        sparsity: Target fraction of dropped channels per layer (the paper's 50% and 70%).
        max_samples: Activation magnitudes kept per layer for the threshold quantile.
        offload: Keep calibration activations on the CPU between layers.
        prefill: See :class:`MlpSparsityConfig`.
        backend: See :class:`MlpSparsityConfig`.
        layers: See :class:`MlpSparsityConfig`.
        mlp: See :class:`MlpSparsityConfig`.
        gate: See :class:`MlpSparsityConfig`.
        up: See :class:`MlpSparsityConfig`.
        down: See :class:`MlpSparsityConfig`.
        act: See :class:`MlpSparsityConfig`.
    """

    technique: ClassVar[str] = "cats"

    max_samples: int = 1 << 20
    offload: bool = False

    def __post_init__(self) -> None:
        super().__post_init__()
        check_positive("max_samples", self.max_samples)


@dataclass(frozen=True)
class DejaVuConfig(MlpSparsityConfig):
    """Deja Vu: a trained low-rank predictor per layer chooses the FFN channels to compute (Liu et al., "Deja Vu:
    Contextual Sparsity for Efficient LLMs at Inference Time", ICML 2023).

    The predictor of a layer maps the hidden state to one logit per FFN channel (``Linear(d, rank)`` then
    ``Linear(rank, ffn)``, the reference's two-layer predictor without a nonlinearity); the ``1 - sparsity``
    fraction of channels with the largest logits is computed and the rest is skipped in the gate, up and down
    maps. The predictors are fitted at attach time (``fit_steps``) and trained further by the objective
    ``dejavu``.

    Args:
        rank: Inner width of each predictor (the reference uses 1000).
        predictor_input: ``"mlp"`` (the MLP's own input) or ``"lookahead"`` (the output of the layer's first norm,
            ``lookahead_norm``, available before attention runs, so the prediction can overlap it).
        lookahead_norm: Attribute of the decoder layer holding the norm read by ``"lookahead"``.
        predictor_dtype: Storage dtype of the predictors.
        fit_steps: Adam steps per layer of the attach-time fit (0: no fit; the predictors stay at their random
            initialization until trained).
        fit_lr: Learning rate of the attach-time fit.
        fit_batch: Tokens per step of the attach-time fit.
        max_tokens: Calibration tokens kept per layer for the fit.
        seed: Seed of the predictor initialization and of the token subsample.
        offload: Keep calibration activations on the CPU between layers.
        sparsity: See :class:`MlpSparsityConfig`.
        prefill: See :class:`MlpSparsityConfig`.
        backend: See :class:`MlpSparsityConfig`.
        layers: See :class:`MlpSparsityConfig`.
        mlp: See :class:`MlpSparsityConfig`.
        gate: See :class:`MlpSparsityConfig`.
        up: See :class:`MlpSparsityConfig`.
        down: See :class:`MlpSparsityConfig`.
        act: See :class:`MlpSparsityConfig`.
    """

    technique: ClassVar[str] = "dejavu"

    rank: int = 1000
    predictor_input: str = "mlp"
    lookahead_norm: str = "input_layernorm"
    predictor_dtype: str = "float32"
    fit_steps: int = 200
    fit_lr: float = 1e-3
    fit_batch: int = 1024
    max_tokens: int = 65536
    seed: int = 0
    offload: bool = False

    def __post_init__(self) -> None:
        super().__post_init__()
        check_positive("rank", self.rank)
        check_choice("predictor_input", self.predictor_input, PREDICTOR_INPUTS)
        _check_name("lookahead_norm", self.lookahead_norm)
        check_choice("predictor_dtype", self.predictor_dtype, PREDICTOR_DTYPES)
        check_non_negative("fit_steps", self.fit_steps)
        check_number("fit_lr", self.fit_lr, low=0, closed_low=False)
        check_positive("fit_batch", self.fit_batch)
        check_positive("max_tokens", self.max_tokens)
        check_non_negative("seed", self.seed)
