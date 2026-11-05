"""DMTD: direct multi-token decoding with cyclical refilling (Luo et al., 2025).

DMTD splits the model into early, middle and late layers and decodes in cycles of ``tau`` tokens. The first token of
a cycle runs all ``L`` layers; the next ``tau - 1`` tokens run only the last ``L_d`` layers, on their embedding. At
the start of the next cycle the early and middle layers run over the tokens that skipped them to refill their K/V
(cyclical refilling), in the same pass as the new cycle's first token. Training is plain next-token cross-entropy on
one pass in which the late layers read the token embedding at the positions of the skipped tokens
(``h_emb + h_think * M`` with the cyclical mask ``M_p = 1`` iff ``p mod tau = 0``); the paper fine-tunes all
weights. The efficiency measure is the percentage of layers per token ``PLT = (L + (tau - 1) L_d) / (tau L)``.

In economize, DMTD is a self-drafter (:class:`DmtdTechnique`) whose draft steps enter the model after layer
``L - L_d`` with the token embedding itself as the state there (:class:`EmbeddingPredictor`, registered as
``"embedding"``), decoded with every draft kept
(:meth:`DmtdTechnique.decode_config`: ``cycle_length = tau``, ``lossless=False``). The exact pass that starts a cycle
runs all layers over the pending tokens, which refills the early K/V and emits the cycle's first token from the
model. It also recomputes the late layers of the drafted tokens, which DMTD keeps from the draft steps:
:meth:`DmtdTechnique.cost_summary` subtracts them (see :mod:`.drafted`), and the decoded drafts attend to the exact
late K/V of earlier cycles, as in training by rollout. The objective ``dmtd`` (:class:`DmtdRollout`) trains rollouts
of ``tau - 1`` steps every ``tau`` tokens, which are the paper's masked positions, by the KL to the model (plus
``ce_weight`` times the cross-entropy).
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar

import torch

from ...core.metadata import Family, Saving, TechniqueInfo
from ...core.technique import register_technique
from ...decoding.config import DecodeConfig
from ...errors import ConfigError
from ...train.objective import Batch, ObjectiveOutput, TeacherOutputs, register_objective
from ...validation import check_positive
from ..jump.config import JumpConfig
from ..jump.objective import JumpRollout
from ..jump.predictor import PREDICTORS, Predictor
from ..jump.technique import JumpTechnique
from .drafted import drafted_cost_summary

if TYPE_CHECKING:
    from collections.abc import Mapping

    from ...arch import ArchAdapter
    from ...core.model import EfficientModel

__all__ = ["DmtdConfig", "DmtdRollout", "DmtdTechnique", "EmbeddingPredictor", "percentage_layers_per_token"]


def percentage_layers_per_token(num_layers: int, decoding_layers: int, cycle_length: int) -> float:
    """DMTD's ``PLT = (L + (tau - 1) L_d) / (tau L)``: the layers run one after another per generated token,
    relative to ordinary decoding."""
    return (num_layers + (cycle_length - 1) * decoding_layers) / (cycle_length * num_layers)


@PREDICTORS.register("embedding")
class EmbeddingPredictor(Predictor):
    """The next token's embedding as its state after layer ``m`` (DMTD: the late layers read ``h_emb``). Has no
    parameters and ignores the source state."""

    def forward(self, source: torch.Tensor, token_embedding: torch.Tensor) -> torch.Tensor:
        return token_embedding.float()


@dataclass(frozen=True)
class DmtdConfig(JumpConfig):
    """Hyperparameters of DMTD.

    Attributes:
        decoding_layers: ``L_d``, the late layers that run for the tokens after a cycle's first (the paper's best
            allocation is 8 of 36).
        cycle_length: ``tau``, tokens per cycle (the paper's default 3).
        target_layer, source_layer: Resolved to ``L - L_d`` when the technique attaches; leave them ``None``.
        adaptation: ``"full"`` (the paper's fine-tuning of all late layers, as trainable copies used on the draft
            steps only), ``"lora"`` (LoRA on the late layers of the draft steps) or ``"frozen"`` (training-free).
        predictor: ``"embedding"`` (the paper's); another registered predictor learns the state after layer ``L - L_d``
            instead.

    The remaining fields are inherited from the base drafter configuration and rarely need changing.
    """

    technique: ClassVar[str] = "dmtd"

    target_layer: int | None = None  # type: ignore[assignment]
    source_layer: int | None = None  # type: ignore[assignment]
    decoding_layers: int = 8
    cycle_length: int = 3
    adaptation: str = "full"
    predictor: str = "embedding"
    use_state: bool = False
    use_token: bool = False

    def __post_init__(self) -> None:
        check_positive("decoding_layers", self.decoding_layers)
        check_positive("cycle_length", self.cycle_length)
        if self.exit_layer is not None:
            raise ConfigError("DMTD's draft steps run through the last layer: leave exit_layer unset")
        if self.extra_source_layers:
            raise ConfigError("DMTD's draft steps read no earlier state: leave extra_source_layers empty")
        if self.target_layer is None or self.source_layer is None:
            if (self.target_layer, self.source_layer) != (None, None):
                raise ConfigError("set both target_layer and source_layer, or neither (DMTD resolves them)")
            dataclasses.replace(self, target_layer=0, source_layer=0)  # the jump checks of the other fields
            return
        super().__post_init__()

    def resolved(self, num_layers: int) -> DmtdConfig:
        """This configuration with ``target_layer = source_layer = L - L_d``."""
        if not self.decoding_layers < num_layers:
            raise ConfigError(f"decoding_layers must be below the model's {num_layers} layers, "
                              f"got {self.decoding_layers}")
        m = num_layers - self.decoding_layers
        if self.target_layer is not None and (self.target_layer, self.source_layer) != (m, m):
            raise ConfigError(f"target_layer and source_layer must be L - decoding_layers = {m}, got "
                              f"{self.target_layer}, {self.source_layer}")
        return dataclasses.replace(self, target_layer=m, source_layer=m)

    def validate(self, adapter: ArchAdapter) -> None:
        JumpConfig.validate(self.resolved(adapter.num_layers), adapter)


@register_technique
class DmtdTechnique(JumpTechnique):
    """DMTD: the late layers decode several tokens per cycle from their embeddings, and the drafts are kept.

    The early and middle layers refill their K/V at the start of the next cycle (see the module docstring).

    Parameters: ``blocks`` (``adaptation="full"``) or ``lora`` (``"lora"``); none with ``"frozen"``.
    """

    name = DmtdConfig.technique
    config_class = DmtdConfig
    info = TechniqueInfo(family=Family.DRAFTING, saves=Saving.LATENCY, lossless=False, trainable=("lora", "blocks"))
    config: DmtdConfig

    def __init__(self, config: DmtdConfig, adapter: ArchAdapter) -> None:
        super().__init__(config.resolved(adapter.num_layers), adapter)

    @property
    def plt(self) -> float:
        """The configuration's :func:`percentage_layers_per_token`."""
        return percentage_layers_per_token(self.adapter.num_layers, self.config.decoding_layers,
                                           self.config.cycle_length)

    def decode_config(self, cycle_length: int | None = None, **options: Any) -> DecodeConfig:
        """DMTD's decoding: cycles of ``cycle_length`` tokens (default :attr:`DmtdConfig.cycle_length`), every draft
        kept. ``options`` are further :class:`~economize.DecodeConfig` fields."""
        tau = self.config.cycle_length if cycle_length is None else cycle_length
        return DecodeConfig(cycle_length=tau, lossless=False, **options)

    def cost_summary(self, result: Any) -> dict[str, Any]:
        """:func:`~.drafted.drafted_cost_summary` of one generation: the drafted tokens keep the late K/V of their
        draft steps. ``relative_sequential_layers`` is the PLT of the generation."""
        return drafted_cost_summary(result, continue_wasted=False)


@register_objective("dmtd", default_for=(DmtdTechnique.name,))
@dataclass
class DmtdRollout(JumpRollout):
    """:class:`~economize.techniques.jump.JumpRollout` with DMTD's cycle: rollouts of ``horizon = tau - 1`` draft
    steps every ``stride = tau`` tokens (``None``: from the technique's :attr:`DmtdConfig.cycle_length`) and no state
    loss (the embedding predictor has nothing to fit)."""

    technique: str = DmtdTechnique.name
    horizon: int | None = None  # type: ignore[assignment]
    stride: int | None = None  # type: ignore[assignment]
    state_weight: float = 0.0
    eval_horizon: int = 8
    eval_stride: int = 8
    eval_first: int = 16

    @classmethod
    def technique_type(cls) -> type[DmtdTechnique]:
        return DmtdTechnique

    def setup(self, model: EfficientModel) -> None:
        self._resolve(model)
        super().setup(model)

    def loss(self, model: EfficientModel, batch: Batch, teacher: TeacherOutputs | None) -> ObjectiveOutput:
        self._resolve(model)
        return super().loss(model, batch, teacher)

    def evaluate(
        self, model: EfficientModel, batch: Batch, teacher: TeacherOutputs | None
    ) -> Mapping[str, torch.Tensor | float]:
        self._resolve(model)
        return super().evaluate(model, batch, teacher)

    def _resolve(self, model: EfficientModel) -> None:
        """Fill ``horizon`` and ``stride`` from the technique's cycle length."""
        tau = self.trained(model).config.cycle_length
        if tau < 2:
            raise ConfigError("DMTD with cycle_length 1 is ordinary decoding and has nothing to train")
        if self.horizon is None:
            self.horizon = tau - 1
        if self.stride is None:
            self.stride = tau
