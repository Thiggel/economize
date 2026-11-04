"""Configuration of cross-layer jump drafting."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar

from ...core.config import TechniqueConfig
from ...errors import ConfigError
from ...modules import COPY_DTYPES
from ...validation import check_choice, check_indices, check_number, check_positive

if TYPE_CHECKING:
    from ...arch import ArchAdapter

__all__ = ["ADAPTATIONS", "JumpConfig"]

#: How the layers on the jump path are adapted: not at all, with LoRA, or as trainable copies.
ADAPTATIONS = ("frozen", "lora", "full")


@dataclass(frozen=True)
class JumpConfig(TechniqueConfig):
    """Hyperparameters of a jump drafter.

    A draft step predicts the next token's hidden state after ``target_layer`` (``m``) from the current token's
    state after ``source_layer`` (``u``) and the next token's embedding, runs layers ``m+1 .. exit_layer`` on it and
    reads out logits. Layers ``1..m`` and ``exit_layer+1..L`` are skipped. Layer ``0`` denotes the embedding.

    Attributes:
        target_layer: ``m``, the layer whose output the predictor produces.
        source_layer: ``u``, the layer whose output conditions the predictor (``m <= u <= exit_layer``).
        exit_layer: ``r``, the last layer run on the jump path (default: the last layer). Below the last layer,
            a readout adapter maps the state before the model's norm and head.
        extra_source_layers: Further layers (below ``u``) whose states are combined with ``u``'s by the mixer.
        mixer: Registered name of the source mixer (see ``MIXERS``). ``None`` means no mixer with a single
            source and ``"concat"`` with several.
        mixer_heads: Attention heads of the ``"layer_attention"`` mixer.
        adaptation: ``"frozen"`` (the jump path uses the model's layers as they are), ``"lora"`` (LoRA updates
            active only on the jump path) or ``"full"`` (trainable copies of layers ``m+1..r``).
        predictor: Registered name of the predictor (see ``PREDICTORS``).
        predictor_width: Hidden width of the predictor's gated layers as a multiple of the model width.
        predictor_depth: Number of gated stages of the ``"mlp"`` predictor.
        predictor_heads: Attention heads of the ``"block"`` and ``"block_linear"`` predictors.
        predictor_layers: Attention layers of the ``"block"`` and ``"block_linear"`` predictors (their depth; other
            predictors ignore it).
        flow_steps: Euler steps of the ``"flow"`` and ``"flow_linear"`` predictors' sampler.
        flow_sigma: Noise scale of the flow predictors' source distribution, in units of the per-channel residual
            scale (see :class:`~economize.techniques.jump.FlowPredictor`); ``0`` makes the flow deterministic.
        flow_width: Hidden width of the flow predictors' velocity network as a multiple of the model width.
        flow_train_noise: The flow predictors sample from a noisy start in training mode (so the divergence of the
            rollout trains on samples); ``False`` starts every training sample at the mean.
        flow_eval_noise: The flow predictors sample from a noisy start in evaluation mode and when drafting;
            ``False`` (the default) drafts deterministically from the mean.
        flow_divergence_grad: The divergence of a rollout also trains the flow predictors' velocity network through
            the sampler. ``False`` (the default) trains it by flow matching alone; the divergence still reaches the
            mean, the input projections and the jump path through the sampler's inputs.
        use_state: Condition on the source state (``False`` ablates it).
        use_token: Condition on the next token's embedding (``False`` ablates it).
        token_residual: Add the next token's embedding to the prediction.
        lora_rank: Rank of the LoRA updates (``adaptation="lora"``).
        lora_alpha: LoRA scaling numerator (default ``2 * lora_rank``).
        copy_dtype: Storage dtype of the trainable copies (``adaptation="full"``): ``"float32"``, or ``"model"``
            for the model's dtype, which halves their memory in a bf16 model and is meant for training with
            float32 master weights (``TrainConfig(offload_optimizer=True)``).
    """

    technique: ClassVar[str] = "jump"

    target_layer: int
    source_layer: int
    exit_layer: int | None = None
    extra_source_layers: tuple[int, ...] = ()
    mixer: str | None = None
    mixer_heads: int = 8
    adaptation: str = "frozen"
    predictor: str = "mlp"
    predictor_width: float = 2.0
    predictor_depth: int = 1
    predictor_heads: int = 8
    predictor_layers: int = 1
    flow_steps: int = 4
    flow_sigma: float = 1.0
    flow_width: float = 1.0
    flow_train_noise: bool = True
    flow_eval_noise: bool = False
    flow_divergence_grad: bool = False
    use_state: bool = True
    use_token: bool = True
    token_residual: bool = True
    lora_rank: int = 16
    lora_alpha: float | None = None
    copy_dtype: str = "float32"

    def __post_init__(self) -> None:
        check_choice("copy_dtype", self.copy_dtype, COPY_DTYPES)
        object.__setattr__(self, "extra_source_layers", tuple(sorted({int(l) for l in self.extra_source_layers})))
        check_choice("adaptation", self.adaptation, ADAPTATIONS)
        if not 0 <= self.target_layer <= self.source_layer:
            raise ConfigError(f"need 0 <= target_layer <= source_layer, got {self.target_layer}, {self.source_layer}")
        if self.exit_layer is not None and not self.source_layer <= self.exit_layer:
            raise ConfigError(f"need source_layer <= exit_layer, got {self.source_layer}, {self.exit_layer}")
        if self.exit_layer is not None and self.exit_layer <= self.target_layer:
            raise ConfigError("the jump path must run at least one layer: need target_layer < exit_layer")
        check_indices("extra_source_layers", self.extra_source_layers, high=self.source_layer)
        for name in ("predictor_depth", "lora_rank", "mixer_heads", "predictor_heads", "predictor_layers",
                     "flow_steps"):
            check_positive(name, getattr(self, name))
        check_number("predictor_width", self.predictor_width, low=0, closed_low=False)
        check_number("flow_width", self.flow_width, low=0, closed_low=False)
        check_number("flow_sigma", self.flow_sigma, low=0)

    # ---- derived ---------------------------------------------------------------------------------------------
    @property
    def source_layers(self) -> tuple[int, ...]:
        """All source layers, ascending; the last one is ``source_layer``."""
        return (*self.extra_source_layers, self.source_layer)

    @property
    def resolved_mixer(self) -> str | None:
        if self.mixer is not None:
            return self.mixer
        return "concat" if self.extra_source_layers else None

    @property
    def resolved_lora_alpha(self) -> float:
        return float(self.lora_alpha) if self.lora_alpha is not None else 2.0 * self.lora_rank

    def exit_layer_of(self, num_layers: int) -> int:
        return num_layers if self.exit_layer is None else self.exit_layer

    def validate(self, adapter: ArchAdapter) -> None:
        num_layers = adapter.num_layers
        r = self.exit_layer_of(num_layers)
        if r > num_layers:
            raise ConfigError(f"exit_layer {r} exceeds the model's {num_layers} layers")
        if r <= self.target_layer or self.source_layer > r:
            raise ConfigError(f"need target_layer < exit_layer and source_layer <= exit_layer, got m, u, r = "
                              f"{self.target_layer}, {self.source_layer}, {r}")
        if self.resolved_mixer == "layer_attention" and adapter.hidden_size % self.mixer_heads:
            raise ConfigError(f"hidden size {adapter.hidden_size} is not divisible by mixer_heads={self.mixer_heads}")
