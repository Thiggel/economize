"""Configuration of structured width pruning (``width_prune``: FLAP, LLM-Pruner, Minitron)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar

from ...core.config import TechniqueConfig
from ...errors import ConfigError
from ...validation import (
    check_choice,
    check_non_negative,
    check_number,
    check_positive,
    check_registered,
    check_unit_interval,
)

if TYPE_CHECKING:
    from ...arch import ArchAdapter

__all__ = ["STRUCTURES", "WidthPruneConfig"]

#: ``"uniform"``: every pruned layer keeps the same number of key/value groups and FFN channels (FLAP's ``UL-UM``,
#: LLM-Pruner's and Minitron's setting); ``"adaptive"``: one global threshold on standardized scores, so the widths
#: differ from layer to layer (FLAP's ``AL-AM``).
STRUCTURES = ("uniform", "adaptive")


@dataclass(frozen=True)
class WidthPruneConfig(TechniqueConfig):
    """Remove attention heads, FFN channels and (Minitron) residual channels, which leaves a smaller dense model.

    Heads are removed in key/value groups: a group is one key and value head and the query heads that share them
    (one query head without GQA), so the attention keeps its layout. The pruned model has the same modules and
    state-dict keys as the original with smaller weights; when every layer keeps the same widths, the config is
    rewritten (``num_attention_heads``, ``num_key_value_heads``, ``intermediate_size``, ``hidden_size`` and an
    explicit ``head_dim``) and the model saves and serves as a plain checkpoint.

    Args:
        method: Registered importance criterion (:data:`~economize.techniques.width_prune.WIDTH_CRITERIA`):
            ``"flap"`` (An et al., AAAI 2024: fluctuation of the inputs times the squared column norms, with a bias
            that compensates for the removed channels), ``"llm_pruner"`` (Ma et al., NeurIPS 2023: first-order
            Taylor importance ``|W * dL/dW|`` summed over each coupled group) or ``"minitron"`` (Muralidharan et
            al., NeurIPS 2024: activation magnitudes, mean over the sequence and L2 over the batch).
        sparsity: Fraction of the prunable units removed. With ``structure="uniform"`` it is the fraction of key/value
            groups and FFN channels removed in each pruned layer (``attn_sparsity`` and ``mlp_sparsity`` override
            it per block type); with ``"adaptive"`` it is the fraction of the attention and FFN parameters of the
            whole model removed (FLAP's ``pruning_ratio``), spread over layers by the global threshold.
        attn_sparsity: Fraction of key/value groups removed per layer (``"uniform"`` only; ``None``: ``sparsity``).
        mlp_sparsity: Fraction of FFN channels removed per layer (``"uniform"`` only; ``None``: ``sparsity``).
        hidden_sparsity: Fraction of residual-stream channels removed in every layer, the embedding and the head
            (criteria that score the residual stream, ``"minitron"``).
        structure: ``"uniform"`` or ``"adaptive"`` (``None``: the criterion's default, ``"adaptive"`` for FLAP and
            ``"uniform"`` for the others).
        keep_first: Leading layers whose heads and channels are kept (``None``: the criterion's default, 4 for
            LLM-Pruner as its Llama scripts, else 0). Residual channels are removed everywhere.
        keep_last: Trailing layers whose heads and channels are kept (``None``: 2 for LLM-Pruner, else 0).
        round_to: Kept FFN channels per layer are a multiple of this (``"uniform"`` only).
        compensate: Add FLAP's bias ``E[x_removed] W_removed^T`` to the attention output and down maps (``None``:
            the criterion's default, ``True`` for FLAP; only criteria that compensate accept it).
        lora_rank: Rank of a LoRA update of the seven maps of every layer, trained after pruning (LLM-Pruner's
            recovery stage; ``0``: none). It is the technique's only trainable part.
        lora_alpha: LoRA scaling numerator (the update is multiplied by ``lora_alpha / lora_rank``).
        merge_lora: Multiply the recovery LoRA into the pruned weights when the technique attaches, so that the model
            runs at the cost of the pruned model alone and later weight transforms (quantization, sparsity) apply to
            the merged weights. The LoRA is then no longer trainable; its parameters stay in the technique's state,
            so a saved technique reloads (and merges) exactly. Set it for inference after the recovery training.
        offload: Keep calibration activations on the CPU between layers.
        gradient_chunk: Pruned layers whose weight gradients a gradient-based criterion (``"llm_pruner"``) holds at
            once. The calibration batches run forward and backward once per chunk of this many layers, and each
            chunk's gradients are reduced to unit scores and freed before the next, which bounds the extra memory to
            the gradients of one chunk (``0``: all pruned layers in one pass). The scores do not depend on it.
    """

    technique: ClassVar[str] = "width_prune"

    method: str = "flap"
    sparsity: float = 0.2
    attn_sparsity: float | None = None
    mlp_sparsity: float | None = None
    hidden_sparsity: float = 0.0
    structure: str | None = None
    keep_first: int | None = None
    keep_last: int | None = None
    round_to: int = 1
    compensate: bool | None = None
    lora_rank: int = 0
    lora_alpha: float = 16.0
    merge_lora: bool = False
    offload: bool = False
    gradient_chunk: int = 0

    def __post_init__(self) -> None:
        from .criteria import WIDTH_CRITERIA

        check_registered("width-pruning method", self.method, WIDTH_CRITERIA)
        check_unit_interval("sparsity", self.sparsity)
        for name in ("attn_sparsity", "mlp_sparsity"):
            value = getattr(self, name)
            if value is not None:
                check_unit_interval(name, value)
        check_unit_interval("hidden_sparsity", self.hidden_sparsity)
        if self.structure is not None:
            check_choice("structure", self.structure, STRUCTURES)
        for name in ("keep_first", "keep_last"):
            value = getattr(self, name)
            if value is not None:
                check_non_negative(name, value)
        check_positive("round_to", self.round_to)
        check_non_negative("lora_rank", self.lora_rank)
        check_number("lora_alpha", self.lora_alpha, low=0, closed_low=False)
        check_non_negative("gradient_chunk", self.gradient_chunk)
        if self.merge_lora and not self.lora_rank:
            raise ConfigError("merge_lora needs a recovery LoRA (lora_rank > 0)")
        criterion = WIDTH_CRITERIA.get(self.method)
        if self.resolved_structure == "adaptive" and (self.attn_sparsity is not None or self.mlp_sparsity is not None):
            raise ConfigError("structure='adaptive' spreads one global sparsity over attention and MLP; "
                              "attn_sparsity / mlp_sparsity need structure='uniform'")
        if self.resolved_structure == "adaptive" and not criterion.adaptive:
            raise ConfigError(f"method {self.method!r} has no adaptive structure (its scores are not comparable "
                              "across layers); use structure='uniform'")
        if self.hidden_sparsity > 0 and not criterion.scores_hidden:
            raise ConfigError(f"method {self.method!r} does not score residual channels; hidden_sparsity needs a "
                              "criterion that does ('minitron')")
        if self.compensate and not criterion.compensates:
            raise ConfigError(f"method {self.method!r} has no bias compensation (FLAP's); leave compensate unset")
        if self.gradient_chunk and not criterion.uses_gradients:
            raise ConfigError(f"method {self.method!r} takes no weight gradients; gradient_chunk needs a criterion "
                              "that does ('llm_pruner')")

    @property
    def resolved_structure(self) -> str:
        """The resolved ``structure``."""
        from .criteria import WIDTH_CRITERIA

        return self.structure or WIDTH_CRITERIA.get(self.method).structure

    @property
    def first(self) -> int:
        """The resolved ``keep_first``."""
        from .criteria import WIDTH_CRITERIA

        return WIDTH_CRITERIA.get(self.method).keep_first if self.keep_first is None else self.keep_first

    @property
    def last(self) -> int:
        """The resolved ``keep_last``."""
        from .criteria import WIDTH_CRITERIA

        return WIDTH_CRITERIA.get(self.method).keep_last if self.keep_last is None else self.keep_last

    @property
    def compensates(self) -> bool:
        """The resolved ``compensate``."""
        from .criteria import WIDTH_CRITERIA

        return WIDTH_CRITERIA.get(self.method).compensates if self.compensate is None else self.compensate

    @property
    def attn_ratio(self) -> float:
        """Fraction of key/value groups removed per pruned layer (``"uniform"``)."""
        return self.sparsity if self.attn_sparsity is None else self.attn_sparsity

    @property
    def mlp_ratio(self) -> float:
        """Fraction of FFN channels removed per pruned layer (``"uniform"``)."""
        return self.sparsity if self.mlp_sparsity is None else self.mlp_sparsity

    @property
    def uniform_widths(self) -> bool:
        """Whether every layer of the pruned model has the same widths, so that its config describes it and it
        serves as a plain checkpoint (``"uniform"`` with no protected layers, or nothing pruned per layer)."""
        if self.resolved_structure == "adaptive":
            return False
        prunes_layers = self.attn_ratio > 0 or self.mlp_ratio > 0
        return not prunes_layers or (self.first == 0 and self.last == 0)

    @property
    def prunes_heads(self) -> bool:
        """Whether attention heads may be removed."""
        return self.sparsity > 0 if self.resolved_structure == "adaptive" else self.attn_ratio > 0

    def pruned_layers(self, num_layers: int) -> list[int]:
        """The 1-indexed layers whose heads and channels are pruned."""
        return list(range(1 + self.first, num_layers - self.last + 1))

    def validate(self, adapter: ArchAdapter) -> None:
        from .._structured.layout import layout_for

        layout_for(adapter, self.technique, residual=self.hidden_sparsity > 0)
        if adapter.tensor_parallel_size > 1:
            raise ConfigError("width_prune rewrites the shapes of the weights and needs the whole model on one process")
        if not self.pruned_layers(adapter.num_layers) and self.hidden_sparsity == 0:
            raise ConfigError(f"keep_first={self.first} and keep_last={self.last} protect all "
                              f"{adapter.num_layers} layers and hidden_sparsity is 0: nothing is pruned")
        if int((1 - self.hidden_sparsity) * adapter.hidden_size) < 1:
            raise ConfigError(f"hidden_sparsity={self.hidden_sparsity} keeps no residual channel")
