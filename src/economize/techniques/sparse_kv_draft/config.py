"""Configurations of long-context self-speculation on a sparse view of the KV cache (MagicDec, TriForce)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar

from ...core.config import TechniqueConfig
from ...errors import ConfigError
from ...validation import check_non_negative, check_positive, check_registered
from .path import sparse_layers
from .views import DRAFT_VIEWS

if TYPE_CHECKING:
    from ...arch import ArchAdapter

__all__ = ["MagicDecConfig", "TriForceConfig"]


def _check_model(name: str, adapter: ArchAdapter, implementations: tuple[str, ...]) -> None:
    """Reject a model without full-attention layers or with an attention implementation the drafter cannot mask."""
    if not sparse_layers(adapter):
        raise ConfigError(f"{name} drafts with a sparse view of the full-attention layers' KV cache, and the model "
                          "has no full-attention layer")
    implementation = adapter.attn_implementation
    if implementation not in implementations:
        raise ConfigError(f"{name} needs one of the attention implementations {list(implementations)} "
                          f"(the model was loaded with {implementation!r})")


@dataclass(frozen=True)
class MagicDecConfig(TechniqueConfig):
    """Hyperparameters of the MagicDec self-drafter (Sadhukhan et al., 2025, arXiv 2408.11049).

    A draft step runs the whole model, and each of its full-attention layers reads a StreamingLLM view of the cache:
    the first ``sinks`` positions and the last ``budget - sinks`` positions up to the query. Sliding-window,
    chunked and recurrent layers run unchanged. The exact pass reads the whole cache, so decoding stays lossless.

    Attributes:
        budget: KV entries one draft query reads in a full-attention layer (sinks plus window).
        sinks: Leading positions every draft query reads (StreamingLLM's attention sinks).
        view: How a full-attention layer reads the budget (:data:`~.views.DRAFT_VIEWS`). ``"gathered"`` gathers the
            budget's entries, so the draft attention reads ``budget`` entries per query at any context length;
            ``"masked"`` is the numerical reference, which masks the other entries and saves their reads only on
            economize attention.
    """

    technique: ClassVar[str] = "magicdec"

    budget: int = 512
    sinks: int = 4
    view: str = "gathered"

    def __post_init__(self) -> None:
        check_positive("budget", self.budget)
        check_non_negative("sinks", self.sinks)
        check_registered("draft view", self.view, DRAFT_VIEWS)
        if self.sinks >= self.budget:
            raise ConfigError(f"sinks={self.sinks} leaves no window in budget={self.budget}")

    @property
    def window(self) -> int:
        """Recent positions a draft query reads, its own included."""
        return self.budget - self.sinks

    def validate(self, adapter: ArchAdapter) -> None:
        _check_model(self.technique, adapter, ("sdpa", "eager", "economize"))


@dataclass(frozen=True)
class TriForceConfig(TechniqueConfig):
    """Hyperparameters of the TriForce retrieval self-drafter (Sun et al., 2024, arXiv 2404.11912).

    The prompt part of each full-attention layer's cache is cut into chunks of ``chunk_size`` slots. Once per
    generation, the first draft step runs densely and scores every chunk by the dot product of the step's query with
    the chunk's mean key. The first chunk and the ``budget // chunk_size - 1`` best others are kept, per KV head;
    the slots after the last whole chunk of the prompt and every generated position are always kept. Later draft
    steps attend to the kept slots only. The exact pass reads the whole cache, so decoding stays lossless.

    Attributes:
        budget: Prompt slots one draft query reads per KV head in a full-attention layer, besides the always-kept
            tail (``budget // chunk_size`` chunks).
        chunk_size: Slots per retrieval chunk.
        view: How a full-attention layer reads the selection (:data:`~.views.DRAFT_VIEWS`). ``"gathered"`` gathers
            the selected chunks and the tail, so the draft attention reads ``budget`` prompt entries plus the tail
            at any prompt length; ``"masked"`` is the numerical reference, which masks the other entries and saves
            their reads only on economize attention (whose decode kernel skips a 64-slot key block when the mask
            hides all of it).
    """

    technique: ClassVar[str] = "triforce"

    budget: int = 4096
    chunk_size: int = 64
    view: str = "gathered"

    def __post_init__(self) -> None:
        check_positive("budget", self.budget)
        check_positive("chunk_size", self.chunk_size)
        check_registered("draft view", self.view, DRAFT_VIEWS)
        if self.budget < self.chunk_size:
            raise ConfigError(f"budget={self.budget} holds no chunk of chunk_size={self.chunk_size}")

    @property
    def select_sets(self) -> int:
        """Chunks kept per KV head (the official code's ``select_sets``)."""
        return self.budget // self.chunk_size

    def validate(self, adapter: ArchAdapter) -> None:
        # the selection step reads the query and keys of scaled_dot_product_attention (economize attention runs
        # its sdpa fallback while the tap is active); eager attention never calls it
        _check_model(self.technique, adapter, ("sdpa", "economize"))
