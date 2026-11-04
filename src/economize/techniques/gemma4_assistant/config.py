"""Configuration of drafting with the Gemma-4 assistant (Google's multi-token-prediction drafter for Gemma 4)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from ...core.config import TechniqueConfig
from ...errors import ConfigError
from ...validation import check_choice

__all__ = ["Gemma4AssistantDraftConfig"]

_DTYPES = ("auto", "float32", "bfloat16", "float16")
_ATTENTION = ("sdpa", "eager")


@dataclass(frozen=True)
class Gemma4AssistantDraftConfig(TechniqueConfig):
    """Gemma 4's assistant model drafts for its base model (``Gemma4AssistantForCausalLM``).

    Google ships one assistant per Gemma-4 checkpoint as a separate repository (``google/gemma-4-31B-it-assistant``
    for ``google/gemma-4-31B-it``; the base configs have no MTP fields). The assistant is a four-layer decoder whose
    layers project no keys and values of their own: they attend over the base model's cache entries of its last
    sliding and its last full attention layer. A draft step reads the embedding of the last token and a state of
    the base width, ``x = W_pre [E[x_t]; h]``, and returns the next token's logits and a new state
    ``h' = W_post Assistant(x)``. The first step of a cycle reads the base model's final normalized state at the last
    committed token, later steps the previous step's state; every step of a cycle runs at the bonus token's position.

    Attributes:
        model: Hugging Face id or local directory of the assistant. Its weights are not part of the economize
            checkpoint; they are loaded from here whenever the technique attaches.
        revision: Revision of ``model`` on the Hub.
        dtype: Compute dtype of the assistant; ``"auto"`` uses the dtype of the base model.
        attn_implementation: Attention implementation the assistant is loaded with (``"sdpa"`` or ``"eager"``: the
            drafter builds its masks over the base model's cache buffers in that format).
    """

    technique: ClassVar[str] = "gemma4_assistant"

    model: str = "google/gemma-4-31B-it-assistant"
    revision: str | None = None
    dtype: str = "auto"
    attn_implementation: str = "sdpa"

    def __post_init__(self) -> None:
        if not self.model:
            raise ConfigError("Gemma4AssistantDraftConfig needs the assistant: model='<hub id or directory>'")
        check_choice("dtype", self.dtype, _DTYPES)
        check_choice("attn_implementation", self.attn_implementation, _ATTENTION)
