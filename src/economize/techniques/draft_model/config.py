"""Configuration of two-model speculative decoding with a separate draft LM."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from ...core.config import TechniqueConfig
from ...errors import ConfigError
from ...validation import check_choice

__all__ = ["DraftModelConfig"]

_DTYPES = ("auto", "float32", "bfloat16", "float16")


@dataclass(frozen=True)
class DraftModelConfig(TechniqueConfig):
    """A smaller causal LM of the same tokenizer drafts for the model (Leviathan et al., 2023; Chen et al., 2023).

    The draft LM keeps its own KV cache. Every cycle it first catches up on the committed tokens it has not seen
    (the token of the last exact pass, plus the last accepted draft after a fully accepted cycle; the prompt in the
    first cycle) in one call that also yields the first draft, then drafts one token per call.

    Attributes:
        model: Hugging Face model id or local directory of the draft LM. Its weights are not part of the economize
            checkpoint; they are loaded from here whenever the technique attaches.
        revision: Revision of ``model`` on the Hub.
        dtype: Compute dtype of the draft LM; ``"auto"`` uses the dtype of the base model.
        attn_implementation: Attention implementation the draft LM is loaded with.
    """

    technique: ClassVar[str] = "draft_model"

    model: str = ""
    revision: str | None = None
    dtype: str = "auto"
    attn_implementation: str = "sdpa"

    def __post_init__(self) -> None:
        if not self.model:
            raise ConfigError("DraftModelConfig needs the draft LM: model='<hub id or directory>'")
        check_choice("dtype", self.dtype, _DTYPES)
