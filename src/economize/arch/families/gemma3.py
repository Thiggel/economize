"""Gemma-3 text decoders and the multimodal wrapper (``gemma3_text``, ``gemma3``).

Both keep one rotary table per layer type. ``Gemma3ForCausalLM`` soft-caps its logits; Hugging Face's
``Gemma3ForConditionalGeneration`` does not (as of ``transformers`` 5.18), and the adapter follows the model it wraps.
"""

from __future__ import annotations

from ..base import register_adapter
from ..standard import SoftcappedLogitsMixin, StandardDecoderAdapter

__all__ = ["Gemma3Adapter", "Gemma3WrapperAdapter"]


@register_adapter("gemma3_text")
class Gemma3Adapter(SoftcappedLogitsMixin, StandardDecoderAdapter):
    """Gemma-3 text decoders (one rotary table per layer type, logit soft-capping)."""


@register_adapter("gemma3")
class Gemma3WrapperAdapter(StandardDecoderAdapter):
    """The multimodal ``Gemma3ForConditionalGeneration``: the text decoder at ``model.model.language_model`` and
    uncapped logits."""
