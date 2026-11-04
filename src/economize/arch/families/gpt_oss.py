"""OpenAI gpt-oss (``gpt_oss``)."""

from __future__ import annotations

from typing import ClassVar

from ..base import register_adapter
from ..standard import StandardDecoderAdapter

__all__ = ["GptOssAdapter"]


@register_adapter("gpt_oss")
class GptOssAdapter(StandardDecoderAdapter):
    """gpt-oss: alternating sliding and full layers with attention sinks. ``sdpa`` ignores the sinks, so the model
    runs on the eager kernel or on ``"economize"``, which adds the sinks to its softmax and sends the calls it cannot
    take (prefill) to the model's eager function."""

    attention_implementations: ClassVar[tuple[str, ...]] = ("eager", "economize")
