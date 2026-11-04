"""OLMo-2 (``olmo2``)."""

from __future__ import annotations

from ..base import register_adapter
from ..standard import StandardDecoderAdapter

__all__ = ["Olmo2Adapter"]


@register_adapter("olmo2")
class Olmo2Adapter(StandardDecoderAdapter):
    """OLMo-2 (post-norms, query/key norms)."""
