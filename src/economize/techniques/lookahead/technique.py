"""The lookahead decoding technique."""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

from ...core.metadata import Family, Saving, TechniqueInfo
from ...core.technique import Capability, Technique, register_technique
from .._drafting import NotWrappable
from .config import LookaheadConfig

if TYPE_CHECKING:
    from ...arch import ArchAdapter
    from ...runtime import LayerRunner
    from .drafter import LookaheadDrafter

__all__ = ["LookaheadTechnique"]


@register_technique
class LookaheadTechnique(NotWrappable, Technique):
    """Lookahead decoding: n-grams from a Jacobi window of the model itself, verified in the same exact pass.

    The technique has no parameters and changes nothing in the base model. It needs a verifier that verifies
    trees (``"greedy"``, ``"rejection"`` or ``"typical"``) and is lossless with the lossless ones. Every exact pass
    also runs the window (``(N - 1) W - 1`` extra tokens once it is full), which the cost log counts. Its drafts
    are trees without logits, so no other drafter can wrap it.
    """

    name = LookaheadConfig.technique
    config_class = LookaheadConfig
    info = TechniqueInfo(family=Family.DRAFTING, saves=Saving.LATENCY, lossless=True)
    capabilities = Capability.DRAFTER
    wrap_hint: ClassVar[str] = " (lookahead drafts trees without logits to restrict)"
    config: LookaheadConfig

    def __init__(self, config: LookaheadConfig, adapter: ArchAdapter) -> None:
        super().__init__(config, adapter)

    def drafter(self, runner: LayerRunner) -> LookaheadDrafter:
        from .drafter import LookaheadDrafter

        return LookaheadDrafter(self.config, self.adapter.vocab_size)

    def extra_repr(self) -> str:
        c = self.config
        return f"window={c.window}, ngram={c.ngram}, guesses={c.pool_size}, pool_from_prompt={c.pool_from_prompt}"
