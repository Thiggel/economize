"""The prompt-lookup technique."""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

from ...core.metadata import Family, Saving, TechniqueInfo
from ...core.technique import Capability, Technique, register_technique
from .._drafting import NotWrappable
from .config import LookupConfig

if TYPE_CHECKING:
    from ...arch import ArchAdapter
    from ...runtime import LayerRunner
    from .drafter import LookupDrafter

__all__ = ["LookupTechnique"]


@register_technique
class LookupTechnique(NotWrappable, Technique):
    """Prompt-lookup drafting: drafts are copied from earlier text that ended in the same n-gram.

    The technique has no parameters and changes nothing in the base model. It is lossless with every lossless
    verifier and helps where the output repeats its context (code edits, retrieval, summaries). Its drafts are
    copied without logits, so a drafter that restricts logits (FR-Spec) cannot wrap it; a hybrid uses it as its
    primary drafter.
    """

    name = LookupConfig.technique
    config_class = LookupConfig
    info = TechniqueInfo(family=Family.DRAFTING, saves=Saving.LATENCY, lossless=True)
    capabilities = Capability.DRAFTER
    wrap_hint: ClassVar[str] = " (lookup drafts are copied, without logits to restrict)"
    config: LookupConfig

    def __init__(self, config: LookupConfig, adapter: ArchAdapter) -> None:
        super().__init__(config, adapter)

    def drafter(self, runner: LayerRunner) -> LookupDrafter:
        from .drafter import LookupDrafter

        return LookupDrafter(self.config, self.adapter.vocab_size)

    def extra_repr(self) -> str:
        c = self.config
        return f"ngram={c.min_ngram}..{c.max_ngram}, max_proposal={c.max_proposal}, match={c.match}"
