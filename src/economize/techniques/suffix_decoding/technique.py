"""The SuffixDecoding technique."""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

from ...core.metadata import Family, Saving, TechniqueInfo
from ...core.technique import Capability, Technique, register_technique
from .._drafting import NotWrappable
from .config import SuffixDecodingConfig

if TYPE_CHECKING:
    from ...arch import ArchAdapter
    from ...runtime import LayerRunner
    from .drafter import SuffixDecodingDrafter

__all__ = ["SuffixDecodingTechnique"]


@register_technique
class SuffixDecodingTechnique(NotWrappable, Technique):
    """SuffixDecoding (Oliaro et al., 2025): drafts grow from the context's matches in a suffix structure over earlier
    responses and over the row's own tokens, as long as their empirical probability stays high.

    The technique has no parameters and changes nothing in the base model; its state is the optional datastore
    folder named by the configuration, which is read (once per process) when decoding starts. It is lossless with
    every lossless verifier and helps where outputs repeat (agentic loops, code, rewriting). Its drafts are copied
    without logits, so a drafter that restricts logits cannot wrap it; a hybrid uses it as its primary drafter.
    """

    name = SuffixDecodingConfig.technique
    config_class = SuffixDecodingConfig
    info = TechniqueInfo(family=Family.DRAFTING, saves=Saving.LATENCY, lossless=True)
    capabilities = Capability.DRAFTER
    wrap_hint: ClassVar[str] = " (SuffixDecoding drafts are retrieved, without logits to restrict)"
    config: SuffixDecodingConfig

    def __init__(self, config: SuffixDecodingConfig, adapter: ArchAdapter) -> None:
        super().__init__(config, adapter)

    def drafter(self, runner: LayerRunner) -> SuffixDecodingDrafter:
        from ..retrieval import load_index
        from .drafter import SuffixDecodingDrafter

        index = load_index(self.config.datastore) if self.config.datastore else None
        return SuffixDecodingDrafter(self.config, index, self.adapter.vocab_size)

    def extra_repr(self) -> str:
        c = self.config
        sources = ", ".join(s for s, on in (("global", bool(c.datastore)), ("local", c.local)) if on)
        return (f"sources={sources}, depth={c.max_depth}, factor={c.max_spec_factor}, "
                f"min_prob={c.min_token_prob}, {'tree' if c.tree else 'path'}")
