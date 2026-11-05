"""The REST technique."""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

from ...core.metadata import Family, Saving, TechniqueInfo
from ...core.technique import Capability, Technique, register_technique
from .._drafting import NotWrappable
from .config import RestConfig

if TYPE_CHECKING:
    from ...arch import ArchAdapter
    from ...runtime import LayerRunner
    from .drafter import RestDrafter

__all__ = ["RestTechnique"]


@register_technique
class RestTechnique(NotWrappable, Technique):
    """REST: drafts retrieved from a datastore of text by the context's longest matching suffix.

    REST, retrieval-based speculative decoding (He et al., 2024): drafts are continuations of the context's
    longest suffix in a datastore of text, verified as a tree.

    The technique has no parameters and changes nothing in the base model; its state is the datastore folder named
    by the configuration, which is read (once per process) when decoding starts. It is lossless with every lossless
    verifier. Its drafts are copied without logits, so a drafter that restricts logits cannot wrap it; a hybrid
    uses it as its primary drafter.
    """

    name = RestConfig.technique
    config_class = RestConfig
    info = TechniqueInfo(family=Family.DRAFTING, saves=Saving.LATENCY, lossless=True)
    capabilities = Capability.DRAFTER
    wrap_hint: ClassVar[str] = " (REST drafts are retrieved, without logits to restrict)"
    config: RestConfig

    def __init__(self, config: RestConfig, adapter: ArchAdapter) -> None:
        super().__init__(config, adapter)

    def drafter(self, runner: LayerRunner) -> RestDrafter:
        from ..retrieval import load_index
        from .drafter import RestDrafter

        return RestDrafter(self.config, load_index(self.config.datastore), self.adapter.vocab_size)

    def extra_repr(self) -> str:
        c = self.config
        mode = f"tree={c.tree_size}" if c.tree else "chain"
        return (f"datastore={c.datastore!r}, span={c.min_span}..{c.max_span}, k={c.max_occurrences}, "
                f"depth={c.max_depth}, {mode}")
