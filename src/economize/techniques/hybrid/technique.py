"""The hybrid technique: prompt lookup with a fallback drafter."""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

from ...core.config import TechniqueConfig
from ...core.metadata import Family, Saving, TechniqueInfo
from ...core.technique import Capability, Technique, register_technique
from ...errors import ConfigError
from .._drafting import NotWrappable, PartialDrafter, WrappingTechnique
from .config import HybridConfig

if TYPE_CHECKING:
    from ...arch import ArchAdapter
    from ...runtime import LayerRunner
    from ..lookup.config import LookupConfig
    from .drafter import HybridDrafter

__all__ = ["HybridTechnique"]


@register_technique
class HybridTechnique(NotWrappable, WrappingTechnique):
    """Prompt lookup (or another primary drafter) where it finds a draft, a fallback drafter elsewhere.

    The fallback technique is built from ``config.fallback`` and held as the submodule :attr:`fallback`, the
    primary one (the lookup unless ``config.primary`` names another) as :attr:`primary`; attaching, detaching and
    refreshing the hybrid does the same to both (:class:`~economize.techniques._drafting.WrappingTechnique`). Train the
    fallback on its own (e.g. an EAGLE-3 drafter with ``eagle3_ttt``) and load its weights with
    ``hybrid.fallback.load_state_dict(...)``, or save a model that already holds the trained fallback inside a
    hybrid. Another drafter cannot wrap the hybrid; it wraps the fallback instead.
    """

    name = HybridConfig.technique
    config_class = HybridConfig
    info = TechniqueInfo(family=Family.DRAFTING, saves=Saving.LATENCY, lossless=True, trainable=("fallback",))
    capabilities = Capability.DRAFTER
    wrap_hint: ClassVar[str] = " (wrap the hybrid's fallback instead)"
    wrapped: ClassVar[dict[str, str]] = {"fallback": "fallback", "primary": "primary"}
    config: HybridConfig
    fallback: Technique
    primary: Technique

    def __init__(self, config: HybridConfig, adapter: ArchAdapter) -> None:
        super().__init__(config, adapter)

    @property
    def lookup_config(self) -> LookupConfig:
        """The configuration of the built-in lookup (the primary drafter unless ``config.primary`` is set)."""
        return self.config.lookup_config()

    def wrapped_config(self, attribute: str) -> TechniqueConfig:
        if attribute == "primary" and self.config.primary is None:
            return self.config.lookup_config()
        return super().wrapped_config(attribute)

    def wraps_drafter(self, attribute: str) -> bool:
        return attribute != "primary"  # the primary hands rows over instead of being wrapped

    def drafter(self, runner: LayerRunner) -> HybridDrafter:
        from .drafter import HybridDrafter

        primary = self.primary.drafter(runner)
        if not isinstance(primary, PartialDrafter):
            raise ConfigError(f"the primary drafter of a hybrid must serve some rows and leave the others "
                              f"(a PartialDrafter); {self.primary.name!r} drafts {type(primary).__name__}")
        return HybridDrafter(primary, self.fallback.drafter(runner),
                             self.config.max_drafts, self.adapter.vocab_size)

    def extra_repr(self) -> str:
        c = self.config
        if c.primary is not None:
            return f"primary={c.primary.technique}, max_drafts={c.max_drafts}"
        return (
            f"ngram={c.min_ngram}..{c.max_ngram}, max_proposal={c.max_proposal}, max_drafts={c.max_drafts}, "
            f"match={c.match}"
        )
