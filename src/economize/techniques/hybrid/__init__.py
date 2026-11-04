"""Hybrid drafting: prompt lookup first, any other drafter where the lookup finds no match.

Example::

    import economize
    from economize.techniques.eagle3 import Eagle3Config
    from economize.techniques.hybrid import HybridConfig

    model = economize.apply(base, HybridConfig(fallback=Eagle3Config()))
"""

from .config import HybridConfig
from .drafter import HybridDrafter, merge_drafts
from .technique import HybridTechnique

__all__ = ["HybridConfig", "HybridDrafter", "HybridTechnique", "merge_drafts"]
