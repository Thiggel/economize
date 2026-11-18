"""Union drafting: several drafters propose for every row, and one exact pass verifies all their drafts as a tree.

Example::

    import economize
    from economize.techniques.dflash import DFlashConfig
    from economize.techniques.eagle3 import Eagle3Config
    from economize.techniques.union import UnionConfig

    model = economize.apply(base, UnionConfig(members=(DFlashConfig(), Eagle3Config())))
    ...
    print(model.techniques["union"].stats.summary())
"""

from .config import UnionConfig, UnionMode
from .drafter import CascadeDrafter, CycleRecord, ForcedRequest, UnionDrafter, UnionStats
from .merge import MergedTree, TreeBuilder, merge_drafts
from .technique import UnionTechnique, union_checkpoint

__all__ = [
    "CascadeDrafter",
    "CycleRecord",
    "ForcedRequest",
    "MergedTree",
    "TreeBuilder",
    "UnionConfig",
    "UnionDrafter",
    "UnionMode",
    "UnionStats",
    "UnionTechnique",
    "merge_drafts",
    "union_checkpoint",
]
