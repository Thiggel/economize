"""The drafter of whole-layer skipping: the self-drafter of early exit, costed by the layers of the skip set."""

from __future__ import annotations

from ..early_exit.drafter import SelfDrafter

__all__ = ["LayerSkipDrafter"]


class LayerSkipDrafter(SelfDrafter):
    """A :class:`~economize.techniques.early_exit.SelfDrafter` that logs the layers a draft step really runs.

    The layers outside the skip set are not one range, so the technique declares them as its ``footprint``, which
    the drafter reads on every access (:attr:`~economize.decoding.Drafter.footprint`); the cost log then counts
    ``L - |skip set|`` layers per draft token and names them, also after the skip set changed.
    """
