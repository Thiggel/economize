"""The drafter of a draft view."""

from __future__ import annotations

from ..early_exit.drafter import SelfDrafter

__all__ = ["DraftWithDrafter"]


class DraftWithDrafter(SelfDrafter):
    """The drafter of :class:`~economize.techniques.draft_with.DraftWithTechnique`: every layer of the draft path
    (the held techniques' copies of the decoder layers, minus the layers a held technique skips), then the model's
    norm and head."""
