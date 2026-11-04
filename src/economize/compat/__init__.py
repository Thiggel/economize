"""Compatibility with checkpoints and logs of economize's predecessors.

* :mod:`.tskip`: the ``jump.pt`` checkpoints of the research code ``tskip`` (the ``"tskip"``, ``"tskip_eagle3"``
  and ``"tskip_medusa"`` checkpoint formats, registered in :data:`economize.core.FORMATS` when this package is
  imported, which ``import economize`` does) and its cost-log naming.
"""

from __future__ import annotations

from . import tskip

__all__ = ["tskip"]
