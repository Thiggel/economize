"""Built-in techniques.

Importing this package imports every subpackage, which registers its techniques in :data:`economize.TECHNIQUES`. A
subpackage that cannot be imported is skipped, so a broken or missing optional dependency of one technique never
breaks the others: a missing third-party module is logged at INFO level, any other error at WARNING level with its
traceback. :data:`unavailable` maps every skipped subpackage to the reason; using one of its techniques raises
:class:`~economize.UnknownNameError` naming the registered ones and the skipped subpackages with their reasons.
Third-party techniques register through the ``economize.techniques`` entry-point group (loaded on first lookup) or by
importing a module that applies :func:`economize.core.register_technique`.
"""

from __future__ import annotations

from ..core.technique import TECHNIQUES
from ..registry import import_submodules

#: Subpackages that could not be imported, mapped to the reason (for diagnostics, e.g.
#: ``economize.techniques.unavailable``).
#: The same dict as ``TECHNIQUES.unavailable``, so that an unknown technique's error lists them.
unavailable: dict[str, str] = TECHNIQUES.unavailable


def _import_subpackages() -> list[str]:
    return import_submodules(__name__, __path__, unavailable=unavailable)


__all__: list[str] = []
__all__.extend(_import_subpackages())
