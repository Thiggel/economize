"""Deprecation warnings for names that moved or were renamed (layer 0: imports nothing of economize).

A module that keeps old names alive sets its module-level ``__getattr__`` to :func:`moved_getattr`. The ``load``
function it passes imports the new home explicitly inside the shim module, so that the layering test sees the edge.
"""

from __future__ import annotations

import sys
import warnings
from collections.abc import Callable, Iterable, Mapping
from typing import Any

__all__ = ["moved_getattr", "warn_moved"]


def warn_moved(old: str, new: str, *, stacklevel: int = 3) -> None:
    """Warn that ``old`` (a dotted name) is deprecated in favour of ``new``.

    The default ``stacklevel`` points at the caller of the shim that calls this function (the module
    ``__getattr__`` or the deprecated method).
    """
    warnings.warn(f"{old} is deprecated; use {new} instead", DeprecationWarning, stacklevel=stacklevel)


def moved_getattr(
    module: str,
    moved: Mapping[str, str] | Iterable[str],
    new_home: str | None = None,
    load: Callable[[], object] | None = None,
) -> Callable[[str], Any]:
    """A module ``__getattr__`` (PEP 562) that resolves deprecated names with a :class:`DeprecationWarning`.

    Args:
        module: the shim module's ``__name__``.
        moved: old name -> name in the new home, or an iterable of names that kept their name.
        new_home: dotted name of the new home, used in the warning (default: ``module`` itself, for renames).
        load: imports and returns the new home (default: ``module`` itself).

    Unknown names raise :class:`AttributeError` as usual. Usage::

        def _tskip() -> ModuleType:
            from ...compat import tskip

            return tskip

        __getattr__ = moved_getattr(__name__, {"OldName": "NewName"}, "economize.compat.tskip", _tskip)
    """
    names = dict(moved) if isinstance(moved, Mapping) else {name: name for name in moved}
    home = new_home or module

    def _getattr(name: str) -> Any:
        if name not in names:
            raise AttributeError(f"module {module!r} has no attribute {name!r}")
        target = load() if load is not None else sys.modules[module]
        warn_moved(f"{module}.{name}", f"{home}.{names[name]}")
        return getattr(target, names[name])

    return _getattr
