"""A small generic registry with lazy discovery of third-party plugins.

Everything in economize that users may extend (techniques, architectures, checkpoint formats, quantization
formats, verifiers, draft-length policies, attention masks) lives in a :class:`Registry`. A new entry is added by
registration, never by editing an economize module::

    from economize.techniques.quant import QUANT_FORMATS, QuantGrid

    class MyGrid(QuantGrid):
        ...

    QUANT_FORMATS.register("my_format", lambda symmetric: MyGrid())

Installed packages can also contribute entries through a Python entry-point group, e.g. in ``pyproject.toml``::

    [project.entry-points."economize.quant_formats"]
    my_format = "my_pkg.formats:my_format"

Entry points are loaded lazily, the first time a name is missing or the full listing is requested. A plugin that
fails to load is logged and recorded in :attr:`Registry.unavailable`, which the error of an unknown name lists.

Built-in entries register when their module is imported; a package of them calls :func:`import_submodules` in its
``__init__``, so adding a module adds its entries and no list is edited.
"""

from __future__ import annotations

import importlib
import logging
import pkgutil
from collections.abc import Callable, Iterable, Iterator
from importlib import metadata
from typing import Generic, TypeVar, overload

from .errors import DuplicateNameError, UnknownNameError

__all__ = ["Registry", "import_submodules"]

T = TypeVar("T")
logger = logging.getLogger(__name__)


class Registry(Generic[T]):
    """Maps names to objects of one kind.

    Args:
        kind: Human-readable name of the kind of entry, used in error messages.
        entry_point_group: Optional entry-point group scanned for plugins on the first lookup miss.

    Attributes:
        unavailable: Sources of entries that could not be loaded (a broken plugin, a module with a missing optional
            dependency), mapped to the reason. Loaders outside the registry add their own failures here; the error
            of an unknown name lists them.
    """

    def __init__(self, kind: str, entry_point_group: str | None = None) -> None:
        self.kind = kind
        self.entry_point_group = entry_point_group
        self.unavailable: dict[str, str] = {}
        self._entries: dict[str, T] = {}
        self._plugins_loaded = entry_point_group is None

    @overload
    def register(self, name: str, obj: None = None, *, override: bool = False) -> Callable[[T], T]: ...

    @overload
    def register(self, name: str, obj: T, *, override: bool = False) -> T: ...

    def register(self, name: str, obj: T | None = None, *, override: bool = False):
        """Register ``obj`` under ``name``. Without ``obj`` this returns a decorator.

        Raises:
            DuplicateNameError: ``name`` is taken and ``override`` is false.
        """

        def add(value: T) -> T:
            if not override and name in self._entries and self._entries[name] is not value:
                raise DuplicateNameError(
                    f"{self.kind} {name!r} is already registered as {self._entries[name]!r}; "
                    "pass override=True to replace it"
                )
            self._entries[name] = value
            return value

        return add if obj is None else add(obj)

    def unregister(self, name: str) -> T:
        """Remove and return the entry ``name`` (plugins are loaded first, so that a plugin entry can be removed and
        stays removed)."""
        if name not in self._entries:
            self._load_plugins()
        try:
            return self._entries.pop(name)
        except KeyError:
            raise self._unknown(name) from None

    def get(self, name: str) -> T:
        """Return the entry ``name``, loading plugins first if it is not registered yet.

        Raises:
            UnknownNameError: no entry of that name exists; the message lists the known names.
        """
        if name not in self._entries:
            self._load_plugins()
        try:
            return self._entries[name]
        except KeyError:
            raise self._unknown(name) from None

    def names(self) -> list[str]:
        """All registered names, plugins included, sorted."""
        self._load_plugins()
        return sorted(self._entries)

    def __contains__(self, name: object) -> bool:
        if name not in self._entries:
            self._load_plugins()
        return name in self._entries

    def __iter__(self) -> Iterator[str]:
        return iter(self.names())

    def __len__(self) -> int:
        return len(self.names())

    def __repr__(self) -> str:
        return f"Registry({self.kind!r}, entries={sorted(self._entries)})"

    def _unknown(self, name: str) -> UnknownNameError:
        known = ", ".join(repr(n) for n in sorted(self._entries)) or "none"
        message = f"unknown {self.kind} {name!r}; registered: {known}"
        if self.unavailable:
            failed = "; ".join(f"{source}: {reason}" for source, reason in sorted(self.unavailable.items()))
            message += f"; could not be loaded: {failed}"
        return UnknownNameError(message)

    def _load_plugins(self) -> None:
        if self._plugins_loaded:
            return
        self._plugins_loaded = True
        if self.entry_point_group is None:  # a registry without plugins
            return
        for ep in metadata.entry_points(group=self.entry_point_group):
            if ep.name in self._entries:
                continue
            try:
                self.register(ep.name, ep.load())
            except Exception as error:  # a broken plugin must not break the whole library
                logger.exception("could not load %s plugin %r from %s", self.kind, ep.name, ep.value)
                self.unavailable[ep.name] = f"plugin {ep.value} failed to load ({type(error).__name__}: {error})"


def import_submodules(package: str, path: Iterable[str], *, packages_only: bool = True,
                      unavailable: dict[str, str] | None = None) -> list[str]:
    """Import the submodules of ``package`` in name order (names with a leading underscore hold shared code and are
    skipped), so that each registers its entries, and return the names of those imported.

    Args:
        package: The package's qualified name (``__name__`` in its ``__init__``).
        path: The package's ``__path__``.
        packages_only: Import subpackages only, not plain modules.
        unavailable: Where a submodule that fails to import is recorded (name -> reason) and skipped: a missing
            third-party module is logged at INFO level, any other error at WARNING level with its traceback. ``None``
            lets the error propagate (for submodules that must import).
    """
    names = []
    for module in pkgutil.iter_modules(path):
        if (packages_only and not module.ispkg) or module.name.startswith("_"):
            continue
        qualified = f"{package}.{module.name}"
        try:
            importlib.import_module(qualified)
        except Exception as error:  # one broken submodule must not break the others
            if unavailable is None:
                raise
            dependency = _missing_dependency(error)
            if dependency is not None:
                reason = f"optional dependency {dependency!r} is not installed ({error})"
                logger.info("economize: %s not available: %s", qualified, reason)
            else:
                reason = f"{type(error).__name__}: {error}"
                logger.warning("economize: %s failed to import and is skipped (%s)", qualified, reason, exc_info=True)
            unavailable[module.name] = reason
        else:
            names.append(module.name)
    return names


def _missing_dependency(error: BaseException) -> str | None:
    """The name of the missing third-party module if ``error`` is just that, else ``None``."""
    if isinstance(error, ModuleNotFoundError) and error.name and not error.name.startswith("economize"):
        return error.name.split(".")[0]
    return None
