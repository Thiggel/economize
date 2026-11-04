"""Argument and configuration checks that raise :class:`~economize.errors.ConfigError`.

Every check names the offending field in its message, so that the error points at the value to change. The module
imports nothing from economize but :mod:`economize.errors`, so every layer of the library can use it.
"""

from __future__ import annotations

from collections.abc import Collection, Iterable, Sequence
from dataclasses import fields, is_dataclass
from numbers import Integral, Real
from typing import Protocol

from .errors import ConfigError, UnregisteredNameError

__all__ = [
    "check_choice",
    "check_globs",
    "check_indices",
    "check_layers",
    "check_non_negative",
    "check_number",
    "check_options",
    "check_positive",
    "check_registered",
    "check_unit_interval",
]


class _HasNumLayers(Protocol):
    """Anything that knows the depth of a model (an :class:`~economize.arch.ArchAdapter`)."""

    @property
    def num_layers(self) -> int: ...


class _NameRegistry(Protocol):
    """A registry as far as :func:`check_registered` needs it (an :class:`~economize.registry.Registry`)."""

    def __contains__(self, name: object, /) -> bool: ...

    def names(self) -> Iterable[str]: ...


def check_globs(name: str, patterns: object, *, allow_empty: bool = False) -> None:
    """``patterns`` must be a tuple of non-empty strings (module-name globs relative to a decoder layer)."""
    if not isinstance(patterns, tuple) or not all(isinstance(p, str) and p for p in patterns):
        raise ConfigError(f"{name} must be a tuple of glob strings such as ('self_attn.*', 'mlp.*'), got {patterns!r}")
    if not patterns and not allow_empty:
        raise ConfigError(f"{name} must not be empty")


def check_layers(layers: object, adapter: _HasNumLayers | None = None) -> None:
    """``layers`` must be a tuple of 1-indexed decoder layers (empty means every layer)."""
    if not isinstance(layers, tuple) or not all(isinstance(i, int) and not isinstance(i, bool) for i in layers):
        raise ConfigError(f"layers must be a tuple of 1-indexed layer numbers, got {layers!r}")
    if any(i < 1 for i in layers):
        raise ConfigError(f"layers are 1-indexed, got {layers!r}")
    if adapter is not None:
        bad = [i for i in layers if not 1 <= i <= adapter.num_layers]
        if bad:
            raise ConfigError(f"layers {bad} outside 1..{adapter.num_layers}")


def check_indices(
    name: str,
    values: object,
    *,
    low: int = 0,
    high: int | None = None,
    distinct: bool = False,
    allow_empty: bool = True,
) -> None:
    """``values`` must be a sequence of integers in ``[low, high)`` (``high=None``: no upper bound), pairwise
    distinct when ``distinct``, and non-empty unless ``allow_empty``."""
    if (isinstance(values, (str, bytes)) or not isinstance(values, Sequence)
            or not all(isinstance(i, Integral) and not isinstance(i, bool) for i in values)):
        raise ConfigError(f"{name} must be a sequence of integers, got {values!r}")
    if not values and not allow_empty:
        raise ConfigError(f"{name} must not be empty")
    bad = [i for i in values if i < low or (high is not None and i >= high)]
    if bad:
        bounds = f">= {low}" if high is None else f"in [{low}, {high})"
        raise ConfigError(f"{name} must hold integers {bounds}, got {values!r}")
    if distinct and len(set(values)) != len(values):
        raise ConfigError(f"{name} must hold distinct integers, got {values!r}")


def check_positive(name: str, value: object) -> None:
    """``value`` must be an integer >= 1 (a Python or NumPy integer, not a bool)."""
    if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
        raise ConfigError(f"{name} must be a positive integer, got {value!r}")


def check_non_negative(name: str, value: object) -> None:
    """``value`` must be an integer >= 0 (a Python or NumPy integer, not a bool)."""
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
        raise ConfigError(f"{name} must be a non-negative integer, got {value!r}")


def check_number(
    name: str,
    value: object,
    *,
    low: float | None = None,
    high: float | None = None,
    closed_low: bool = True,
    closed_high: bool = True,
) -> None:
    """``value`` must be a real number within the given bounds (``None``: unbounded on that side)."""
    if not isinstance(value, bool) and isinstance(value, Real):
        number = float(value)
        above = low is None or (number >= low if closed_low else number > low)
        below = high is None or (number <= high if closed_high else number < high)
        if above and below:
            return
    if low is not None and high is not None:
        bounds = f"lie in {'[' if closed_low else '('}{low:g}, {high:g}{']' if closed_high else ')'}"
    elif low == 0 and not closed_low:
        bounds = "be positive (a number > 0)"
    elif low is not None:
        bounds = f"be a number {'>=' if closed_low else '>'} {low:g}"
    elif high is not None:
        bounds = f"be a number {'<=' if closed_high else '<'} {high:g}"
    else:
        bounds = "be a real number"
    raise ConfigError(f"{name} must {bounds}, got {value!r}")


def check_unit_interval(name: str, value: object, *, closed_low: bool = True, closed_high: bool = False) -> None:
    """``value`` must be a real number in the unit interval (half-open ``[0, 1)`` by default)."""
    check_number(name, value, low=0, high=1, closed_low=closed_low, closed_high=closed_high)


def check_choice(name: str, value: object, choices: Collection[object]) -> None:
    """``value`` must be one of ``choices``."""
    if value not in choices:
        raise ConfigError(f"{name} must be one of {list(choices)}, got {value!r}")


def check_registered(kind: str, name: object, registry: _NameRegistry) -> None:
    """``name`` must be registered in ``registry``; ``kind`` says what it names ("pruning method").

    Raises:
        UnregisteredNameError: A :class:`ConfigError` that is also a ``KeyError``, as a registry lookup raises.
    """
    if name not in registry:
        raise UnregisteredNameError(f"unknown {kind} {name!r}; registered: {list(registry.names())}")


def check_options(config: object, owner: str, used: Collection[str], options: Collection[str]) -> None:
    """Reject options of ``config`` that ``owner`` ignores.

    ``options`` names the fields of the dataclass ``config`` that only some registered methods (or backends) read;
    ``used`` is the subset that ``owner`` reads. A field outside ``used`` must keep its default, so that a knob set
    for a method that ignores it fails loudly instead of being silently dropped.
    """
    if not is_dataclass(config):
        raise TypeError(f"check_options needs a dataclass instance, got {type(config).__name__}")
    defaults = {f.name: f.default for f in fields(config)}
    for name in options:
        if name not in used and getattr(config, name) != defaults[name]:
            reads = ", ".join(sorted(used)) or "none of them"
            raise ConfigError(
                f"{name}={getattr(config, name)!r} has no effect with {owner} (of the options {sorted(options)} "
                f"it reads {reads}); leave {name} at its default {defaults[name]!r}"
            )
