"""Narrowing for the export helpers of the drafters: an exporter hands over the technique it writes as a plain
:class:`~torch.nn.Module`, and the spec of each drafter reads the attributes of its own technique class."""

from __future__ import annotations

from typing import TypeVar

from torch import nn

__all__ = ["technique_as"]

T = TypeVar("T", bound=nn.Module)


def technique_as(technique: nn.Module, *classes: type[T]) -> T:
    """``technique``, which must be an instance of one of ``classes``.

    Raises:
        TypeError: It is none of them (the exporter was registered for another technique).
    """
    if not isinstance(technique, classes):
        names = " or ".join(cls.__name__ for cls in classes)
        raise TypeError(f"expected a {names} technique, got {type(technique).__name__}")
    return technique
