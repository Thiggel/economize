"""Narrowing for the export helpers of the drafters: an exporter hands over the technique it writes as a plain
:class:`~torch.nn.Module`, and the spec of each drafter reads the attributes of its own technique class."""

from __future__ import annotations

from typing import TypeVar

from torch import nn

__all__ = ["technique_as"]

T = TypeVar("T", bound=nn.Module)


def technique_as(technique: nn.Module, *classes: type[T]) -> T:
    """``technique``, narrowed to ``classes``: each exporter is registered for its own technique classes, so any
    other technique is a registration bug (asserted)."""
    assert isinstance(technique, classes), f"expected one of {[c.__name__ for c in classes]}, got {type(technique)}"
    return technique
