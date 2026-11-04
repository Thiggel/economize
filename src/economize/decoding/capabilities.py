"""Optional capabilities of a token selector, and the helpers through which the loop and the verifiers use them.

A :class:`~economize.decoding.TokenSelector` needs only ``greedy`` and ``__call__``. The protocols below name the
members a selector may add to unlock faster or lossless paths. A selector gains a capability by defining the member;
there is nothing to register or subclass, and code that needs a capability asks for it through the helper functions
here, so the probing lives in one place.

=============================  =========================================================================================
Protocol                       Unlocks
=============================  =========================================================================================
:class:`BatchSelector`         ``select_many``: one call per batch instead of one per row (selectors that ignore the
                               history).
:class:`DistributionSelector`  ``distribution``: the ``"rejection"`` and ``"typical"`` verifiers.
:class:`BatchDistributions`    ``distributions``: those verifiers on a whole batch at once, and draft trees.
:class:`SeededSelector`        ``generator_for``: the selector's own random stream, which the verifiers draw from too.
=============================  =========================================================================================
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import torch

from ..errors import ConfigError

__all__ = [
    "BatchDistributions",
    "BatchSelector",
    "DistributionSelector",
    "SeededSelector",
    "selector_distribution",
    "selector_distributions",
    "selector_generator",
    "selector_many",
]


@runtime_checkable
class BatchSelector(Protocol):
    """A selector that chooses for many rows in one call (its choice ignores the history)."""

    def select_many(self, logits: torch.Tensor) -> torch.Tensor:
        """One choice per row of ``logits`` ``[B, V]``, ``[B]``."""
        ...


@runtime_checkable
class DistributionSelector(Protocol):
    """A selector that exposes the probabilities ``__call__`` samples from."""

    def distribution(self, logits: torch.Tensor, history: torch.Tensor) -> torch.Tensor:
        """The distribution ``[V]`` behind the choice after ``history``."""
        ...


@runtime_checkable
class BatchDistributions(Protocol):
    """A selector that gives the distributions of many rows in one call (its choice ignores the history)."""

    def distributions(self, logits: torch.Tensor) -> torch.Tensor:
        """The distribution of every row of ``logits`` ``[..., V]``, ``[..., V]``."""
        ...


@runtime_checkable
class SeededSelector(Protocol):
    """A sampling selector with its own random stream on every device it samples on."""

    def generator_for(self, device: torch.device) -> torch.Generator:
        """The generator that draws on ``device``."""
        ...


def selector_many(selector: object, logits: torch.Tensor) -> torch.Tensor | None:
    """The choice for every row of ``logits`` ``[..., V]`` (shape ``[...]``), or ``None`` when the selector has no
    :class:`BatchSelector` member and must choose row by row with the history."""
    if not isinstance(selector, BatchSelector):
        return None
    return selector.select_many(logits.reshape(-1, logits.shape[-1])).view(logits.shape[:-1])


def selector_distributions(selector: object, logits: torch.Tensor) -> torch.Tensor | None:
    """The selector's float distributions for every row of ``logits`` ``[..., V]`` on the logits' device, or ``None``
    when it has no :class:`BatchDistributions` member."""
    if not isinstance(selector, BatchDistributions):
        return None
    return selector.distributions(logits).float()


def selector_distribution(selector: object, logits: torch.Tensor, history: torch.Tensor, user: str) -> torch.Tensor:
    """The selector's float distribution ``[V]`` for ``logits`` ``[V]`` after ``history``.

    Raises:
        ConfigError: The selector has no ``distribution`` (``user`` names the component that needs it).
    """
    if not isinstance(selector, DistributionSelector):
        raise ConfigError(f"the {user!r} verifier needs a selector with distribution(), got {type(selector).__name__}")
    return selector.distribution(logits, history).float()


def selector_generator(selector: object, device: torch.device) -> torch.Generator | None:
    """The generator to draw with on ``device``: the selector's :meth:`SeededSelector.generator_for`, else its
    ``generator`` attribute, else ``None`` (torch's global generator).

    Raises:
        ConfigError: The selector's ``generator`` attribute lives on another device.
    """
    if isinstance(selector, SeededSelector):
        return selector.generator_for(device)
    generator: torch.Generator | None = getattr(selector, "generator", None)
    if generator is not None and not _same_device(generator.device, device):
        raise ConfigError(
            f"{type(selector).__name__}.generator is on {generator.device} but the logits are on {device}; "
            "create it there, or implement generator_for(device)"
        )
    return generator


def _same_device(a: torch.device, b: torch.device) -> bool:
    if a.type != b.type:
        return False
    return a.index is None or b.index is None or a.index == b.index
