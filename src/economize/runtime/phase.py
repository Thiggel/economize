"""Which part of decoding is running: a signal for techniques that treat prefill, decoding and drafting differently.

The decoding loop sets the phase around every model call. Hooks of ACTIVATIONS techniques read it with
:func:`current_phase`; for example activation sparsity keeps exact refresh passes dense so that verification stays
lossless.

==============  ==================================================================================================
Phase           Model calls
==============  ==================================================================================================
``"prefill"``   The first exact pass of a generation, over the prompt.
``"decode"``    Later exact passes when no drafter runs (``cycle_length == 1``): one token per row.
``"refresh"``   Later exact passes with a drafter (``cycle_length > 1``): they verify the drafts and rewrite their
                cache entries.
``"draft"``     Drafter steps.
``None``        Outside :meth:`economize.decoding.CycleDecoder.decode` (training, plain ``forward`` calls).
==============  ==================================================================================================
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Literal

from ..errors import ConfigError

__all__ = ["PHASES", "Phase", "current_phase", "phase"]

Phase = Literal["prefill", "decode", "refresh", "draft"]

#: Every phase the decoding loop sets.
PHASES: tuple[Phase, ...] = ("prefill", "decode", "refresh", "draft")

_PHASE: ContextVar[Phase | None] = ContextVar("economize_phase", default=None)


def current_phase() -> Phase | None:
    """The phase of the model call in progress, or ``None`` outside decoding."""
    return _PHASE.get()


@contextmanager
def phase(name: Phase | None) -> Iterator[None]:
    """Set the phase inside the block (nested blocks restore the outer phase on exit)."""
    if name is not None and name not in PHASES:
        raise ConfigError(f"unknown phase {name!r}; expected one of {PHASES}")
    token = _PHASE.set(name)
    try:
        yield
    finally:
        _PHASE.reset(token)
