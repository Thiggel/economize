"""Refusal of ``transformers``' plain ``save_pretrained`` on a base model that it would save wrongly.

A weight technique that keeps part of its transform outside the weights (QuaRot's online Hadamard rotations are
forward hooks) or that replaces modules by ones ``transformers`` cannot rebuild (low-rank factors) leaves a base
model whose ``save_pretrained`` writes a checkpoint that ``from_pretrained`` loads into a different function, without
any error. :func:`guard_plain_save` replaces that method on the model instance with one that raises
:class:`~economize.errors.ConfigError` and names the way to save instead. A call with an explicit ``state_dict``
(what the serving exporters write) still goes through, since its caller chose the tensors.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Mapping, Sequence
from typing import cast

from torch import Tensor, nn

from ..errors import ConfigError

__all__ = ["PlainSaveGuard", "guard_plain_save"]


class PlainSaveGuard:
    """The instance-level ``save_pretrained`` of a guarded model (a plain object, so ``copy.deepcopy`` of the model
    guards the copy, and pickling works)."""

    def __init__(self, model: nn.Module, reasons: Sequence[str]) -> None:
        self.model = model
        self.reasons = tuple(reasons)

    def message(self) -> str:
        return (
            f"save_pretrained would write a checkpoint that from_pretrained loads wrongly: {'; '.join(self.reasons)}. "
            "Save with economize.save(model, path) (economize.load replays the techniques on the base model) or "
            "export with a serving backend; pass state_dict= explicitly to write chosen tensors anyway"
        )

    def __call__(
        self,
        save_directory: str | os.PathLike[str],
        *args: object,
        state_dict: Mapping[str, Tensor] | None = None,
        **kwargs: object,
    ) -> object:
        if state_dict is None:
            raise ConfigError(self.message())
        save = cast("Callable[..., object]", getattr(type(self.model), "save_pretrained"))  # noqa: B009
        return save(self.model, save_directory, *args, state_dict=state_dict, **kwargs)


def guard_plain_save(models: Sequence[nn.Module], reasons: Sequence[str]) -> None:
    """Make ``save_pretrained`` of each model (those that have one) refuse to write without an explicit
    ``state_dict``, giving ``reasons``. Nothing happens without reasons."""
    if not reasons:
        return
    for model in models:
        if callable(getattr(type(model), "save_pretrained", None)):
            object.__setattr__(model, "save_pretrained", PlainSaveGuard(model, reasons))
