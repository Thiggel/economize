"""Refusal of the base model's own ``generate`` while techniques act only in economize's decoding loop.

KV-cache techniques (eviction, quantization, sparse attention over the cache) wrap the cache of economize's decoder,
and dynamic-depth techniques give it a layer policy. ``transformers``' ``generate`` on the base model runs neither,
so it would decode the unmodified model without any error. :func:`guard_plain_generate` replaces that method on the
model instance with one that raises :class:`~economize.errors.ConfigError` and points to the
:class:`~economize.EfficientModel`'s ``generate``; :func:`release_plain_generate` removes it again (``detach``).
economize's own ``generate`` calls the class's method and is unaffected.
"""

from __future__ import annotations

from collections.abc import Sequence

from torch import nn

from ..errors import ConfigError

__all__ = ["PlainGenerateGuard", "guard_plain_generate", "release_plain_generate"]


class PlainGenerateGuard:
    """The instance-level ``generate`` of a guarded model (a plain object, so ``copy.deepcopy`` of the model guards
    the copy, and pickling works)."""

    def __init__(self, reasons: Sequence[str]) -> None:
        self.reasons = tuple(reasons)

    def message(self) -> str:
        return (
            f"the base model's own generate would ignore the technique(s) {', '.join(self.reasons)}, which act only "
            "in economize's decoding loop: call generate on the EfficientModel that economize.apply returned, or "
            "detach() it"
        )

    def __call__(self, *args: object, **kwargs: object) -> object:
        raise ConfigError(self.message())


def guard_plain_generate(model: nn.Module, reasons: Sequence[str]) -> None:
    """Make ``model.generate`` (when its class has one) raise, naming the techniques ``reasons``. Nothing happens
    without reasons or when the instance already has its own ``generate``."""
    if reasons and callable(getattr(type(model), "generate", None)) and "generate" not in vars(model):
        object.__setattr__(model, "generate", PlainGenerateGuard(reasons))


def release_plain_generate(model: nn.Module) -> None:
    """Remove the guard of :func:`guard_plain_generate` from ``model``, if it has one."""
    if isinstance(vars(model).get("generate"), PlainGenerateGuard):
        del vars(model)["generate"]
