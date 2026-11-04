"""Base class of the techniques that rewrite base weights once at attach time."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, ClassVar, Protocol, cast

from torch import Tensor

from ...core.calibration import calibration_source, token_batches
from ...core.technique import Technique
from ...errors import EconomizeError
from ...registry import Registry

__all__ = ["CalibratedWeightTransform", "WeightTransform"]


class _MethodConfig(Protocol):
    """The config of a :class:`CalibratedWeightTransform`: it names the method to run."""

    method: str


class WeightTransform(Technique):
    """Shared behaviour of techniques that rewrite base weights in place once at attach time (pruning,
    quantization, low rank): they are lossy, report what they did in :attr:`summary` and cannot be detached."""

    #: What the transform did (method, layers, sparsity / bits / rank, ...), set by ``attach``.
    summary: dict[str, Any]

    def detach(self) -> None:
        raise EconomizeError(
            f"{self.name!r} rewrote the base weights in place and cannot be detached; reload the model"
        )


class CalibratedWeightTransform(WeightTransform):
    """A :class:`WeightTransform` whose ``attach`` validates the config against the model, draws calibration
    batches from the context when :meth:`needs_calibration`, and stores what :meth:`transform` returns as
    :attr:`summary`. A subclass names its method registry in :attr:`methods` (entries with a ``needs_calibration``
    flag, looked up by ``config.method``) and implements :meth:`transform`."""

    #: The registry of the technique's methods, looked up by ``config.method``.
    methods: ClassVar[Registry[Any] | None] = None

    def needs_calibration(self) -> bool:
        """Whether the configured method reads calibration data."""
        assert self.methods is not None, f"{type(self).__name__} sets no methods registry"
        return bool(self.methods.get(cast(_MethodConfig, self.config).method).needs_calibration)

    def transform(self, adapter: Any, batches: Sequence[Tensor] | None) -> dict[str, Any]:
        """Rewrite the weights of ``adapter`` in place and return the summary."""
        raise NotImplementedError

    def attach(self, model: Any, adapter: Any, context: Any = None) -> None:
        self.config.validate(adapter)
        batches = token_batches(calibration_source(context, self.name)) if self.needs_calibration() else None
        self.summary = self.transform(adapter, batches)
