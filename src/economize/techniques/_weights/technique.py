"""Base class of the techniques that rewrite base weights once at attach time."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, ClassVar, Protocol, cast

from torch import Tensor, nn

from ...core.calibration import calibration_source, token_batches
from ...core.technique import Technique
from ...errors import EconomizeError

if TYPE_CHECKING:
    from ...arch import ArchAdapter
    from ...core.technique import AttachContext

__all__ = ["CalibratedWeightTransform", "WeightTransform"]


class _MethodConfig(Protocol):
    """The config of a :class:`CalibratedWeightTransform`: it names the method to run."""

    method: str


class _Method(Protocol):
    """An entry of the method registry of a :class:`CalibratedWeightTransform` (a method class)."""

    @property
    def needs_calibration(self) -> bool:
        """Whether the method reads calibration data."""
        ...


class _MethodRegistry(Protocol):
    """The method registry of a :class:`CalibratedWeightTransform` (a :class:`~economize.registry.Registry` of
    method classes), as far as the base class reads it."""

    def get(self, name: str) -> _Method:
        """The method registered as ``name``."""
        ...


class WeightTransform(Technique):
    """Shared behaviour of techniques that rewrite base weights in place once at attach time (pruning,
    quantization, low rank): they are lossy, report what they did in :attr:`summary` and cannot be detached."""

    #: What the transform did (method, layers, sparsity / bits / rank, ...), set by ``attach``.
    summary: dict[str, object]

    def detach(self) -> None:
        raise EconomizeError(
            f"{self.name!r} rewrote the base weights in place and cannot be detached; reload the model"
        )

    def detach_draft(self) -> None:
        """Nothing to undo: on a draft path the transform rewrote the view's copies, which the view drops."""


class CalibratedWeightTransform(WeightTransform):
    """A :class:`WeightTransform` whose ``attach`` validates the config against the model, draws calibration
    batches from the context when :meth:`needs_calibration`, and stores what :meth:`transform` returns as
    :attr:`summary`. A subclass names its method registry in :attr:`methods` (entries with a ``needs_calibration``
    flag, looked up by ``config.method``) and implements :meth:`transform`."""

    #: The registry of the technique's methods, looked up by ``config.method``.
    methods: ClassVar[_MethodRegistry | None] = None

    def needs_calibration(self) -> bool:
        """Whether the configured method reads calibration data."""
        assert self.methods is not None, f"{type(self).__name__} sets no methods registry"
        return bool(self.methods.get(cast(_MethodConfig, self.config).method).needs_calibration)

    def transform(self, adapter: ArchAdapter, batches: Sequence[Tensor] | None) -> dict[str, object]:
        """Rewrite the weights of ``adapter`` in place and return the summary."""
        raise NotImplementedError

    def calibration_batches(self, context: AttachContext | None) -> Sequence[Tensor] | None:
        """The calibration batches of ``context`` (an :class:`~economize.core.AttachContext`) when
        :meth:`needs_calibration`, else ``None``."""
        return token_batches(calibration_source(context, self.name)) if self.needs_calibration() else None

    def attach(self, model: nn.Module, adapter: ArchAdapter, context: AttachContext | None = None) -> None:
        self.config.validate(adapter)
        self.summary = self.transform(adapter, self.calibration_batches(context))
