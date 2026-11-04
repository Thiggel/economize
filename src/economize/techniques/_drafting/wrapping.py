"""Drafter techniques built around other drafter techniques (FR-Spec's draft vocabulary, hybrid drafting).

* :class:`WrappingConfig`: a configuration with nested technique configurations (coerced from and written to
  their JSON form).
* :class:`WrappingTechnique`: holds the wrapped techniques as submodules and attaches, detaches, refreshes and
  checks them with itself.
* :class:`DelegatingDrafter`: forwards every drafter member to the wrapped drafter; a subclass overrides only how
  tokens are proposed.
* :class:`NotWrappable`: the marker of drafter techniques another drafter cannot wrap, so wrappers refuse them by
  capability (a new technique opts out by inheriting it, never by being listed in a wrapper).
* :class:`PartialDrafter`: a drafter that serves only some rows of a request without a model call and leaves the
  others to a fallback (prompt lookup, the primary of a hybrid).
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, Any, ClassVar, Protocol, runtime_checkable

import torch
from torch import nn

from ...core.config import TechniqueConfig, nested_config, nested_dict
from ...core.technique import TECHNIQUES, AttachContext, Capability, Technique
from ...decoding.protocols import Draft, Drafter, Verifier, WrappableDrafter, draft_footprint
from ...errors import ConfigError
from .drafters import DraftModules

if TYPE_CHECKING:
    from ...arch import ArchAdapter
    from ...decoding.cost import LayerFootprint
    from ...decoding.request import DraftRequest
    from ...runtime import ExactPass, KVCache, LayerRunner

__all__ = [
    "DelegatingDrafter",
    "NotWrappable",
    "PartialDrafter",
    "WrappingConfig",
    "WrappingTechnique",
    "drafting_class",
]


class NotWrappable:
    """Marker base of drafter techniques whose drafters another drafter cannot wrap: they propose tokens without
    logits (prompt lookup) or hand rows to a drafter of their own (hybrid). List it before
    :class:`~economize.core.technique.Technique`.

    Attributes:
        wrap_hint: Appended to the error a wrapper raises (e.g. what to wrap instead).
    """

    wrap_hint: ClassVar[str] = ""


def drafting_class(config: TechniqueConfig, owner: str, role: str, *, wrappable: bool = True) -> type[Technique]:
    """The registered technique class of ``config``, checked to draft tokens and, with ``wrappable``, to allow
    wrapping (not :class:`NotWrappable`).

    Args:
        config: The nested configuration.
        owner: Name of the wrapping technique (for errors).
        role: What the nested technique is to the owner (for errors), e.g. ``"drafter"``.
        wrappable: Refuse :class:`NotWrappable` techniques.

    Raises:
        ConfigError: The technique does not draft tokens, or cannot be wrapped.
    """
    cls = TECHNIQUES.get(config.technique)
    if Capability.DRAFTER not in cls.capabilities:
        raise ConfigError(f"the {role} of {owner} must draft tokens; {cls.name!r} does not")
    if wrappable and issubclass(cls, NotWrappable):
        raise ConfigError(f"{owner} cannot wrap a {cls.name!r} drafter as its {role}{cls.wrap_hint}")
    return cls


@dataclasses.dataclass(frozen=True)
class WrappingConfig(TechniqueConfig):
    """A technique configuration with fields that hold other technique configurations.

    Subclasses name those fields in :attr:`nested_fields` and call :meth:`coerce_nested` first in
    ``__post_init__``. The fields accept a configuration object or its JSON form (:func:`nested_dict`), which
    ``dataclasses.replace`` and :meth:`from_dict` pass through, and :meth:`to_dict` writes the JSON form. A field
    may be ``None`` when its type allows it.
    """

    nested_fields: ClassVar[tuple[str, ...]] = ()

    def coerce_nested(self) -> None:
        """Turn the JSON form of every nested field into a configuration and check its type.

        Raises:
            ConfigError: A nested field holds neither a configuration nor its JSON form.
        """
        for name in self.nested_fields:
            value = getattr(self, name)
            if isinstance(value, dict):
                value = nested_config(value)
                object.__setattr__(self, name, value)
            if value is not None and not isinstance(value, TechniqueConfig):
                raise ConfigError(f"{name} must be a technique configuration, got {type(value).__name__}")

    def nested(self) -> dict[str, TechniqueConfig]:
        """The nested configurations that are set, by field name."""
        return {name: value for name in self.nested_fields if (value := getattr(self, name)) is not None}

    def validate(self, adapter: ArchAdapter) -> None:
        for config in self.nested().values():
            config.validate(adapter)

    def to_dict(self) -> dict[str, Any]:
        data = super().to_dict()
        for name, config in self.nested().items():
            data[name] = nested_dict(config)
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Any:
        data = dict(data)
        for name in cls.nested_fields:
            if isinstance(data.get(name), dict):
                data[name] = nested_config(data[name])
        return super().from_dict(data)


class WrappingTechnique(Technique):
    """A drafter technique that holds other drafter techniques as submodules.

    Subclasses name the submodules in :attr:`wrapped` (attribute name -> field of the configuration); their
    parameters are saved under the attribute name. :meth:`wrapped_config` gives the configuration of each, and a
    subclass overrides it for a submodule with a derived configuration. The wrapper checks every wrapped
    technique's compatibility with the recipe, attaches and refreshes them in order, detaches them in reverse order,
    lists their draft modules under the attribute name, and allows CUDA graphs only when all of them do.
    """

    #: Submodule attribute -> configuration field of every wrapped technique.
    wrapped: ClassVar[Mapping[str, str]] = {}

    def __init__(self, config: TechniqueConfig, adapter: ArchAdapter) -> None:
        super().__init__(config, adapter)
        for attribute in self.wrapped:
            sub = self.wrapped_config(attribute)
            cls = drafting_class(sub, self.name, attribute, wrappable=self.wraps_drafter(attribute))
            setattr(self, attribute, cls(sub, adapter))
        self.graph_safe = all(t.graph_safe for t in self.wrapped_techniques())  # type: ignore[misc]

    def wrapped_config(self, attribute: str) -> TechniqueConfig:
        """The configuration of the wrapped technique held as ``attribute``."""
        return getattr(self.config, self.wrapped[attribute])

    def wraps_drafter(self, attribute: str) -> bool:
        """Whether the wrapper wraps the drafter of ``attribute`` (it then must not be :class:`NotWrappable`)."""
        return True

    def wrapped_techniques(self) -> list[Technique]:
        """The wrapped techniques, in the order of :attr:`wrapped`."""
        return [getattr(self, attribute) for attribute in self.wrapped]

    @classmethod
    def conflict(
        cls, config: TechniqueConfig | None, other: type[Technique], other_config: TechniqueConfig | None
    ) -> str | None:
        """The wrapper's own conflict with ``other``, else the first conflict of a wrapped technique."""
        reason = super().conflict(config, other, other_config)
        if reason is None and isinstance(config, WrappingConfig):
            for nested in config.nested().values():
                reason = TECHNIQUES.get(nested.technique).conflict(nested, other, other_config)
                if reason is not None:
                    break
        return reason

    def attach(self, model: nn.Module, adapter: ArchAdapter, context: AttachContext | None = None) -> None:
        for technique in self.wrapped_techniques():
            technique.attach(model, adapter, context)

    def detach(self) -> None:
        for technique in reversed(self.wrapped_techniques()):
            technique.detach()

    def refresh(self) -> None:
        for technique in self.wrapped_techniques():
            technique.refresh()

    def draft_modules(self) -> dict[str, nn.Module]:
        """The wrapped techniques' own draft modules (for draft-only transforms), under their attribute names. A
        wrapped technique without :class:`~economize.techniques._drafting.DraftModules` contributes its whole module
        tree, as an unwrapped one does."""
        modules: dict[str, nn.Module] = {}
        for attribute in self.wrapped:
            technique = getattr(self, attribute, None)
            if technique is None:
                continue
            own = technique.draft_modules() if isinstance(technique, DraftModules) else dict(technique.named_modules())
            modules.update({f"{attribute}.{name}": module for name, module in own.items() if name})
        return modules


class DelegatingDrafter(Drafter):
    """A drafter around another drafter (:attr:`inner`) that forwards every member of
    :class:`~economize.decoding.Drafter` to it: the layers it reads and runs (``source_layers``, ``layer_span``,
    ``footprint``, ``observed_layers``, ``streams_prompt``), :meth:`begin`, :meth:`observe`, :meth:`observe_logits`,
    :meth:`step`, :meth:`propose`, :meth:`reorder`, :meth:`draft_capacity`, :meth:`enable_graphs` and
    :meth:`drafts_trees`. A subclass overrides what it changes.

    Args:
        inner: The wrapped drafter.
        owner: Name of the wrapping technique (for errors).

    Raises:
        ConfigError: ``inner`` is not a :class:`~economize.decoding.WrappableDrafter`.
    """

    def __init__(self, inner: Drafter, owner: str) -> None:
        if not isinstance(inner, WrappableDrafter):
            raise ConfigError(f"{owner} wraps a drafter (a WrappableDrafter), got {type(inner).__name__}")
        self.inner = inner
        self.source_layers = tuple(inner.source_layers)

    @property
    def layer_span(self) -> tuple[int, int]:  # type: ignore[override]
        return self.inner.layer_span

    @property
    def footprint(self) -> LayerFootprint:  # type: ignore[override]
        """The inner drafter's layers, read on every access (they may change, e.g. a new skip set)."""
        return draft_footprint(self.inner)

    @property
    def observed_layers(self) -> tuple[int, ...]:
        return tuple(self.inner.observed_layers)

    @property
    def streams_prompt(self) -> bool:  # type: ignore[override]
        return self.inner.streams_prompt

    def begin(self, runner: LayerRunner, cache: KVCache, batch_size: int) -> None:
        self.inner.begin(runner, cache, batch_size)

    def observe(self, exact: ExactPass, columns: torch.Tensor) -> None:
        self.inner.observe(exact, columns)

    def observe_logits(self, logits: torch.Tensor) -> None:
        self.inner.observe_logits(logits)

    def step(self, tokens: torch.Tensor, positions: torch.Tensor, valid: torch.Tensor | None) -> torch.Tensor:
        return self.inner.step(tokens, positions, valid)

    def propose(self, request: DraftRequest) -> Draft:
        return self.inner.propose(request)

    def reorder(self, rows: torch.Tensor) -> None:
        self.inner.reorder(rows)

    def draft_capacity(self, cycle_length: int) -> int:
        return self.inner.draft_capacity(cycle_length)

    def enable_graphs(self, wrap: Callable[[Callable[..., Any]], Callable[..., Any]]) -> bool:
        return self.inner.enable_graphs(wrap)

    def drafts_trees(self, verifier: Verifier) -> bool:
        return self.inner.drafts_trees(verifier)


@runtime_checkable
class PartialDrafter(Protocol):
    """A drafter that drafts some rows of a request without a model call and leaves the others to a fallback (the
    primary drafter of :class:`~economize.techniques.hybrid.HybridConfig`).

    :meth:`propose_partial` returns a chain draft in which the rows it does not serve have no tokens; it neither
    runs a model layer nor keeps state between cycles beyond what :meth:`~economize.decoding.Drafter.observe` and
    :meth:`~economize.decoding.Drafter.reorder` maintain.
    """

    def begin(self, runner: LayerRunner, cache: KVCache, batch_size: int) -> None: ...

    def observe(self, exact: ExactPass, columns: torch.Tensor) -> None: ...

    def reorder(self, rows: torch.Tensor) -> None: ...

    def propose_partial(self, request: DraftRequest) -> Draft: ...

