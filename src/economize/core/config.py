"""Base class of technique configurations."""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, Any, ClassVar, TypeVar

from ..errors import ConfigError

if TYPE_CHECKING:
    from ..arch import ArchAdapter

__all__ = ["TechniqueConfig", "nested_config", "nested_dict"]

C = TypeVar("C", bound="TechniqueConfig")


@dataclasses.dataclass(frozen=True)
class TechniqueConfig:
    """Hyperparameters of one technique. Subclasses are frozen dataclasses and set :attr:`technique`.

    A configuration is plain data: it can be written to JSON and read back, and it fully determines the
    parameter shapes of the technique it builds.
    """

    #: Registered name of the technique this configuration builds.
    technique: ClassVar[str] = ""

    def validate(self, adapter: ArchAdapter) -> None:
        """Check the configuration against a concrete model. Raise :class:`ConfigError` if it cannot apply."""

    def to_dict(self) -> dict[str, Any]:
        return {f.name: _to_json(getattr(self, f.name)) for f in dataclasses.fields(self)}

    @classmethod
    def from_dict(cls: type[C], data: dict[str, Any]) -> C:
        """The configuration written by :meth:`to_dict` (lists become tuples).

        Raises:
            ConfigError: ``data`` names a field the class does not have, or lacks a field without a default.
        """
        fields = dataclasses.fields(cls)
        names = {f.name for f in fields}
        unknown = set(data) - names
        if unknown:
            raise ConfigError(f"unknown {cls.__name__} fields {sorted(unknown)}; known: {sorted(names)}")
        missing = [
            f.name for f in fields
            if f.init and f.name not in data
            and f.default is dataclasses.MISSING and f.default_factory is dataclasses.MISSING
        ]
        if missing:
            raise ConfigError(f"{cls.__name__} needs the fields {missing}")
        return cls(**{k: tuple(v) if isinstance(v, list) else v for k, v in data.items()})


def _to_json(value: Any) -> Any:
    return list(value) if isinstance(value, tuple) else value


def nested_dict(config: TechniqueConfig) -> dict[str, Any]:
    """The JSON form of a technique configuration nested in another one (the drafter of FR-Spec, the fallback of a
    hybrid): ``{"technique": name, "config": fields}``."""
    return {"technique": config.technique, "config": config.to_dict()}


def nested_config(data: dict[str, Any] | TechniqueConfig) -> TechniqueConfig:
    """The configuration written by :func:`nested_dict` (a configuration object is returned as it is).

    Raises:
        ConfigError: ``data`` lacks the keys ``technique`` and ``config`` or holds others.
    """
    if isinstance(data, TechniqueConfig):
        return data
    from .technique import TECHNIQUES

    if set(data) != {"technique", "config"}:
        raise ConfigError(
            f"a nested technique configuration needs the keys 'technique' and 'config', got {sorted(data)}"
        )
    return TECHNIQUES.get(data["technique"]).config_class.from_dict(dict(data["config"]))
