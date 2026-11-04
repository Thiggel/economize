"""Exception hierarchy. Every error raised on purpose by economize derives from :class:`EconomizeError`."""

from __future__ import annotations

__all__ = [
    "CheckpointFormatError",
    "ConfigError",
    "DuplicateNameError",
    "EconomizeError",
    "IncompatibleTechniquesError",
    "SimulatedPathWarning",
    "TeacherMismatchWarning",
    "UnknownNameError",
    "UnregisteredNameError",
    "UnsupportedModelError",
]


class EconomizeError(Exception):
    """Base class of all economize errors."""


class ConfigError(EconomizeError, ValueError):
    """A configuration value is invalid, alone or for the model it is applied to."""


class UnknownNameError(EconomizeError, KeyError):
    """A registry lookup failed. The message lists the registered names."""

    def __str__(self) -> str:  # KeyError would otherwise quote the message
        return str(self.args[0]) if self.args else ""


class UnregisteredNameError(ConfigError, UnknownNameError):
    """A configuration field names an entry its registry lacks: a :class:`ConfigError` and an
    :class:`UnknownNameError` (so ``except ValueError`` and ``except KeyError`` both catch it)."""


class DuplicateNameError(EconomizeError, ValueError):
    """A name was registered twice without ``override=True``."""


class UnsupportedModelError(EconomizeError, TypeError):
    """The model's architecture, attention implementation or cache layout is not supported."""


class IncompatibleTechniquesError(EconomizeError, ValueError):
    """Two requested techniques cannot be combined on one model."""


class CheckpointFormatError(EconomizeError, ValueError):
    """A checkpoint could not be recognised or does not match its configuration."""


class SimulatedPathWarning(UserWarning):
    """A technique attached on a path that reproduces its outputs without realizing its saving (fake quantization,
    masked eviction; see :class:`~economize.Simulation`). Quality numbers are valid; memory and speed numbers of the
    path are not those of the technique."""


class TeacherMismatchWarning(UserWarning):
    """The base model's weights differ from those a checkpoint was saved with (see ``teacher_fingerprint``).

    Lossless decoding stays exact; trained drafters may be accepted less often and lossy decoding may degrade.
    """
