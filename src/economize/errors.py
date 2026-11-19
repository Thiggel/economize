"""Exception hierarchy. Every error raised on purpose by economize derives from :class:`EconomizeError`."""

from __future__ import annotations

__all__ = [
    "CalibrationMismatchWarning",
    "CheckpointFormatError",
    "ConfigError",
    "DuplicateNameError",
    "EconomizeError",
    "IncompatibleTechniquesError",
    "NonFiniteLossError",
    "SimulatedPathWarning",
    "TeacherMismatchWarning",
    "UnknownFieldWarning",
    "UnknownNameError",
    "UnregisteredNameError",
    "UnsupportedEngineVersionError",
    "UnsupportedModelError",
    "UntestedEngineWarning",
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


class NonFiniteLossError(EconomizeError, FloatingPointError):
    """A training step's loss is ``NaN`` or infinite (see ``TrainConfig.nonfinite``)."""


class UnsupportedEngineVersionError(EconomizeError, ImportError):
    """An engine whose private classes economize patches (vLLM, SGLang, TRL) is missing or installed in a release
    outside the tested range (see ``economize._compat.ENGINES``)."""


class UntestedEngineWarning(UserWarning):
    """An engine release outside the tested range runs because ``economize._compat.ALLOW_UNTESTED_ENV`` allows it."""


class SimulatedPathWarning(UserWarning):
    """A technique attached on a path that reproduces its outputs without realizing its saving (fake quantization,
    masked eviction; see :class:`~economize.Simulation`). Quality numbers are valid; memory and speed numbers of the
    path are not those of the technique."""


class CalibrationMismatchWarning(UserWarning):
    """A checkpoint's weight techniques, replayed on load, produced weights other than those it was saved with.

    The base model matches the checkpoint's, so the calibration differed: other calibration data, or a device or
    library version that rounds differently. Pass the calibration data the checkpoint was made with.
    """


class UnknownFieldWarning(UserWarning):
    """A saved configuration names fields this version of economize does not know (typically a checkpoint written by
    a newer release). The fields are ignored and the configuration is built from the fields that are known."""


class TeacherMismatchWarning(UserWarning):
    """The base model's weights differ from those a checkpoint was saved with (see ``teacher_fingerprint``).

    Lossless decoding stays exact; trained drafters may be accepted less often and lossy decoding may degrade.
    """
