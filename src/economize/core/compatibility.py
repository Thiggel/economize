"""Which registered techniques combine, and on which backend each one is real, asked without a model.

Both answers derive from the techniques' :class:`~economize.TechniqueInfo` declarations (and their
:meth:`~economize.Technique.conflict` rules), the same ones a recipe checks when it attaches techniques:

* :func:`incompatibility`, :func:`compatible` and :func:`compatibility_matrix` say which pairs a recipe accepts;
* :func:`backend_support` says, per backend, whether the technique runs for real, only simulates its saving or does
  not run. Each backend is one entry of :data:`BACKEND_SUPPORT`; the package registers ``"transformers"`` here and
  the serving backends (``"vllm"``, ``"sglang"``) in its composition root, so that answering imports no engine.

A technique is named by a registered name, a technique class, a configuration or a built technique. A configuration
(or a built technique) is checked as configured; a name or a class is checked with its configuration class's
defaults when those form a valid configuration, and otherwise with the rules that need no configuration.
"""

from __future__ import annotations

import enum
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass

from ..errors import ConfigError
from ..registry import Registry
from .config import TechniqueConfig
from .metadata import Simulation
from .technique import TECHNIQUES, Technique, listed_techniques

__all__ = [
    "BACKEND_SUPPORT",
    "BackendSupport",
    "SupportFn",
    "SupportStatus",
    "TechniqueRef",
    "backend_support",
    "compatibility_matrix",
    "compatible",
    "incompatibility",
    "resolve",
]

#: A technique as the functions of this module accept it.
TechniqueRef = str | type[Technique] | TechniqueConfig | Technique


def resolve(ref: TechniqueRef) -> tuple[type[Technique], TechniqueConfig | None]:
    """The technique class ``ref`` names and its configuration (``None`` when its defaults are not valid)."""
    if isinstance(ref, Technique):
        return type(ref), ref.config
    if isinstance(ref, TechniqueConfig):
        return TECHNIQUES.get(ref.technique), ref
    technique = TECHNIQUES.get(ref) if isinstance(ref, str) else ref
    try:
        return technique, technique.config_class()
    except (TypeError, ConfigError):  # required fields, or defaults the config rejects
        return technique, None


def incompatibility(a: TechniqueRef, b: TechniqueRef) -> str | None:
    """Why a recipe rejects techniques ``a`` and ``b`` together, ``None`` if it accepts the pair.

    The answer covers the pair: a technique that :attr:`~economize.TechniqueInfo.requires` a third family (such as
    ``draft_quant``, which needs a drafter) still needs one in the recipe.
    """
    first, first_config = resolve(a)
    second, second_config = resolve(b)
    if first.name == second.name:
        return f"technique {first.name!r} is applied twice"
    return first.conflict(first_config, second, second_config) or second.conflict(second_config, first, first_config)


def compatible(a: TechniqueRef, b: TechniqueRef) -> bool:
    """Whether a recipe accepts techniques ``a`` and ``b`` together (see :func:`incompatibility`)."""
    return incompatibility(a, b) is None


def compatibility_matrix(techniques: Iterable[TechniqueRef] | None = None) -> dict[str, dict[str, str | None]]:
    """``matrix[a][b]``: :func:`incompatibility` of every pair of ``techniques`` (default: every listed one),
    keyed by technique name."""
    refs = list(techniques) if techniques is not None else listed_techniques()
    names = [resolve(ref)[0].name for ref in refs]
    return {name: {other: incompatibility(ref, other_ref) for other, other_ref in zip(names, refs)}
            for name, ref in zip(names, refs)}


class SupportStatus(enum.Enum):
    """How a backend runs a technique."""

    #: The backend realizes the technique's saving (:attr:`~economize.TechniqueInfo.saves`).
    REAL = "real"
    #: The backend reproduces the technique's outputs without (all of) its saving.
    SIMULATED = "simulated"
    #: The backend cannot run the technique.
    UNSUPPORTED = "unsupported"


@dataclass(frozen=True)
class BackendSupport:
    """How one backend runs one configured technique.

    Attributes:
        status: Real, simulated or unsupported.
        detail: What the backend runs, or why it cannot.
        simulation: For a simulated path, the saving it does not realize.
    """

    status: SupportStatus
    detail: str = ""
    simulation: Simulation | None = None

    @property
    def simulated(self) -> bool:
        return self.status is SupportStatus.SIMULATED

    @classmethod
    def of(cls, simulation: Simulation | None, detail: str = "") -> BackendSupport:
        """Real without ``simulation``, simulated with it."""
        if simulation is None:
            return cls(SupportStatus.REAL, detail)
        return cls(SupportStatus.SIMULATED, detail or str(simulation), simulation)


#: How a backend runs a technique class with a configuration (``None``: its defaults are not valid).
SupportFn = Callable[[type[Technique], TechniqueConfig | None], BackendSupport]

#: The backends :func:`backend_support` reports, by name. Other packages register through the
#: ``economize.backend_support`` entry points.
BACKEND_SUPPORT: Registry[SupportFn] = Registry("backend", "economize.backend_support")


@BACKEND_SUPPORT.register("transformers")
def _transformers(technique: type[Technique], config: TechniqueConfig | None) -> BackendSupport:
    """The Hugging Face path of :class:`~economize.EfficientModel` (see :meth:`Technique.simulation_for`)."""
    return BackendSupport.of(technique.simulation_for(config))


def backend_support(technique: TechniqueRef) -> Mapping[str, BackendSupport]:
    """How every registered backend runs ``technique``, by backend name.

    A benchmark of memory or speed on a backend needs :attr:`SupportStatus.REAL` there.
    """
    cls, config = resolve(technique)
    return {name: BACKEND_SUPPORT.get(name)(cls, config) for name in BACKEND_SUPPORT.names()}
