"""How a serving engine runs a technique, read from its adapter registry, for :func:`economize.backend_support`.

A serving backend (vLLM, SGLang) registers one adapter per technique; the adapter's :class:`Support` level says
whether the engine serves the technique and how. :func:`serving_support` turns that level into a
:class:`~economize.core.compatibility.BackendSupport`:

* no adapter, or level ``UNSUPPORTED``: :attr:`~SupportStatus.UNSUPPORTED`, with the adapter's reason;
* level ``DENSIFIED``: :attr:`~SupportStatus.SIMULATED`; the engine runs the technique's weights multiplied out, so
  none of the technique's savings (:attr:`~economize.TechniqueInfo.saves`) is realized;
* an adapter that declares a ``simulation`` (or returns one from ``simulation_of`` for the configuration: FP4
  weights before Blackwell; KIVI's cache keeps the dense entries beside the codes, pruned weights
  are served dense): :attr:`~SupportStatus.SIMULATED` with it;
* otherwise :attr:`~SupportStatus.REAL`.

The level is the configured one when the configuration is known (an adapter's ``level`` reads only
``technique.config``), else the adapter's class-level ``support``. Refusals that depend on the other techniques of
a recipe or on the model are raised by the engine's own checks when a model is served.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..core.compatibility import BackendSupport, SupportStatus
from ..core.config import TechniqueConfig
from ..core.metadata import Simulation
from ..core.technique import Technique
from ..registry import Registry
from .vllm.serving.base import Support

__all__ = ["adapter_support", "serving_support"]


@dataclass(frozen=True)
class _Configured:
    """What an adapter's ``level`` reads of a technique: its configuration."""

    config: TechniqueConfig


def serving_support(
    engine: str, adapters: Registry[Any], technique: type[Technique], config: TechniqueConfig | None
) -> BackendSupport:
    """How ``engine``, whose adapters by technique name are ``adapters``, runs ``technique`` with ``config``."""
    if technique.name not in adapters:
        return BackendSupport(SupportStatus.UNSUPPORTED, f"{engine} has no adapter for {technique.name!r}")
    adapter = adapters.get(technique.name)()
    level = adapter.level(_Configured(config)) if config is not None else adapter.support
    hook = getattr(adapter, "simulation_of", None)
    simulation: Simulation | None = (
        hook(_Configured(config)) if hook is not None and config is not None else getattr(adapter, "simulation", None)
    )
    return adapter_support(engine, technique, level, adapter.reason, simulation)


def adapter_support(
    engine: str, technique: type[Technique], level: Support, reason: str, simulation: Simulation | None = None
) -> BackendSupport:
    """The :class:`~economize.core.compatibility.BackendSupport` of an adapter's support level.

    Parameters
    ----------
    engine : str
        The engine or runtime in messages.
    technique : type[Technique]
        The technique class.
    level : Support
        The adapter's level for the configured technique.
    reason : str
        The adapter's reason (the refusal for ``UNSUPPORTED``).
    simulation : Simulation, optional
        What a non-densified adapter does not realize.

    Returns
    -------
    BackendSupport
        Real, simulated or unsupported, with the reason.
    """
    if level is Support.UNSUPPORTED:
        return BackendSupport(SupportStatus.UNSUPPORTED, reason)
    if level is Support.DENSIFIED:
        simulation = Simulation(technique.info.saves, f"{engine} serves its dense equivalent")
    return BackendSupport.of(simulation.within(technique.info.saves) if simulation else None, reason)
