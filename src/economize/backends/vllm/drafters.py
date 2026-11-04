"""What vLLM needs to serve one economize drafter technique, registered per technique in :data:`VLLM_DRAFTERS`.

The vLLM plugin (:mod:`economize.backends.vllm.plugin`) knows no technique. It reads this registry and, for every
:class:`VllmDrafter`, registers the draft architecture with vLLM, teaches vLLM's ``speculators`` config updater the
technique's config block, and routes the construction of vLLM's proposer classes to the technique's replacements
when the draft model of a run is one of its exports. A new servable drafter technique is therefore one new
subpackage ``economize/backends/vllm/<technique>/`` (found by :mod:`pkgutil` when :mod:`economize.backends.vllm` is
imported, so no list is edited) that registers its exporter and its :class:`VllmDrafter`::

    from economize.backends.vllm.drafters import VllmDrafter, register_vllm_drafter

    register_vllm_drafter(VllmDrafter(
        technique="my_drafter",
        config_key="economize_my_drafter",                           # the block its exporter writes into config.json
        architecture="EconomizeMyDrafterEagle3",                     # the draft architecture in config.json
        model_class="my_pkg.vllm_model:MyDrafterForCausalLM",     # imported by vLLM, lazily
        proposers={"vllm.v1.worker.gpu.spec_decode.eagle.speculator:EagleSpeculator":
                   "my_pkg.vllm_proposer:MySpeculator"},           # optional: vLLM class -> economize subclass
    ))

A separate package names its :class:`VllmDrafter` in the ``economize.vllm_drafters`` entry-point group, which
:data:`VLLM_DRAFTERS` loads before the plugin installs anything::

    [project.entry-points."economize.vllm_drafters"]
    my_drafter = "my_pkg.vllm:MY_DRAFTER"      # a VllmDrafter instance

Every class path is a ``"module:Class"`` string, so registering a drafter never imports vLLM.
:attr:`VllmDrafter.method` is the ``speculative_config["method"]`` that
:func:`economize.backends.vllm.speculative_config` writes for an export of the technique.
"""

from __future__ import annotations

import importlib
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

from ...errors import ConfigError
from ...registry import Registry

__all__ = [
    "SYNC_PREFIX",
    "VLLM_DRAFTERS",
    "VllmDrafter",
    "drafter_for_architectures",
    "drafter_of",
    "import_path",
    "register_vllm_drafter",
    "registered_drafters",
]

#: Name prefix that routes a tensor of an RL weight sync of the target to the drafter (a re-distilled drafter),
#: see :func:`economize.backends.vllm.drafter_weights`.
SYNC_PREFIX = "economize_drafter."


def _check_path(name: str, value: object) -> None:
    if not isinstance(value, str) or value.count(":") != 1 or not all(value.split(":")):
        raise ConfigError(f"{name} must be a 'module:Class' import path, got {value!r}")


def import_path(path: str) -> Any:
    """The object a ``"module:attribute"`` path names (imports the module)."""
    module, _, attribute = path.partition(":")
    return getattr(importlib.import_module(module), attribute)


@dataclass(frozen=True)
class VllmDrafter:
    """How vLLM recognizes and runs the exported drafter of one technique.

    Attributes:
        technique: The technique name (the :data:`VLLM_DRAFTERS` key, as in ``economize.backends.EXPORTERS``).
        config_key: The block the technique's exporter writes into ``config.json`` (``"economize_jump"``).
        architecture: The draft architecture of the exported ``config.json`` (``"EconomizeJumpEagle3"``).
        model_class: ``"module:Class"`` of the draft model class vLLM builds for :attr:`architecture`.
        proposers: vLLM proposer classes whose construction is routed to a replacement for this drafter, as
            ``{"vllm.module:VllmClass": "module:Replacement"}``. A replacement subclasses the vLLM class, so every
            ``isinstance`` check of the model runner still holds. Empty: vLLM's own proposer drafts.
        speculators_algorithm: The ``speculators`` algorithm whose config updater must keep :attr:`architecture`
            for a ``speculators``-format export (``"eagle3"``), or ``None``.
        method: vLLM's speculative method that runs the draft model (``speculative_config["method"]``; vLLM
            routes ``"eagle3"`` drafts through its EAGLE proposer).
    """

    technique: str
    config_key: str
    architecture: str
    model_class: str
    proposers: Mapping[str, str] = field(default_factory=dict, hash=False)
    speculators_algorithm: str | None = "eagle3"
    method: str = "eagle3"

    def __post_init__(self) -> None:
        for name in ("technique", "config_key", "architecture", "method"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise ConfigError(f"VllmDrafter.{name} must be a non-empty string, got {value!r}")
        _check_path("VllmDrafter.model_class", self.model_class)
        for target, replacement in self.proposers.items():
            _check_path("a VllmDrafter.proposers key", target)
            _check_path(f"VllmDrafter.proposers[{target!r}]", replacement)
        object.__setattr__(self, "proposers", MappingProxyType(dict(self.proposers)))

    def matches(self, vllm_config: Any) -> bool:
        """Whether the speculative config of ``vllm_config`` drafts with an export of this technique (its draft
        architecture, or its config block in the draft model's ``hf_config``)."""
        draft = getattr(getattr(vllm_config, "speculative_config", None), "draft_model_config", None)
        if draft is None:
            return False
        if self.architecture in (getattr(draft, "architectures", None) or ()):
            return True
        hf_config = getattr(draft, "hf_config", None)
        architectures = getattr(hf_config, "architectures", None) or ()
        return self.architecture in architectures or hasattr(hf_config, self.config_key)

    def owns_speculators_config(self, config_dict: Mapping[str, Any], pre_trained_config: Mapping[str, Any]) -> bool:
        """Whether a ``speculators`` config (vLLM's updater arguments) is an export of this technique."""
        return (config_dict.get(self.config_key) or pre_trained_config.get(self.config_key)) is not None


#: The vLLM drafter of each servable technique, by technique name (entry-point group ``economize.vllm_drafters``).
VLLM_DRAFTERS: Registry[VllmDrafter] = Registry("vLLM drafter", "economize.vllm_drafters")


def register_vllm_drafter(drafter: VllmDrafter, *, override: bool = False) -> VllmDrafter:
    """Register ``drafter`` under its technique name.

    Raises:
        DuplicateNameError: The technique already has a vLLM drafter and ``override`` is false.
    """
    return VLLM_DRAFTERS.register(drafter.technique, drafter, override=override)


def registered_drafters() -> Iterator[VllmDrafter]:
    """The registered drafters in technique-name order."""
    return (VLLM_DRAFTERS.get(name) for name in VLLM_DRAFTERS.names())


def drafter_for_architectures(architectures: Iterable[str]) -> VllmDrafter | None:
    """The registered drafter whose draft architecture is among ``architectures`` (a ``config.json`` list)."""
    names = set(architectures)
    return next((drafter for drafter in registered_drafters() if drafter.architecture in names), None)


def drafter_of(vllm_config: Any) -> VllmDrafter | None:
    """The registered drafter whose export ``vllm_config`` drafts with, or ``None`` (any other draft model)."""
    return next((drafter for drafter in registered_drafters() if drafter.matches(vllm_config)), None)
