"""What a technique is, declared once per technique class (:class:`TechniqueInfo`).

Every registered technique sets the class attribute ``info``::

    @register_technique
    class MyKVQuant(KVCacheTechnique):
        info = TechniqueInfo(family=Family.KV_QUANTIZATION, saves=Saving.KV_MEMORY, lossless=False)

The declaration is the single source of three things:

* the recipe's compatibility rules (:meth:`Technique.check_compatible <economize.Technique.check_compatible>`,
  :func:`economize.compatible`): two techniques of one exclusive :class:`Family` never combine, an
  :attr:`TechniqueInfo.excludes` entry rejects a family, an :attr:`TechniqueInfo.requires` entry needs one;
* whether a path only *simulates* a saving (:class:`Simulation`): the technique's effect on the outputs is exact,
  but the resource it claims to save is still spent (fake quantization keeps the dequantized weights, an evicting
  cache masks entries it keeps in memory). The recipe warns once per attach (:class:`~economize.SimulatedPathWarning`),
  and :meth:`EfficientModel.simulations <economize.EfficientModel.simulations>` lists the simulated techniques of a
  model, which a benchmark of memory or speed checks first;
* the documentation's technique table (field names are stable; new fields get defaults).

Families, savings and kernel choices are open: a new family is one ``Family(...)`` value, a kernel registry that
offers a simulated kernel gives that kernel a ``simulation`` attribute.
"""

from __future__ import annotations

import enum
from collections.abc import Iterable
from dataclasses import dataclass, replace
from typing import Any, ClassVar

from ..registry import Registry

__all__ = [
    "SINGLE_OWNER",
    "Capability",
    "Exclusion",
    "Family",
    "KernelChoice",
    "Saving",
    "Simulation",
    "TechniqueInfo",
    "check_declaration",
    "conflict",
    "saving_names",
    "unmet_requirement",
]


class Capability(enum.Flag):
    """What a technique contributes. A recipe orders and combines techniques by their capabilities."""

    NONE = 0
    #: Proposes tokens between exact passes (at most one drafter per model).
    DRAFTER = enum.auto()
    #: Rewrites the base weights once at attach time (pruning, low rank, quantization). Cannot be detached.
    WEIGHTS = enum.auto()
    #: Changes activations on the fly with hooks (activation sparsity).
    ACTIVATIONS = enum.auto()
    #: Wraps or replaces the KV cache (:meth:`Technique.wrap_cache`).
    KV_CACHE = enum.auto()
    #: Decides per token and layer whether a layer runs (:meth:`Technique.layer_policy`).
    DEPTH = enum.auto()


class Saving(enum.Flag):
    """A resource a technique spends less of."""

    NONE = 0
    #: Bytes of the model's weights in device memory.
    WEIGHT_MEMORY = enum.auto()
    #: Bytes of the KV cache in device memory.
    KV_MEMORY = enum.auto()
    #: Floating-point operations per generated token.
    COMPUTE = enum.auto()
    #: Wall-clock time per generated token.
    LATENCY = enum.auto()


@dataclass(frozen=True)
class Family:
    """A kind of technique and the part of the model it owns.

    Attributes:
        name: Short name used in tables and errors (``"drafting"``).
        capability: The :class:`Capability` every technique of the family has (``NONE`` for none).
        action: What a technique of the family does, as a verb phrase that completes "they both ..."
            (``"propose the tokens the model verifies"``).
        exclusive: At most one technique of the family per model.

    The built-in families are class attributes (``Family.DRAFTING``). A third-party technique of a new kind declares
    its own ``Family(...)``; equal families are equal values.
    """

    name: str
    capability: Capability
    action: str
    exclusive: bool = True

    DRAFTING: ClassVar[Family]
    DYNAMIC_DEPTH: ClassVar[Family]
    WEIGHT_COMPRESSION: ClassVar[Family]
    ACTIVATION_SPARSITY: ClassVar[Family]
    KV_QUANTIZATION: ClassVar[Family]
    KV_EVICTION: ClassVar[Family]
    DRAFT_COMPRESSION: ClassVar[Family]
    SPARSE_DECODE: ClassVar[Family]
    SPARSE_PREFILL: ClassVar[Family]

    def __str__(self) -> str:
        return self.name


Family.DRAFTING = Family("drafting", Capability.DRAFTER, "propose the tokens the model verifies")
Family.DYNAMIC_DEPTH = Family("dynamic-depth", Capability.DEPTH, "choose the layers each token runs")
#: Not exclusive: weight transforms compose in attach order (prune, then quantize).
Family.WEIGHT_COMPRESSION = Family("weight-compression", Capability.WEIGHTS, "rewrite the base weights", False)
Family.ACTIVATION_SPARSITY = Family(
    "activation-sparsity", Capability.ACTIVATIONS, "sparsify the inputs of the linear maps"
)
Family.KV_QUANTIZATION = Family("KV-quantization", Capability.KV_CACHE, "quantize the KV cache")
Family.KV_EVICTION = Family("KV-eviction", Capability.KV_CACHE, "evict KV-cache entries")
Family.DRAFT_COMPRESSION = Family("draft-compression", Capability.NONE, "compress the drafter's weights")
Family.SPARSE_DECODE = Family(
    "sparse-decode", Capability.KV_CACHE, "choose the cache entries each decoding query reads"
)
Family.SPARSE_PREFILL = Family("sparse-prefill", Capability.NONE, "choose the keys each prompt query attends to")


@dataclass(frozen=True)
class Simulation:
    """A path that reproduces a technique's outputs without realizing (all of) its saving.

    Attributes:
        savings: The :class:`Saving` members the path does not realize.
        reason: Why, in a few words (``"keeps the dequantized weights"``).
    """

    savings: Saving
    reason: str

    def __str__(self) -> str:
        return f"{self.reason}, so it saves no {saving_names(self.savings)}"

    def within(self, saves: Saving) -> Simulation | None:
        """This simulation as it applies to a technique that saves ``saves``: a kernel's simulation names every
        saving the kernel loses, a technique loses only those it declares (``None`` if it loses none)."""
        savings = self.savings & saves
        return replace(self, savings=savings) if savings else None


def saving_names(savings: Saving) -> str:
    """``"weight memory or latency"`` for ``Saving.WEIGHT_MEMORY | Saving.LATENCY`` (``"nothing"`` for none)."""
    names = [str(m.name).lower().replace("_", " ").replace("kv", "KV") for m in Saving if m.value and m in savings]
    if not names:
        return "nothing"
    return names[0] if len(names) == 1 else f"{', '.join(names[:-1])} or {names[-1]}"


@dataclass(frozen=True)
class Exclusion:
    """A family a technique never combines with, and why."""

    family: Family
    reason: str


@dataclass(frozen=True)
class KernelChoice:
    """The configuration field that selects a technique's kernel, and the registry it names.

    A registry entry whose ``simulation`` attribute is a :class:`Simulation` only simulates the technique (the
    ``"fake"`` quantization backend); every other entry realizes it.
    """

    field: str
    registry: Registry[Any]

    def simulation(self, config: Any) -> Simulation | None:
        """The simulation of the kernel ``config`` selects (``config`` may be a config class: its default)."""
        name = getattr(config, self.field)
        if name not in self.registry:
            return None
        simulation = getattr(self.registry.get(name), "simulation", None)
        return simulation if isinstance(simulation, Simulation) else None


@dataclass(frozen=True)
class TechniqueInfo:
    """What a technique is and does (the class attribute ``info`` of every registered technique).

    Attributes:
        family: Its :class:`Family`; an exclusive family admits one technique per model.
        saves: What it saves when it runs on a real path.
        lossless: Whether greedy decoding with it reproduces the base model's greedy output exactly.
        trainable: Names of the components it trains, empty for a training-free technique.
        requires: Families it needs another technique of (``draft_quant`` needs a drafter).
        excludes: Families it never combines with, beyond the exclusive-family rule.
        simulated: How the Hugging Face path simulates it, if it does for every configuration.
        kernel: The configuration field that picks its kernel; the kernel's own ``simulation`` then decides.
        listed: Whether listings (:func:`~economize.listed_techniques`, compatibility matrices, the documentation)
            show it. An unlisted technique still builds and loads by name.
    """

    family: Family
    saves: Saving
    lossless: bool
    trainable: tuple[str, ...] = ()
    requires: tuple[Family, ...] = ()
    excludes: tuple[Exclusion, ...] = ()
    simulated: Simulation | None = None
    kernel: KernelChoice | None = None
    listed: bool = True

    def simulation(self, config: Any) -> Simulation | None:
        """How the Hugging Face path with ``config`` (an instance or the config class) simulates the technique."""
        simulation = (self.kernel.simulation(config) if self.kernel is not None else None) or self.simulated
        return simulation.within(self.saves) if simulation is not None else None


#: Capabilities of which a model has one owner: a decoder follows one drafter and one layer policy. A technique with
#: one of them belongs to an exclusive family of that capability.
SINGLE_OWNER = Capability.DRAFTER | Capability.DEPTH


def check_declaration(owner: str, capabilities: Capability, info: Any) -> None:
    """Raise ``TypeError`` unless ``info`` is a :class:`TechniqueInfo` that fits the class's ``capabilities``."""
    if not isinstance(info, TechniqueInfo):
        raise TypeError(f"{owner}.info must be a TechniqueInfo, got {type(info).__name__}")
    family = info.family
    if family.capability not in capabilities:
        raise TypeError(f"{owner} belongs to the {family} family but lacks its capability {family.capability.name}")
    for capability in Capability:
        owned = capability.value and capability in SINGLE_OWNER and capability in capabilities
        if owned and (family.capability != capability or not family.exclusive):
            raise TypeError(
                f"{owner} has the single-owner capability {capability.name}, so its family must be an exclusive "
                f"family of that capability, got {family}"
            )


def conflict(name: str, info: TechniqueInfo, other_name: str, other: TechniqueInfo) -> str | None:
    """Why the declarations of techniques ``name`` and ``other_name`` forbid combining them, ``None`` if they allow
    it. Symmetric up to the wording."""
    if info.family == other.family and info.family.exclusive:
        return (
            f"{name} cannot be combined with {other_name}: they both {info.family.action} "
            f"(at most one {info.family} technique per model)"
        )
    for first, second, first_info, second_info in ((name, other_name, info, other), (other_name, name, other, info)):
        for exclusion in first_info.excludes:
            if exclusion.family == second_info.family:
                return f"{first} cannot be combined with {second}: {exclusion.reason}"
    return None


def unmet_requirement(name: str, info: TechniqueInfo, others: Iterable[TechniqueInfo]) -> str | None:
    """Why technique ``name`` cannot run next to ``others`` alone (a family it requires is missing), ``None`` if it
    can."""
    present = {other.family for other in others}
    for family in info.requires:
        if family not in present:
            return f"{name} needs a technique that {family.action} ({family}), and the recipe has none"
    return None
