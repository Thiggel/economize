"""Base class and registry of efficiency techniques."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, ClassVar, TypeVar

from torch import nn

from ..errors import ConfigError, IncompatibleTechniquesError
from ..registry import Registry
from .config import TechniqueConfig
from .metadata import Capability, Simulation, TechniqueInfo, check_declaration, conflict, unmet_requirement

if TYPE_CHECKING:
    from ..arch import ArchAdapter
    from ..decoding import Drafter
    from ..runtime import KVCache, LayerPolicy, LayerRunner

__all__ = [
    "CAPABILITY_ORDER",
    "TECHNIQUES",
    "AttachContext",
    "Capability",
    "Technique",
    "attach_rank",
    "listed_techniques",
    "register_technique",
    "technique_for",
]


#: Attach order of the capability groups; inside a group techniques attach by :attr:`Technique.order`.
CAPABILITY_ORDER: tuple[Capability, ...] = (
    Capability.WEIGHTS,
    Capability.ACTIVATIONS,
    Capability.KV_CACHE,
    Capability.DEPTH,
    Capability.DRAFTER,
)


@dataclass(frozen=True)
class AttachContext:
    """What a technique may use while it attaches.

    Attributes:
        techniques: Every technique of the recipe, in attach order (attached or not yet attached).
        calibration: Calibration token batches passed to :func:`economize.apply` (``None`` if not given). Techniques
            that need data raise :class:`~economize.ConfigError` without it.
        options: Free-form options for third-party techniques.
    """

    techniques: tuple[Technique, ...] = ()
    calibration: Any = None
    options: dict[str, Any] = field(default_factory=dict)

    def layer_copies(self) -> dict[int, list[nn.Module]]:
        """The :meth:`Technique.layer_copies` of every technique of the recipe, by 1-based layer index."""
        copies: dict[int, list[nn.Module]] = {}
        for technique in self.techniques:
            for index, layer in technique.layer_copies().items():
                copies.setdefault(int(index), []).append(layer)
        return copies


class Technique(nn.Module):
    """An efficiency technique attached to a base model.

    A technique owns its added parameters (an ``nn.Module`` tree). Its life cycle inside a recipe is

    1. construction from its config and the model's :class:`~economize.arch.ArchAdapter` (no side effects on the
       base model),
    2. :meth:`check_compatible` with the other techniques of the recipe,
    3. loading of a saved state (``load_state_dict``) when the recipe comes from a checkpoint,
    4. :meth:`attach`, in recipe order (see :data:`CAPABILITY_ORDER` and :attr:`order`),
    5. :meth:`refresh` whenever the base model changed after attach (optional),
    6. :meth:`detach`, which undoes every change made to the base model.

    Techniques with :attr:`Capability.WEIGHTS` rewrite the base weights in place; their :meth:`detach` raises.

    Subclasses set :attr:`name`, :attr:`config_class`, :attr:`capabilities`, :attr:`info` and optionally
    :attr:`order`, and are registered with :func:`register_technique`.
    """

    name: ClassVar[str]
    config_class: ClassVar[type[TechniqueConfig]]
    capabilities: ClassVar[Capability] = Capability.NONE
    #: What the technique is: its family, what it saves, whether it is lossless and which paths only simulate it
    #: (see :class:`~economize.TechniqueInfo`). Every technique that is built declares it; the compatibility rules
    #: (:meth:`conflict`) and the documentation's technique table derive from it.
    info: ClassVar[TechniqueInfo]
    #: Whether greedy decoding with this technique reproduces the base model's greedy output exactly (a copy of
    #: ``info.lossless``).
    lossless: ClassVar[bool]
    #: Attach order inside the technique's capability group (lower attaches first).
    order: ClassVar[int] = 50
    #: Whether the modules and hooks the technique adds may be recorded in CUDA graphs (``DecodeConfig.cuda_graphs``):
    #: no host synchronization and no Python-side state that changes between calls. ``False`` makes the model
    #: decode eagerly. A technique that wraps the KV cache (:meth:`wrap_cache`) is graph-safe only if its wrapper
    #: works under the graphs (see :class:`~economize.techniques._kv.CacheWrapper`).
    graph_safe: ClassVar[bool] = True

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        info = getattr(cls, "info", None)
        if info is not None:
            check_declaration(cls.__name__, cls.capabilities, info)
            cls.lossless = info.lossless

    def __init__(self, config: TechniqueConfig, adapter: ArchAdapter) -> None:
        super().__init__()
        if getattr(type(self), "info", None) is None:
            raise ConfigError(
                f"technique {type(self).__name__} declares no TechniqueInfo; set the class attribute info "
                "(see economize.TechniqueInfo)"
            )
        if not isinstance(config, self.config_class):
            raise ConfigError(
                f"{type(self).__name__} needs a {self.config_class.__name__}, got {type(config).__name__}"
            )
        config.validate(adapter)
        self.config = config
        self.adapter = adapter  # plain attribute: the base model is not part of the technique's state

    @classmethod
    def conflict(
        cls, config: TechniqueConfig | None, other: type[Technique], other_config: TechniqueConfig | None
    ) -> str | None:
        """Why this technique, built from ``config``, cannot be combined with technique ``other`` built from
        ``other_config``; ``None`` if it can.

        The default applies the rules of the two :attr:`info` declarations (one technique per exclusive family,
        :attr:`~economize.TechniqueInfo.excludes`). A technique with a rule that depends on its configuration
        extends it; ``None`` for a configuration means the configuration is not known (:func:`economize.compatible`
        on names), and such a rule then does not apply.
        """
        return conflict(cls.name, cls.info, other.name, other.info)

    def check_compatible(self, others: Sequence[Technique]) -> None:
        """Raise :class:`~economize.IncompatibleTechniquesError` if this technique cannot be combined with
        ``others``: a :meth:`conflict` with one of them, or a family it :attr:`~economize.TechniqueInfo.requires`
        missing among them."""
        for other in others:
            reason = self.conflict(self.config, type(other), other.config)
            if reason is not None:
                raise IncompatibleTechniquesError(reason)
        reason = unmet_requirement(self.name, self.info, (other.info for other in others))
        if reason is not None:
            raise IncompatibleTechniquesError(reason)

    @classmethod
    def trainable_for(cls, config: TechniqueConfig | None = None) -> tuple[str, ...]:
        """The components (:attr:`~economize.TechniqueInfo.trainable`) this technique has with ``config``;
        ``None``: every component it can have. A technique whose components depend on its configuration
        extends it."""
        return cls.info.trainable

    @classmethod
    def simulation_for(cls, config: TechniqueConfig | None = None) -> Simulation | None:
        """How the Hugging Face path of this technique with ``config`` (default: the config class's defaults)
        simulates its saving, ``None`` if it realizes it (see :class:`~economize.Simulation`)."""
        return cls.info.simulation(config if config is not None else cls.config_class)

    def simulation(self) -> Simulation | None:
        """:meth:`simulation_for` this technique's configuration."""
        return self.simulation_for(self.config)

    def attach(self, model: nn.Module, adapter: ArchAdapter, context: AttachContext | None = None) -> None:
        """Install the technique on the base ``model`` (hooks, rewritten weights). Called once, in recipe order."""

    def detach(self) -> None:
        """Undo every change made to the base model."""

    def layer_copies(self) -> Mapping[int, nn.Module]:
        """Copies of base-model decoder layers (1-based index to module) that the technique runs in place of the
        layer on some path, reading and writing that layer's KV cache (e.g. a drafter's trainable copies).

        A weight transform that changes the shapes of layer ``i`` (width pruning) makes the same change to every copy
        of it before the copies' owner attaches, so that the copies keep fitting the base model's cache. The recipe
        then carries the changed state over to the owner it builds again on the transformed model. Default: none.
        """
        return {}

    def refresh(self) -> None:
        """Recompute what the technique derived from the base model at attach time (hooks on modules that were
        since replaced, fused weights). :meth:`economize.EfficientModel.refresh` calls it after the base model's
        weights or modules changed, e.g. after a policy update in reinforcement learning."""

    def drafter(self, runner: LayerRunner) -> Drafter:
        """A new drafter (techniques with :attr:`Capability.DRAFTER`).

        :class:`~economize.EfficientModel` asks for one per generation. The one exception is a decoder that records CUDA
        graphs: it keeps its drafter across generations (see :meth:`~economize.decoding.Drafter.enable_graphs`), so
        :meth:`~economize.decoding.Drafter.begin` must reset everything that belongs to one generation.
        """
        raise NotImplementedError(f"technique {self.name!r} does not draft tokens")

    def layer_policy(self) -> LayerPolicy | None:
        """The per-token layer policy of the exact path (techniques with :attr:`Capability.DEPTH`)."""
        return None

    def wrap_cache(self, cache: KVCache, previous: KVCache | None = None) -> KVCache:
        """Wrap the KV cache of one generation (techniques with :attr:`Capability.KV_CACHE`).

        ``previous`` is this technique's wrapper of the decoder's previous generation when that generation ran on
        the same static cache of CUDA graphs (``cache`` is then the same object as before). A graph-safe technique
        restarts and returns it, so that graphs recorded over the wrapper stay valid; any technique may also ignore
        it and wrap anew, which only costs the decoder its recorded graphs.
        """
        return cache


def attach_rank(technique: Technique | type[Technique]) -> tuple[int, int]:
    """Sort key of the recipe: capability group (:data:`CAPABILITY_ORDER`, ``NONE`` last), then :attr:`order`."""
    group = next(
        (i for i, capability in enumerate(CAPABILITY_ORDER) if capability in technique.capabilities),
        len(CAPABILITY_ORDER),
    )
    return group, technique.order


TECHNIQUES: Registry[type[Technique]] = Registry("technique", "economize.techniques")

T = TypeVar("T", bound=type[Technique])


def register_technique(cls: T) -> T:
    """Class decorator registering a technique under ``cls.name``.

    Raises:
        ConfigError: The configuration class names another technique.
        TypeError: The class declares no :class:`~economize.TechniqueInfo` ``info``, or one that does not fit its
            capabilities.
    """
    check_declaration(cls.__name__, cls.capabilities, getattr(cls, "info", None))
    if cls.config_class.technique != cls.name:
        raise ConfigError(
            f"{cls.config_class.__name__}.technique is {cls.config_class.technique!r}, expected {cls.name!r}"
        )
    TECHNIQUES.register(cls.name, cls)
    return cls


def listed_techniques() -> list[str]:
    """Names of the registered techniques that listings show (:attr:`~economize.TechniqueInfo.listed`), sorted."""
    return [name for name in TECHNIQUES.names() if TECHNIQUES.get(name).info.listed]


def technique_for(config: TechniqueConfig) -> type[Technique]:
    """The registered technique class that builds ``config``."""
    return TECHNIQUES.get(config.technique)
