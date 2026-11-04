"""The contract between training objectives and the training loops.

An :class:`Objective` turns a batch of token sequences into a scalar loss for the parameters it trains. Both
training loops, :func:`economize.train.fit` and :class:`economize.train.EffTrainer`, drive objectives through this
interface only, so a technique gains training support by registering an objective, never by editing a loop::

    from dataclasses import dataclass
    from economize.train import Objective, ObjectiveOutput, ParamGroup, TeacherRequest, register_objective

    @register_objective("my_drafter_kl")
    @dataclass
    class MyDrafterKL(Objective):
        temperature: float = 1.0                     # hyperparameters are dataclass fields

        def teacher_request(self, model):
            return TeacherRequest(hidden_layers=(model.adapter.num_layers,))

        def parameter_groups(self, model):
            return [ParamGroup("drafter", list(model.techniques["my_drafter"].parameters()))]

        def loss(self, model, batch, teacher):
            ...
            return ObjectiveOutput(loss, {"kl": kl.detach()})

A loop calls :meth:`Objective.setup` once, freezes every parameter of the model except those returned by
:meth:`Objective.parameter_groups`, builds one optimizer group per :class:`ParamGroup`, and then, per batch, runs the
teacher pass described by :meth:`Objective.teacher_request` under ``torch.no_grad()`` and calls
:meth:`Objective.loss`. Optimizer settings not given by the user come from :attr:`Objective.defaults` and from each
group's own ``lr`` / ``weight_decay``.

The teacher is the target model with the weights it has when training starts: drafters that adapt layers through
LoRA (active only inside the drafter's path) or trainable copies of layers leave the target's own pass unchanged.

Objectives are found by name in :data:`OBJECTIVES`; installed packages add entries through the entry-point group
``economize.objectives``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field, fields, replace
from typing import (
    TYPE_CHECKING,
    ClassVar,
    Generic,
    Literal,
    Protocol,
    TypeVar,
    cast,
    get_args,
    get_origin,
    runtime_checkable,
)

import torch
from torch import nn

from ..errors import ConfigError
from ..registry import Registry

if TYPE_CHECKING:
    from ..core.model import EfficientModel
    from ..core.technique import Technique
    from ..runtime import KVCache

__all__ = [
    "DEFAULT_OBJECTIVES",
    "OBJECTIVES",
    "Batch",
    "Objective",
    "ObjectiveOutput",
    "OptimDefaults",
    "ParamGroup",
    "TeacherOutputs",
    "TeacherRequest",
    "TechniqueObjective",
    "TrainRequestProvider",
    "attached_technique",
    "build_objective",
    "default_objective",
    "register_default_objective",
    "register_objective",
    "require_teacher",
    "teacher_cache",
    "trainable",
    "unique_parameters",
]

GroupKind = Literal["head", "backbone"]


@dataclass
class Batch:
    """Token sequences of one training step.

    Attributes:
        input_ids: ``[B, T]`` token ids.
        attention_mask: ``[B, T]``, 1 for real tokens and 0 for right padding (``None``: no padding).
        loss_mask: ``[B, T]``, 1 where a position's next-token prediction counts in the loss (``None``: every
            position with a next token; chat data masks the prompt here).
    """

    input_ids: torch.Tensor
    attention_mask: torch.Tensor | None = None
    loss_mask: torch.Tensor | None = None

    def to(self, device: torch.device | str) -> Batch:
        """The batch on ``device``."""

        def move(t: torch.Tensor | None) -> torch.Tensor | None:
            return None if t is None else t.to(device, non_blocking=True)

        return Batch(self.input_ids.to(device, non_blocking=True), move(self.attention_mask), move(self.loss_mask))

    @property
    def shape(self) -> torch.Size:
        return self.input_ids.shape


@dataclass(frozen=True)
class TeacherRequest:
    """What an objective needs from the teacher pass over a batch.

    Attributes:
        hidden_layers: Residual-stream states to keep: ``0`` is the embedding output, ``l`` the output of block
            ``l`` (1-based), ``num_layers`` the last block's output before the final norm.
        cache: Keep every layer's keys and values of the pass in a :class:`~economize.runtime.KVCache` (rollouts that
            attend to the teacher's prefix need them).
        logits: Materialize the full ``[B, T, V]`` teacher logits. Most objectives leave this off and call
            :meth:`TeacherOutputs.log_probs` on the positions they score, which is far cheaper for large
            vocabularies.
        own_pass: Whether the objective reads the trained model's own pass (its ``hidden`` states or cache) when
            a larger teacher replaces the distributions. Objectives that read only ``final``, ``head`` and
            ``logits`` set it to ``False``: with a teacher head that can run alone (:func:`larger_teacher`) the
            loops then skip the trained model's exact pass, whose outputs would all be replaced. Without a
            teacher head the pass always runs.
    """

    hidden_layers: tuple[int, ...] = ()
    cache: bool = False
    logits: bool = False
    own_pass: bool = True

    def merge(self, other: TeacherRequest) -> TeacherRequest:
        """The union of two requests (for objectives that combine others)."""
        return TeacherRequest(
            tuple(sorted(set(self.hidden_layers) | set(other.hidden_layers))),
            self.cache or other.cache,
            self.logits or other.logits,
            self.own_pass or other.own_pass,
        )


@runtime_checkable
class TrainRequestProvider(Protocol):
    """Implemented by a technique that states what its training needs from the teacher pass (drafters do).
    :class:`TechniqueObjective` uses it as its default :meth:`~Objective.teacher_request`."""

    def train_request(self, model: EfficientModel) -> TeacherRequest:
        """What training this technique needs from the teacher pass over a batch."""
        ...


@dataclass
class TeacherOutputs:
    """Results of the teacher pass, computed without gradients.

    Attributes:
        hidden: Residual-stream states ``[B, T, D]`` by layer index, as listed in the request.
        final: Last block's output ``[B, T, D]`` (before the final norm).
        head: Maps last-block outputs to logits (final norm and LM head of the teacher).
        cache: The pass's keys and values, slot ``t`` holding position ``t`` (when requested).
        logits: Full ``[B, T, V]`` logits (when requested).
    """

    hidden: Mapping[int, torch.Tensor]
    final: torch.Tensor
    head: Callable[[torch.Tensor], torch.Tensor]
    cache: KVCache | None = None
    logits: torch.Tensor | None = None

    def log_probs(self, *index: torch.Tensor | slice) -> torch.Tensor:
        """Float32 log-probabilities of the teacher at ``final[index]`` (e.g. ``(slice(None), positions)``). The
        head runs in the model's own dtypes, also when an objective calls this inside the loop's autocast region."""
        if self.logits is not None:
            return self.logits[_on(index, self.logits.device)].float().log_softmax(-1)
        index = _on(index, self.final.device)  # under ``device_map`` the final state sits with the last layer
        with torch.no_grad(), torch.autocast(self.final.device.type, enabled=False):
            return self.head(self.final[index]).float().log_softmax(-1)


def _on(index: tuple[torch.Tensor | slice, ...], device: torch.device) -> tuple[torch.Tensor | slice, ...]:
    """``index`` with its tensors on ``device``."""
    return tuple(i.to(device) if isinstance(i, torch.Tensor) else i for i in index)


def require_teacher(teacher: TeacherOutputs | None, owner: str) -> TeacherOutputs:
    """``teacher``, checked to be present: ``owner`` (an objective or technique) asked for a teacher pass.

    Raises:
        ConfigError: ``teacher`` is ``None``.
    """
    if teacher is None:
        raise ConfigError(f"{owner} needs the outputs of a teacher pass (see its teacher_request()), got None")
    return teacher


def teacher_cache(teacher: TeacherOutputs, owner: str) -> KVCache:
    """The K/V cache of the teacher pass, which ``owner`` asked for with ``TeacherRequest(cache=True)``.

    Raises:
        ConfigError: The teacher pass kept no cache.
    """
    if teacher.cache is None:
        raise ConfigError(f"{owner} needs the teacher pass's K/V cache: request it with TeacherRequest(cache=True)")
    return teacher.cache


@dataclass
class ParamGroup:
    """Parameters an objective trains, with their optimizer settings.

    Attributes:
        name: Label used in logs and checkpoints.
        params: The parameters (all of them get ``requires_grad=True``).
        kind: ``"head"`` for newly added modules (learning rate ``lr`` of the run), ``"backbone"`` for adapted
            or copied pretrained layers (learning rate ``lr_backbone`` of the run).
        lr: Default learning rate of the group, used when the run sets no rate for its kind.
        weight_decay: Weight decay of the group (``None``: the objective's default).
    """

    name: str
    params: list[nn.Parameter]
    kind: GroupKind = "head"
    lr: float | None = None
    weight_decay: float | None = None


@dataclass(frozen=True)
class OptimDefaults:
    """Optimizer and schedule settings an objective recommends; the run's arguments override each of them.

    Attributes:
        lr: Learning rate of ``"head"`` groups.
        lr_backbone: Learning rate of ``"backbone"`` groups (``None``: each group's own ``lr``, else ``lr``).
        weight_decay: Weight decay of groups that set none.
        betas: AdamW betas.
        warmup_steps: Linear warmup from 0.
        schedule: Decay after warmup, a name in :data:`~economize.train.SCHEDULES` (``"cosine"``, ``"linear"``,
            ``"constant"`` or a registered one).
        min_lr_ratio: Final learning rate as a fraction of the peak (cosine and linear).
        grad_clip: Maximum global gradient norm (``None``: no clipping).
    """

    lr: float = 5e-4
    lr_backbone: float | None = None
    weight_decay: float = 0.01
    betas: tuple[float, float] = (0.9, 0.95)
    warmup_steps: int = 100
    schedule: str = "cosine"
    min_lr_ratio: float = 0.1
    grad_clip: float | None = 1.0

    def __post_init__(self) -> None:
        from .schedule import SCHEDULES

        SCHEDULES.get(self.schedule)  # fail early on unknown names


@dataclass
class ObjectiveOutput:
    """The loss of one batch and the metrics to log.

    Attributes:
        loss: Scalar tensor to minimize.
        metrics: Detached scalar tensors (or floats) by name; loops average them over accumulation steps.
    """

    loss: torch.Tensor
    metrics: Mapping[str, torch.Tensor | float] = field(default_factory=dict)


_O = TypeVar("_O", bound="Objective")
_T = TypeVar("_T")
_TT = TypeVar("_TT", bound="Technique")


class Objective(ABC):
    """A training objective. Subclasses are dataclasses whose fields are the objective's hyperparameters.

    Class attributes:
        name: Registry name (set by :func:`register_objective`).
        defaults: Optimizer and schedule settings the objective recommends.
    """

    name: ClassVar[str] = ""
    defaults: ClassVar[OptimDefaults] = OptimDefaults()

    def setup(self, model: EfficientModel) -> None:
        """Prepare ``model`` once before the first step (check that the needed technique is attached, install
        training-only hooks). The default does nothing."""

    def teardown(self, model: EfficientModel) -> None:
        """Undo :meth:`setup` after training. The default does nothing."""

    def begin_step(self, step: int, total_steps: int) -> None:
        """Called by the loops before each optimizer step (0-based ``step`` of ``total_steps``), for objectives
        with curricula or schedules of their own. The default does nothing."""

    @abstractmethod
    def parameter_groups(self, model: EfficientModel) -> list[ParamGroup]:
        """The parameters to train, in optimizer groups. Everything else in ``model`` stays frozen."""

    def modules(self, model: EfficientModel) -> Sequence[nn.Module]:
        """Modules the objective trains that live outside ``model`` (gates or probes whose result is exported as
        a file, not attached as a technique). The loops register them with the trained module, so that data
        parallel wrappers synchronize their gradients, and checkpoints save and restore their states. Called once,
        after :meth:`setup`. The default is none."""
        return ()

    def teacher_request(self, model: EfficientModel) -> TeacherRequest | None:
        """What the teacher pass must provide, or ``None`` when the objective needs no teacher (plain
        cross-entropy). The default requests the last block's output (teacher logits through
        :meth:`TeacherOutputs.log_probs`)."""
        return TeacherRequest(hidden_layers=(model.adapter.num_layers,))

    @abstractmethod
    def loss(self, model: EfficientModel, batch: Batch, teacher: TeacherOutputs | None) -> ObjectiveOutput:
        """The loss of ``batch`` (called with gradients enabled, inside the loop's autocast)."""

    def evaluate(
        self, model: EfficientModel, batch: Batch, teacher: TeacherOutputs | None
    ) -> Mapping[str, torch.Tensor | float]:
        """Evaluation metrics of ``batch`` (called under ``torch.no_grad()``). Tensors may be vectors, e.g. one
        value per draft step for a teacher-forced curve. The default returns the training metrics and loss."""
        output = self.loss(model, batch, teacher)
        return {"loss": output.loss.detach(), **output.metrics}

    # ---- helpers -----------------------------------------------------------------------------------------------
    def hyperparameters(self) -> dict[str, object]:
        """The dataclass fields of the objective (its constructor arguments) and their values."""
        return {f.name: getattr(self, f.name) for f in fields(self) if f.init}  # type: ignore[arg-type]

    def with_options(self: _O, **changes: object) -> _O:
        """A copy with some hyperparameters changed."""
        return replace(self, **changes)  # type: ignore[type-var]


class TechniqueObjective(Objective, Generic[_TT]):
    """An objective that trains one attached technique of a known type (the drafters' and dynamic-depth techniques'
    own objectives).

    Subclasses are dataclasses with a field ``technique`` (the attachment name, defaulting to the technique's
    registry name :attr:`Technique.name <economize.core.Technique.name>`, which is its config's ``technique``). The
    type argument names the class they train (:meth:`technique_type`), and they get the lookup (:meth:`trained`),
    a :meth:`setup` that fails early when the technique is missing, the technique's own teacher request
    (:meth:`teacher_request`) and one ``"head"`` optimizer group with all of the technique's parameters
    (:meth:`parameter_groups`, named :attr:`group`)::

        @register_objective("my_drafter_kl", default_for=(MyDrafter.name,))
        @dataclass
        class MyDrafterKL(TechniqueObjective[MyDrafter]):
            technique: str = MyDrafter.name

            def loss(self, model, batch, teacher):
                drafter = self.trained(model)
                ...

    Attributes:
        technique: Name under which the trained technique is attached.

    Class attributes:
        group: Name of the optimizer group of :meth:`parameter_groups` (``""``: the attachment name).
    """

    technique: str
    group: ClassVar[str] = ""
    #: The type argument of the nearest ``TechniqueObjective[...]`` base (``None`` while it is a type variable).
    _technique_type: ClassVar[type | None] = None

    def __init_subclass__(cls, **kwargs: object) -> None:
        super().__init_subclass__(**kwargs)
        for base in cls.__dict__.get("__orig_bases__", ()):
            origin = get_origin(base)
            if isinstance(origin, type) and issubclass(origin, TechniqueObjective):
                concrete = [arg for arg in get_args(base) if isinstance(arg, type)]
                cls._technique_type = concrete[0] if len(concrete) == 1 else None

    @classmethod
    def technique_type(cls) -> type[_TT]:
        """The class (or a base class) of the techniques this objective trains: by default the type argument of
        the class (``MyDrafterKL(TechniqueObjective[MyDrafter])`` trains a ``MyDrafter``).

        Raises:
            NotImplementedError: The class gives no concrete type argument and does not override this method.
        """
        if cls._technique_type is None:
            raise NotImplementedError(f"{cls.__name__} names no technique class: subclass "
                                      "TechniqueObjective[MyTechnique] or override technique_type()")
        return cast("type[_TT]", cls._technique_type)

    def trained(self, model: EfficientModel) -> _TT:
        """The trained technique.

        Raises:
            IncompatibleTechniquesError: Nothing is attached as :attr:`technique`, or it is no
                :meth:`technique_type`.
        """
        kind = self.technique_type()
        return attached_technique(model, self.technique, kind,
                                  f"objective {self.name!r} needs a {kind.__name__} attached as {self.technique!r}")

    def setup(self, model: EfficientModel) -> None:
        """Check that the trained technique is attached."""
        self.trained(model)

    def teacher_request(self, model: EfficientModel) -> TeacherRequest | None:
        """The trained technique's own request when it is a :class:`TrainRequestProvider`, else the default of
        :meth:`Objective.teacher_request` (the last block's output)."""
        technique = self.trained(model)
        if isinstance(technique, TrainRequestProvider):
            return technique.train_request(model)
        return super().teacher_request(model)

    def parameter_groups(self, model: EfficientModel) -> list[ParamGroup]:
        """All parameters of the trained technique, as one ``"head"`` group named :attr:`group`.

        Raises:
            IncompatibleTechniquesError: The technique has no parameters.
        """
        from ..errors import IncompatibleTechniquesError

        params = list(self.trained(model).parameters())
        if not params:
            raise IncompatibleTechniquesError(f"technique {self.technique!r} has no parameters to train")
        return [ParamGroup(self.group or self.technique, params, "head")]


#: Training objectives by name (entry-point group ``economize.objectives``).
OBJECTIVES: Registry[type[Objective]] = Registry("training objective", "economize.objectives")


class _DefaultObjectives(Registry[str]):
    """Objective names by technique name. Also reads and writes like the ``dict`` it replaced
    (``DEFAULT_OBJECTIVES[name]``, ``DEFAULT_OBJECTIVES[name] = objective``, ``del``, ``setdefault``, ``pop``)."""

    def __getitem__(self, name: str) -> str:
        return self.get(name)

    def __setitem__(self, name: str, objective: str) -> None:
        self.register(name, objective, override=True)

    def __delitem__(self, name: str) -> None:
        self.unregister(name)

    def setdefault(self, name: str, objective: str) -> str:
        if name not in self:
            self.register(name, objective)
        return self.get(name)

    def pop(self, name: str, *default: str) -> str:
        if default and name not in self:
            return default[0]
        return self.unregister(name)


#: The objective that trains each technique when none is named, by technique name (the ``name`` of a technique class;
#: :func:`default_objective` also finds the entry of a base class). Plugins add entries through
#: :func:`register_objective` (``default_for``), :func:`register_default_objective` or the entry-point group
#: ``economize.default_objectives`` (an entry ``my_drafter = "my_pkg:OBJECTIVE_NAME"`` loads the objective's name).
DEFAULT_OBJECTIVES: Registry[str] = _DefaultObjectives("default objective", "economize.default_objectives")


def register_default_objective(technique: str, objective: str) -> None:
    """Train ``technique`` (a technique name) with the registered objective ``objective`` when none is named."""
    DEFAULT_OBJECTIVES.register(technique, objective, override=True)


def register_objective(name: str, *, default_for: Sequence[str] = ()) -> Callable[[type[_O]], type[_O]]:
    """Class decorator registering an :class:`Objective` subclass in :data:`OBJECTIVES` under ``name``.

    Args:
        name: Registry name.
        default_for: Technique names this objective trains when :func:`default_objective` is asked for one.
    """

    def add(cls: type[_O]) -> type[_O]:
        cls.name = name
        OBJECTIVES.register(name, cls)
        for technique in default_for:
            register_default_objective(technique, name)
        return cls

    return add


def default_objective(model: EfficientModel) -> Objective:
    """The default objective of the first attached technique that has one (e.g. ``eagle3_ttt`` for ``eagle3``).

    The lookup follows the technique's type: a subclass attached under its own name (``class MyEagle3(Eagle3Technique):
    name = "my_eagle3"``) trains with the default objective of the nearest base class that has one, and the objective
    is pointed at the attachment name.

    Raises:
        ConfigError: No attached technique has a default objective.
    """
    for attached, technique in model.techniques.items():
        for cls in type(technique).__mro__:
            name = cls.__dict__.get("name")
            if isinstance(name, str) and name in DEFAULT_OBJECTIVES:
                return OBJECTIVES.get(DEFAULT_OBJECTIVES.get(name))(technique=attached)  # type: ignore[call-arg]
    raise ConfigError(
        f"no default objective for the attached techniques {list(model.techniques)}; name one of "
        f"{OBJECTIVES.names()}"
    )


def attached_technique(model: EfficientModel, name: str, kind: type[_T], error: str) -> _T:
    """The technique attached to ``model`` as ``name``, checked to be a ``kind`` (a class, or a runtime-checkable
    protocol such as :class:`~economize.train.objectives.StepDrafter`). Objectives look their technique up with it.

    Raises:
        IncompatibleTechniquesError: With message ``error``, when nothing is attached as ``name`` or the attached
            technique is no ``kind``.
    """
    from ..errors import IncompatibleTechniquesError

    technique = model.techniques[name] if name in model.techniques else None  # noqa: SIM401  (nn.ModuleDict has no get)
    if not isinstance(technique, kind):
        raise IncompatibleTechniquesError(error)
    return technique


def build_objective(objective: str | Objective, **options: object) -> Objective:
    """An objective instance from a registered name and hyperparameters, or ``objective`` itself."""
    if isinstance(objective, Objective):
        return objective.with_options(**options) if options else objective
    return OBJECTIVES.get(objective)(**options)


def unique_parameters(params: Iterable[nn.Parameter]) -> list[nn.Parameter]:
    """``params`` without repeats (e.g. tied embeddings), in order of first appearance."""
    seen: set[int] = set()
    out = []
    for p in params:
        if id(p) not in seen:
            seen.add(id(p))
            out.append(p)
    return out


def trainable(groups: Sequence[ParamGroup]) -> list[nn.Parameter]:
    """The distinct parameters of ``groups``, in order."""
    return unique_parameters(p for group in groups for p in group.params)
