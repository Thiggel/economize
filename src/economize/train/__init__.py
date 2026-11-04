"""Training of the components that techniques add (drafters, routers, adapters) and of adapted base models.

The contract is :class:`Objective` (:mod:`economize.train.objective`): registered objectives declare their trainable
parameters, what they need from the teacher pass and their recommended optimizer settings, and both loops,
:func:`fit` (a plain loop, optionally on ``accelerate``) and :class:`EffTrainer` (``transformers.Trainer``), drive
any registered objective. Importing this package registers the technique-agnostic objectives ``distill_kl``,
``step_distill`` and ``ce``. Technique-specific objectives (``layerskip``, ``early_exit_kl``, ``eagle3_ttt``,
``medusa_ce``, ...) live with their technique under :mod:`economize.techniques` and register when ``economize`` is
imported. Objective classes that earlier releases exported from this package stay importable here as deprecated
aliases that resolve to the registered classes through :data:`OBJECTIVES` and warn (``economize.train.LayerSkip``
moved to :mod:`economize.techniques.early_exit`, for example), so this package never imports a concrete technique.
:class:`EffTrainer` imports
``transformers.Trainer`` (and thereby ``accelerate``) only when first accessed. :mod:`economize.train.rl` holds the
pieces of on-policy RL with a drafter inside the policy (exact and hybrid log-probabilities with gradients,
acceptance on the policy's samples).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from . import rl
from .checkpoint import RunCheckpointer, latest_checkpoint, load_trained, save_trained
from .data import (
    BatchSource,
    Example,
    LoaderSource,
    PasskeyRetrieval,
    SequenceSource,
    TensorSource,
    TokenStream,
    as_batch,
    as_source,
    collate,
    fixed_windows,
    generate_self_distillation,
    shard_source,
    tokenize_chats,
    tokenize_texts,
)
from .evaluate import MetricMeans, evaluate, evaluate_batch
from .loop import TrainConfig, TrainResult, fit
from .module import ObjectiveModule
from .objective import (
    DEFAULT_OBJECTIVES,
    OBJECTIVES,
    Batch,
    Objective,
    ObjectiveOutput,
    OptimDefaults,
    ParamGroup,
    TeacherOutputs,
    TeacherRequest,
    TechniqueObjective,
    TrainRequestProvider,
    attached_technique,
    build_objective,
    default_objective,
    register_default_objective,
    register_objective,
    trainable,
    unique_parameters,
)
from .objectives import (
    DIVERGENCES,
    STATE_LOSSES,
    CrossEntropy,
    DistillKL,
    StepDistill,
    StepDrafter,
    TrainableDrafter,
)
from .offload import OffloadedAdamW
from .optim import build_optimizer, build_scheduler
from .schedule import SCHEDULES, lr_factor
from .teacher import (
    LargerTeacher,
    StandaloneTeacherHead,
    TeacherHead,
    larger_teacher,
    policy_pass,
    run_teacher,
    teacher_pass,
)

if TYPE_CHECKING:
    from .trainer import EffCollator, EffTrainer, EffTrainingArguments, economize_training_args

__all__ = [
    "DEFAULT_OBJECTIVES",
    "DIVERGENCES",
    "OBJECTIVES",
    "SCHEDULES",
    "STATE_LOSSES",
    "Batch",
    "BatchSource",
    "CrossEntropy",
    "DistillKL",
    "EffCollator",
    "EffTrainer",
    "EffTrainingArguments",
    "Example",
    "LargerTeacher",
    "LoaderSource",
    "MetricMeans",
    "Objective",
    "ObjectiveModule",
    "ObjectiveOutput",
    "OffloadedAdamW",
    "OptimDefaults",
    "ParamGroup",
    "PasskeyRetrieval",
    "RunCheckpointer",
    "SequenceSource",
    "StandaloneTeacherHead",
    "StepDistill",
    "StepDrafter",
    "TeacherHead",
    "TeacherOutputs",
    "TeacherRequest",
    "TechniqueObjective",
    "TensorSource",
    "TokenStream",
    "TrainConfig",
    "TrainRequestProvider",
    "TrainResult",
    "TrainableDrafter",
    "as_batch",
    "as_source",
    "attached_technique",
    "build_objective",
    "build_optimizer",
    "build_scheduler",
    "collate",
    "default_objective",
    "economize_training_args",
    "evaluate",
    "evaluate_batch",
    "fit",
    "fixed_windows",
    "generate_self_distillation",
    "larger_teacher",
    "latest_checkpoint",
    "load_trained",
    "lr_factor",
    "policy_pass",
    "register_default_objective",
    "register_objective",
    "rl",
    "run_teacher",
    "save_trained",
    "shard_source",
    "teacher_pass",
    "tokenize_chats",
    "tokenize_texts",
    "trainable",
    "unique_parameters",
]

_LAZY = {"EffCollator", "EffTrainer", "EffTrainingArguments", "economize_training_args"}

#: Deprecated names of objectives that moved to their technique's subpackage: ``name -> (registered name, new home)``.
_REGISTERED = {
    "JumpRollout": ("jump_rollout", "economize.techniques.jump.JumpRollout"),
    "JumpAware": ("jump_aware", "economize.techniques.jump.JumpAware"),
    "LayerSkip": ("layerskip", "economize.techniques.early_exit.LayerSkip"),
}


def __getattr__(name: str) -> Any:
    """Resolve the lazy exports: the ``transformers`` trainer, and the technique objectives kept under their
    deprecated old names (these warn with a :class:`DeprecationWarning`).

    Args:
        name: The attribute looked up on :mod:`economize.train`.

    Returns:
        The trainer class or function, or the registered objective class.

    Raises:
        AttributeError: If ``name`` is no lazy export or its objective is not registered.
    """
    if name in _LAZY:
        from . import trainer

        return getattr(trainer, name)
    if name in _REGISTERED:
        from .._deprecation import warn_moved
        from ..errors import UnknownNameError

        registered, new = _REGISTERED[name]
        try:
            objective = OBJECTIVES.get(registered)
        except UnknownNameError as error:
            raise AttributeError(f"module {__name__!r} has no attribute {name!r} ({error})") from error
        warn_moved(f"{__name__}.{name}", new)
        return objective
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
