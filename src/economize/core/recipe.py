"""Building, ordering and attaching the techniques of one model (the *recipe*).

A recipe attaches techniques in a fixed order (:data:`~economize.core.technique.CAPABILITY_ORDER`, then
:attr:`Technique.order`): weight transforms first, then activation hooks, KV-cache wrappers, dynamic depth, the
drafter, and techniques without a capability (e.g. ``draft_quant``, which needs the drafter in place).

Weight transforms can change the model's shape (depth pruning removes layers, low rank replaces linear maps).
Every other technique is therefore built again on the transformed model before it attaches, so that its modules
see the final architecture, and loads the state of its first build (a transform may have reshaped that build's
copies of base layers, see :meth:`~economize.core.technique.Technique.layer_copies`). A saved state that does not fit
the first build's copies was saved from a transformed model; it is loaded into the rebuilt technique instead. The
first build is kept although it is discarded: the compatibility checks and the loading of saved states need technique
instances, and they must pass before a transform rewrites the base weights, which cannot be undone.
Construction has no side effects and allocates only the techniques' own modules (draft LMs load in ``attach``).
"""

from __future__ import annotations

import sys
import warnings
from collections.abc import Mapping, Sequence
from typing import Any

import torch

from ..arch import ArchAdapter
from ..errors import CheckpointFormatError, IncompatibleTechniquesError, SimulatedPathWarning
from ..modules.sharding import model_tensor_parallel_group, replicate_from_first_rank, shard_state_like
from .config import TechniqueConfig
from .technique import AttachContext, Capability, Technique, attach_rank, technique_for

__all__ = ["attach_techniques", "build_techniques"]

States = Mapping[str, Mapping[str, torch.Tensor]]


def build_techniques(
    adapter: ArchAdapter, configs: Sequence[TechniqueConfig], states: States | None = None
) -> list[Technique]:
    """Construct the techniques of ``configs`` in attach order, check that they combine and load ``states``.

    Nothing is attached and the base model is not modified. On a model split by tensor parallelism the techniques'
    replicated parameters are made equal to those of the first process (a collective call, see
    :func:`~economize.modules.sharding.replicate_from_first_rank`).

    Raises:
        IncompatibleTechniquesError: A technique appears twice, or a technique's
            :meth:`~Technique.check_compatible` rejects the others (two techniques of one exclusive family such as
            drafting or dynamic depth, an excluded family, a missing required family).
        CheckpointFormatError: ``states`` do not match the techniques' parameters.
    """
    techniques = [technique_for(config)(config, adapter) for config in configs]
    names = [t.name for t in techniques]
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise IncompatibleTechniquesError(f"technique {duplicates[0]!r} is applied twice")
    techniques.sort(key=attach_rank)
    for technique in techniques:
        technique.check_compatible([t for t in techniques if t is not technique])
    if adapter.tensor_parallel_size > 1:
        group = model_tensor_parallel_group(adapter.model)
        if group is not None:
            for technique in techniques:
                replicate_from_first_rank(technique, group)
    if states is not None:
        _load_states(techniques, states)
    return techniques


def attach_techniques(
    model: torch.nn.Module,
    adapter: ArchAdapter,
    configs: Sequence[TechniqueConfig],
    *,
    states: States | None = None,
    calibration: Any = None,
    options: Mapping[str, Any] | None = None,
) -> list[Technique]:
    """Build the techniques of ``configs``, load ``states`` (before anything attaches) and attach them in order.

    Args:
        model: The base model (unwrapped).
        adapter: Its architecture adapter.
        configs: One configuration per technique, in any order.
        states: Saved state per technique name (from a checkpoint); every technique must have one.
        calibration: Calibration data for techniques that need it (see :class:`AttachContext`).
        options: Free-form options for third-party techniques.

    Returns:
        The attached techniques in attach order.

    A technique that attaches on a simulated path (:meth:`Technique.simulation`) gives a
    :class:`~economize.SimulatedPathWarning`.

    If an attach fails, the techniques attached so far are detached in reverse order before the error propagates.
    Weight transforms cannot be undone; the error then notes that the base model was (or, for the transform that
    failed, may have been) modified. A detach that fails during this roll-back is noted on the original error.
    """
    extra = dict(options or {})
    rewriting = any(Capability.WEIGHTS in technique_for(c).capabilities for c in configs)
    techniques = build_techniques(adapter, configs, None if rewriting else states)
    rewritten = _load_first_build(techniques, states) if rewriting and states is not None else {}
    rewrites = [t for t in techniques if Capability.WEIGHTS in t.capabilities]
    attached: list[Technique] = []
    attaching: Technique | None = None  # the technique inside ``attach`` (for the roll-back)
    try:
        if rewrites:
            context = AttachContext(tuple(techniques), calibration, extra)
            for attaching in rewrites:
                _attach(attaching, model, adapter, context)
                attached.append(attaching)
            attaching = None
            # rebuilt from the requested configs: a technique may resolve its config against the old depth
            rest = [c for c in configs if Capability.WEIGHTS not in technique_for(c).capabilities]
            after = _states_after(techniques, rewrites, states)
            rebuilt = build_techniques(adapter, rest, _subset(None if after is None else {**after, **rewritten}, rest))
            for technique in rebuilt:
                technique.check_compatible([t for t in (*rewrites, *rebuilt) if t is not technique])
            techniques = sorted([*rewrites, *rebuilt], key=attach_rank)
        context = AttachContext(tuple(techniques), calibration, extra)
        for attaching in techniques:
            if attaching in attached:
                continue
            _attach(attaching, model, adapter, context)
            attached.append(attaching)
    except BaseException as error:
        _roll_back(attached, attaching if attaching not in attached else None, error)
        raise
    return techniques


def _attach(technique: Technique, model: torch.nn.Module, adapter: ArchAdapter, context: AttachContext) -> None:
    technique.attach(model, adapter, context)
    simulation = technique.simulation()
    if simulation is not None:
        warnings.warn(
            f"{technique.name} runs on a simulated path: it {simulation}. Its outputs and its other savings are "
            "those of the technique",
            SimulatedPathWarning,
            stacklevel=2,
        )


def _load_states(techniques: Sequence[Technique], states: States) -> None:
    _check_known(techniques, states)
    for technique in techniques:
        _load_state(technique, states.get(technique.name, {}))


def _load_first_build(techniques: Sequence[Technique], states: States) -> dict[str, Mapping[str, torch.Tensor]]:
    """Load ``states`` into the first build of a recipe with weight transforms, except the states that were saved
    after a transform had reshaped the technique's copies of base layers (a checkpoint of a pruned model): those fit
    only the rebuilt technique and are returned by technique name."""
    _check_known(techniques, states)
    rewritten = {}
    for technique in techniques:
        state = states.get(technique.name, {})
        try:
            _load_state(technique, state)
        except CheckpointFormatError:
            if Capability.WEIGHTS in technique.capabilities or not technique.layer_copies():
                raise
            rewritten[technique.name] = state
    return rewritten


def _check_known(techniques: Sequence[Technique], states: States) -> None:
    unknown = set(states) - {t.name for t in techniques}
    if unknown:
        raise CheckpointFormatError(
            f"the checkpoint holds weights of techniques that are not applied: {sorted(unknown)}"
        )


def _load_state(technique: Technique, state: Mapping[str, torch.Tensor]) -> None:
    try:
        technique.load_state_dict(shard_state_like(dict(state), technique), strict=True)
    except RuntimeError as error:
        raise CheckpointFormatError(f"weights of technique {technique.name!r} do not match: {error}") from error


def _states_after(
    techniques: Sequence[Technique], rewrites: Sequence[Technique], states: States | None
) -> States | None:
    """The saved states as the weight transforms left them: a transform may have reshaped a technique's copies of
    base layers (:meth:`Technique.layer_copies`), so the first build's current state replaces the saved one."""
    if states is None:
        return None
    return {**states, **{t.name: t.state_dict() for t in techniques if t not in rewrites and t.name in states}}


def _subset(states: States | None, configs: Sequence[TechniqueConfig]) -> States | None:
    if states is None:
        return None
    names = {c.technique for c in configs}
    return {name: state for name, state in states.items() if name in names}


def _add_note(error: BaseException, note: str) -> None:
    """``error.add_note(note)``; ``BaseException.add_note`` is new in Python 3.11, so on 3.10 the note is appended to
    ``__notes__`` directly (the same attribute 3.11 fills)."""
    if sys.version_info >= (3, 11):
        error.add_note(note)
    else:
        error.__notes__ = [*getattr(error, "__notes__", []), note]  # type: ignore[attr-defined]


def _roll_back(attached: list[Technique], failed: Technique | None, error: BaseException) -> None:
    """Detach ``attached`` in reverse order and note on ``error`` what could not be undone.

    ``failed`` is the technique whose attach raised (it may have changed the model before raising).
    """
    for technique in reversed(attached):
        if Capability.WEIGHTS in technique.capabilities:
            continue
        try:
            technique.detach()
        except Exception as detach_error:  # keep the original error; report this one on it
            _add_note(error, f"economize: detaching {technique.name!r} during the roll-back failed: {detach_error!r}")
    modified = [t.name for t in attached if Capability.WEIGHTS in t.capabilities]
    if modified:
        _add_note(error, f"economize: {modified} already rewrote the base weights in place; reload the base model")
    if failed is not None and Capability.WEIGHTS in failed.capabilities:
        _add_note(
            error,
            f"economize: {failed.name!r} failed while rewriting the base weights and may have changed some of them; "
            "reload the base model"
        )
