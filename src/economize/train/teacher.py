"""The teacher pass: one exact forward of the target over a batch, keeping what an objective asked for."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from typing import TYPE_CHECKING, Protocol, runtime_checkable

import torch
from torch import nn

from .objective import Batch, TeacherOutputs, TeacherRequest

if TYPE_CHECKING:
    from ..core.model import EfficientModel

__all__ = [
    "LargerTeacher",
    "StandaloneTeacherHead",
    "TeacherHead",
    "larger_teacher",
    "make_teacher_head",
    "policy_pass",
    "run_teacher",
    "teacher_pass",
]

#: Replaces the teacher's last-block states and head after the teacher pass of the trained model, given the batch
#: and that pass's outputs (see :func:`larger_teacher`).
TeacherHead = Callable[[Batch, TeacherOutputs], TeacherOutputs]


@runtime_checkable
class StandaloneTeacherHead(Protocol):
    """A :data:`TeacherHead` that can also produce its outputs without the trained model's pass, for requests with
    ``own_pass=False`` (:class:`LargerTeacher` is one)."""

    def __call__(self, batch: Batch, outputs: TeacherOutputs) -> TeacherOutputs: ...

    def standalone(self, batch: Batch, request: TeacherRequest) -> TeacherOutputs:
        """The teacher's outputs for ``batch`` with empty ``hidden`` states and no cache."""
        ...


@torch.no_grad()
def run_teacher(model: EfficientModel, batch: Batch, request: TeacherRequest) -> TeacherOutputs:
    """Run the unmodified model over ``batch`` and keep the states, cache and logits listed in ``request``.

    The pass goes through :class:`~economize.runtime.LayerRunner` with the model's own layers (no technique is active),
    so its states are the reference that drafters are trained to reproduce. It runs in the model's own dtypes, also
    inside a caller's autocast region. Rows may be right-padded
    (``batch.attention_mask``); padding tokens write to the cache's trash slot and no real token attends to them.

    Args:
        model: The model; its base weights are the teacher.
        batch: Token ids ``[B, T]``.
        request: What to keep (see :class:`TeacherRequest`).
    """
    with torch.autocast(model.adapter.device.type, enabled=False):
        return _exact_pass(model, batch, request)


def policy_pass(model: EfficientModel, batch: Batch, request: TeacherRequest) -> TeacherOutputs:
    """:func:`run_teacher` with gradients: the same exact pass, but its states, cache and logits depend on the base
    weights, so that a loss on them (a policy gradient, or a drafter rollout that starts from them) trains the
    model. Used by on-policy RL (:mod:`economize.train.rl`); :func:`economize.train.rl.detach_outputs` turns the result
    into a teacher for drafter objectives."""
    with torch.enable_grad():
        return _exact_pass(model, batch, request)


def teacher_pass(model: EfficientModel, batch: Batch, request: TeacherRequest,
                 teacher_head: TeacherHead | None = None) -> TeacherOutputs:
    """The teacher outputs an objective trains or evaluates against: :func:`run_teacher` of ``model``, followed by
    ``teacher_head`` when one is given. A request with ``own_pass=False`` and a :class:`StandaloneTeacherHead` skip
    the trained model's pass, whose outputs the head would replace anyway."""
    if teacher_head is None:
        return run_teacher(model, batch, request)
    if not request.own_pass and isinstance(teacher_head, StandaloneTeacherHead):
        return teacher_head.standalone(batch, request)
    return teacher_head(batch, run_teacher(model, batch, request))


def _exact_pass(model: EfficientModel, batch: Batch, request: TeacherRequest) -> TeacherOutputs:
    runner, adapter = model.runner, model.adapter
    ids = batch.input_ids.to(adapter.device)
    rows, length = ids.shape
    cache = runner.new_cache(length, rows)
    valid = None if batch.attention_mask is None else batch.attention_mask.to(ids.device).bool()
    ctx = runner.context(cache, torch.arange(length, device=ids.device), valid, trusted=True)
    last = adapter.num_layers
    exact = runner.forward(ids, ctx, capture=set(request.hidden_layers) - {last})
    hidden = dict(exact.hidden)
    if last in request.hidden_layers:
        hidden[last] = exact.final
    logits = runner.logits(exact.final) if request.logits else None
    return TeacherOutputs(hidden, exact.final, runner.logits, cache if request.cache else None, logits)


class LargerTeacher:
    """Distill from another model with the same tokenizer (DistillSpec-style); see :func:`larger_teacher`.

    Called with the outputs of the trained model's teacher pass, it runs ``teacher_model`` over the batch and
    replaces the last-block states and the head, so that :meth:`TeacherOutputs.log_probs` gives the other model's
    distributions; the intermediate states and the cache stay those of the trained model (rollouts start from its
    own states). Only ``final``, ``head`` and ``logits`` change. Every entry of ``hidden``, including
    ``hidden[num_layers]`` (the trained model's own last-block output, which now differs from ``final``), stays the
    trained model's, so state losses always target the states of the model the drafter runs on.

    :meth:`standalone` runs the other model alone, for objectives that read only its distributions (requests with
    ``own_pass=False``, such as ``logit_distill``).

    Args:
        teacher_model: An :class:`~economize.EfficientModel` or a ``transformers`` causal LM, on any device.
        vocab_size: Truncate the teacher's logits to this many entries (models of one family often pad their
            embedding matrices to different sizes). ``None``: no truncation.
    """

    def __init__(self, teacher_model: nn.Module, vocab_size: int | None = None) -> None:
        from ..core.model import EfficientModel, apply

        self.teacher = teacher_model if isinstance(teacher_model, EfficientModel) else apply(teacher_model)
        self.vocab_size = vocab_size
        self._request = TeacherRequest(hidden_layers=(self.teacher.adapter.num_layers,))

    def _head(self, final: torch.Tensor, device: torch.device) -> Callable[[torch.Tensor], torch.Tensor]:
        head, limit = self.teacher.runner.logits, self.vocab_size

        def logits(hidden: torch.Tensor) -> torch.Tensor:
            out = head(hidden.to(final.device)).to(device)
            return out if limit is None else out[..., :limit]

        return logits

    def __call__(self, batch: Batch, outputs: TeacherOutputs) -> TeacherOutputs:
        device = outputs.final.device
        big = run_teacher(self.teacher, batch, self._request)
        logits = self._head(big.final, device)
        full = None if outputs.logits is None else logits(big.final)
        return replace(outputs, final=big.final.to(device), head=logits, logits=full)

    def standalone(self, batch: Batch, request: TeacherRequest) -> TeacherOutputs:
        big = run_teacher(self.teacher, batch, self._request)
        logits = self._head(big.final, big.final.device)
        return TeacherOutputs({}, big.final, logits, None, logits(big.final) if request.logits else None)


def larger_teacher(teacher_model: nn.Module, vocab_size: int | None = None) -> LargerTeacher:
    """Distill from another model with the same tokenizer (DistillSpec-style): the :class:`LargerTeacher` of
    ``teacher_model``, a :data:`TeacherHead` that replaces the distributions of the trained model's teacher pass
    with those of ``teacher_model`` (truncated to ``vocab_size`` entries when given)."""
    return LargerTeacher(teacher_model, vocab_size)


def make_teacher_head(model: EfficientModel, teacher: nn.Module | TeacherHead | None) -> TeacherHead | None:
    """The teacher head the loops use for their ``teacher_model`` argument: ``None`` stays ``None``, a model
    becomes :func:`larger_teacher` truncated to ``model``'s vocabulary, and a function without ``forward`` is taken
    as a ready head."""
    if teacher is None:
        return None
    if not isinstance(teacher, nn.Module) and callable(teacher):
        return teacher
    return larger_teacher(teacher, model.adapter.vocab_size)
