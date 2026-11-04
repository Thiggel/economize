"""Pieces shared by the distillation objectives: the per-step term, the weighting over steps, chunked scoring, the
state losses and the lookup of the trained drafter.

A new state loss is added by registration::

    @STATE_LOSSES.register("l1")
    def l1(predicted, target):
        return (predicted.float() - target).abs().mean(-1)
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any, NamedTuple, Protocol

import torch
from torch.utils.checkpoint import checkpoint

from ...errors import ConfigError, IncompatibleTechniquesError
from ...registry import Registry
from ..objective import attached_technique
from .divergences import Divergence, masked_mean, token_nll

if TYPE_CHECKING:
    from ...core.model import EfficientModel
    from ...core.technique import Technique

__all__ = [
    "STATE_LOSSES",
    "DistillParts",
    "StateLoss",
    "check_loss_weights",
    "checkpointed_chunks",
    "decayed_mean",
    "distill_parts",
    "distill_term",
    "find_drafter",
    "normalized_mse",
]


class StateLoss(Protocol):
    def __call__(self, predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Per-token loss ``[...]`` between a predicted state ``[..., D]`` and the float32 exact state."""
        ...


#: Losses between a predicted and an exact hidden state, per token, by name (entry-point group
#: ``economize.state_losses``).
STATE_LOSSES: Registry[StateLoss] = Registry("state loss", "economize.state_losses")


@STATE_LOSSES.register("nmse")
def normalized_mse(predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Squared error normalized by the target's mean square."""
    return (predicted.float() - target).pow(2).mean(-1) / target.pow(2).mean(-1)


@STATE_LOSSES.register("cos")
def cosine_distance(predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """One minus the cosine similarity."""
    return 1 - torch.nn.functional.cosine_similarity(predicted.float(), target, dim=-1)


class DistillParts(NamedTuple):
    """One distillation term and its unweighted parts (``None``: the part is off)."""

    loss: torch.Tensor
    divergence: torch.Tensor | None
    cross_entropy: torch.Tensor | None

    def metrics(self, ce_name: str, divergence_name: str) -> dict[str, torch.Tensor]:
        """The parts that are on, detached, under the given metric names (cross-entropy first)."""
        named = ((ce_name, self.cross_entropy), (divergence_name, self.divergence))
        return {name: value.detach() for name, value in named if value is not None}


def check_loss_weights(ce_weight: float, kl_weight: float) -> None:
    """Both weights must be ``>= 0`` and not both ``0``.

    Raises:
        ConfigError: When they are not.
    """
    if ce_weight < 0 or kl_weight < 0 or ce_weight + kl_weight == 0:
        raise ConfigError(f"ce_weight and kl_weight must be >= 0 and not both 0, got {ce_weight}, {kl_weight}")


def distill_parts(
    student: torch.Tensor,
    weights: torch.Tensor | None,
    *,
    teacher: torch.Tensor | None = None,
    divergence: Divergence | None = None,
    targets: torch.Tensor | None = None,
    kl_weight: float = 1.0,
    ce_weight: float = 0.0,
) -> DistillParts:
    """``kl_weight * divergence + ce_weight * cross-entropy`` of the drafter's log-probabilities, each part a masked
    mean.

    Args:
        student: Drafter log-probabilities ``[..., V]``.
        weights: Mask of the counted positions ``[...]`` (``None``: all).
        teacher: Teacher log-probabilities ``[..., V]`` (``None``: no divergence part).
        divergence: Per-token divergence (an entry of :data:`DIVERGENCES`); needed with ``teacher``.
        targets: True tokens ``[...]``; needed when ``ce_weight`` is set.
        kl_weight: Weight of the divergence (``0``: none).
        ce_weight: Weight of the cross-entropy (``0``: none).
    """
    terms: list[torch.Tensor] = []
    div = ce = None
    if teacher is not None and kl_weight:
        if divergence is None:
            raise ConfigError("a divergence part needs a divergence")
        div = masked_mean(divergence(teacher, student), weights)
        terms.append(div if kl_weight == 1 else kl_weight * div)
    if ce_weight:
        if targets is None:
            raise ConfigError("a cross-entropy part needs the targets")
        ce = masked_mean(token_nll(student, targets), weights)
        terms.append(ce_weight * ce)
    loss = sum(terms[1:], terms[0]) if terms else student.new_zeros(())
    return DistillParts(loss, div, ce)


def distill_term(
    teacher: torch.Tensor,
    student: torch.Tensor,
    targets: torch.Tensor,
    divergence: Divergence,
    ce_weight: float,
    weights: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """``(divergence + ce_weight * cross-entropy, divergence)`` of one draft step, each a masked mean (see
    :func:`distill_parts`).

    Args:
        teacher: Teacher log-probabilities ``[..., V]``.
        student: Drafter log-probabilities ``[..., V]`` (float32).
        targets: True tokens ``[...]`` for the cross-entropy.
        divergence: Per-token divergence (an entry of :data:`DIVERGENCES`).
        ce_weight: Weight of the cross-entropy (``0``: none).
        weights: Mask of the counted positions ``[...]`` (``None``: all).
    """
    parts = distill_parts(student, weights, teacher=teacher, divergence=divergence, targets=targets,
                          ce_weight=ce_weight)
    assert parts.divergence is not None
    return parts.loss, parts.divergence


def decayed_mean(terms: Sequence[torch.Tensor], decay: float, exponents: Sequence[int] | None = None) -> torch.Tensor:
    """Weighted mean of per-step scalar ``terms``, term ``i`` weighted by ``decay ** exponents[i]`` (default
    ``decay ** i``, so that draft step ``k`` gets ``decay ** (k - 1)``)."""
    stacked = torch.stack(list(terms))
    powers = (torch.arange(len(stacked), dtype=torch.float32, device=stacked.device) if exponents is None
              else torch.tensor(list(exponents), dtype=torch.float32, device=stacked.device))
    weights = decay ** powers
    return (stacked * weights).sum() / weights.sum()


def checkpointed_chunks(
    fn: Callable[..., torch.Tensor],
    columns: Sequence[tuple[torch.Tensor | None, int]],
    *extra: object,
    rows: int,
    chunk: int,
) -> torch.Tensor:
    """Sum of ``fn(*column_chunks, *extra)`` over chunks of at most ``chunk`` tokens, each under activation
    checkpointing (the chunk's logits are recomputed in backward instead of kept).

    Args:
        fn: Returns a tensor that adds up over chunks (a sum, or a mean scaled by the chunk's share).
        columns: ``(tensor, dim)`` pairs split along ``dim``, which has the same length in all of them (``None``
            tensors are passed through).
        extra: Further arguments passed unchanged.
        rows: Tokens per column (a chunk spans ``max(chunk // rows, 1)`` columns).
        chunk: Maximum tokens per chunk (``0``: one chunk).
    """
    tensor, dim = next((t, d) for t, d in columns if t is not None)
    count = tensor.shape[dim]
    if not chunk or rows * count <= chunk:
        return checkpoint(fn, *(t for t, _ in columns), *extra, use_reentrant=False)
    size = max(chunk // rows, 1)
    total: torch.Tensor | None = None
    for begin in range(0, count, size):
        width = min(size, count - begin)
        parts = [None if t is None else t.narrow(d, begin, width) for t, d in columns]
        value = checkpoint(fn, *parts, *extra, use_reentrant=False)
        total = value if total is None else total + value
    assert total is not None
    return total


def find_drafter(
    model: EfficientModel, name: str | None, kind: type[Any], objective: str, requirement: str
) -> Technique:
    """The technique an objective trains: the one attached as ``name``, else the only attached ``kind``.

    Args:
        model: The efficient model.
        name: Attachment name (``None``: find the one technique that is a ``kind``).
        kind: A class or runtime-checkable protocol the technique must implement (it may be abstract).
        objective: The objective's name, for the error.
        requirement: What ``kind`` stands for, for the error (e.g. ``"train_request/train_logits"``).

    Raises:
        IncompatibleTechniquesError: ``name`` is not attached or no ``kind``; or, without ``name``, there is not
            exactly one ``kind``.
    """
    if name is not None:
        return attached_technique(
            model, name, kind, f"{objective} needs an attached technique {name!r} that implements {requirement}"
        )
    found = [t for t in model.techniques.values() if isinstance(t, kind)]
    if len(found) != 1:
        raise IncompatibleTechniquesError(
            f"{objective} needs exactly one attached technique that implements {requirement}, found "
            f"{[t.name for t in found]}"
        )
    return found[0]
