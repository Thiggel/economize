"""Building blocks for on-policy RL with a drafter inside the policy.

A policy-gradient step needs the log-probabilities of the sampled tokens under the sampling distribution, with
gradients. :func:`policy_pass` (:mod:`economize.train.teacher`) runs the exact pass with gradients, and from its
outputs

* :func:`token_log_probs` gives the exact policy's log-probabilities and entropies, chunked over positions so that
  the ``[tokens, vocab]`` float32 buffers stay small;
* :func:`detach_outputs` turns the pass into a stop-gradient teacher, so that a drafter's distillation
  objective reuses it without a second full forward;
* :func:`acceptance` and :func:`expected_tokens` turn draft and target distributions into the acceptance
  ``sum_v min(p, q)`` of every draft step and the expected tokens per cycle of speculative sampling.

The pieces that run a drafter live with the drafter and build on :func:`token_log_probs`, :func:`cycle_starts` and
:func:`acceptance`.
"""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

import torch
from torch.utils.checkpoint import checkpoint

from ..runtime import KVCache
from .objective import TeacherOutputs
from .teacher import policy_pass

if TYPE_CHECKING:
    from collections.abc import Callable

__all__ = [
    "acceptance",
    "cycle_starts",
    "detach_outputs",
    "detached_cache",
    "expected_tokens",
    "policy_pass",
    "scored_tokens",
    "token_log_probs",
]


# ---- stop-gradient views ---------------------------------------------------------------------------------------------
def detached_cache(cache: KVCache) -> KVCache:
    """A cache of the same kind over the detached buffers of ``cache`` (no copy): :meth:`KVCache.detached`.

    The two kinds :func:`policy_pass` creates, :class:`~economize.runtime.cache.PositionalKVCache` and
    :class:`~economize.runtime.cache.HybridCache` (whose conv windows and recurrent states are detached too), support
    it.

    Raises:
        TypeError: ``cache`` does not declare :attr:`~economize.runtime.KVCache.copyable` (e.g. a compressing wrapper,
            whose buffers are not the cache's whole state).
    """
    return cache.detached()


def detach_outputs(outputs: TeacherOutputs) -> TeacherOutputs:
    """The outputs of a :func:`policy_pass` without their autograd history: a teacher for drafter objectives.

    The head is kept as it is; :meth:`TeacherOutputs.log_probs` evaluates it without gradients.
    """
    return replace(
        outputs,
        hidden={layer: h.detach() for layer, h in outputs.hidden.items()},
        final=outputs.final.detach(),
        cache=None if outputs.cache is None else detached_cache(outputs.cache),
        logits=None if outputs.logits is None else outputs.logits.detach(),
    )


# ---- log-probabilities -----------------------------------------------------------------------------------------------
def scored_tokens(logits: torch.Tensor, targets: torch.Tensor, temperature: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Log-probability of ``targets`` under ``logits / temperature`` and the distribution's entropy (float32)."""
    log_probs = (logits.float() / temperature).log_softmax(-1)
    chosen = log_probs.gather(-1, targets[..., None])[..., 0]
    entropy = -(log_probs.exp() * log_probs).sum(-1)
    return chosen, entropy


def _chunked(fn: Callable[[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]], states: torch.Tensor,
             targets: torch.Tensor, chunk: int) -> tuple[torch.Tensor, torch.Tensor]:
    """``fn`` over chunks of at most ``chunk`` positions along dim 1, each under activation checkpointing when
    gradients are on (the chunk's logits are recomputed in backward instead of kept)."""
    rows, length = states.shape[:2]
    size = length if chunk <= 0 else max(chunk // max(rows, 1), 1)
    parts = []
    for begin in range(0, length, size):
        s, t = states[:, begin:begin + size], targets[:, begin:begin + size]
        if torch.is_grad_enabled() and s.requires_grad:
            parts.append(checkpoint(fn, s, t, use_reentrant=False))
        else:
            parts.append(fn(s, t))
    return torch.cat([p[0] for p in parts], 1), torch.cat([p[1] for p in parts], 1)


def token_log_probs(outputs: TeacherOutputs, input_ids: torch.Tensor, *, temperature: float = 1.0,
                    chunk: int = 4096) -> tuple[torch.Tensor, torch.Tensor]:
    """Log-probability of every next token under the exact policy, and the policy's entropy there.

    Args:
        outputs: A :func:`policy_pass` (or :func:`~economize.train.run_teacher`) over ``input_ids``.
        input_ids: ``[B, T]`` tokens.
        temperature: Sampling temperature.
        chunk: Tokens per chunk of logits (``0``: all at once).

    Returns:
        ``(log_prob, entropy)``, both ``[B, T - 1]`` float32: entry ``t`` scores token ``t + 1`` given ``0..t``.
    """
    targets = input_ids[:, 1:].to(outputs.final.device)
    return _chunked(lambda h, y: scored_tokens(outputs.head(h), y, temperature), outputs.final[:, :-1], targets, chunk)


def cycle_starts(prompt_length: int, length: int, cycle: int, device: torch.device | str | None = None) -> torch.Tensor:
    """Positions of the exact passes' last tokens of lossy cycle decoding after a prompt of ``prompt_length``
    tokens, as far as a whole cycle fits into ``length`` positions: ``prompt_length - 1 + j * cycle``.

    Cycle ``j`` emits token ``s_j + 1`` from the exact logits at ``s_j`` and tokens ``s_j + 2 .. s_j + cycle`` from
    the ``cycle - 1`` jump steps at positions ``s_j + 1 .. s_j + cycle - 1``. Empty when no whole cycle fits."""
    first = prompt_length - 1
    return torch.arange(first, max(length - cycle, first), cycle, device=device)


# ---- acceptance ------------------------------------------------------------------------------------------------------
def acceptance(log_p: torch.Tensor, log_q: torch.Tensor) -> torch.Tensor:
    """Probability ``sum_v min(p_v, q_v)`` that speculative sampling accepts a draft from ``q`` for target ``p``
    (log-probabilities over the last dim)."""
    return torch.minimum(log_p, log_q).exp().sum(-1)


def expected_tokens(accept: torch.Tensor) -> torch.Tensor:
    """Expected tokens per cycle of speculative sampling with ``g = 1..K`` drafts, from the acceptance ``[K, ...]``
    of draft steps ``1..K`` (draft ``k`` is used only if drafts ``1..k-1`` were accepted):
    ``tau_g = 1 + sum_{k <= g} prod_{j <= k} a_j``. Returns ``[K, ...]`` (entry ``g - 1`` is ``tau_g``)."""
    return 1 + accept.cumprod(0).cumsum(0)
