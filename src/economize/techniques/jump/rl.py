"""On-policy RL with a jump drafter inside the policy: the jump-specific pieces of :mod:`economize.train.rl`.

* :func:`hybrid_log_probs` gives the log-probabilities of the *hybrid* policy that lossy cycle decoding samples from
  (``DecodeConfig(cycle_length=N, lossless=False)``): the exact distribution at the first position of every cycle and
  the jump drafter's distribution at the ``N - 1`` positions after it, all from one teacher-forced pass plus one
  parallel rollout (:func:`~economize.techniques.jump.jump_rollout`);
* :func:`jump_acceptance` measures, teacher-forced on the policy's own samples, the acceptance ``sum_v min(p, q)``
  of every draft step (turned into expected tokens per cycle by :func:`economize.train.rl.expected_tokens`).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch.utils.checkpoint import checkpoint

from ...errors import ConfigError
from ...train.rl import acceptance, cycle_starts, scored_tokens, token_log_probs
from .rollout import jump_rollout

if TYPE_CHECKING:
    from ...core.model import EfficientModel
    from ...train.objective import TeacherOutputs
    from .technique import JumpTechnique

__all__ = ["hybrid_log_probs", "jump_acceptance"]


def hybrid_log_probs(
    model: EfficientModel,
    technique: JumpTechnique,
    input_ids: torch.Tensor,
    outputs: TeacherOutputs,
    prompt_length: int,
    cycle: int,
    *,
    temperature: float = 1.0,
    chunk: int = 4096,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Log-probability of every next token under the distribution lossy cycle decoding sampled it from.

    Decoding with ``DecodeConfig(cycle_length=cycle, lossless=False)`` after a prompt of ``prompt_length`` tokens
    draws token ``s + 1`` from the exact logits at every cycle start ``s`` (:func:`~economize.train.rl.cycle_starts`)
    and the next ``cycle - 1`` tokens from the jump drafter, which starts from the exact states at ``s``. A rollout
    of horizon ``cycle - 1`` from every cycle start reproduces those drafts in one parallel pass. All rows share the
    prompt length (one GRPO group). Gradients flow wherever ``outputs`` has them: into the exact pass (through the
    drafter's source states and the cached keys and values) and into every layer of the jump path, including the
    drafter.

    Args:
        model: The model the drafter is attached to.
        technique: The jump technique used for drafting.
        input_ids: ``[B, T]`` prompts followed by the sampled tokens (right-padded). Positions after the last
            whole cycle that fits into ``T`` get the exact distribution, so pad the batch by ``cycle`` tokens past
            its longest row.
        outputs: A :func:`~economize.train.policy_pass` over ``input_ids`` that kept the drafter's source states and
            the cache.
        prompt_length: Number of prompt tokens of every row.
        cycle: Cycle length ``N`` (``1``: the exact policy).
        temperature: Sampling temperature.
        chunk: Tokens per chunk of exact logits.

    Returns:
        ``(log_prob, entropy)``, both ``[B, T - 1]`` float32, entry ``t`` scoring token ``t + 1``.
    """
    log_prob, entropy = token_log_probs(outputs, input_ids, temperature=temperature, chunk=chunk)
    if cycle <= 1:
        return log_prob, entropy
    if outputs.cache is None:
        raise ConfigError("hybrid_log_probs needs the cache of the policy pass (TeacherRequest(cache=True))")
    ids = input_ids.to(outputs.final.device)
    starts = cycle_starts(prompt_length, ids.shape[1], cycle, ids.device)
    if len(starts) == 0:
        return log_prob, entropy

    def scored(h_r_hat: torch.Tensor, targets: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return scored_tokens(technique.logits(h_r_hat), targets, temperature)

    def step(k: int, positions: torch.Tensor, h_m_hat: torch.Tensor, h_r_hat: torch.Tensor):
        targets = ids[:, positions + 1]
        if torch.is_grad_enabled() and h_r_hat.requires_grad:
            return positions, *checkpoint(scored, h_r_hat, targets, use_reentrant=False)
        return positions, *scored(h_r_hat, targets)

    steps = jump_rollout(model, technique, ids, dict(outputs.hidden), outputs.cache, starts, cycle - 1, step)
    positions = torch.cat([p for p, _, _ in steps])
    log_prob = log_prob.index_copy(1, positions, torch.cat([lp for _, lp, _ in steps], 1))
    entropy = entropy.index_copy(1, positions, torch.cat([e for _, _, e in steps], 1))
    return log_prob, entropy


@torch.no_grad()
def jump_acceptance(
    model: EfficientModel,
    technique: JumpTechnique,
    input_ids: torch.Tensor,
    outputs: TeacherOutputs,
    starts: torch.Tensor,
    horizon: int,
    *,
    temperature: float = 1.0,
) -> torch.Tensor:
    """Teacher-forced acceptance of the jump drafter's steps ``1..horizon`` from every start, ``[K, B, S]``.

    Step ``k`` from start ``s`` drafts token ``s + k + 1``; its acceptance is ``sum_v min(p, q)`` between the
    exact distribution ``p`` at ``s + k`` (from ``outputs``) and the drafter's ``q`` (as in E13a). ``outputs`` must
    hold the drafter's source states and the cache (detached or not, nothing is differentiated)."""
    ids = input_ids.to(outputs.final.device)

    def step(k: int, positions: torch.Tensor, h_m_hat: torch.Tensor, h_r_hat: torch.Tensor) -> torch.Tensor:
        log_p = (outputs.head(outputs.final[:, positions]).float() / temperature).log_softmax(-1)
        log_q = (technique.logits(h_r_hat).float() / temperature).log_softmax(-1)
        return acceptance(log_p, log_q)

    cache = outputs.cache
    if cache is None:
        raise ConfigError("jump_acceptance needs the cache of the exact pass")
    hidden = {layer: h.detach() for layer, h in outputs.hidden.items()}
    return torch.stack(jump_rollout(model, technique, ids, hidden, cache, starts, horizon, step))
