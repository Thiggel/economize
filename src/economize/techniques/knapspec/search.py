"""The search of KnapSpec: a knapsack dynamic program over sublayers and skipped latency, then the draft length.

Every sublayer ``i`` (attention or MLP of a layer) has an integer latency weight ``w_i = round(t_i / Delta)`` with
``Delta = min(t_attn(n), t_mlp)``. The program keeps, for every skipped weight ``j``, one candidate draft path: its
hidden states on the recent tokens after sublayer ``i``. Executing sublayer ``i`` maps ``g[i - 1, j]`` to
``g[i, j]``; skipping it moves ``g[i - 1, j - w_i]`` to ``g[i, j]`` unchanged. Of the two, the candidate with the
higher cosine similarity to the full model's state ``X(i) = g[i, 0]`` survives; candidates below
``min_similarity`` are dropped, and ``j`` stays within ``max_skip_fraction`` of the model's total weight ``K``.
All candidates of a sublayer run as one batched layer call (the budgets are processed in parallel), and one host
synchronization per sublayer reads their similarities.

After the last sublayer, every surviving budget ``j`` is a skip set with an acceptance rate ``alpha_j`` (the share
of recent tokens whose greedy token it agrees on with the full model). The search returns the set and draft length
``gamma <= max_draft`` that maximize the modelled tokens per unit time
``sum_{k=0..gamma} alpha^k / (gamma * t_draft + t_target)``, with ``t_draft = (K - j) Delta`` and
``t_target = L (t_attn(n) + t_mlp)``.

Approximation: a candidate's attention reads the exact K/V of the recent tokens from the cache (its own K/V are not
written), so every candidate attends over the full model's keys and values of earlier positions. The paper does not
say how the candidates' attention gets its keys; recomputing them per candidate would run the whole context.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

from ...runtime.context import ContextLayout, LayerContext
from .latency import SublayerLatency
from .sublayer import SublayerSkipper

__all__ = ["KnapsackSearch", "SearchStats", "SearchTokens", "SkipPlan"]

_TOKEN_CHUNK = 4096  # tokens per batched candidate call (copies of the recent tokens)


@dataclass(frozen=True)
class SkipPlan:
    """A skip set and draft length chosen by the search.

    Attributes:
        attention: Layers (1-indexed) whose attention a draft step skips.
        mlp: Layers whose MLP a draft step skips (a layer in both is skipped whole).
        draft_length: Draft tokens per cycle (``gamma``).
        acceptance: Estimated acceptance rate of a draft token (``alpha``).
        skipped_weight: Skipped latency weight ``j`` (in units of ``Delta``).
        speed: Modelled tokens per second of the choice.
        similarity: Mean cosine similarity of the draft path's last state to the full model's.
    """

    attention: frozenset[int] = frozenset()
    mlp: frozenset[int] = frozenset()
    draft_length: int = 1
    acceptance: float = 1.0
    skipped_weight: int = 0
    speed: float = 0.0
    similarity: float = 1.0

    @property
    def key(self) -> tuple[tuple[int, ...], tuple[int, ...]]:
        """What identifies the draft step of this plan (its CUDA graph)."""
        return tuple(sorted(self.attention)), tuple(sorted(self.mlp))

    def run_layers(self, num_layers: int) -> tuple[int, ...]:
        """Layers a draft step runs at least one sublayer of."""
        whole = self.attention & self.mlp
        return tuple(i for i in range(1, num_layers + 1) if i not in whole)


@dataclass
class SearchTokens:
    """The recent tokens a search runs on: their layer inputs and where they sit.

    Attributes:
        hidden: Embeddings (input of layer 1) ``[B, R, d]``.
        positions: Positions ``[B, R]`` (``0`` where invalid).
        valid: Which entries are real tokens ``[B, R]``.
        token_inputs: Per-token inputs of the family ``[B, R, ...]`` (Gemma-4's per-layer embeddings).
    """

    hidden: torch.Tensor
    positions: torch.Tensor
    valid: torch.Tensor
    token_inputs: dict[str, torch.Tensor] = field(default_factory=dict)


@dataclass(frozen=True)
class SearchStats:
    """What a search computed, for the cost log.

    Attributes:
        layer_tokens: Token-sublayer evaluations divided by two (layer-equivalents), summed over the batch.
        head_tokens: Tokens read by the LM head, summed over the batch.
        position: Mean position of the recent tokens.
    """

    layer_tokens: int
    head_tokens: int
    position: int


class KnapsackSearch:
    """KnapSpec's dynamic program on one model (see the module docstring).

    Args:
        skipper: Partial-layer calls and skippable sublayers of the model.
        candidates: Layers whose sublayers may be skipped.
        min_similarity: Pruning threshold ``tau`` on the mean cosine similarity.
        max_skip_fraction: Largest skipped share of the total weight ``K``.
        max_draft: Largest draft length ``D``.
    """

    def __init__(
        self,
        skipper: SublayerSkipper,
        candidates: tuple[int, ...],
        min_similarity: float,
        max_skip_fraction: float,
        max_draft: int,
    ) -> None:
        self.skipper = skipper
        self.adapter = skipper.adapter
        allowed = set(candidates)
        self.searchable = tuple(flag and (s // 2 + 1) in allowed for s, flag in enumerate(skipper.skippable))
        self.min_similarity = min_similarity
        self.max_skip_fraction = max_skip_fraction
        self.max_draft = max_draft

    def weights(self, latency: SublayerLatency, length: float) -> tuple[int, int, float]:
        """Integer weights of an attention and an MLP sublayer at context length ``length``, and ``Delta``."""
        attention, mlp = latency.attention(length), latency.mlp
        delta = min(attention, mlp)
        return max(1, round(attention / delta)), max(1, round(mlp / delta)), delta

    @torch.no_grad()
    def __call__(self, runner, cache, tokens: SearchTokens, latency: SublayerLatency) -> tuple[SkipPlan, SearchStats]:
        """The best plan on ``tokens``, whose K/V the cache holds (``runner``: the model's layer runner)."""
        L = self.adapter.num_layers
        valid = tokens.valid
        count = valid.sum()
        weight = (valid.float() / count.clamp(min=1)).to(tokens.hidden.device)
        mean_position = float((tokens.positions.float() * valid).sum() / count.clamp(min=1))
        if float(count) == 0:
            return SkipPlan(), SearchStats(0, 0, 0)
        w_attention, w_mlp, delta = self.weights(latency, mean_position + 1)
        total = L * (w_attention + w_mlp)
        budget = math.floor(total * self.max_skip_fraction)
        contexts: dict[int, LayerContext] = {}
        states: dict[int, torch.Tensor] = {0: tokens.hidden}
        choices: list[dict[int, bool]] = []  # per sublayer: budget -> skipped
        similarity: dict[int, float] = {0: 1.0}
        layer_tokens = 0
        B, R = tokens.hidden.shape[:2]
        for s in range(2 * L):
            index, w = s // 2 + 1, w_attention if s % 2 == 0 else w_mlp
            live = sorted(states)
            executed = self._execute(runner, cache, tokens, contexts, index, self.skipper.sublayer_call(s),
                                     [states[j] for j in live])
            layer_tokens += B * R * _padded(len(live), R)
            target = executed[0]
            moves = self.searchable[s]
            stacked = torch.stack(executed + ([states[j] for j in live] if moves else []))
            cos = (F.cosine_similarity(stacked.float(), target.float()[None], dim=-1) * weight).sum((1, 2)).tolist()
            cos_run = dict(zip(live, cos[:len(live)]))
            cos_keep = dict(zip(live, cos[len(live):])) if moves else {}
            new_states: dict[int, torch.Tensor] = {}
            choice: dict[int, bool] = {}
            reachable = set(live) | ({j + w for j in live if j + w <= budget} if moves else set())
            for j in sorted(reachable):
                run = cos_run.get(j, -math.inf)
                keep = cos_keep.get(j - w, -math.inf)
                best = max(run, keep)
                if j > 0 and best < self.min_similarity:
                    continue
                skipped = keep > run
                new_states[j] = states[j - w] if skipped else executed[live.index(j)]
                choice[j] = skipped
                similarity[j] = best
            states = new_states
            choices.append(choice)
        acceptance, head_tokens = self._acceptance(tokens, states)
        plan = self._choose(choices, states, acceptance, similarity, total, delta, latency, mean_position + 1,
                            w_attention, w_mlp)
        return plan, SearchStats(layer_tokens // 2, head_tokens, int(mean_position))

    # ---- pieces --------------------------------------------------------------------------------------------------
    def _context(self, runner, cache, tokens: SearchTokens, contexts: dict[int, LayerContext], copies: int
                 ) -> LayerContext:
        """The context of ``copies`` copies of the recent tokens (memoized per count: their masks are reused)."""
        if copies not in contexts:
            positions = tokens.positions.repeat(1, copies)
            valid = tokens.valid.repeat(1, copies)
            layout = ContextLayout(positions, torch.full_like(positions, cache.trash_slot), valid)
            inputs = {name: t.repeat(1, copies, *([1] * (t.dim() - 2))) for name, t in tokens.token_inputs.items()}
            contexts[copies] = runner.context_from_layout(cache, layout, inputs)
        return contexts[copies]

    def _execute(self, runner, cache, tokens: SearchTokens, contexts: dict[int, LayerContext], index: int, call,
                 inputs: list[torch.Tensor]) -> list[torch.Tensor]:
        """``call`` (layer ``index`` with one sublayer) on every state in ``inputs``, batched along the sequence."""
        R = tokens.hidden.shape[1]
        per_call = max(1, _TOKEN_CHUNK // max(R, 1))
        outputs: list[torch.Tensor] = []
        for start in range(0, len(inputs), per_call):
            chunk = inputs[start:start + per_call]
            copies = _padded(len(chunk), R)
            padded = chunk + [chunk[-1]] * (copies - len(chunk))
            ctx = self._context(runner, cache, tokens, contexts, copies)
            out, _ = runner.run(torch.cat(padded, 1), ctx, index, index, overrides={index: call})
            outputs.extend(out.split(R, 1)[:len(chunk)])
        return outputs

    def _acceptance(self, tokens: SearchTokens, states: Mapping[int, torch.Tensor]) -> tuple[dict[int, float], int]:
        """Share of the valid recent tokens on which each candidate's greedy token is the full model's."""
        valid = tokens.valid
        count = max(int(valid.sum()), 1)
        reference = self.adapter.logits(states[0]).argmax(-1)
        agree = {0: 1.0}
        others = [j for j in states if j != 0]
        values = []
        for j in others:
            same = (self.adapter.logits(states[j]).argmax(-1) == reference) & valid
            values.append(same.sum())
        if values:
            agree.update(zip(others, (torch.stack(values).float() / count).tolist()))
        return agree, tokens.hidden.shape[0] * tokens.hidden.shape[1] * len(states)

    def _choose(self, choices: list[dict[int, bool]], states: Mapping[int, torch.Tensor], acceptance: dict[int, float],
                similarity: dict[int, float], total: int, delta: float, latency: SublayerLatency, length: float,
                w_attention: int, w_mlp: int) -> SkipPlan:
        """The (budget, draft length) pair with the highest modelled tokens per second, as a plan."""
        L = self.adapter.num_layers
        t_target = L * (latency.attention(length) + latency.mlp)
        best: tuple[float, int, int] | None = None
        for j in states:
            alpha = acceptance[j]
            t_draft = (total - j) * delta
            for gamma in range(1, self.max_draft + 1):
                speed = sum(alpha ** k for k in range(gamma + 1)) / (gamma * t_draft + t_target)
                if best is None or speed > best[0] + 1e-12 * abs(best[0]):
                    best = (speed, j, gamma)
        assert best is not None
        speed, j, gamma = best
        attention: set[int] = set()
        mlp: set[int] = set()
        for s in range(len(choices) - 1, -1, -1):
            if choices[s].get(j, False):
                (attention if s % 2 == 0 else mlp).add(s // 2 + 1)
                j -= w_attention if s % 2 == 0 else w_mlp
        return SkipPlan(frozenset(attention), frozenset(mlp), gamma, acceptance[best[1]], best[1], speed,
                        similarity.get(best[1], 1.0))


def _padded(count: int, length: int) -> int:
    """Copies per batched call for ``count`` states: the next power of two (bounded by the chunk size), so a search
    builds the masks of a few copy counts only."""
    per_call = max(1, _TOKEN_CHUNK // max(length, 1))
    count = min(count, per_call)
    return min(1 << (count - 1).bit_length(), per_call)
