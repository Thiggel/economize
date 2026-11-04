"""Speculative speculative decoding (SSD) on two devices: the drafter prepares the next round while the target
verifies the current one.

Ordinary speculative decoding alternates: the drafter proposes ``k`` tokens, then the target verifies them, and
each side waits for the other. SSD (Kumar et al., 2026, "Speculative Speculative Decoding") overlaps the two. While
the target verifies the spine ``d_1..d_k``, the drafter guesses the verification outcome ``(a, y)`` (the target
keeps ``a`` drafts and appends ``y``) and drafts the next chain behind every guess. When the actual outcome is among
the guesses (a *hit*) the next round starts at once; otherwise (a *miss*) the drafter drafts the next chain after
the verification, as ordinary speculative decoding would.

The guesses for outcome ``a < k`` are the drafter's ``fanout`` likeliest tokens at depth ``a`` other than
``d_{a+1}`` (which the target rejected), and for ``a = k`` its likeliest tokens after ``d_k``. All
``(k + 1) * fanout`` next chains come from one tree pass per depth on the drafter's device, so the speculation costs
``k + 1`` drafter calls of width ``(k + 1) * fanout``.

The output is exactly that of the target alone: verification is economize's greedy verifier (temperature 0) or
speculative sampling (temperature above 0), and the drafts only decide how many tokens a round commits. Another
registered verifier can be named in :attr:`SSDConfig.verifier` (e.g. ``"typical"``, which is lossy). Both
devices receive their work from one host thread; the overlap comes from CUDA's asynchronous launches, so the target
and the drafter should sit on different GPUs. Time a generation with
``economize.bench.time_call(..., devices=[target_device, drafter_device])``: the default times only the current device
and misses the drafter's still-queued speculation.

Example::

    target = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-7B-Instruct", dtype=torch.bfloat16).to("cuda:0")
    draft = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct", dtype=torch.bfloat16).to("cuda:1")
    decoder = AsyncSpeculativeDecoder(target, draft, SSDConfig(draft_length=4, fanout=2))
    result = decoder.generate(prompt_ids, max_new_tokens=256)
    result.tokens, result.hit_rate, result.acceptance_length

Stop conditions and results are those of :class:`~economize.decoding.CycleDecoder`: ``generate`` takes a
:data:`~economize.decoding.StopCondition` and returns an :class:`SSDResult`, a :class:`~economize.decoding.DecodeResult`
with the SSD statistics added.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Literal

import torch
from torch import nn

from ..decoding.capabilities import selector_many
from ..decoding.config import DecodeConfig
from ..decoding.cost import CostLog
from ..decoding.protocols import Draft, TokenSelector, Verifier
from ..decoding.result import DecodeResult
from ..decoding.selectors import GreedySelector, TopPSampler
from ..decoding.stop import StopChecker, StopCondition
from ..decoding.verifiers import VERIFIERS
from ..errors import ConfigError
from ..validation import check_choice, check_positive
from .workers import Chooser, ModelWorker, Speculation, SpeculationWorker, Spine, TargetWorker

__all__ = ["AsyncSpeculativeDecoder", "SSDConfig", "SSDResult"]

Mode = Literal["async", "sync"]


@dataclass(frozen=True)
class SSDConfig:
    """Settings of :class:`AsyncSpeculativeDecoder`.

    Attributes:
        draft_length: Drafts per round ``k``.
        fanout: Guessed tokens per verification outcome (used by :class:`ModelWorker` built from a model).
        mode: ``"async"`` speculates on the outcome while the target verifies; ``"sync"`` is ordinary two-model
            speculative decoding with the same models (the reference SSD is measured against).
        temperature: ``0`` decodes greedily; above ``0`` samples losslessly with speculative sampling.
        top_p: Nucleus mass when sampling.
        seed: Seed of the sampling generators. Every :meth:`~AsyncSpeculativeDecoder.generate` call starts them
            afresh from it, so repeated calls on one prompt return the same sample.
        verifier: Name of a registered verifier (see :data:`~economize.decoding.VERIFIERS`); ``None`` picks
            ``"greedy"`` at temperature 0 and ``"rejection"`` (speculative sampling) above it.
        verifier_options: Options of the verifier, as in :attr:`DecodeConfig.verifier_options`.
    """

    draft_length: int = 4
    fanout: int = 2
    mode: Mode = "async"
    temperature: float = 0.0
    top_p: float = 1.0
    seed: int = 0
    verifier: str | None = None
    verifier_options: Mapping[str, Any] = field(default_factory=dict, hash=False)

    def __post_init__(self) -> None:
        check_positive("draft_length", self.draft_length)
        check_positive("fanout", self.fanout)
        check_choice("mode", self.mode, ("async", "sync"))
        if self.temperature < 0 or not 0 < self.top_p <= 1:
            raise ConfigError(f"need temperature >= 0 and 0 < top_p <= 1, got {self.temperature}, {self.top_p}")

    @property
    def sampling(self) -> bool:
        return self.temperature > 0

    @property
    def verifier_name(self) -> str:
        return self.verifier or ("rejection" if self.sampling else "greedy")


@dataclass
class SSDResult(DecodeResult):
    """One generation: a :class:`~economize.decoding.DecodeResult` (``acceptance`` holds ``(k, accepted)`` per round,
    ``cost`` the target's calls as ``"exact"`` events in target layers) plus the drafter's side.

    Attributes:
        hits: Rounds whose outcome the drafter had speculated (their next chain was ready).
        draft_cost: The drafter's calls (``"draft"`` events, in drafter layers).
    """

    hits: int = 0
    draft_cost: CostLog = field(default_factory=CostLog)

    @property
    def accepted(self) -> list[int]:
        """Drafts the target kept in every round."""
        return [accepted for _, accepted in self.acceptance]

    @property
    def rounds(self) -> int:
        """Verification passes after the prompt."""
        return len(self.acceptance)

    @property
    def hit_rate(self) -> float:
        """Share of rounds after which the next chain was already drafted."""
        return self.hits / max(1, self.rounds)

    @property
    def acceptance_length(self) -> float:
        """Tokens committed per verification pass (accepted drafts plus the target's own token)."""
        return 1 + self.mean_accepted

    def summary(self) -> dict[str, Any]:
        """The statistics above and the target's cost summary."""
        return {
            "generated": self.tokens.numel(),
            "rounds": self.rounds,
            "acceptance_length": self.acceptance_length,
            "hit_rate": self.hit_rate,
            "draft_calls": len(self.draft_cost.events),
            **self.cost_summary(),
        }


class AsyncSpeculativeDecoder:
    """Lossless speculative decoding with a separate draft model that drafts ahead on its own device.

    Args:
        target: The model whose output is reproduced, on its device.
        drafter: A draft model on another device (wrapped in a :class:`ModelWorker`), or any
            :class:`SpeculationWorker`.
        config: Decoding settings.

    Batch size is one: a round commits a different number of tokens per row, and SSD's gain is latency.

    Raises:
        ConfigError: The verifier needs greedy decoding but ``temperature > 0``.
    """

    def __init__(self, target: nn.Module, drafter: nn.Module | SpeculationWorker,
                 config: SSDConfig | None = None) -> None:
        self.config = c = config or SSDConfig()
        self.drafter = drafter if isinstance(drafter, SpeculationWorker) else ModelWorker(drafter, c.fanout)
        self.target = self._target_worker(target)
        name = c.verifier_name
        self.verifier: Verifier = VERIFIERS.get(name)(
            DecodeConfig(verifier=name, verifier_options=c.verifier_options))
        if self.verifier.requires_greedy and c.sampling:
            raise ConfigError(f"verifier {name!r} is exact only for greedy decoding; use temperature=0 or another "
                              "verifier")

    def _target_worker(self, target: nn.Module) -> TargetWorker:
        """The worker that verifies on ``target`` (built after :attr:`drafter`, so a subclass can fit it to the
        drafter, as :class:`~economize.distributed.linked.LinkedSpeculativeDecoder` does)."""
        return TargetWorker(target)

    # ---- token choice --------------------------------------------------------------------------------------------
    def _selectors(self) -> tuple[TokenSelector, TokenSelector]:
        """The target's and the drafter's selectors, with fresh generators (see :attr:`SSDConfig.seed`)."""
        c = self.config
        if not c.sampling:
            return GreedySelector(), GreedySelector()
        return TopPSampler(c.temperature, c.top_p, c.seed), TopPSampler(c.temperature, c.top_p, c.seed + 1)

    def _chooser(self, selector: TokenSelector) -> Chooser:
        verifier = self.verifier

        def choose(logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
            drawn = verifier.draft_many(logits, selector)
            assert drawn is not None  # both built-in selectors choose for a whole batch
            return drawn

        return choose

    # ---- generation ----------------------------------------------------------------------------------------------
    @torch.no_grad()
    def generate(self, prompt: torch.Tensor, max_new_tokens: int, stop_tokens: StopCondition = None) -> SSDResult:
        """Generate up to ``max_new_tokens`` tokens after ``prompt`` (``[P]`` or ``[1, P]`` token ids).

        Generation also ends after the first token that meets ``stop_tokens`` (a stop token id, a collection of
        them, or a predicate on all tokens so far; the token is kept), as in :class:`~economize.decoding.CycleDecoder`.
        """
        prompt = prompt.reshape(-1)
        if prompt.numel() == 0:
            raise ConfigError("the prompt must hold at least one token")
        k, length = self.config.draft_length, prompt.numel()
        target_selector, draft_selector = self._selectors()
        choose = self._chooser(draft_selector)
        target, drafter, stop = self.target, self.drafter, StopChecker(stop_tokens)
        total = length + max_new_tokens + k + 1
        history = torch.zeros(total, dtype=torch.long, device=target.device)  # committed tokens, then drafts
        history[:length] = prompt.to(target.device)
        generated: list[int] = []

        def commit(tokens: list[int], position: int) -> bool:
            """Commit ``tokens`` from ``position`` on; return whether the generation ends (it keeps the tokens up
            to the stop or the budget)."""
            history[position : position + len(tokens)] = torch.tensor(tokens, device=target.device)
            for i, token in enumerate(tokens):
                generated.append(token)
                if len(generated) >= max_new_tokens or stop(token, history, position + i):
                    return True
            return False

        last = target.begin(prompt, total + 1)
        drafter.begin(prompt, total + drafter.capacity_overhead(k))
        root = selector_many(target_selector, last[None])
        assert root is not None
        result = SSDResult(torch.empty(0, dtype=torch.long), target.cost, prompt_length=length,
                           num_layers=target.num_layers)
        finished = commit([int(root[0])], length)
        n = length + 1  # committed tokens; the root sits at position n - 1
        spine = None if finished else drafter.draft(root, n - 1, k, choose)

        while not finished:
            assert spine is not None
            drafts = spine.tokens.to(target.device)
            history[n : n + k] = drafts  # the verifier reads the drafts at their positions; commits overwrite them
            logits = target.verify(history[n - 1 : n + k], n - 1)
            draft = Draft(drafts[None], [k], probs=None if spine.probs is None else spine.probs[None])
            accepted_t, next_t = self.verifier.verify_batch(logits[None], draft, target_selector, history[None], [n])
            speculation: Speculation | None = None
            if self.config.mode == "async":
                speculation = drafter.speculate(spine, n, choose)  # queued on the drafter's device
            outcome = torch.cat([accepted_t, next_t, drafts]).tolist()  # waits for the target only
            accepted, token = outcome[0], outcome[1]
            committed = [*outcome[2 : 2 + accepted], token]
            result.acceptance.append((k, accepted))
            finished = commit(committed, n)
            n += accepted + 1
            following: Spine | None = None
            if speculation is not None:
                following = drafter.adopt(speculation, accepted, token)
                result.hits += following is not None
            if finished:
                break
            if following is None:  # a miss: draft after y, and after d_k too when all drafts were kept
                tail = committed[-2:] if accepted == k else [token]
                context = torch.tensor(tail, device=drafter.device)
                following = drafter.draft(context, n - len(tail), k, choose)
            spine = following

        result.tokens = torch.tensor(generated[: max(max_new_tokens, 0)], dtype=torch.long)
        result.cost, result.draft_cost = target.cost, drafter.cost
        return result
