"""Log-likelihood of given continuations under the efficient model's own model calls.

A plain ``forward`` of an :class:`~economize.EfficientModel` is the base model's pass: weight transforms (pruning,
quantization, low rank) and activation hooks act on it, but the techniques that act inside the decoding loop do
not. :func:`score_continuations` scores a continuation the way the model would generate it instead. The context is
prefilled, then the continuation is teacher-forced one token per exact pass (ordinary decoding,
``cycle_length=1``), so every exact pass sees the same layer policy, KV cache and decoding phase as in generation:

=====================================  =========================================================================
Technique group (capability)           What scoring through decoding changes
=====================================  =========================================================================
dynamic depth (``DEPTH``)              The layer policy decides per token and layer, in the ``prefill`` phase for
                                       the context and the ``decode`` phase for every continuation token, exactly
                                       as while generating (``skip_prompt`` options included).
KV eviction / quantization             The context's K/V go through the cache wrapper after the prefill (evicted,
(``KV_CACHE``)                         quantized), and each continuation token attends to the wrapped cache.
activation sparsity (``ACTIVATIONS``)  The phase policy of the hooks applies per phase (``prefill`` / ``decode``)
                                       instead of the policy for calls outside decoding.
weight transforms (``WEIGHTS``)        Same as a plain forward (the weights are rewritten in place).
drafters (``DRAFTER``)                 Not used: scores are those of the exact passes. A lossless drafter does not
                                       change them; lossy drafting has no score of its own.
=====================================  =========================================================================

The cost is one model call per continuation token, so long rolling windows (perplexity tasks) are slow.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
from typing import Any

import torch

from ..core.technique import Capability
from ..errors import ConfigError

__all__ = ["DECODE_ONLY", "ForcedSelector", "needs_decode_scoring", "score_continuations"]

#: Capabilities whose techniques act only inside the decoding loop, so a plain forward does not see them.
DECODE_ONLY = Capability.DEPTH | Capability.KV_CACHE


def needs_decode_scoring(model: Any) -> bool:
    """Whether a technique of ``model`` acts only while decoding (see :data:`DECODE_ONLY`)."""
    techniques = getattr(model, "techniques", None) or {}
    return any(technique.capabilities & DECODE_ONLY for technique in techniques.values())


class ForcedSelector:
    """Token selector that commits a fixed continuation and records the log-probability of every token.

    Args:
        targets: The continuation's token ids.
        prompt_length: Length of the prompt it follows (``history`` holds the prompt, then the continuation).
    """

    greedy = True  # the choice is a deterministic function of the history
    generator: torch.Generator | None = None

    def __init__(self, targets: Sequence[int], prompt_length: int) -> None:
        self.targets = [int(token) for token in targets]
        self.prompt_length = prompt_length
        self._logprobs: dict[int, float] = {}
        self._argmax: dict[int, bool] = {}

    def __call__(self, logits: torch.Tensor, history: torch.Tensor) -> int:
        index = history.shape[0] - self.prompt_length
        target = self.targets[index]
        self._logprobs[index] = float(torch.log_softmax(logits.float(), -1)[target])
        self._argmax[index] = int(logits.argmax(-1)) == target
        return target

    def distribution(self, logits: torch.Tensor, history: torch.Tensor) -> torch.Tensor:
        """A one-hot distribution on the forced token."""
        probs = torch.zeros(logits.shape[-1], dtype=torch.float32, device=logits.device)
        probs[self(logits, history)] = 1.0
        return probs

    def result(self) -> tuple[float, bool]:
        """``(sum of log-probabilities, whether every token was the argmax)``, lm-eval's loglikelihood answer."""
        missing = [i for i in range(len(self.targets)) if i not in self._logprobs]
        if missing:
            raise RuntimeError(f"the decoding loop scored no logits for continuation positions {missing}")
        return sum(self._logprobs.values()), all(self._argmax.values())


@torch.no_grad()
def score_continuations(
    model: Any,
    prompts: Sequence[torch.Tensor],
    continuations: Sequence[Sequence[int]],
    *,
    decoding: Any = None,
) -> list[tuple[float, bool]]:
    """``(log-likelihood, is_greedy)`` of every continuation after its prompt, under the model's decoding loop.

    Args:
        model: An :class:`~economize.EfficientModel` (anything with ``decode_batch`` and ``decoding``).
        prompts: 1-D token tensors on the model's device (at least one token each).
        continuations: Token ids to score after each prompt (at least one each).
        decoding: Decoding options (default: the model's); the drafter is turned off (``cycle_length=1``).
    """
    if len(prompts) != len(continuations):
        raise ConfigError(f"{len(prompts)} prompts but {len(continuations)} continuations")
    if not prompts:
        return []
    config = replace(decoding or model.decoding, cycle_length=1)
    selectors = [ForcedSelector(cont, prompt.shape[-1]) for prompt, cont in zip(prompts, continuations)]
    model.decode_batch(list(prompts), [len(cont) for cont in continuations], decoding=config, selectors=selectors)
    return [selector.result() for selector in selectors]
