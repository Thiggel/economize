"""Offline search of the layers a self-drafter skips, scored on calibration data.

A search is registered in :data:`SKIP_SEARCHES` (open for extension)::

    from economize.techniques.layer_skip import SKIP_SEARCHES, SkipSearch

    @SKIP_SEARCHES.register("my_search")
    class MySearch(SkipSearch):
        def __call__(self, scorer, candidates, size):
            ...
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import TYPE_CHECKING, ClassVar

import torch

from ...errors import ConfigError
from ...registry import Registry
from .metrics import SKIP_METRICS, SkipMetric
from .policy import SkipLayers

if TYPE_CHECKING:
    from ...runtime import LayerRunner

__all__ = ["SKIP_SEARCHES", "GreedySearch", "SkipScorer", "SkipSearch", "UniformSearch"]


class SkipScorer:
    """Scores skip sets on calibration batches: the mean over all positions of a registered metric
    (:data:`SKIP_METRICS`) of the skipped model against the full model.

    Every candidate runs as one parallel pass of the model over each batch with the candidate's layers skipped for
    every token (Draft & Verify scores its skip sets the same way). The full model's pass over each batch runs once
    per scorer: the metric's :meth:`~SkipMetric.reference` of it is cached (``[B, T, V]`` log-probabilities for
    ``"kl"``, only the greedy tokens for ``"mismatch"``), unless ``cache_reference=False``.

    Attributes:
        passes: Model passes run so far (reference passes included).
    """

    def __init__(self, runner: LayerRunner, batches: Sequence[torch.Tensor], metric: str | SkipMetric = "kl", *,
                 cache_reference: bool = True) -> None:
        self.runner = runner
        self.batches = [b.to(runner.adapter.device) for b in batches]
        self.metric = SKIP_METRICS.get(metric)() if isinstance(metric, str) else metric
        self.cache_reference = cache_reference
        self.passes = 0
        self._references: dict[int, torch.Tensor] = {}

    @torch.no_grad()
    def log_probs(self, ids: torch.Tensor, skip: frozenset[int]) -> torch.Tensor:
        """Next-token log-probabilities ``[B, T, V]`` of the model with ``skip`` skipped."""
        runner = self.runner
        rows, length = ids.shape
        cache = runner.new_cache(length, rows)
        ctx = runner.context(cache, torch.arange(length, device=ids.device), None, trusted=True)
        policy = SkipLayers(skip) if skip else None
        hidden, _ = runner.run(runner.embed(ids, ctx), ctx, 1, runner.adapter.num_layers, policy=policy)
        self.passes += 1
        return runner.logits(hidden).float().log_softmax(-1)

    def reference(self, index: int) -> torch.Tensor:
        """The metric's reference of the full model on batch ``index`` (computed once when cached)."""
        cached = self._references.get(index)
        if cached is not None:
            return cached
        reference = self.metric.reference(self.log_probs(self.batches[index], frozenset()))
        if self.cache_reference:
            self._references[index] = reference
        return reference

    @torch.no_grad()
    def score(self, sets: Sequence[frozenset[int]]) -> list[float]:
        """The metric of every skip set in ``sets`` (lower is better)."""
        totals = [0.0] * len(sets)
        count = 0
        for index, ids in enumerate(self.batches):
            reference = self.reference(index)
            for i, skip in enumerate(sets):
                totals[i] += self.metric(reference, self.log_probs(ids, skip)).sum().item()
            count += ids.numel()
        return [total / count for total in totals]


class SkipSearch(ABC):
    """Chooses ``size`` layers among ``candidates``."""

    #: Whether the search scores sets on calibration data (it then receives a :class:`SkipScorer`).
    needs_data: ClassVar[bool] = True

    @abstractmethod
    def __call__(self, scorer: SkipScorer | None, candidates: Sequence[int], size: int) -> tuple[int, ...]:
        """The chosen layers, sorted."""


#: Registered skip-set searches.
SKIP_SEARCHES: Registry[type[SkipSearch]] = Registry("skip-set search", "economize.skip_searches")


@SKIP_SEARCHES.register("greedy")
class GreedySearch(SkipSearch):
    """Forward selection: add, one layer at a time, the candidate whose skipping raises the metric least.

    It costs ``size * len(candidates)`` scored sets. This is a whole-layer greedy skip, an approximation of the
    published searches: Draft & Verify uses Bayesian optimization and skips attention and MLP sub-layers
    separately, SWIFT optimizes the set on the fly during decoding, and CLaSp chooses it by dynamic programming on
    the last verified states, again for every cycle. Greedy selection over whole layers, fixed after calibration,
    is the deterministic stand-in offered here.

    Attributes:
        trace: The layer added in each round of the last search with its score, in order (empty before a search).
    """

    def __init__(self) -> None:
        self.trace: list[tuple[int, float]] = []

    def __call__(self, scorer: SkipScorer | None, candidates: Sequence[int], size: int) -> tuple[int, ...]:
        if scorer is None:
            raise ConfigError(f"the {type(self).__name__} skip-set search needs calibration data (a SkipScorer)")
        chosen: list[int] = []
        self.trace = []
        for _ in range(size):
            options = [c for c in candidates if c not in chosen]
            scores = scorer.score([frozenset([*chosen, c]) for c in options])
            best = min(range(len(options)), key=scores.__getitem__)
            chosen.append(options[best])
            self.trace.append((options[best], scores[best]))
        return tuple(sorted(chosen))


@SKIP_SEARCHES.register("uniform")
class UniformSearch(SkipSearch):
    """Spread the skipped layers evenly over the candidates (no data needed)."""

    needs_data = False

    def __call__(self, scorer: SkipScorer | None, candidates: Sequence[int], size: int) -> tuple[int, ...]:
        n = len(candidates)
        return tuple(candidates[int((i + 0.5) * n / size)] for i in range(size))
