"""How a candidate skip set is scored against the full model, one registered metric per score.

A metric is registered in :data:`SKIP_METRICS` (open for extension)::

    from economize.techniques.layer_skip import SKIP_METRICS, SkipMetric

    @SKIP_METRICS.register("top5_miss")
    class Top5Miss(SkipMetric):
        def reference(self, log_probs):
            return log_probs.argmax(-1)

        def __call__(self, reference, log_probs):
            return (log_probs.topk(5, -1).indices != reference[..., None]).all(-1).float()
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch

from ...registry import Registry

__all__ = ["SKIP_METRICS", "KLMetric", "MismatchMetric", "SkipMetric"]


class SkipMetric(ABC):
    """A per-position score of a skipped model against the full model (lower is better).

    The scorer computes the full model's log-probabilities once per calibration batch, keeps
    :meth:`reference` of them and compares every candidate's log-probabilities with that.
    """

    def reference(self, log_probs: torch.Tensor) -> torch.Tensor:
        """What the metric keeps of the full model's log-probabilities ``[B, T, V]`` (by default all of them)."""
        return log_probs

    @abstractmethod
    def __call__(self, reference: torch.Tensor, log_probs: torch.Tensor) -> torch.Tensor:
        """The score ``[B, T]`` of the skipped model's log-probabilities ``[B, T, V]`` at every position."""


#: Registered skip-set metrics.
SKIP_METRICS: Registry[type[SkipMetric]] = Registry("skip-set metric", "economize.skip_metrics")


@SKIP_METRICS.register("kl")
class KLMetric(SkipMetric):
    """KL(full || skipped) of the next-token distributions."""

    def __call__(self, reference: torch.Tensor, log_probs: torch.Tensor) -> torch.Tensor:
        return (reference.exp() * (reference - log_probs)).sum(-1)


@SKIP_METRICS.register("mismatch")
class MismatchMetric(SkipMetric):
    """1 where the skipped model's greedy token differs from the full model's (Draft & Verify's acceptance proxy).

    It keeps only the full model's greedy tokens, so its cache is ``V`` times smaller than KL's.
    """

    def reference(self, log_probs: torch.Tensor) -> torch.Tensor:
        return log_probs.argmax(-1)

    def __call__(self, reference: torch.Tensor, log_probs: torch.Tensor) -> torch.Tensor:
        return (log_probs.argmax(-1) != reference).float()
