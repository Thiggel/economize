"""The drafter of a draft view."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from ...decoding.depth_steps import DepthSteps
from ...decoding.graphs import DecodeGraphs
from ...decoding.protocols import Drafter
from ..early_exit.drafter import SelfDrafter

if TYPE_CHECKING:
    from ...decoding.protocols import Draft
    from ...decoding.request import DraftRequest
    from ...runtime import KVCache, LayerRunner
    from .._drafting import StepWrap
    from .technique import DraftWithTechnique

__all__ = ["DraftWithDrafter"]


class DraftWithDrafter(SelfDrafter):
    """The drafter of :class:`~economize.techniques.draft_with.DraftWithTechnique`: every layer of the draft path
    (the held techniques' copies of the decoder layers, minus the layers a held technique skips), then the model's
    norm and head.

    A draft path with a dynamic-depth policy (a held :class:`~economize.techniques.dynamic_depth.DepthTechnique`,
    the technique used as a drafter) decides per token which layers its draft steps run. Its decisions are host
    decisions, so a whole draft step cannot be one CUDA graph. The drafter runs the draft steps of a single row
    through :class:`~economize.decoding.DepthSteps` instead (skipped layers launch no kernel; with the decoder's CUDA
    graphs the segments between the policy's probes are graphs), and other batches eagerly through the policy's
    eager path. The policy counts generated positions from the end of each prompt
    (:meth:`~economize.techniques.dynamic_depth.DepthPolicy.begin_generation`, at the first draft of a generation:
    the committed tokens then are the prompt and the token of the prefill).
    """

    technique: DraftWithTechnique

    def __init__(self, technique: DraftWithTechnique, runner: LayerRunner) -> None:
        super().__init__(technique, runner)
        self._steps: DepthSteps | None = None
        self._started = False

    @property
    def depth_steps(self) -> DepthSteps | None:
        """The step engine of the draft steps (built by :meth:`enable_graphs` or the first single-row draft), or
        ``None`` (no layer policy, a recurrent model, or no single-row draft yet)."""
        return self._steps

    @property
    def _depth(self) -> bool:
        """Whether the draft path has a layer policy (decisions on the host)."""
        return self.technique.policy is not None

    def begin(self, runner: LayerRunner, cache: KVCache, batch_size: int) -> None:
        super().begin(runner, cache, batch_size)
        self._started = False

    def enable_graphs(self, wrap: StepWrap) -> bool:
        policy = self.technique.policy
        if policy is None:
            return super().enable_graphs(wrap)
        if not policy.supports_steps or self.runner.has_recurrent_layers:
            return False  # eager draft steps
        graphs = getattr(wrap, "__self__", None)  # the decoder's DecodeGraphs.wrap
        self._steps = DepthSteps(self.runner, policy, graphs if isinstance(graphs, DecodeGraphs) else None,
                                 overrides=self.technique.overrides, head=self._draft_logits)
        return self._steps.graphed

    def propose(self, request: DraftRequest) -> Draft:
        if not self._depth:
            return super().propose(request)
        if not self._started:
            self._start(request)
        steps, cache = self._eager_steps() if self._steps is None else self._steps, self._cache
        if steps is None or request.batch_size != 1 or cache is None or cache.active_batch_size != 1:
            return Drafter.propose(self, request)  # eager passes through the policy's decide/after
        while request.drafting:
            position = request.ends[0] - 1 + request.index
            request.extend(steps.draft(cache, request.tokens[:, None], position))
        return request.draft()

    def _eager_steps(self) -> DepthSteps | None:
        """Step engine without graphs (no decoder graphs: a CPU model, or ``cuda_graphs`` off)."""
        policy = self.technique.policy
        if policy is None or not policy.supports_steps or self.runner.has_recurrent_layers:
            return None
        self._steps = DepthSteps(self.runner, policy, None, overrides=self.technique.overrides,
                                 head=self._draft_logits)
        return self._steps

    def _start(self, request: DraftRequest) -> None:
        """Start the policy's generation: the prompts end one before the committed tokens of the first draft."""
        policy = self.technique.policy
        assert policy is not None
        begin = getattr(policy, "begin_generation", None)
        if begin is not None:
            begin(torch.tensor([end - 1 for end in request.ends], device=request.device))
        self._started = True

    def reorder(self, rows: torch.Tensor) -> None:
        policy = self.technique.policy
        if policy is not None and self._started:
            policy.reorder(rows)

    def supports_tree_steps(self) -> bool:
        return not self._depth and super().supports_tree_steps()
