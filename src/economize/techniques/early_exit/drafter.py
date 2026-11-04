"""The drafter of self-speculative techniques (early exit, layer skipping), which draft with the model's own layers."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Protocol

import torch

from ...decoding.protocols import Drafter
from .._drafting import AdapterHeadSlicing, GraphedStep, tree_step_context

if TYPE_CHECKING:
    from ...arch import ArchAdapter
    from ...decoding.cost import LayerFootprint
    from ...decoding.request import DraftRequest
    from ...runtime import ExactPass, KVCache, LayerContext, LayerRunner

__all__ = ["EarlyExitDrafter", "SelfDraftStep", "SelfDrafter"]


class SelfDraftStep(Protocol):
    """What a self-speculative technique provides for one draft step."""

    @property
    def layer_span(self) -> tuple[int, int]:
        """First and last decoder layer of a draft step, for the cost log."""

    @property
    def footprint(self) -> LayerFootprint | None:
        """The layers a draft step runs when they are not the whole :attr:`layer_span` (``None``: the span)."""

    @property
    def adapter(self) -> ArchAdapter:
        """The model's adapter (its final norm and LM head read the draft path's output)."""

    def run(self, runner: LayerRunner, embedding: torch.Tensor, ctx: LayerContext) -> torch.Tensor:
        """Draft-path hidden states for token embeddings ``[B, S, d]``, writing the K/V of the layers it runs."""

    def logits(self, hidden: torch.Tensor) -> torch.Tensor:
        """Next-token logits from the draft-path hidden states."""


class SelfDrafter(AdapterHeadSlicing, Drafter):
    """Self-speculative drafting for a batch of rows: embed the last token, run the technique's draft path, read logits.

    The drafter keeps no state between steps: the K/V that the draft path wrote for earlier draft tokens are in the
    shared cache, and the next exact pass overwrites them. Every step is one batched call per layer with fixed
    shapes ``[B, 1, d]`` and no host synchronization, so it can be replayed from a CUDA graph. Rows that stopped
    drafting run on the cache's trash slot. :attr:`layer_span` and :attr:`footprint` are read from the technique on
    every access, so a technique whose draft layers change (a new skip set) is costed by its current layers.

    The drafter is :class:`~economize.techniques._drafting.HeadSliceable` when the technique exposes the state its LM
    head reads (:class:`~economize.techniques._drafting.HeadInput`).
    """

    def __init__(self, technique: SelfDraftStep, runner: LayerRunner) -> None:
        self.technique = technique
        self.runner = runner
        self._cache: KVCache | None = None
        self._step_fn = GraphedStep(self._step)
        self._head: Callable[[torch.Tensor], torch.Tensor] | None = None

    @property
    def layer_span(self) -> tuple[int, int]:  # type: ignore[override]
        return self.technique.layer_span

    @property
    def footprint(self) -> LayerFootprint | None:  # type: ignore[override]
        return self.technique.footprint

    def begin(self, runner: LayerRunner, cache: KVCache, batch_size: int) -> None:
        self.runner, self._cache = runner, cache

    def observe(self, exact: ExactPass, columns: torch.Tensor) -> None:
        pass

    def enable_graphs(self, wrap: Callable[[Callable[..., Any]], Callable[..., Any]]) -> bool:
        self._step_fn.enable(wrap)
        return True

    def step(self, tokens: torch.Tensor, positions: torch.Tensor, valid: torch.Tensor | None) -> torch.Tensor:
        if self._cache is None:
            raise RuntimeError(f"{type(self).__name__}.step needs begin() first")
        return self._step_fn(tokens, positions, valid)

    def _step(self, tokens: torch.Tensor, positions: torch.Tensor, valid: torch.Tensor | None) -> torch.Tensor:
        assert self._cache is not None
        ctx = self.runner.context(self._cache, positions[:, None], None if valid is None else valid[:, None],
                                  trusted=True)
        hidden = self.technique.run(self.runner, self.runner.embed(tokens[:, None], ctx), ctx)
        return self._draft_logits(hidden)


    # ---- dynamic trees (economize.decoding.dynamic_tree) ----------------------------------------------------
    def supports_tree_steps(self) -> bool:
        """True unless a subclass changes how a step or a proposal runs (KnapSpec's adaptive chains)."""
        cls = type(self)
        return cls._step is SelfDrafter._step and cls.propose is Drafter.propose

    def tree_begin(self, request: DraftRequest) -> None:
        if self._cache is None:
            raise RuntimeError(f"{type(self).__name__} needs begin() before drafting a tree")

    def tree_state(self) -> tuple[torch.Tensor, ...]:
        return ()

    def tree_step(self, tokens: torch.Tensor, parents: torch.Tensor, base: torch.Tensor,
                  *states: torch.Tensor) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
        """The draft path over the frontier of a tree (see :class:`~economize.decoding.TreeStepDrafter`); its layers
        write the frontier's K/V at the nodes' slots and attend to the committed prefix and their ancestors."""
        assert self._cache is not None
        ctx = tree_step_context(self.runner, self._cache, parents, base, tokens.shape[1])
        hidden = self.technique.run(self.runner, self.runner.embed(tokens, ctx), ctx)
        return self._tree_logits(hidden), ()

class EarlyExitDrafter(SelfDrafter):
    """The drafter of :class:`~economize.techniques.early_exit.EarlyExitTechnique` (layers ``1..r``, then the exit)."""
