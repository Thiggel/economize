"""Static depth pruning: remove whole decoder layers.

* ShortGPT: Men et al., "ShortGPT: Layers in Large Language Models are More Redundant Than You Expect", 2024
  (arXiv:2403.03853). Block Influence ``BI_i = 1 - E_t cos(x_t^{(i-1)}, x_t^{(i)})`` is measured once on the full
  model, and the layers with the lowest BI are removed.
* SLEB: Song et al., "SLEB: Streamlining LLMs through Redundancy Verification and Elimination of Transformer
  Blocks", ICML 2024 (https://github.com/jiwonsong-dev/SLEB). Layers are removed one at a time; each step removes
  the layer whose removal gives the lowest calibration perplexity of the current model.

Both criteria run on the layer-wise calibration engine (:mod:`economize.core.calibration`), so they work on models
split over several devices with ``device_map``: every layer runs on its own device with its call arguments moved
there. A criterion may be restricted to candidate layers (:attr:`DepthPruneConfig.keep_first` and
:attr:`~DepthPruneConfig.keep_last` protect the outermost layers; SLEB never removes the first or the last layer
by default).

:func:`remove_layers` deletes layers in place through :meth:`ArchAdapter.remove_layers`, which keeps the config
(``num_hidden_layers``, per-layer lists, layer-index fields) and every ``layer_idx`` consistent, so ``generate``,
the KV cache and ``save_pretrained`` / ``from_pretrained`` keep working.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Hashable, Iterable, Sequence
from typing import TYPE_CHECKING, ClassVar

import torch
import torch.nn.functional as F
from torch import Tensor

from ...core.calibration import (
    CalibrationStep,
    LayerCall,
    LayerwiseCalibration,
    capture_inputs,
    layer_device,
    layer_types,
    token_batches,
)
from ...registry import Registry

if TYPE_CHECKING:
    from ...arch import ArchAdapter

__all__ = [
    "DEPTH_CRITERIA",
    "SLEB",
    "DepthCriterion",
    "ShortGPT",
    "block_influence",
    "remove_layers",
    "shortgpt_select",
    "sleb_select",
]


@torch.no_grad()
def block_influence(adapter: ArchAdapter, batches: Sequence[Tensor]) -> Tensor:
    """``BI_i`` for every layer ``i = 1..L`` (index ``i - 1`` of the result), averaged over all calibration tokens.

    The states are the residual stream entering and leaving each layer of the unmodified model (the engine's
    ``propagate="original"``; the last entry of ``output_hidden_states`` has the final norm applied, so it is not
    used). Padded tokenizer batches are split into their sequences first, so padding never enters the average.
    """
    batches = token_batches(batches)
    totals = torch.zeros(adapter.num_layers, dtype=torch.float64)
    for step in LayerwiseCalibration(adapter, batches, propagate="original"):
        assert step.first_outputs is not None
        for x, y in zip(step.inputs, step.first_outputs):
            cos = F.cosine_similarity(x.to(y.device).float(), y.float(), dim=-1)
            totals[step.index - 1] += (1 - cos).sum().double().cpu()
    tokens = sum(batch.numel() for batch in batches)
    return (totals / max(tokens, 1)).float()


def shortgpt_select(influence: Tensor, num_blocks: int, candidates: Sequence[int] | None = None) -> list[int]:
    """1-indexed layers with the ``num_blocks`` lowest Block Influence among ``candidates`` (default: all), sorted."""
    allowed = list(range(1, influence.numel() + 1)) if candidates is None else list(candidates)
    ranked = sorted(allowed, key=lambda i: (float(influence[i - 1]), i))
    return sorted(ranked[:num_blocks])


@torch.no_grad()
def _nll(adapter: ArchAdapter, hidden: Tensor, ids: Tensor) -> float:
    logits = adapter.logits(hidden)[:, :-1].float()
    target = ids[:, 1:].to(logits.device)
    return float(F.cross_entropy(logits.reshape(-1, logits.shape[-1]), target.reshape(-1), reduction="sum"))


class _LayerRunner:
    """Runs single decoder layers on calibration states, each on its own device (``device_map``) with its call
    arguments moved there once per (layer key, device)."""

    def __init__(self, adapter: ArchAdapter, batches: Sequence[Tensor]) -> None:
        self.adapter = adapter
        self.hidden, self._calls = capture_inputs(adapter, batches)
        self._types = layer_types(adapter)
        self._moved: dict[tuple[Hashable, torch.device], list[LayerCall]] = {}

    def run(self, index: int, states: list[Tensor]) -> list[Tensor]:
        """Outputs of layer ``index`` (1-indexed) for ``states`` (one tensor per batch, on any device)."""
        device = layer_device(self.adapter, index)
        key = (self._types[index - 1], device)
        if key not in self._moved:
            self._moved[key] = [c[key[0]].to(device) for c in self._calls]
        return CalibrationStep(index, self.adapter.layer(index), states, self._moved[key], device).forward()


@torch.no_grad()
def sleb_select(
    adapter: ArchAdapter, batches: Sequence[Tensor], num_blocks: int, candidates: Sequence[int] | None = None
) -> list[int]:
    """SLEB's iterative selection among ``candidates`` (1-indexed, default: every layer). Returns the 1-indexed
    layers (of the original model) to remove, sorted.

    Each step evaluates every remaining candidate for removal. Layers are visited in depth order and candidates share
    the hidden state of the prefix before them, which gives exactly the perplexities of the naive re-evaluation at
    about half the cost.
    """
    batches = token_batches(batches)
    runner = _LayerRunner(adapter, batches)
    allowed = set(range(1, adapter.num_layers + 1) if candidates is None else candidates)
    alive = list(range(1, adapter.num_layers + 1))
    removed: list[int] = []
    for _ in range(num_blocks):
        best, best_loss = -1, float("inf")
        prefix = runner.hidden
        for pos, layer in enumerate(alive):
            if layer in allowed:
                states = prefix
                for later in alive[pos + 1 :]:
                    states = runner.run(later, states)
                loss = sum(_nll(adapter, x, b) for x, b in zip(states, batches))
                if loss < best_loss:
                    best, best_loss = layer, loss
            prefix = runner.run(layer, prefix)
        alive.remove(best)
        allowed.discard(best)
        removed.append(best)
    return sorted(removed)


class DepthCriterion(ABC):
    """Chooses which layers to remove.

    Attributes:
        keep_first: Leading layers the criterion never removes unless the config says otherwise.
        keep_last: Trailing layers the criterion never removes unless the config says otherwise.
    """

    keep_first: ClassVar[int] = 0
    keep_last: ClassVar[int] = 0

    @abstractmethod
    def select(
        self, adapter: ArchAdapter, batches: Sequence[Tensor], num_blocks: int, candidates: Sequence[int] | None = None
    ) -> list[int]:
        """1-indexed layers to remove, taken from ``candidates`` (default: every layer)."""


DEPTH_CRITERIA: Registry[type[DepthCriterion]] = Registry("depth-pruning criterion", "economize.depth_criteria")


@DEPTH_CRITERIA.register("shortgpt")
class ShortGPT(DepthCriterion):
    """Lowest Block Influence, measured once (every layer is a candidate, as in the paper)."""

    def select(
        self, adapter: ArchAdapter, batches: Sequence[Tensor], num_blocks: int, candidates: Sequence[int] | None = None
    ) -> list[int]:
        return shortgpt_select(block_influence(adapter, batches), num_blocks, candidates)


@DEPTH_CRITERIA.register("sleb")
class SLEB(DepthCriterion):
    """Iterative lowest-perplexity removal. The reference implementation never removes the first or the last block."""

    keep_first: ClassVar[int] = 1
    keep_last: ClassVar[int] = 1

    def select(
        self, adapter: ArchAdapter, batches: Sequence[Tensor], num_blocks: int, candidates: Sequence[int] | None = None
    ) -> list[int]:
        return sleb_select(adapter, batches, num_blocks, candidates)


def remove_layers(adapter: ArchAdapter, indices: Iterable[int]) -> None:
    """Delete the 1-indexed decoder layers ``indices`` from the model in place (:meth:`ArchAdapter.remove_layers`;
    the family adapter renumbers its layer-indexed config fields through :meth:`ArchAdapter.renumber_config`)."""
    adapter.remove_layers(indices)
