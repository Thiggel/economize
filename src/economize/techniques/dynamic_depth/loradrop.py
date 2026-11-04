"""LoRA-Drop (Rajabzadeh et al. 2026, arXiv 2601.02569): droppable layers run in full only every ``period`` tokens;
in between a token reuses the layer output of the last full token plus a LoRA correction.

Decoding goes in cycles of ``period`` generated tokens. The first token of a cycle (generated position ``g`` with
``g % period == 0``, counted from the end of the row's prompt) runs every layer. At a droppable layer ``l`` every
other token skips the layer and leaves it as ``h_l(anchor) + B A x * alpha / rank``: the output of layer ``l`` at the
cycle's full token (the *anchor*) plus a low-rank map of the token's own layer input, trained (objective ``loradrop``)
to imitate the layer. The default droppable layers are the paper's: every layer except the first three and the last.
:func:`redundant_layers` ranks layers by the cosine similarity of their input and output on calibration data. The
prompt runs every layer.

Anchors are kept per row with their positions, so a pass over several pending tokens (a verification pass after
drafting) reuses the same anchors as one-token decoding, and anchors of rejected draft tokens are discarded when the
next pass starts at or before their position. A token without any anchor (the schedule's first full token was never
seen, e.g. a dropped first position) falls back to ``x + B A x * alpha / rank``.

Deviation from the paper. Dropped tokens still write the K/V of the layers they skip (the paper writes no K/V there,
which needs the routed-K/V runtime support described in :func:`~.base.runtime_supports_routed_kv`).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar, cast

import torch
import torch.nn.functional as F
from torch import nn

from ...core.config import TechniqueConfig
from ...core.metadata import Family, Saving, TechniqueInfo
from ...core.technique import register_technique
from ...errors import ConfigError
from ...runtime import LayerRunner
from ...validation import check_positive
from .base import DepthPolicy, DepthTechnique, LayeredDepthConfig, layer_modules, module_at, valid_mask

if TYPE_CHECKING:
    from ...runtime import LayerContext

__all__ = ["AnchorStore", "LoRADropConfig", "LoRADropPolicy", "LoRADropTechnique", "LowRankUpdate",
           "redundant_layers"]

_MIN = torch.iinfo(torch.long).min
_MAX = torch.iinfo(torch.long).max


@dataclass(frozen=True)
class LoRADropConfig(LayeredDepthConfig):
    """LoRA-Drop.

    Attributes:
        layers: Droppable layers (empty: all but the first three and the last, as in the paper;
            :func:`redundant_layers` ranks layers on calibration data).
        period: A droppable layer runs in full for every ``period``-th generated token (``1``: always, the model
            is unchanged). The paper's ``k + 1``.
        rank: Rank of the low-rank update.
        alpha: Scale ``alpha / rank`` of the update.
        skip_prompt: Drop layers in the prompt too (``False``: the prompt runs every layer).
    """

    technique: ClassVar[str] = "loradrop"
    period: int = 2
    rank: int = 16
    alpha: float = 16.0
    skip_prompt: bool = False

    def __post_init__(self) -> None:
        super().__post_init__()
        check_positive("period", self.period)
        check_positive("rank", self.rank)

    def default_layers(self, num_layers: int) -> tuple[int, ...]:
        return tuple(range(4, num_layers))

    def droppable_layers(self, num_layers: int) -> tuple[int, ...]:
        """The droppable layers (:meth:`layer_set`)."""
        return self.layer_set(num_layers)


class LowRankUpdate(nn.Module):
    """``B A x * alpha / rank`` with ``B`` zero-initialized."""

    def __init__(self, hidden_size: int, rank: int, alpha: float) -> None:
        super().__init__()
        self.a = nn.Linear(hidden_size, rank, bias=False)
        self.b = nn.Linear(rank, hidden_size, bias=False)
        nn.init.kaiming_uniform_(self.a.weight, a=5**0.5)
        nn.init.zeros_(self.b.weight)
        self.scale = alpha / rank

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.b(self.a(x.float())) * self.scale


class AnchorStore:
    """The layer outputs of the full tokens of one layer, with their positions, per row.

    :meth:`lookup` gives every token of a pass the output of the latest full token at or before it: from the pass
    itself, or else the latest stored one before the pass. Stored entries at or after the pass's first position
    are dropped first (they belonged to tokens that are being recomputed). Since the next pass starts after this
    pass's first token, only the latest entry at or before that token and the later ones are kept.
    """

    def __init__(self) -> None:
        self.states: torch.Tensor | None = None  # [B, M, D]
        self.positions: torch.Tensor | None = None  # [B, M]
        self.filled: torch.Tensor | None = None  # [B, M]

    def reorder(self, rows: torch.Tensor) -> None:
        if self.states is None or self.positions is None or self.filled is None:  # nothing stored yet
            return
        rows = rows.to(self.states.device)
        self.states = self.states.index_select(0, rows)
        self.positions = self.positions.index_select(0, rows)
        self.filled = self.filled.index_select(0, rows)

    def lookup(self, out: torch.Tensor, full: torch.Tensor, positions: torch.Tensor, valid: torch.Tensor,
               keep: bool = True) -> tuple[torch.Tensor, torch.Tensor]:
        """``(anchor [B, S, D], found [B, S])`` for the tokens of a pass with outputs ``out``, full-token mask
        ``full`` and schedule positions ``positions``; stores the pass's full outputs when ``keep``."""
        batch, length, width = out.shape
        full = full & valid
        index = torch.where(full, torch.arange(length, device=out.device), -1).cummax(-1).values
        anchor = out.gather(1, index.clamp_min(0)[..., None].expand(batch, length, width))
        found = index >= 0
        start = torch.where(valid, positions, _MAX).min(-1).values
        states, stored_positions, filled = out.detach(), positions, full
        old_states, old_positions, old_filled = self.states, self.positions, self.filled  # set together
        if (old_states is not None and old_positions is not None and old_filled is not None
                and len(old_states) == batch and old_states.shape[-1] == width):
            kept = old_filled & (old_positions < start[:, None])
            latest = torch.where(kept, old_positions, _MIN).argmax(-1)
            previous = old_states[torch.arange(batch, device=out.device), latest].to(out.dtype)
            anchor = torch.where(found[..., None], anchor, previous[:, None])
            found = found | kept.any(-1)[:, None]
            states = torch.cat([old_states, states], 1)
            stored_positions = torch.cat([old_positions, positions], 1)
            filled = torch.cat([kept, full], 1)
        if keep:
            oldest = torch.where(filled & (stored_positions <= start[:, None]), stored_positions, _MIN).max(-1).values
            filled = filled & (stored_positions >= oldest[:, None])
            columns = filled.any(0)
            self.states, self.positions, self.filled = (states[:, columns], stored_positions[:, columns],
                                                        filled[:, columns])
        return anchor, found


class LoRADropPolicy(DepthPolicy):
    """Skips droppable layers between full runs and builds the dropped tokens' outputs from the anchors. In
    training mode every token runs every layer and the tokens the schedule drops take the anchor plus the low-rank
    update in place of the layer's output, so training sees the inference schedule."""

    def __init__(self, technique: LoRADropTechnique) -> None:
        super().__init__(technique.adapter.num_layers, technique.config.phases)
        self.technique = technique
        self._full: torch.Tensor | None = None
        self._positions: torch.Tensor | None = None
        self._anchors: dict[int, AnchorStore] = {}

    def begin_prompt(self) -> None:
        self._anchors.clear()

    def begin_pass(self, hidden: torch.Tensor, ctx: LayerContext) -> None:
        self._positions = self.generated_positions(hidden, ctx)
        self._full = self._positions % self.technique.config.period == 0

    def reorder(self, rows: torch.Tensor) -> None:
        super().reorder(rows)
        for store in self._anchors.values():
            store.reorder(rows)

    def choose(self, index: int, hidden: torch.Tensor, ctx: LayerContext) -> torch.Tensor | None:
        return self._full if self.technique.droppable(index) else None

    def transform(self, index: int, x: torch.Tensor, out: torch.Tensor, ran: torch.Tensor | None,
                  ctx: LayerContext) -> torch.Tensor | None:
        if not self.technique.droppable(index):
            return None
        assert self._full is not None and self._positions is not None
        full = self._full.to(out.device)
        store = AnchorStore() if self.training else self._anchors.setdefault(index, AnchorStore())
        anchor, found = store.lookup(out, full, self._positions.to(out.device), valid_mask(x, ctx),
                                     keep=not self.training)
        if bool(full.all()):
            return None
        base = torch.where(found[..., None], anchor, x)
        dropped = base + self.technique.update(index, x).to(x.dtype)
        return torch.where(full[..., None], out, dropped)


@register_technique
class LoRADropTechnique(DepthTechnique):
    """LoRA-Drop (see :class:`LoRADropConfig`); trained by the objective ``loradrop``."""

    name: ClassVar[str] = LoRADropConfig.technique
    config_class: ClassVar[type[TechniqueConfig]] = LoRADropConfig
    info = TechniqueInfo(
        family=Family.DYNAMIC_DEPTH,
        saves=Saving.COMPUTE | Saving.LATENCY,
        lossless=False,
        trainable=("updates",),
    )
    config: LoRADropConfig
    order: ClassVar[int] = 35

    def __init__(self, config: LoRADropConfig, adapter: Any) -> None:
        super().__init__(config, adapter)
        self.updates = layer_modules(
            adapter,
            config.layer_set(adapter.num_layers),
            lambda _: LowRankUpdate(adapter.hidden_size, config.rank, config.alpha),
        )

    def droppable(self, index: int) -> bool:
        return module_at(self.updates, index) is not None

    def update(self, index: int, hidden: torch.Tensor) -> torch.Tensor:
        """The low-rank update (fp32) of layer ``index``."""
        module = cast(LowRankUpdate, self.updates[str(index)])
        return module(hidden.to(module.a.weight.device)).to(hidden.device)

    def make_policy(self) -> LoRADropPolicy:
        return LoRADropPolicy(self)


@torch.no_grad()
def redundant_layers(
    adapter: Any, batches: Sequence[torch.Tensor], count: int, *, exclude_first: int = 1
) -> tuple[int, ...]:
    """The ``count`` layers whose output is most similar to their input on ``batches`` (mean cosine similarity over
    tokens), in increasing order. The first ``exclude_first`` layers are never chosen."""
    if not 0 < count <= adapter.num_layers - exclude_first:
        raise ConfigError(f"count must lie in 1..{adapter.num_layers - exclude_first}, got {count}")
    runner = LayerRunner(adapter)
    layers = range(1, adapter.num_layers + 1)
    total = torch.zeros(adapter.num_layers + 1, dtype=torch.float64)
    tokens = 0
    for batch in batches:
        batch = batch.view(1, -1) if batch.dim() == 1 else batch
        cache = runner.new_cache(batch.shape[1], batch.shape[0])
        ctx = runner.context(cache, torch.arange(batch.shape[1]), trusted=True)
        captured = runner.forward(batch, ctx, capture=(0, *layers)).hidden
        for index in layers:
            x, out = captured[index - 1].float(), captured[index].float().to(captured[index - 1].device)
            total[index] += float(F.cosine_similarity(x, out, dim=-1).sum())
        tokens += batch.numel()
    scores = total[1:] / max(tokens, 1)
    scores[:exclude_first] = float("-inf")
    return tuple(sorted(int(i) + 1 for i in scores.topk(count).indices))
