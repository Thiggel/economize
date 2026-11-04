"""Cost accounting of dynamic-depth methods that decode through economize's drafting cycle.

FREE and DMTD run some tokens through part of the model only and fill the rest of their layers later in one
parallel pass. economize expresses both as a drafter with every draft kept (``DecodeConfig(lossless=False)``): the
draft steps run the partial path and the next exact pass runs all layers over the pending tokens. That exact pass
recomputes the layers a draft step already ran on the same token, which the original methods reuse:

* FREE's shallow layers are the model's first layers, so the exact pass recomputes them identically, and a token
  whose shallow exit was rejected continues from its shallow state instead of starting over;
* DMTD keeps the late layers' K/V that its draft steps wrote.

:func:`drafted_cost_summary` subtracts what the method reuses from the cost log, so the reported cost is the
method's own.
"""

from __future__ import annotations

from typing import Any

from ...decoding.cost import CostKind

__all__ = ["drafted_cost_summary"]


def drafted_cost_summary(result: Any, *, continue_wasted: bool) -> dict[str, Any]:
    """``result.cost_summary()`` with the layer applications the method reuses subtracted.

    Every exact pass reuses the decoded model's layers that the draft steps since the previous exact pass ran on its
    tokens (at most one draft step per token of the pass, so a generation that ends on drafts reuses nothing).

    Keys added:
        ``reused_layer_tokens``: layer applications the exact passes recompute and the method reuses.
        ``effective_layer_tokens``, ``relative_effective_layer_tokens``: ``layer_tokens`` minus the reused ones, and
            that count relative to plain decoding of the same output (prompt included, as ``relative_layer_tokens``).
        ``sequential_layers``: the decoded model's layers run one after another after the prefill (one per layer of
            every call but the first), with a rejected draft step's layers counted once when ``continue_wasted``
            (FREE: the token that did not exit continues through the deep layers from its shallow state).
        ``relative_sequential_layers``: ``sequential_layers / ((n - 1) * L)`` for ``n`` generated tokens, the
            sequential depth relative to plain decoding.
    """
    summary = dict(result.cost_summary())
    events = result.cost.events
    reused = 0
    sequential = 0
    pending: list[int] = []  # layers of the draft steps since the last exact pass
    last_wasted = 0
    for index, event in enumerate(events):
        if event.module is not None:  # a separate network's layers have another size
            continue
        if event.kind == CostKind.EXACT:
            reused += sum(pending[: event.tokens])
            if continue_wasted:
                sequential -= last_wasted
            pending, last_wasted = [], 0
        elif event.kind in (CostKind.DRAFT, CostKind.DRAFT_WASTED):
            pending.append(len(event.layer_indices))
            last_wasted = len(event.layer_indices) if event.kind == CostKind.DRAFT_WASTED else 0
        if index > 0:
            sequential += len(event.layer_indices)
    effective = summary["layer_tokens"] - reused
    relative = summary["relative_layer_tokens"]
    plain = summary["layer_tokens"] / relative if relative else 0
    generated = result.tokens.numel()
    summary["reused_layer_tokens"] = reused
    summary["effective_layer_tokens"] = effective
    summary["relative_effective_layer_tokens"] = effective / plain if plain else 0.0
    summary["sequential_layers"] = sequential
    summary["relative_sequential_layers"] = sequential / max(1, (generated - 1) * result.num_layers)
    return summary
