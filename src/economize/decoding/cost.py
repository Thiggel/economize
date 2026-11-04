"""Exact cost accounting: every model call made while decoding is recorded as one event."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field, replace
from typing import Any, Final

__all__ = ["CostEvent", "CostKind", "CostLog", "LayerFootprint"]


class CostKind:
    """The kinds of call the library records. Kinds are plain strings, so a technique may record kinds of its own."""

    EXACT: Final = "exact"
    """The unmodified model over one or more tokens."""
    DRAFT: Final = "draft"
    """One draft step."""
    DRAFT_WASTED: Final = "draft_wasted"
    """A draft step whose proposal a draft policy discarded."""
    REPLAY: Final = "replay"
    """Exact tokens run again to bring the recurrent states of a hybrid model in line with the committed tokens."""
    DRAFT_CATCHUP: Final = "draft_catchup"
    """A separate draft network run over committed tokens it has not seen, before it drafts."""
    DRAFT_OBSERVE: Final = "draft_observe"
    """A draft module run over committed tokens to update its own cache (EAGLE-3, MTP)."""


@dataclass(frozen=True)
class LayerFootprint:
    """The layers one call runs, as a drafter declares them for the cost log.

    Attributes:
        first_layer: First decoder layer run (1-indexed).
        last_layer: Last decoder layer run (``first_layer - 1`` for none).
        module: The separate network the layers belong to, ``None`` for the decoded model (see :class:`CostEvent`).
        layers: The layers run when they are not one contiguous range.
    """

    first_layer: int
    last_layer: int
    module: str | None = None
    layers: tuple[int, ...] | None = None

    @classmethod
    def of(cls, layers: Iterable[int], module: str | None = None) -> LayerFootprint:
        """The footprint of an arbitrary set of layers, stored as a range when it is one."""
        ordered = tuple(sorted(set(layers)))
        if not ordered:
            return cls(1, 0, module)
        contiguous = ordered == tuple(range(ordered[0], ordered[-1] + 1))
        return cls(ordered[0], ordered[-1], module, None if contiguous else ordered)

    @classmethod
    def none(cls, module: str | None = None) -> LayerFootprint:
        """No decoder layers (a drafter without layers of its own, or one that runs only heads)."""
        return cls(1, 0, module)


@dataclass(frozen=True)
class CostEvent:
    """One model call.

    Attributes:
        kind: What the call did, one of the :class:`CostKind` constants or a technique's own kind.
        tokens: Tokens processed in parallel.
        position: Position of the first token.
        first_layer: First decoder layer run (1-indexed).
        last_layer: Last decoder layer run.
        step: Index of a draft step within its cycle, counted from 0 for every row (``None`` for other calls, and
            for draft steps a drafter records without :meth:`~economize.decoding.DraftRequest.extend` and without a
            step). The loop uses it to drop the steps a row made after it finished.
        module: ``None`` when the layers are the decoded model's own. Otherwise the name of the separate network
            that ran (``"draft_model"``, ``"eagle3"``, ``"mtp"``, ``"medusa"``), and the layers are that
            network's, counted from 1. A network without decoder layers (Medusa heads) records an empty range.
        layers: The decoder layers run when they are not one contiguous range (a draft that skips layers), in
            which case ``first_layer`` and ``last_layer`` are its smallest and largest entries.
    """

    kind: str
    tokens: int
    position: int
    first_layer: int
    last_layer: int
    step: int | None = None
    module: str | None = None
    layers: tuple[int, ...] | None = None

    @property
    def layer_indices(self) -> tuple[int, ...]:
        """The decoder layers the call ran, in order."""
        if self.layers is not None:
            return self.layers
        return tuple(range(max(1, self.first_layer), self.last_layer + 1))

    @property
    def layer_tokens(self) -> int:
        """Decoder-layer applications, the unit in which the cost of a call is compared."""
        return self.tokens * len(self.layer_indices)


@dataclass
class CostLog:
    """Every model call of one generation, in order."""

    events: list[CostEvent] = field(default_factory=list)

    def record(
        self, kind: str, tokens: int, position: int, first_layer: int, last_layer: int, step: int | None = None, *,
        module: str | None = None, layers: tuple[int, ...] | None = None,
    ) -> None:
        """Append one call (see :class:`CostEvent` for the fields)."""
        self.events.append(CostEvent(kind, tokens, position, first_layer, last_layer, step, module, layers))

    def record_footprint(
        self, kind: str, tokens: int, position: int, footprint: LayerFootprint, step: int | None = None
    ) -> None:
        """Append one call over the layers of ``footprint``."""
        self.record(
            kind, tokens, position, footprint.first_layer, footprint.last_layer, step,
            module=footprint.module, layers=footprint.layers,
        )

    def mark_last_wasted(self) -> None:
        """Mark the last event as wasted (its proposal was discarded)."""
        self.events[-1] = replace(self.events[-1], kind=f"{self.events[-1].kind}_wasted")

    def truncate(self, length: int) -> None:
        """Keep the first ``length`` events (drops calls a finished row made while the rest of the batch ran on)."""
        del self.events[length:]

    def truncate_after_step(self, step: int, start: int = 0) -> None:
        """Drop the events that follow draft step ``step`` among the events from index ``start`` on (the cycle's
        first event). Keeps every event when that step was not recorded.

        A drafter may log other calls besides its draft steps (the draft model's catch-up), so the cut cannot be
        counted in events.
        """
        for index in range(start, len(self.events)):
            event = self.events[index]
            if event.kind == CostKind.DRAFT and event.step == step:
                del self.events[index + 1 :]
                return

    def count(self, kind: str) -> int:
        """Number of events of ``kind``."""
        return sum(1 for e in self.events if e.kind == kind)

    def summary(self, num_layers: int, prompt_tokens: int, generated: int) -> dict[str, Any]:
        """Aggregate counts, including layer applications relative to plain decoding of the same output.

        Plain decoding runs all ``L`` layers once over the prompt and once over every generated token but the last,
        ``(prompt_tokens + generated - 1) * L`` layer applications in total. ``layer_tokens`` counts the decoded
        model's layers only. Layers of separate networks have other sizes, so they are counted per network in
        ``module_layer_tokens`` (price both with :class:`~economize.bench.CostModel`).
        """
        layer_tokens = sum(e.layer_tokens for e in self.events if e.module is None)
        modules: Counter[str] = Counter()
        for event in self.events:
            if event.module is not None:
                modules[event.module] += event.layer_tokens
        plain = max(1, (prompt_tokens + generated - 1) * num_layers)
        return {
            "exact_calls": self.count(CostKind.EXACT),
            "exact_tokens": sum(e.tokens for e in self.events if e.kind == CostKind.EXACT),
            "draft_steps": self.count(CostKind.DRAFT),
            "wasted_draft_steps": self.count(CostKind.DRAFT_WASTED),
            "replay_calls": self.count(CostKind.REPLAY),
            "layer_tokens": layer_tokens,
            "module_layer_tokens": dict(modules),
            "sequential_calls": len(self.events),
            "relative_layer_tokens": layer_tokens / plain,
        }

    def as_tuples(self, kinds: Mapping[str, str] | None = None) -> list[tuple[str, int, int]]:
        """The ``(kind, tokens, position)`` of every event, with kinds renamed by ``kinds`` (kinds it does not name
        keep theirs). Layers and modules are dropped. Used to compare economize's log with another decoder's."""
        kinds = kinds or {}
        return [(kinds.get(e.kind, e.kind), e.tokens, e.position) for e in self.events]

    def __len__(self) -> int:
        return len(self.events)
