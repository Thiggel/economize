"""User-facing decoding options."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, fields
from typing import Any

from ..errors import ConfigError
from ..validation import check_positive, check_unit_interval

__all__ = ["DecodeConfig"]


@dataclass(frozen=True)
class DecodeConfig:
    """How a model with a drafting technique generates text.

    One *cycle* runs the unmodified model once, in parallel, over all tokens that have not been through an exact
    pass yet. This pass also yields one exact next token. Then the drafter proposes up to ``cycle_length - 1``
    further tokens cheaply.

    Attributes:
        cycle_length: Tokens per cycle (``N``). ``1`` is ordinary decoding.
        lossless: Verify drafts against the exact pass and keep only those the model itself would have produced
            (speculative decoding, output identical to greedy decoding). With ``False`` every draft is kept and
            the exact pass only refreshes the cache (lossy, cheaper). ``None`` (the default) means ``True`` unless
            ``verifier`` names another verifier; an explicit ``True`` together with a verifier that commits drafts
            unverified (``"accept_all"``) raises :class:`~economize.errors.ConfigError` when the decoder is built.
        min_confidence: Stop drafting early when the drafter's top probability falls below this value (the first
            draft of a cycle is always made).
        verifier: Advanced. Name of a registered verifier, used instead of the one ``lossless`` picks.
        draft_policy: Advanced. Name of a registered draft-length policy, overriding ``min_confidence``.
        verifier_options: Advanced. Keyword options of the verifier, e.g. ``{"threshold": 0.09, "alpha": 0.3}``
            for ``verifier="typical"``. Each verifier documents and checks its own options.
        cuda_graphs: Record the draft steps and the exact passes of a chain decoder as CUDA graphs and replay them
            (CUDA only). Configurations a graph cannot express (draft trees, dynamic-depth layer policies, KV-cache
            techniques, recurrent layers) fall back to eager execution per call.
        sync_interval: Ordinary decoding (``cycle_length=1``) reads the new tokens back to the host and checks the
            stop conditions every ``sync_interval`` steps, so the GPU never waits for the host in between. A row
            that stopped earlier is cut at its stop token: tokens and cost logs are those of reading every step,
            and a stopped row costs at most ``sync_interval - 1`` extra steps of compute. A sampler shared by
            several rows may draw different random numbers after one of them stops (the distribution is the same).
            Verify observers need every step and imply ``1``.
        record_confidence: Log the drafter's top probability of every drafted token in
            :attr:`DecodeResult.confidence <economize.decoding.DecodeResult.confidence>` (for calibration metrics).
            It costs one softmax over the draft logits per draft step, kept on the device and read back once per
            row when the row finishes; tokens and acceptance are unchanged.
        record_distributions: Compare the drafter's distribution with the verifier's at every drafted token of a
            chain draft and log ``[top-1 agreement, KL(verifier || drafter), TV]`` in
            :attr:`DecodeResult.match <economize.decoding.DecodeResult.match>`. Both distributions are a sampling
            selector's (temperature and truncation applied) when it exposes them, the plain softmax otherwise; the
            drafter's are the ones it sampled from when the verifier needs them. It keeps every draft step's logits
            on the device until the exact pass; tokens and acceptance are unchanged.
        draft_loop_graph: With ``cuda_graphs``, record a whole chain proposal of ``k`` draft steps as one CUDA graph
            (one graph per ``k``) for drafters that support it, with the greedy choice of every step computed by a
            fused LM-head kernel that never writes the logits. It applies to greedy drafting at a fixed draft length
            (no ``min_confidence``, no ``record_distributions``) where every row drafts the same number of tokens;
            other proposals run step by step. Decoding stays lossless; a draft differs from the step-by-step one
            only where two logits lie within the rounding of the head's matrix product.
        draft_policy_options: Advanced. Keyword options of the draft policy, e.g.
            ``{"depth": 6, "topk": 8, "total_tokens": 60}`` for ``draft_policy="dynamic_tree"``. Each policy
            documents and checks its own options. Lists (also nested ones, as JSON gives them) are stored as tuples,
            so the configuration stays hashable.
        prefill_chunk: Run the prefill of prompts longer than this many tokens in consecutive chunks of this size
            (activations and attention scores then scale with the chunk, not the prompt). The tokens and logits are
            those of one pass up to rounding. It needs prompts of equal length, a drafter that reads only the hidden
            states of the last accepted tokens (:attr:`Drafter.streams_prompt
            <economize.decoding.Drafter.streams_prompt>`), no layer policy, no verify observer and a model without
            recurrent layers. A KV-cache technique sees the chunks as one pass over the whole prompt.
        window_cache: With ``prefill_chunk``, keep the sliding-window layers in a ring of slightly more than their
            window (:class:`~economize.runtime.RingSpec`) instead of a slot per position, so long prompts fit on
            fewer devices. Chains only (no draft trees); the cache must declare ``supports_ring``. Tokens are those
            of the full cache. A KV-cache technique then acts on the full-length layers only, since the ring already
            bounds the sliding ones.
    """

    cycle_length: int = 4
    lossless: bool | None = None
    min_confidence: float | None = None
    verifier: str | None = None
    draft_policy: str | None = None
    verifier_options: Mapping[str, Any] = field(default_factory=dict, hash=False)
    cuda_graphs: bool = False
    sync_interval: int = 8
    record_confidence: bool = False
    record_distributions: bool = False
    draft_loop_graph: bool = False
    draft_policy_options: Mapping[str, Any] = field(default_factory=dict, hash=False)
    prefill_chunk: int | None = None
    window_cache: bool = False

    def __post_init__(self) -> None:
        check_positive("cycle_length", self.cycle_length)
        check_positive("sync_interval", self.sync_interval)
        if self.prefill_chunk is not None:
            check_positive("prefill_chunk", self.prefill_chunk)
        if self.window_cache and self.prefill_chunk is None:
            raise ConfigError("window_cache needs prefill_chunk (an unchunked prefill writes the whole prompt at once)")
        if self.min_confidence is not None:
            check_unit_interval("min_confidence", self.min_confidence, closed_high=True)
        if not isinstance(self.verifier_options, Mapping):
            raise ConfigError(f"verifier_options must be a mapping, got {type(self.verifier_options).__name__}")
        object.__setattr__(self, "verifier_options", dict(self.verifier_options))
        if not isinstance(self.draft_policy_options, Mapping):
            raise ConfigError(
                f"draft_policy_options must be a mapping, got {type(self.draft_policy_options).__name__}")
        object.__setattr__(self, "draft_policy_options",
                           {name: _frozen(value) for name, value in self.draft_policy_options.items()})

    def __hash__(self) -> int:
        options = ("verifier_options", "draft_policy_options")
        return hash((*(getattr(self, f.name) for f in fields(self) if f.name not in options),
                     *(tuple(sorted(getattr(self, name).items())) for name in options)))

    @property
    def verifier_name(self) -> str:
        return self.verifier or ("accept_all" if self.lossless is False else "greedy")

    @property
    def draft_policy_name(self) -> str:
        return self.draft_policy or ("fixed" if self.min_confidence is None else "confidence")

    def to_dict(self) -> dict[str, Any]:
        data = {f.name: getattr(self, f.name) for f in fields(self)}
        data["verifier_options"] = dict(self.verifier_options)
        data["draft_policy_options"] = dict(self.draft_policy_options)
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DecodeConfig:
        unknown = set(data) - set(cls.__dataclass_fields__)
        if unknown:
            raise ConfigError(f"unknown DecodeConfig fields {sorted(unknown)}")
        return cls(**data)


def _frozen(value: Any) -> Any:
    """``value`` with every list or tuple in it turned into a tuple, recursively (so it can be hashed)."""
    if isinstance(value, (list, tuple)):
        return tuple(_frozen(v) for v in value)
    return value
