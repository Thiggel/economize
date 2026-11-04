"""Analytic cost of a generation: floating-point operations and memory traffic per model call, and a roofline time.

The decoding loop records every model call as a :class:`~economize.decoding.CostEvent` (kind, tokens, position, the
layers run and the module that ran them). This module prices those events for a concrete architecture::

    shape = ModelShape.from_config(model.config)
    costs = CostModel(shape)
    report = costs.log(result.cost)                      # every call of one generation
    costs.relative(result.cost, prompt_tokens, generated, "a100-80gb")
    # {"flops": 0.71, "bytes": 0.43, "seconds": 0.45, "speedup": 2.2, ...}
    CostModel(shape, draft=CallSpec(shape=draft_shape))  # a separate draft model (module "draft_model")
    CostModel(shape, modules={"eagle3": CallSpec(shape=eagle_shape)})  # any other draft network

Counting rules, per call of ``T`` tokens at positions ``p .. p+T-1`` over the decoder layers it ran (any subset):

* Linear layers cost ``2`` FLOPs per weight per token (a mixture of experts counts its router, its shared expert and
  ``k`` routed experts). Attention costs ``4 * heads * head_dim`` FLOPs per token and attended key, and token ``i``
  attends to ``p + i + 1`` keys (capped at the window in a sliding-window layer, and to the keys of its own chunk in
  a chunked-attention layer). Norms, rotary embeddings and activations are elementwise and left out.
* Memory traffic is the weights of the layers once per call (for experts, the expected number of distinct experts
  that ``T`` tokens route to under uniform routing), the key/value cache read once per call (the union of the keys
  the call's tokens attend to) and the new keys and values written. Activations stay in caches and are left out.
* The LM head costs ``2 * hidden * rows`` FLOPs per token that gets logits and its weights once per call. Which
  tokens get logits depends on the kind of call (:class:`CallSpec`); the prompt's exact pass computes logits for
  its last token only, as plain decoding does.
* A call takes ``max(flops / peak, bytes / bandwidth) + overhead`` seconds on a :class:`Hardware` (a roofline per
  call). Kernel launches and the host are not modelled beyond the constant overhead; measured times come from
  :mod:`economize.bench.timing`.

A hybrid stack whose layers hold different parts (Nemotron-H: one Mamba-2, attention, MLP or MoE block per layer)
names them per layer in :attr:`ModelShape.layer_parts`. A recurrent or linear-attention mixer is counted with its
weights only (:attr:`ModelShape.mixer_params`: projections and convolution); its scan and its state traffic are left
out. Multi-head latent attention is counted as grouped-query attention with the config's head counts, so the model is
approximate for those architectures.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field, replace
from typing import Any, Literal

from ..arch.config import config_value, text_config
from ..decoding.cost import CostEvent, CostKind, CostLog
from ..errors import ConfigError
from ..registry import Registry
from .shape_readers import SHAPE_READERS, ShapeReader, family_fields
from .shape_readers import first_set as _first

__all__ = [
    "DRAFT_MODEL",
    "HARDWARE",
    "SHAPE_READERS",
    "CallCost",
    "CallSpec",
    "CostModel",
    "CostReport",
    "Hardware",
    "ModelShape",
    "ShapeReader",
    "get_hardware",
]

_SLIDING = "sliding_attention"
_FULL = "full_attention"
_CHUNKED = "chunked_attention"
_ATTENTION = (_FULL, _SLIDING, _CHUNKED)
#: The parts a layer may hold (:attr:`ModelShape.layer_parts`), and those of a standard decoder layer.
_LAYER_PARTS = frozenset({"attention", "mlp", "mixer"})
_STANDARD_PARTS = frozenset({"attention", "mlp"})
#: The module name a separate draft model records (``CostModel(draft=...)`` prices it).
DRAFT_MODEL = "draft_model"


@dataclass(frozen=True)
class ModelShape:
    """The sizes of a decoder-only transformer that its cost depends on.

    Attributes:
        num_layers: Decoder layers ``L``.
        hidden: Width of the residual stream ``d``.
        heads: Query heads.
        kv_heads: Key/value heads.
        head_dim: Width of one head.
        intermediate: Width of a dense MLP.
        vocab: Rows of the LM head.
        gated: Whether the MLP is gated (three matrices, as SwiGLU) or not (two).
        experts: Routed experts per mixture-of-experts layer (``0`` for a dense model).
        active_experts: Experts each token routes to.
        expert_intermediate: Width of one routed expert.
        shared_intermediate: Total width of the shared experts (``0`` if none).
        shared_gate: Whether a sigmoid gate (``hidden -> 1``) scales the shared experts (Qwen2-MoE) or not
            (DeepSeek, Llama-4).
        dense_layers: Layers (1-indexed) with a dense MLP in a mixture-of-experts model.
        layer_types: Attention type per layer (``"full_attention"``, ``"sliding_attention"``,
            ``"chunked_attention"`` or another type that is counted without attention over the context); empty
            means full attention everywhere.
        sliding_window: Keys a sliding-window layer attends to (including the query's own).
        chunk_size: Tokens per chunk of a chunked-attention layer (Llama-4): a query attends to the earlier keys of
            its own chunk.
        layer_parts: The parts of each layer, out of ``"attention"`` (the query, key, value and output projections),
            ``"mlp"`` (a dense MLP, or the mixture of experts of an MoE layer) and ``"mixer"`` (a token mixer of
            ``mixer_params`` weights, such as a Mamba-2 block); empty means attention and MLP in every layer.
            Attention over the context follows ``layer_types``.
        mixer_params: Weight entries of one ``"mixer"`` part.
        expert_hidden: Width the routed experts read and write when a latent MoE projects the hidden state down
            before them and back up after them (Nemotron-H's ``moe_latent_size``); ``0``: no projection, the experts
            read the hidden state.
    """

    num_layers: int
    hidden: int
    heads: int
    kv_heads: int
    head_dim: int
    intermediate: int
    vocab: int
    gated: bool = True
    experts: int = 0
    active_experts: int = 0
    expert_intermediate: int = 0
    shared_intermediate: int = 0
    shared_gate: bool = True
    dense_layers: frozenset[int] = field(default_factory=frozenset)
    layer_types: tuple[str, ...] = ()
    sliding_window: int | None = None
    chunk_size: int | None = None
    layer_parts: tuple[frozenset[str], ...] = ()
    mixer_params: int = 0
    expert_hidden: int = 0

    def __post_init__(self) -> None:
        if min(self.num_layers, self.hidden, self.heads, self.kv_heads, self.head_dim, self.vocab) < 1:
            raise ConfigError(f"every size of a model shape must be positive: {self}")
        if self.experts and not 0 < self.active_experts <= self.experts:
            raise ConfigError(f"need 0 < active_experts <= experts, got {self.active_experts} of {self.experts}")
        if self.layer_types and len(self.layer_types) != self.num_layers:
            raise ConfigError(f"{len(self.layer_types)} layer types for {self.num_layers} layers")
        if self.layer_parts and len(self.layer_parts) != self.num_layers:
            raise ConfigError(f"{len(self.layer_parts)} layer parts for {self.num_layers} layers")
        unknown = sorted({part for parts in self.layer_parts for part in parts} - _LAYER_PARTS)
        if unknown:
            raise ConfigError(f"unknown layer parts {unknown}; a layer holds some of {sorted(_LAYER_PARTS)}")

    @classmethod
    def from_config(cls, config: Any, **overrides: Any) -> ModelShape:
        """Read the shape from a ``transformers`` config (the text config of a multimodal one).

        The generic reading knows the common config spellings; the family's corrections then fix what it spells
        differently (the adapter's ``shape_fields``, then the reader registered in :data:`SHAPE_READERS` for the
        config's ``model_type``, see :mod:`.shape_readers`), and ``overrides`` replace fields that the config does
        not describe well.
        """
        config = text_config(config)
        hidden = config_value(config, "hidden_size")
        heads = config_value(config, "num_attention_heads")
        layers = config_value(config, "num_hidden_layers")
        intermediate = _first(config, "intermediate_size", "ffn_dim", "n_inner") or 4 * hidden
        experts = _first(config, "num_experts", "num_local_experts", "n_routed_experts") or 0
        expert_intermediate = _first(config, "moe_intermediate_size") or intermediate
        shared, shared_gate = _first(config, "shared_expert_intermediate_size") or 0, True
        if not shared and (shared_experts := _first(config, "n_shared_experts")):
            shared, shared_gate = shared_experts * expert_intermediate, False
        dense: set[int] = set()
        if experts:
            moe_layers = config_value(config, "moe_layers")
            if moe_layers is not None:  # 0-indexed MoE layers (Llama-4)
                dense.update(i + 1 for i in range(layers) if i not in set(moe_layers))
            dense.update(i + 1 for i in _first(config, "mlp_only_layers") or ())
            step = _first(config, "decoder_sparse_step") or 1
            dense.update(i + 1 for i in range(layers) if (i + 1) % step)
            dense.update(range(1, (_first(config, "first_k_dense_replace") or 0) + 1))
        layer_types = tuple(_first(config, "layer_types") or ())
        window = _first(config, "sliding_window")
        if not layer_types and window and config_value(config, "use_sliding_window", True):
            layer_types = (_SLIDING,) * layers
        fields: dict[str, Any] = dict(
            num_layers=layers, hidden=hidden, heads=heads,
            kv_heads=_first(config, "num_key_value_heads") or heads,
            head_dim=_first(config, "head_dim") or hidden // heads,
            intermediate=intermediate, vocab=config_value(config, "vocab_size"),
            gated=True,
            experts=experts, active_experts=_first(config, "num_experts_per_tok") or 0 if experts else 0,
            expert_intermediate=expert_intermediate if experts else 0, shared_intermediate=shared,
            shared_gate=shared_gate,
            dense_layers=frozenset(dense), layer_types=layer_types,
            sliding_window=window if _SLIDING in layer_types else None,
            chunk_size=_first(config, "attention_chunk_size") if _CHUNKED in layer_types else None,
        )
        fields = family_fields(config, fields)
        fields.update(overrides)
        return cls(**fields)

    # ---- weights ---------------------------------------------------------------------------------------------
    @property
    def attention_params(self) -> int:
        """Weights of the query, key, value and output projections of one layer."""
        return 2 * self.hidden * self.head_dim * (self.heads + self.kv_heads)

    def _mlp(self, width: int, inner: int | None = None) -> int:
        return (3 if self.gated else 2) * (inner or self.hidden) * width

    def parts(self, layer: int) -> frozenset[str]:
        """The parts of ``layer`` (1-indexed), see :attr:`layer_parts`."""
        return self.layer_parts[layer - 1] if self.layer_parts else _STANDARD_PARTS

    def is_moe(self, layer: int) -> bool:
        """Whether ``layer`` (1-indexed) has a mixture-of-experts MLP."""
        return bool(self.experts) and layer not in self.dense_layers and "mlp" in self.parts(layer)

    def _expert(self) -> int:
        """Weights of one routed expert."""
        return self._mlp(self.expert_intermediate, self.expert_hidden)

    def _moe_fixed(self) -> int:
        """Router, shared experts and their gate, and the latent projections."""
        shared = self._mlp(self.shared_intermediate) + self.hidden * self.shared_gate if self.shared_intermediate else 0
        return self.hidden * self.experts + shared + 2 * self.hidden * self.expert_hidden

    def _fixed(self, layer: int) -> int:
        """Weights of ``layer`` that every token reads: its attention and mixer, and the fixed part of its MoE."""
        parts = self.parts(layer)
        fixed = self.attention_params * ("attention" in parts) + self.mixer_params * ("mixer" in parts)
        return fixed + (self._moe_fixed() if self.is_moe(layer) else 0)

    def layer_params(self, layer: int = 1) -> int:
        """Stored weight-matrix entries of ``layer`` (1-indexed); biases and norms are left out."""
        if self.is_moe(layer):
            return self._fixed(layer) + self.experts * self._expert()
        return self._fixed(layer) + (self._mlp(self.intermediate) if "mlp" in self.parts(layer) else 0)

    def active_params(self, layer: int = 1) -> int:
        """Weight-matrix entries one token is multiplied with in ``layer``."""
        if not self.is_moe(layer):
            return self.layer_params(layer)
        return self._fixed(layer) + self.active_experts * self._expert()

    def touched_params(self, layer: int, tokens: int) -> float:
        """Expected weight entries that ``tokens`` tokens read in ``layer`` (distinct experts under uniform
        routing)."""
        if tokens <= 0:
            return 0.0
        if not self.is_moe(layer):
            return float(self.layer_params(layer))
        distinct = self.experts * (1 - (1 - self.active_experts / self.experts) ** tokens)
        return self._fixed(layer) + distinct * self._expert()

    @property
    def head_params(self) -> int:
        """Weights of the LM head."""
        return self.hidden * self.vocab

    @property
    def total_params(self) -> int:
        """Weight matrices of all layers plus the head (a tied embedding counts once)."""
        return sum(self.layer_params(layer) for layer in range(1, self.num_layers + 1)) + self.head_params

    # ---- attention -------------------------------------------------------------------------------------------
    def layer_type(self, layer: int) -> str:
        return self.layer_types[layer - 1] if self.layer_types else _FULL

    def attended_keys(self, layer: int, position: int, tokens: int) -> int:
        """Keys attended in total by ``tokens`` tokens starting at ``position`` (token ``i`` sees ``position+i+1``,
        capped at the window of a sliding layer, ``(position+i) % chunk_size + 1`` in a chunked layer; ``0`` for a
        layer without attention over the context)."""
        kind = self.layer_type(layer)
        first, last = position + 1, position + tokens
        if kind == _CHUNKED and self.chunk_size:
            return _chunked_keys(position + tokens, self.chunk_size) - _chunked_keys(position, self.chunk_size)
        window = self.sliding_window
        if kind in (_FULL, _CHUNKED) or (kind == _SLIDING and not window):
            return (first + last) * tokens // 2
        if kind != _SLIDING or window is None:
            return 0
        if last <= window:
            return (first + last) * tokens // 2
        if first > window:
            return window * tokens
        return (first + window) * (window - first + 1) // 2 + window * (last - window)

    def cached_keys(self, layer: int, end: int, tokens: int = 1) -> int:
        """Key/value entries that a call of ``tokens`` tokens ending at position ``end`` reads from the cache of
        ``layer``: the union of the keys its tokens attend to (the window of the first token reaches back
        ``window - 1`` keys, the chunk of the first token starts at a multiple of the chunk size)."""
        kind = self.layer_type(layer)
        if kind == _SLIDING and self.sliding_window:
            return min(end, self.sliding_window + tokens - 1)
        if kind == _CHUNKED and self.chunk_size:
            return end - max(end - tokens, 0) // self.chunk_size * self.chunk_size
        return end if kind in _ATTENTION else 0


def _chunked_keys(end: int, chunk: int) -> int:
    """Keys attended in total by the tokens at positions ``0 .. end-1`` of a chunked-attention layer."""
    rest = end % chunk
    return end // chunk * (chunk * (chunk + 1) // 2) + rest * (rest + 1) // 2


@dataclass(frozen=True)
class Hardware:
    """Peak dense throughput and memory bandwidth of an accelerator.

    Attributes:
        name: Display name.
        flops: Peak dense FLOP/s at the weights' precision (bf16 tensor cores for the presets).
        bandwidth: Memory bandwidth in bytes/s.
        call_overhead: Seconds added to every call (kernel launches, host work).
    """

    name: str
    flops: float
    bandwidth: float
    call_overhead: float = 0.0

    @property
    def ridge(self) -> float:
        """FLOPs per byte above which a call is compute bound."""
        return self.flops / self.bandwidth


#: Accelerator presets (extensible by registration or the ``economize.hardware`` entry-point group).
HARDWARE: Registry[Hardware] = Registry("hardware", "economize.hardware")
for _name, _tflops, _gbps in (
    ("a40", 149.7, 696), ("a100-40gb", 312, 1555), ("a100-80gb", 312, 2039), ("h100-sxm", 989, 3350),
    ("h100-pcie", 756, 2000), ("l40", 181, 864), ("l40s", 362, 864), ("rtx-a6000", 154.8, 768),
):
    HARDWARE.register(_name, Hardware(_name, _tflops * 1e12, _gbps * 1e9))


def get_hardware(hardware: Hardware | str) -> Hardware:
    """A :class:`Hardware` itself, or the preset of that name."""
    return hardware if isinstance(hardware, Hardware) else HARDWARE.get(hardware)


@dataclass(frozen=True)
class CallCost:
    """FLOPs and bytes moved by one or more calls."""

    flops: float = 0.0
    bytes: float = 0.0
    calls: int = 1

    def __add__(self, other: CallCost) -> CallCost:
        return CallCost(self.flops + other.flops, self.bytes + other.bytes, self.calls + other.calls)

    @property
    def intensity(self) -> float:
        """FLOPs per byte."""
        return self.flops / self.bytes if self.bytes else float("inf")

    def seconds(self, hardware: Hardware | str) -> float:
        """Roofline time of one call with these totals (sum per-call times for several calls, as
        :meth:`CostReport.seconds` does)."""
        hw = get_hardware(hardware)
        return max(self.flops / hw.flops, self.bytes / hw.bandwidth) + self.calls * hw.call_overhead


@dataclass(frozen=True)
class CallSpec:
    """How the calls of one kind are priced.

    Attributes:
        shape: Whose decoder layers the call's layer range refers to (``None``: the target model's).
        logits: Which tokens get logits: ``"all"``, ``"last"`` or ``"none"``.
        head_rows: Rows of the LM head applied (``None``: the shape's vocabulary; fewer for a reduced draft
            vocabulary).
        extra_params: Weight entries applied per token besides the decoder layers and the head (a readout
            adapter, Medusa heads, an EAGLE layer's fusion).
    """

    shape: ModelShape | None = None
    logits: Literal["all", "last", "none"] = "all"
    head_rows: int | None = None
    extra_params: int = 0

    def __post_init__(self) -> None:
        if self.logits not in ("all", "last", "none"):
            raise ConfigError(f"logits must be 'all', 'last' or 'none', got {self.logits!r}")


def _catchup(spec: CallSpec) -> CallSpec:
    """A draft network run over committed tokens: its layers only, without logits or extra weights."""
    return replace(spec, logits="none", extra_params=0)


@dataclass(frozen=True)
class CostReport:
    """The priced calls of one generation."""

    calls: tuple[tuple[str, CallCost], ...]

    @property
    def flops(self) -> float:
        return sum(cost.flops for _, cost in self.calls)

    @property
    def bytes(self) -> float:
        return sum(cost.bytes for _, cost in self.calls)

    def seconds(self, hardware: Hardware | str) -> float:
        """Sum of the per-call roofline times."""
        hw = get_hardware(hardware)
        return sum(cost.seconds(hw) for _, cost in self.calls)

    def by_kind(self) -> dict[str, CallCost]:
        """Totals per kind of call."""
        totals: dict[str, CallCost] = {}
        for kind, cost in self.calls:
            totals[kind] = totals[kind] + cost if kind in totals else cost
        return totals

    def to_dict(self, hardware: Hardware | str | None = None) -> dict[str, Any]:
        result: dict[str, Any] = {"calls": len(self.calls), "flops": self.flops, "bytes": self.bytes,
                                  "by_kind": {k: {"calls": c.calls, "flops": c.flops, "bytes": c.bytes}
                                              for k, c in self.by_kind().items()}}
        if hardware is not None:
            result["seconds"] = self.seconds(hardware)
        return result


class CostModel:
    """Prices the cost log of a generation for one architecture.

    An event of the decoded model (``module`` ``None``) is priced by its kind through ``specs``. An event of a
    separate network (``module`` set, such as ``"draft_model"``, ``"eagle3"`` or ``"mtp"``) is priced by
    that network's entry in ``modules``: its draft steps as the entry says, and its other calls (catching up on or
    observing committed tokens) without logits. Either way the call runs exactly the layers in
    :attr:`~economize.decoding.CostEvent.layer_indices`, which need not be contiguous.

    Args:
        target: The shape of the model being decoded.
        draft: How draft steps are priced. It prices the ``"draft"`` and ``"draft_wasted"`` events of the decoded
            model (default: the target's layers in the event's range, logits over the full vocabulary), and is the
            default of ``modules["draft_model"]``, so a separate draft model passes its own shape here.
        modules: Pricing of the calls of separate networks, by module name. ``shape`` is the network's own (its
            layers count from 1); ``None`` uses the target's shape, for modules that run no layers of their own
            (Medusa heads, given as ``extra_params``).
        specs: Pricing of further kinds of events of the decoded model, or replacements of the defaults.
        exact_logits: Which tokens of an exact call (other than the prompt's) get logits: ``"all"`` when the call
            verifies drafts, ``"last"`` when it only refreshes the cache as in lossy decoding.
        weight_bytes: Bytes per weight entry (``2`` for bf16, ``1`` for fp8 or int8, ``0.5`` for int4).
        kv_bytes: Bytes per cached key or value entry.

    Raises:
        ConfigError: (when pricing) an event names a module without an entry in ``modules``.
    """

    #: Kinds whose calls produce draft logits; a module's other calls run without them.
    DRAFT_KINDS: frozenset[str] = frozenset({CostKind.DRAFT, CostKind.DRAFT_WASTED})

    def __init__(self, target: ModelShape, *, draft: CallSpec | None = None,
                 modules: Mapping[str, CallSpec] | None = None, specs: Mapping[str, CallSpec] | None = None,
                 exact_logits: Literal["all", "last"] = "all", weight_bytes: float = 2.0,
                 kv_bytes: float = 2.0) -> None:
        if exact_logits not in ("all", "last"):
            raise ConfigError(f"exact_logits must be 'all' or 'last', got {exact_logits!r}")
        self.target, self.weight_bytes, self.kv_bytes = target, weight_bytes, kv_bytes
        self.modules: dict[str, CallSpec] = {} if draft is None else {DRAFT_MODEL: draft}
        self.modules.update(modules or {})
        draft = draft or CallSpec()
        self.specs: dict[str, CallSpec] = {
            CostKind.EXACT: CallSpec(logits=exact_logits),
            CostKind.DRAFT: draft,
            CostKind.DRAFT_WASTED: draft,
            CostKind.DRAFT_CATCHUP: _catchup(draft),
            CostKind.REPLAY: CallSpec(logits="none"),
        }
        self.specs.update(specs or {})

    def spec(self, kind: str, module: str | None = None) -> CallSpec:
        """The pricing of a call of ``kind`` by ``module`` (``None``: the decoded model, where an unknown kind runs
        the target's layers without logits).

        Raises:
            ConfigError: ``module`` has no entry in :attr:`modules`.
        """
        if module is None:
            return self.specs.get(kind, CallSpec(logits="none"))
        if module not in self.modules:
            raise ConfigError(
                f"the cost log has {kind!r} calls of module {module!r}, which CostModel cannot price; pass "
                f"modules={{{module!r}: CallSpec(...)}} (known modules: {sorted(self.modules)})"
            )
        spec = self.modules[module]
        if kind in self.DRAFT_KINDS:
            return spec
        return _catchup(spec) if kind == CostKind.DRAFT_CATCHUP else replace(spec, logits="none")

    def event(self, event: CostEvent) -> CallCost:
        """FLOPs and bytes of one call."""
        spec = self.spec(event.kind, event.module)
        shape = spec.shape or self.target
        layers = event.layer_indices
        bad = [layer for layer in layers if not 1 <= layer <= shape.num_layers]
        if bad:
            where = f"module {event.module!r}" if event.module is not None else "the decoded model"
            edge = bad[-1] if bad[-1] > shape.num_layers else bad[0]  # the farthest layer outside the model
            raise ConfigError(f"{event.kind} event of {where} runs layer {edge} of a {shape.num_layers}-layer model")
        tokens, position = event.tokens, event.position
        flops = bytes_ = 0.0
        kv_width = 2 * shape.kv_heads * shape.head_dim
        for layer in layers:
            keys = shape.attended_keys(layer, position, tokens)
            flops += 2 * shape.active_params(layer) * tokens + 4 * shape.heads * shape.head_dim * keys
            bytes_ += shape.touched_params(layer, tokens) * self.weight_bytes
            if shape.layer_type(layer) in _ATTENTION:
                bytes_ += kv_width * (shape.cached_keys(layer, position + tokens, tokens) + tokens) * self.kv_bytes
        head_tokens = self._head_tokens(spec, event)
        if head_tokens:
            rows = shape.vocab if spec.head_rows is None else spec.head_rows
            flops += 2 * shape.hidden * rows * head_tokens
            bytes_ += shape.hidden * rows * self.weight_bytes
        if spec.extra_params and tokens:
            flops += 2 * spec.extra_params * tokens
            bytes_ += spec.extra_params * self.weight_bytes
        return CallCost(flops, bytes_)

    @staticmethod
    def _head_tokens(spec: CallSpec, event: CostEvent) -> int:
        if spec.logits == "none" or event.tokens <= 0:
            return 0
        if spec.logits == "last" or (event.kind == CostKind.EXACT and event.position == 0):
            return 1
        return event.tokens

    def log(self, log: CostLog | Iterable[CostEvent]) -> CostReport:
        """Every call of a generation, priced."""
        events = log.events if isinstance(log, CostLog) else log
        return CostReport(tuple((event.kind, self.event(event)) for event in events))

    def plain_log(self, prompt_tokens: int, generated: int) -> CostLog:
        """The calls of plain decoding: the prompt in one exact call, then one call per generated token but the
        last (the convention of :meth:`CostLog.summary`)."""
        log = CostLog()
        layers = self.target.num_layers
        log.record(CostKind.EXACT, prompt_tokens, 0, 1, layers)
        for position in range(prompt_tokens, prompt_tokens + generated - 1):
            log.record(CostKind.EXACT, 1, position, 1, layers)
        return log

    def plain(self, prompt_tokens: int, generated: int) -> CostReport:
        """Plain decoding of ``generated`` tokens after a prompt of ``prompt_tokens``, priced."""
        return self.log(self.plain_log(prompt_tokens, generated))

    def relative(self, log: CostLog | Iterable[CostEvent], prompt_tokens: int, generated: int,
                 hardware: Hardware | str | None = None) -> dict[str, float]:
        """A generation's cost relative to plain decoding of the same output.

        Returns:
            ``flops`` and ``bytes`` ratios (below 1 is cheaper) and, with ``hardware``, the ``seconds`` ratio of the
            roofline times and its inverse ``speedup``.
        """
        report, plain = self.log(log), self.plain(prompt_tokens, generated)
        result = {"flops": report.flops / plain.flops, "bytes": report.bytes / plain.bytes,
                  "calls": len(report.calls) / len(plain.calls)}
        if hardware is not None:
            seconds = report.seconds(hardware) / plain.seconds(hardware)
            result.update(seconds=seconds, speedup=1 / seconds)
        return result
