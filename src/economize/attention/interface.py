"""The ``"economize"`` attention implementation: :func:`~.decode.decode_attention` behind the ``transformers``
attention interface.

Decoding steps, verification passes and draft trees (at most :data:`MAX_QUERIES` queries per row, with an explicit
boolean mask or a single query, optionally with gpt-oss attention sinks) go to the split-K Triton kernel; everything
else falls back per call. A long query (prefill) whose mask is causal over a prefix of the buffer runs on vLLM's
FlashAttention (:mod:`.flash`, FA3 on Hopper, FA2 on Ampere / Ada) when it is installed. On Hopper and Blackwell, a
short query whose mask carries economize's prefix declaration (:func:`~.flash.declare_prefix`, attached by the chain
passes of :class:`~economize.runtime.LayerContext`) runs on FA3 / FA4 instead of the split-K kernel when its shape
favours FlashAttention (:func:`~.flash.decode_prefers_flash`: larger batches and verification windows), logged on
path ``"flash<version>"``. A prefill under a torch function mode that watches ``sdpa`` runs on FlashAttention only
when the mode is an :class:`AttentionObserver` (the score tap of ``kv_evict``), which is then told about the call.
The other calls (other
prefills, training with dropout, ``output_attentions``, CPU tensors, any other function mode) fall
back to ``transformers``' own ``sdpa`` function (with contiguous K/V where PyTorch 2.13 errs, see
:mod:`.sdpa_guard`), or to the model's eager function when the call has attention
sinks or asks for the weights (fp8 K/V upcast to the query's dtype first), so results never depend on which path ran
beyond floating-point rounding.
A fresh prompt (no cache before it) of an attention module that carries a sparse-prefill attachment
(:mod:`.prefill`, MInference's patterns) attends over the attachment's indexed keys only: on the Triton kernel of
:mod:`.minference` (path ``"kernel_<name>"``), or on ``sdpa`` with the index expanded into a mask (reason
``"<name>_<why>"``) where the kernel cannot run.
The implementation belongs to the ``sdpa`` family: economize builds the same boolean masks for it.
"""

from __future__ import annotations

import os
import sys
from collections import Counter
from typing import Any, Protocol, runtime_checkable

import torch
from torch.overrides import _get_current_function_mode_stack
from torch.utils._device import DeviceContext

from .banded import StructuredMask
from .decode import _FP8, decode_attention, out_of_resources, triton_available
from .flash import _offsets, flash_decode, flash_prefill, prefix_hint
from .packed import packed_of
from .prefill import prefill_of
from .registry import ATTENTION_KERNELS, AttentionKernel
from .sdpa_guard import sdpa_attention_forward

__all__ = [
    "FLASH_PREFILL",
    "MAX_PREFILL_MASK",
    "MAX_QUERIES",
    "NAME",
    "PACKED_FORMATS",
    "AttentionObserver",
    "economize_attention_forward",
    "install",
    "kernel_calls",
    "kernel_usage",
    "usage_summary",
]

#: Name of the implementation in ``transformers`` and in :data:`~.registry.ATTENTION_KERNELS`.
NAME = "economize"
#: Longest query (tokens per row) the kernel takes; longer ones (prefill) run on FlashAttention (:mod:`.flash`) when
#: vLLM's build is installed and the mask is causal over a prefix, else on ``sdpa``.
MAX_QUERIES = 64
#: Whether long queries try FlashAttention before ``sdpa``. Debug override ``ECONOMIZE_FLASH_PREFILL=0`` turns it
#: off.
FLASH_PREFILL = os.environ.get("ECONOMIZE_FLASH_PREFILL", "1") != "0"
#: Packed K/V formats (:mod:`.packed`) the kernel reads; the call is logged on path ``"kernel_<format>"``.
PACKED_FORMATS = frozenset({"kivi", "gathered", "quest"})

_calls = {"kernel": 0, "fallback": 0, "flash": 0}
_usage: Counter[tuple[int | None, int, str, str]] = Counter()


def kernel_calls() -> dict[str, int]:
    """How many attention calls ran on the kernel, on FlashAttention (long queries) and on the fallback since import
    (for tests and benchmarks)."""
    return dict(_calls)


def kernel_usage(reset: bool = False) -> list[dict[str, Any]]:
    """Which path each attention call took, grouped by layer, query length and fallback reason.

    Every Python-level call is counted (eager steps and CUDA-graph captures; replays run the captured path). A
    benchmark stores this list next to its timings so that a row can show which layers and shapes ran on the
    kernel and which fell back to ``sdpa``, and why.

    Args:
        reset: Clear the log after reading it.

    Returns:
        One record per ``(layer, queries, path, reason)`` with its call count, sorted by layer and query length.
    """
    rows = [
        {"layer": layer, "queries": queries, "path": path, "reason": reason, "calls": n}
        for (layer, queries, path, reason), n in _usage.items()
    ]
    rows.sort(key=lambda r: (-1 if r["layer"] is None else r["layer"], r["queries"], r["path"], r["reason"]))
    if reset:
        _usage.clear()
    return rows


def usage_summary() -> dict[str, Any]:
    """Compact form of :func:`kernel_usage`: calls per path, per fallback reason, and the query lengths per path."""
    paths: Counter[str] = Counter()
    reasons: Counter[str] = Counter()
    lengths: dict[str, set[int]] = {}
    for (_, queries, path, reason), n in _usage.items():
        paths[path] += n
        if not path.startswith(("kernel", "flash")):  # the kernel, its packed-K/V variants and FlashAttention
            reasons[reason] += n
        lengths.setdefault(path, set()).add(queries)
    return {
        "calls": dict(paths),
        "fallback_reasons": dict(reasons),
        "queries": {path: sorted(q) for path, q in lengths.items()},
    }


@runtime_checkable
class AttentionObserver(Protocol):
    """A torch function mode that watches ``sdpa`` calls and can be told about an attention call that ran elsewhere.

    A long query under such a mode runs on FlashAttention; :meth:`observe_attention` then receives the inputs
    ``sdpa`` would have seen (``[B, H_q, Q, D]`` queries, ``[B, H_kv, S, D]`` keys, the boolean mask cut to ``S``
    keys), so the mode records the same thing either way.
    """

    def observe_attention(
        self, query: torch.Tensor, key: torch.Tensor, attn_mask: torch.Tensor | None, scale: float
    ) -> None:
        """Record an attention call of ``query`` over ``key`` under ``attn_mask`` with softmax scale ``scale``."""
        ...


def _active_modes() -> list[Any]:
    return [mode for mode in _get_current_function_mode_stack() if not isinstance(mode, DeviceContext)]


def _tapped() -> bool:
    """Whether a torch function mode other than a device context is active (it may intercept ``sdpa``)."""
    return bool(_active_modes())


def _observers() -> list[AttentionObserver] | None:
    """The active function modes when every one is an :class:`AttentionObserver` (``[]`` for none), else ``None``."""
    modes = _active_modes()
    return modes if all(isinstance(mode, AttentionObserver) for mode in modes) else None


def _sdpa(*args: Any, **kwargs: Any) -> tuple[torch.Tensor, Any]:
    _calls["fallback"] += 1
    return sdpa_attention_forward(*args, **kwargs)  # contiguous K/V where PyTorch's masked kernel errs (sdpa_guard)


def _sinks(kwargs: dict[str, Any]) -> Any:
    return kwargs.get("s_aux") if kwargs.get("s_aux") is not None else kwargs.get("sinks")


def _eager_function(module: torch.nn.Module, kwargs: dict[str, Any]) -> Any:
    """The model's own ``eager_attention_forward`` when the call needs it: ``sdpa`` ignores attention sinks and
    returns no attention weights, so such calls must not fall back to it."""
    if _sinks(kwargs) is None and not kwargs.get("output_attentions", False):
        return None
    return getattr(sys.modules.get(type(module).__module__), "eager_attention_forward", None)


def _fallback(
    module: torch.nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    dropout: float,
    scaling: float | None,
    is_causal: bool | None,
    reason: str,
    kwargs: dict[str, Any],
) -> tuple[torch.Tensor, Any]:
    """Run one call on ``sdpa``, or on the model's eager function for sinks and attention weights; log it."""
    layer = getattr(module, "layer_idx", None)
    restrict = getattr(packed_of(key), "restrict_mask", None)
    if restrict is not None:  # a sparse cache (quest) hides what its kernel would not read
        restricted = restrict(query, key, attention_mask, bool(is_causal))
        if restricted is not attention_mask:
            attention_mask, is_causal = restricted, False
    if key.dtype in _FP8:  # an fp8 cache handed its buffers over (kv_fp8_fused); the fallbacks need the query's dtype
        key, value = key.to(query.dtype), value.to(query.dtype)
    eager = _eager_function(module, kwargs)
    if eager is None:
        _usage[(layer, query.shape[2], "sdpa", reason)] += 1
        return _sdpa(
            module, query, key, value, attention_mask, dropout=dropout, scaling=scaling, is_causal=is_causal, **kwargs
        )
    _usage[(layer, query.shape[2], "eager", reason)] += 1
    _calls["fallback"] += 1
    mask = attention_mask
    if mask is not None and mask.dtype == torch.bool:  # eager functions add the mask to the scores
        mask = mask[..., : key.shape[2]]
        mask = torch.zeros(mask.shape, dtype=query.dtype, device=query.device).masked_fill(
            ~mask, torch.finfo(query.dtype).min
        )
    scale = query.shape[-1] ** -0.5 if scaling is None else scaling
    return eager(module, query, key, value, mask, scaling=scale, dropout=dropout, **kwargs)


def _fallback_reason(
    module: torch.nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    attention_mask: torch.Tensor | None,
    dropout: float,
    kwargs: dict[str, Any],
) -> str | None:
    """Why this call cannot run on the kernel, or ``None`` when it can. Decided per call, never per model."""
    batch, q_heads, queries, _ = query.shape
    if not query.is_cuda:
        return "cpu"
    if not triton_available():
        return "no_triton"
    if queries > MAX_QUERIES:
        return "long_query"
    if dropout > 0.0 and module.training:
        return "dropout"
    sinks = _sinks(kwargs)
    if sinks is not None and (not torch.is_tensor(sinks) or sinks.numel() != q_heads):
        return "sinks"
    if kwargs.get("output_attentions", False):
        return "output_attentions"
    if query.dtype not in (torch.bfloat16, torch.float16, torch.float32):
        return "dtype"
    if key.dtype != query.dtype and key.dtype not in _FP8:
        return "kv_dtype"
    if q_heads % key.shape[1]:
        return "heads"
    if attention_mask is None:
        return None if queries == 1 else "no_mask"
    if attention_mask.dtype != torch.bool:
        return "float_mask"
    if attention_mask.dim() != 4:
        return "mask_rank"
    m_batch, m_heads, m_queries, m_keys = attention_mask.shape
    if m_batch not in (1, batch) or m_heads not in (1, q_heads) or m_queries not in (1, queries):
        return "mask_shape"
    if m_keys < key.shape[2]:
        return "mask_keys"
    if _tapped():
        return "tapped"
    return None


def _flash(
    module: torch.nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    dropout: float,
    scaling: float | None,
    kwargs: dict[str, Any],
) -> torch.Tensor | None:
    """A long query on FlashAttention, logged on path ``"flash<version>"``, or ``None`` (then ``sdpa`` runs it, logged
    with reason ``"long_query"``). fp8 K/V are upcast to the query's dtype first: a prefill chunk reads the cache once,
    so the copy costs far less than falling back to ``sdpa``. A packed attachment (KIVI's codes, say) is ignored, since
    the dense buffers hold the values it encodes, and a sparse one's ``restrict_mask`` is applied to the mask first
    (FlashAttention then refuses any mask that is not a causal prefix)."""
    if (dropout > 0.0 and module.training) or _sinks(kwargs) is not None or kwargs.get("output_attentions", False):
        return None
    observers = _observers()
    if observers is None:
        return None
    restrict = getattr(packed_of(key), "restrict_mask", None)
    if restrict is not None:
        attention_mask = restrict(query, key, attention_mask, False)
    if key.dtype in _FP8:  # kv_fp8_fused: FlashAttention takes the query's dtype; one upcast per prefill chunk
        key, value = key.to(query.dtype), value.to(query.dtype)
    scale = query.shape[-1] ** -0.5 if scaling is None else scaling
    out, path = flash_prefill(
        query, key, value, attention_mask, scale, softcap=kwargs.get("softcap"), window=kwargs.get("sliding_window")
    )
    if out is None:
        return None
    if observers:
        mask = attention_mask[..., : key.shape[2]] if attention_mask is not None else None
        for observer in observers:
            observer.observe_attention(query, key, mask, scale)
    _calls["flash"] += 1
    _usage[(getattr(module, "layer_idx", None), query.shape[2], path, "")] += 1
    return out


#: Largest ``B * Hq * Q * S`` a sparse-prefill fallback expands into a boolean mask; larger CPU prompts attend
#: densely.
MAX_PREFILL_MASK = 2**31


def _sparse_prefill(
    module: torch.nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    dropout: float,
    scaling: float | None,
    kwargs: dict[str, Any],
) -> tuple[torch.Tensor, Any] | None:
    """A fresh prompt through the module's sparse-prefill attachment, or ``None`` when the call attends densely
    (no attachment, a short or non-fresh query, a sliding-window layer, a CUDA-graph capture)."""
    prefill = prefill_of(module)
    batch, q_heads, queries, _ = query.shape
    if prefill is None or queries < max(2, prefill.min_queries) or kwargs.get("sliding_window") is not None:
        return None
    if query.is_cuda and torch.cuda.is_current_stream_capturing():
        return None
    if q_heads % key.shape[1] or key.shape[2] < queries:
        return None
    if attention_mask is None:
        if key.shape[2] != queries:
            return None
    elif attention_mask.dtype != torch.bool or attention_mask.dim() != 4 or attention_mask.shape[1] != 1:
        return None
    else:
        first = _offsets(attention_mask, batch, key.shape[2], None)
        if first is None or bool((first != 0).any()):
            return None  # a prompt after cached tokens, or a left-padded batch: dense
    scale = query.shape[-1] ** -0.5 if scaling is None else scaling
    softcap = kwargs.get("softcap")
    layer = getattr(module, "layer_idx", None)
    k_est = key[:, :, :queries]
    if k_est.dtype != query.dtype:
        k_est = k_est.to(query.dtype)
    index = prefill.index(query, k_est, scale)
    why = _prefill_kernel_reason(module, query, key, dropout, kwargs)
    if why is None:
        from .minference import sparse_prefill_attention

        try:
            out = sparse_prefill_attention(query, key, value, index, scale, softcap=softcap)
        except out_of_resources():
            why = "out_of_resources"
        else:
            _calls["kernel"] += 1
            _usage[(layer, queries, f"kernel_{prefill.name}", "")] += 1
            return out, None
    if batch * q_heads * queries * key.shape[2] > MAX_PREFILL_MASK:
        return None
    from .minference import mask_from_index

    mask = mask_from_index(index, key.shape[2])
    return _fallback(module, query, key, value, mask, dropout, scaling, False, f"{prefill.name}_{why}", kwargs)


def _prefill_kernel_reason(
    module: torch.nn.Module, query: torch.Tensor, key: torch.Tensor, dropout: float, kwargs: dict[str, Any]
) -> str | None:
    """Why a sparse prefill cannot run on its kernel, or ``None`` when it can."""
    if not query.is_cuda:
        return "cpu"
    if not triton_available():
        return "no_triton"
    if dropout > 0.0 and module.training:
        return "dropout"
    if _sinks(kwargs) is not None:
        return "sinks"
    if kwargs.get("output_attentions", False):
        return "output_attentions"
    if query.dtype not in (torch.bfloat16, torch.float16, torch.float32):
        return "dtype"
    if key.dtype != query.dtype and key.dtype not in _FP8:
        return "kv_dtype"
    if packed_of(key) is not None:
        return "packed"
    if _tapped():
        return "tapped"
    return None


def economize_attention_forward(
    module: torch.nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    dropout: float = 0.0,
    scaling: float | None = None,
    is_causal: bool | None = None,
    **kwargs: Any,
) -> tuple[torch.Tensor, None]:
    """``transformers`` attention function (``[B, H, Q, D]`` in, ``[B, Q, H, D]`` out) on the decode kernel.

    The choice between the kernel and the ``sdpa`` fallback is made for every call, so a prefill or a tapped exact
    pass falls back without moving the decoding steps of the same model off the kernel; :func:`kernel_usage` logs
    each choice by layer, query length and reason. A :class:`~.banded.StructuredMask` attends on ``sdpa`` (path
    ``"sdpa"``, reason ``"structured"``).
    """
    if isinstance(attention_mask, StructuredMask):
        _usage[(getattr(module, "layer_idx", None), query.shape[2], "sdpa", "structured")] += 1
        _calls["fallback"] += 1
        return attention_mask.attend(sdpa_attention_forward, module, query, key, value, dropout=dropout,
                                     scaling=scaling, **kwargs)
    sparse = _sparse_prefill(module, query, key, value, attention_mask, dropout, scaling, kwargs)
    if sparse is not None:
        return sparse
    queries = query.shape[2]
    reason = _fallback_reason(module, query, key, attention_mask, dropout, kwargs)
    layer = getattr(module, "layer_idx", None)
    if reason == "long_query" and FLASH_PREFILL:
        out = _flash(module, query, key, value, attention_mask, dropout, scaling, kwargs)
        if out is not None:
            return out, None
    if reason is not None:
        return _fallback(module, query, key, value, attention_mask, dropout, scaling, is_causal, reason, kwargs)
    hint = prefix_hint(attention_mask)
    if attention_mask is not None and attention_mask.shape[-1] != key.shape[2]:
        attention_mask = attention_mask[..., : key.shape[2]]
    scale = query.shape[-1] ** -0.5 if scaling is None else scaling
    softcap = kwargs.get("softcap")
    packed = packed_of(key)
    if hint is not None and packed is None and _sinks(kwargs) is None:
        flash_out, flash_path = flash_decode(query, key, value, hint, scale, softcap)
        if flash_out is not None:
            _calls["flash"] += 1
            _usage[(layer, queries, flash_path, "")] += 1
            return flash_out, None
    path = "kernel"
    try:
        if packed is not None and packed.format in PACKED_FORMATS:
            path = f"kernel_{packed.format}"
            out = packed.attend(query, key, value, attention_mask, scale, softcap=softcap, sinks=_sinks(kwargs))
        else:
            window = kwargs.get("sliding_window") if attention_mask is not None else None
            out = decode_attention(
                query, key, value, attention_mask, scale, softcap=softcap, sinks=_sinks(kwargs), window=window
            )
    except out_of_resources():  # raised at compile time, before any launch (e.g. fp32 at head dim 512 on Turing)
        return _fallback(module, query, key, value, attention_mask, dropout, scaling, is_causal, "out_of_resources",
                         kwargs)
    _calls["kernel"] += 1
    _usage[(layer, queries, path, "")] += 1
    return out, None


def _register_economize() -> None:
    if NAME not in ATTENTION_KERNELS:
        ATTENTION_KERNELS.register(
            NAME,
            AttentionKernel(
                family="sdpa",
                forward=economize_attention_forward,
                description=(
                    "split-K Triton decode attention with GQA, mask-block skipping, fp8 and KIVI K/V, Quest pages,"
                    " MInference sparse prefill"
                ),
                fp8_kv=True,
                packed_kv=PACKED_FORMATS,
                sparse_prefill=True,
                structured_masks=True,
            ),
        )


def install(name: str | None = None) -> list[str]:
    """Register attention kernels with ``transformers`` (all of :data:`ATTENTION_KERNELS`, or the one ``name``).

    Afterwards ``AutoModelForCausalLM.from_pretrained(..., attn_implementation="economize")`` or
    ``model.set_attn_implementation("economize")`` selects the kernel. Idempotent.

    The registration is process-wide because ``transformers`` keeps its attention functions in one class-level
    registry that every model reads. It adds named functions and holds no economize state: which kernel a model
    runs stays a choice of that model (its ``attn_implementation``), and models that do not name an economize
    kernel are unaffected.

    Returns:
        The names registered.
    """
    from transformers import AttentionInterface
    from transformers.masking_utils import AttentionMaskInterface

    names = [name] if name is not None else ATTENTION_KERNELS.names()
    for each in names:
        entry = ATTENTION_KERNELS.get(each)
        AttentionInterface.register(each, entry.forward)
        mask = entry.mask_function or AttentionMaskInterface._global_mapping[entry.family]
        AttentionMaskInterface.register(each, mask)
    return names


_register_economize()
