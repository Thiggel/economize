"""The arithmetic of TEAL inside vLLM's linear maps, free of vLLM so that it is tested on CPU.

vLLM fuses the maps that read one input (``q_proj``, ``k_proj`` and ``v_proj`` into ``qkv_proj``, ``gate_proj`` and
``up_proj`` into ``gate_up_proj``), while TEAL calibrates one threshold per HF map. A fused map therefore holds one
threshold per output shard (:func:`shard_thresholds`) and thresholds its shared input once per shard
(:func:`teal_linear`), which gives the outputs of the unfused HF maps exactly.

A forward pass of vLLM batches the tokens of many requests: decoding requests contribute one row each, prefilling
requests (and verification passes under speculative decoding) several. TEAL's runtime chooses the mode per call from
its phase; vLLM's equivalent is per row, from the attention metadata of the pass (:func:`single_rows`). TEAL's
``"second_half"`` mode thresholds the last ``max(1, n // 2)`` rows of each request's ``n`` rows in the pass
(:func:`second_half_rows`), which are the last half of the positions of the HF forward call when vLLM runs the
request's tokens in one pass, as it does for a prompt without chunked prefill.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from itertools import pairwise
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor

from ...exporters import UnsupportedBackendError

__all__ = [
    "ALL",
    "DENSE",
    "MODES",
    "SECOND_HALF",
    "bounds",
    "map_names",
    "row_mode",
    "second_half_rows",
    "shard_thresholds",
    "single_rows",
    "sparse_rows",
    "teal_linear",
]

#: Modes a row class can take on vLLM: ``"dense"``, ``"all"`` (every input of the row is thresholded) or
#: ``"second_half"`` (the rows of the last half of each request's rows in the pass).
MODES = ("dense", "all", "second_half")
#: The integer codes of :data:`MODES` (vLLM's custom ops take ints): :func:`row_mode` maps a mode to its code.
DENSE, ALL, SECOND_HALF = range(len(MODES))


def map_names(prefix: str, packed: Mapping[str, Sequence[str]], *tables: Mapping[str, Any]) -> list[str] | None:
    """The HF map names that vLLM's linear map ``prefix`` computes, when any table names one of them.

    Args:
        prefix: The vLLM module name of the map, e.g. ``model.layers.0.self_attn.qkv_proj``.
        packed: The model's ``packed_modules_mapping`` (fused name -> the HF names of its shards, in output order).
        tables: Per-map settings by module name in vLLM's naming (after the model's weights mapper).

    Returns:
        ``[prefix]`` when a table names the map itself (an unfused map, or one fused in the checkpoint as well), the
        names of its shards in output order when a table names one of them, else ``None``.
    """
    if any(prefix in table for table in tables):
        return [prefix]
    parent, _, leaf = prefix.rpartition(".")
    shards = packed.get(leaf)
    if not shards:
        return None
    names = [f"{parent}.{shard}" if parent else shard for shard in shards]
    return names if any(name in table for table in tables for name in names) else None


def shard_thresholds(thresholds: Mapping[str, float], prefix: str,
                     packed: Mapping[str, Sequence[str]]) -> list[float] | None:
    """The thresholds of the output shards of vLLM's linear map ``prefix``, or ``None`` when TEAL leaves it dense.

    Args:
        thresholds: TEAL's thresholds by module name in vLLM's naming (after the model's weights mapper).
        prefix: The vLLM module name of the map, e.g. ``model.layers.0.self_attn.qkv_proj``.
        packed: The model's ``packed_modules_mapping`` (fused name -> the HF names of its shards, in output order).

    Returns:
        One threshold per output shard (a single entry for an unfused map). A shard without a threshold gets 0, which
        keeps every nonzero input, as the HF runtime leaves an unhooked map dense. ``None`` when no shard has a
        positive threshold.
    """
    names = map_names(prefix, packed, thresholds)
    if names is None:
        return None
    values = [max(float(thresholds.get(name, 0.0)), 0.0) for name in names]
    return values if any(value > 0 for value in values) else None


def row_mode(mode: str) -> int:
    """The code (:data:`DENSE`, :data:`ALL` or :data:`SECOND_HALF`) of a mode of :data:`MODES`.

    Raises:
        UnsupportedBackendError: Any other mode.
    """
    if mode not in MODES:
        raise UnsupportedBackendError(f"TEAL on vLLM runs the modes {', '.join(map(repr, MODES))}, got {mode!r}")
    return MODES.index(mode)


def _first_metadata(attn_metadata: Any) -> Any:
    """One attention metadata object of the pass (every KV cache group sees the same requests), or ``None``."""
    if attn_metadata is None:
        return None
    if isinstance(attn_metadata, list):
        raise UnsupportedBackendError(
            "TEAL with different modes for decoding and prefilling rows does not run with micro-batching (dual "
            "batch overlap); serve it without enable_dbo, or with one mode for every row")
    if isinstance(attn_metadata, Mapping):
        return next(iter(attn_metadata.values()), None)
    return attn_metadata


def single_rows(metadata: Any, rows: int) -> bool | Tensor:
    """Which rows of a pass belong to requests that run one token in it (a decoding step).

    Args:
        metadata: The pass's attention metadata of one layer, from vLLM's forward context.
        rows: The number of rows of the pass (padded rows of a CUDA graph included).

    Returns:
        ``True`` (every row), ``False`` (no row) or a boolean ``[rows]`` tensor. Padding rows take the class of the
        last request.

    Raises:
        UnsupportedBackendError: The metadata holds neither query offsets nor a decode count.
    """
    query_start = getattr(metadata, "query_start_loc", None)
    longest = getattr(metadata, "max_query_len", None)
    if query_start is not None and longest is not None:
        if int(longest) <= 1:
            return True
        requests = query_start.numel() - 1
        ends = query_start[1:]
        lengths = ends - query_start[:-1]
        index = torch.arange(rows, device=query_start.device, dtype=query_start.dtype)
        owner = torch.searchsorted(ends, index, right=True).clamp_(max=max(requests - 1, 0))
        return lengths[owner] == 1
    decodes = getattr(metadata, "num_decodes", None)
    decode_tokens = getattr(metadata, "num_decode_tokens", None)
    prefills = getattr(metadata, "num_prefills", None)
    if decodes is not None and decode_tokens is not None and prefills is not None:
        if int(decode_tokens) != int(decodes):
            raise UnsupportedBackendError(
                "the attention backend groups several tokens per request as decodes; TEAL on vLLM tells decoding "
                "rows from prefilling rows only when a decode is one token")
        if int(prefills) == 0:
            return True
        if int(decodes) == 0:
            return False
        return torch.arange(rows) < int(decodes)
    raise UnsupportedBackendError(
        f"TEAL on vLLM cannot tell decoding rows from prefilling rows with {type(metadata).__name__}; use one mode "
        "for every row (prefill='all' with decoding sparsified, or a drafter) or an attention backend with "
        "query_start_loc (FLASH_ATTN, TRITON_ATTN, FLEX_ATTENTION) or a decode count (FLASHINFER)")


def second_half_rows(metadata: Any, rows: int, single: bool) -> bool | Tensor:
    """The rows of a pass in the last half of their request's rows (``max(1, n // 2)`` of ``n``).

    A one-token request's row is its own last half; ``single`` replaces that rule for one-token requests (the mode
    of decoding rows when it differs). Padding rows take the class of the last request.

    Raises:
        UnsupportedBackendError: The metadata has no query offsets (``query_start_loc``).
    """
    query_start = getattr(metadata, "query_start_loc", None)
    if query_start is None:
        raise UnsupportedBackendError(
            f"TEAL's 'second_half' mode on vLLM needs the query offsets of each request in the pass, which "
            f"{type(metadata).__name__} lacks; use an attention backend with query_start_loc (FLASH_ATTN, "
            "TRITON_ATTN, FLEX_ATTENTION)")
    longest = getattr(metadata, "max_query_len", None)
    if longest is not None and int(longest) <= 1:
        return single
    requests = query_start.numel() - 1
    starts, ends = query_start[:-1], query_start[1:]
    lengths = ends - starts
    index = torch.arange(rows, device=query_start.device, dtype=query_start.dtype)
    owner = torch.searchsorted(ends, index, right=True).clamp_(max=max(requests - 1, 0))
    length = lengths[owner]
    last = (index - starts[owner]) >= length - (length // 2).clamp_(min=1)
    return torch.where(length == 1, torch.full_like(last, single), last)


def sparse_rows(attn_metadata: Any, rows: int, single: int, multi: int) -> bool | Tensor:
    """Which rows of a pass to threshold: mode ``single`` for one-token requests, mode ``multi`` for the others.

    Args:
        attn_metadata: The forward context's attention metadata (``None`` in vLLM's profiling and dummy runs, where
            every row counts as a multi-token row).
        rows: The number of rows of the pass.
        single: Mode code (:func:`row_mode`) of the rows of one-token requests; :data:`SECOND_HALF` there equals
            :data:`ALL`.
        multi: Mode code of the rows of the other requests.
    """
    single = ALL if single == SECOND_HALF else single
    if single == multi:
        return single == ALL
    metadata = _first_metadata(attn_metadata)
    if metadata is None:
        return multi != DENSE
    if multi == SECOND_HALF:
        return second_half_rows(metadata, rows, single == ALL)
    one = single_rows(metadata, rows)
    if isinstance(one, bool):
        return (single if one else multi) == ALL
    return one if single == ALL else ~one


def _masked(x: Tensor, threshold: Tensor, rows: bool | Tensor) -> Tensor:
    """TEAL's ``x * (|x| > t)`` on the rows to threshold (all rows for ``True``); other rows pass unchanged."""
    keep = x.abs() > threshold
    if isinstance(rows, Tensor):
        keep = keep | ~rows.to(keep.device)[:, None]
    return x * keep


def _kernel_eligible(x: Tensor, weight: Tensor) -> bool:
    """Whether TEAL's sparse GEMV takes the call (16-bit CUDA inputs, at most 16 rows, column-major weight)."""
    from ....techniques.sparsity.teal_gemv import MAX_ROWS

    return (x.is_cuda and x.shape[0] <= MAX_ROWS and x.dtype in (torch.float16, torch.bfloat16)
            and weight.dtype == x.dtype and weight.stride(0) == 1)


def bounds(sizes: Sequence[int]) -> list[tuple[int, int]]:
    """The ``(start, end)`` offsets of consecutive blocks of the given sizes."""
    starts = [0]
    for size in sizes:
        starts.append(starts[-1] + int(size))
    return list(pairwise(starts))


def teal_linear(x: Tensor, weight: Tensor, bias: Tensor | None, thresholds: Tensor, sizes: Sequence[int],
                rows: bool | Tensor) -> Tensor:
    """``y = s_t(x) W^T + b`` with one threshold per output shard, on the rows that ``rows`` selects.

    Args:
        x: Inputs ``[..., in]``.
        weight: vLLM's weight ``[out, in]`` (column-major on CUDA, so that ``weight.t()`` is TEAL's ``W^T`` layout).
        bias: Bias ``[out]`` or ``None``.
        thresholds: float32 ``[shards]`` on the device of ``x``.
        sizes: Output size of each shard (summing to ``out``).
        rows: ``True`` (threshold every row), ``False`` (dense) or a boolean mask over the flattened rows.

    Returns:
        ``[..., out]`` in the dtype of ``x``. Rows that are thresholded match the HF runtime's TEAL map (its dense
        backend outside the kernel's reach, its Triton backend inside).
    """
    if rows is False:
        return F.linear(x, weight, bias)
    shape = x.shape
    flat = x.reshape(-1, shape[-1])
    shards = bounds(sizes)
    if rows is True and _kernel_eligible(flat, weight):
        from ....techniques.sparsity.teal_gemv import teal_gemv

        weight_t = weight.t()
        parts = []
        for index, (start, end) in enumerate(shards):
            part = teal_gemv(flat, weight_t if len(shards) == 1 else weight_t[:, start:end], thresholds[index])
            parts.append(part if bias is None else part + bias[start:end])
        out = parts[0] if len(parts) == 1 else torch.cat(parts, dim=-1)
    elif len(shards) == 1:
        out = F.linear(_masked(flat, thresholds[0], rows), weight, bias)
    else:
        out = torch.cat([
            F.linear(_masked(flat, thresholds[index], rows), weight[start:end],
                     None if bias is None else bias[start:end])
            for index, (start, end) in enumerate(shards)
        ], dim=-1)
    return out.reshape(*shape[:-1], out.shape[-1])
