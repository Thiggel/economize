"""``transformers``' ``sdpa`` attention with contiguous K/V where PyTorch's masked kernel is wrong.

``transformers.integrations.sdpa_attention.sdpa_attention_forward`` repeats grouped K/V heads with ``repeat_kv``
(expand, then reshape) whenever it passes a mask. With a single K/V head the reshape stays a view whose head
stride is 0. PyTorch 2.13.0 (cu130) computes masked attention over such K/V wrongly on CUDA when the head dim
exceeds 128 and the query length is one more than a multiple of 32 (33, 65, ..., 673, ...), at any key length and in
every dtype (output errors of order one in fp32). Head dims up to 128, several K/V heads (``repeat_kv`` then copies),
``enable_gqa``, the math backend and PyTorch 2.14 are not affected. Gemma-4 E2B (one K/V head, head dims 256 and
512) and the Gemma-3 270M / 1B models fall into it during every prefill or long exact pass of such a length.

:func:`sdpa_attention_forward` is a drop-in replacement that materializes the repeated heads before calling
``transformers``' function for every call that could hit the error (a single K/V head, head dim above 128, more than
32 queries, CUDA), and passes all other calls through unchanged. The ``"economize"`` implementation's ``sdpa``
fallback uses it; :func:`install` puts it under ``"sdpa"`` in ``transformers`` for models loaded with plain
``sdpa``.

It also takes a :class:`~.banded.StructuredMask` (a :class:`~.banded.BandMask`) in place of the mask tensor and lets
it attend through the same function, one dense-masked or causal call per piece.

A query longer than :data:`QUERY_CHUNK` with a dense mask attends in query chunks of that length over the same keys
(the repeated K/V heads are materialized once for all chunks). PyTorch's masked kernels convert a boolean mask to the
query's dtype in one piece, which is 2 GB at 32K queries over 32K keys; in chunks the conversion and the scores stay
bounded and the result is the same, since every query row attends independently.
"""

from __future__ import annotations

from functools import partial
from typing import Any

import torch

from .banded import StructuredMask

__all__ = ["QUERY_CHUNK", "WIDE_HEAD_DIM", "install", "needs_dense_kv", "sdpa_attention_forward"]

#: Head dims above this are affected; 64 and 128 are computed correctly.
WIDE_HEAD_DIM = 128
#: Query lengths up to this are never affected (the error needs 33, 65, ... queries).
_SHORT_QUERIES = 32
#: Queries per call of a long, dense-masked query (see the module docstring).
QUERY_CHUNK = 4096


def needs_dense_kv(module: torch.nn.Module, query: torch.Tensor, key: torch.Tensor, *, cuda_only: bool = True) -> bool:
    """Whether ``transformers`` would hand ``sdpa`` a stride-0 K/V view that PyTorch may attend to wrongly.

    True for a single K/V head shared by several query heads, head dim above :data:`WIDE_HEAD_DIM` and more than 32
    queries, on CUDA (``cuda_only=False`` drops the device condition). The test is a superset of the failing
    lengths, so that a PyTorch build with a different tile size is covered too.
    """
    groups = getattr(module, "num_key_value_groups", 1)
    return (
        isinstance(groups, int)
        and groups > 1
        and key.shape[1] == 1
        and key.shape[-1] > WIDE_HEAD_DIM
        and query.shape[2] > _SHORT_QUERIES
        and (query.is_cuda or not cuda_only)
    )


class _Ungrouped:
    """``module`` as ``sdpa_attention_forward`` sees it once K/V hold one head per query head."""

    num_key_value_groups = 1

    def __init__(self, module: torch.nn.Module) -> None:
        self._module = module

    def __getattr__(self, name: str) -> Any:
        return getattr(self._module, name)


def sdpa_attention_forward(
    module: torch.nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    *args: Any,
    dense_kv: bool | None = None,
    **kwargs: Any,
) -> tuple[torch.Tensor, Any]:
    """``transformers``' ``sdpa_attention_forward`` with contiguous K/V when :func:`needs_dense_kv` (or
    ``dense_kv``) says so; same signature and result. A :class:`~.banded.StructuredMask` attends through this
    function."""
    if isinstance(attention_mask, StructuredMask):
        kwargs.update(zip(("dropout", "scaling", "is_causal"), args))
        return attention_mask.attend(partial(sdpa_attention_forward, dense_kv=dense_kv), module, query, key, value,
                                     **kwargs)
    from transformers.integrations.sdpa_attention import sdpa_attention_forward as forward

    chunked = _chunked(query, attention_mask, kwargs)
    if chunked or (dense_kv if dense_kv is not None else needs_dense_kv(module, query, key)):
        key, value = _ungrouped(key, query.shape[1]), _ungrouped(value, query.shape[1])
        module = _Ungrouped(module)  # type: ignore[assignment]
    if not chunked:
        return forward(module, query, key, value, attention_mask, *args, **kwargs)
    assert attention_mask is not None  # _chunked holds only for a dense mask
    pieces = [
        forward(module, query[:, :, start : start + QUERY_CHUNK], key, value,
                attention_mask[:, :, start : start + QUERY_CHUNK], *args, **kwargs)[0]
        for start in range(0, query.shape[2], QUERY_CHUNK)
    ]
    return torch.cat(pieces, dim=1), None


def _ungrouped(states: torch.Tensor, heads: int) -> torch.Tensor:
    """``states`` ``[B, Hkv, S, D]`` with each K/V head repeated for its query heads, contiguous (``repeat_kv``)."""
    batch, kv_heads, length, dim = states.shape
    if kv_heads == heads:
        return states.contiguous()
    return states[:, :, None].expand(batch, kv_heads, heads // kv_heads, length, dim).reshape(batch, heads, length, dim)


def _chunked(query: torch.Tensor, attention_mask: Any, kwargs: dict[str, Any]) -> bool:
    """Whether the call attends in query chunks: a long query with a dense per-query mask and no position bias."""
    return (
        query.shape[2] > QUERY_CHUNK
        and kwargs.get("position_bias") is None
        and torch.is_tensor(attention_mask)
        and attention_mask.dim() == 4
        and attention_mask.shape[2] == query.shape[2]
    )


def install() -> None:
    """Register :func:`sdpa_attention_forward` as ``transformers``' ``"sdpa"`` attention (process-wide, idempotent).

    Only needed for models loaded with plain ``sdpa``; the ``"economize"`` implementation already uses it.
    """
    from transformers import AttentionInterface

    AttentionInterface.register("sdpa", sdpa_attention_forward)
