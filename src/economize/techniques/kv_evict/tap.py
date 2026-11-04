"""Attention probabilities of fused attention calls, for score-based eviction.

``scaled_dot_product_attention`` never materializes its attention matrix. :class:`AttentionTap` is a
:class:`~torch.overrides.TorchFunctionMode` that lets every call run unchanged and, when its sink asks for it,
recomputes the probabilities from the same query, key and mask in fp32 (:func:`attention_probs`). Only the query rows
from the sink's :meth:`~AttentionSink.first_query` on are recomputed, in chunks of at most :data:`CHUNK_BYTES` of
probabilities, so a long prompt never materializes its whole ``[B, H_q, Q, K]`` matrix. Eager attention returns its
probabilities, so models loaded with ``attn_implementation="eager"`` need no tap.

The mode is an :class:`~economize.attention.AttentionObserver`: under the ``"economize"`` attention a long query
(prefill) runs on FlashAttention, and the tap recomputes the same rows from the inputs the attention reports to
:meth:`AttentionTap.observe_attention`, so a long prompt never runs on ``sdpa`` because of the tap.

The mode is active only while an exact pass of a score-based eviction method runs; it intercepts every torch
function call in that window, which makes those passes slower. economize simulates eviction, so this is accepted.
"""

from __future__ import annotations

import math
from collections.abc import Iterator
from typing import Any, Protocol

import torch
import torch.nn.functional as F
from torch.overrides import TorchFunctionMode

__all__ = ["CHUNK_BYTES", "AttentionSink", "AttentionTap", "attention_probs", "iter_attention_probs"]

_SDPA = F.scaled_dot_product_attention
#: Parameter order of ``scaled_dot_product_attention`` (a builtin without an inspectable signature).
_SDPA_PARAMETERS = ("query", "key", "value", "attn_mask", "dropout_p", "is_causal", "scale", "enable_gqa")
#: Size bound of one chunk of fp32 probabilities the tap hands to its sink.
CHUNK_BYTES = 1 << 28


class AttentionSink(Protocol):
    """Receives the attention probabilities of tapped calls."""

    def wants_probs(self) -> bool:
        """Whether the current call's probabilities are needed (they are computed only then)."""
        ...

    def first_query(self, queries: int) -> int:
        """The first of the call's ``queries`` rows whose probabilities can count (``queries`` if none can)."""
        ...

    def observe(self, probs: torch.Tensor, query_start: int = 0) -> None:
        """Probabilities ``[B, H_q, Q', K]`` (fp32) of the call's query rows ``query_start .. query_start + Q'``.

        A call reports its rows from :meth:`first_query` on, possibly in several consecutive chunks.
        """
        ...


class AttentionTap(TorchFunctionMode):
    """Reports the probabilities of every ``scaled_dot_product_attention`` call to ``sink``."""

    def __init__(self, sink: AttentionSink) -> None:
        super().__init__()
        self.sink = sink

    def __torch_function__(self, func: Any, types: Any, args: tuple = (), kwargs: dict | None = None) -> Any:
        kwargs = kwargs or {}
        out = func(*args, **kwargs)
        if func is _SDPA and self.sink.wants_probs():
            self._report(_bind(args, kwargs))
        return out

    def observe_attention(
        self, query: torch.Tensor, key: torch.Tensor, attn_mask: torch.Tensor | None, scale: float
    ) -> None:
        """Report a call that ran outside ``sdpa`` (FlashAttention) as if ``sdpa`` had run it."""
        if self.sink.wants_probs():
            self._report({"query": query, "key": key, "attn_mask": attn_mask, "scale": scale})

    def _report(self, bound: dict[str, Any]) -> None:
        start = self.sink.first_query(bound["query"].shape[-2])
        for offset, probs in iter_attention_probs(**bound, query_start=start, chunk_bytes=CHUNK_BYTES):
            self.sink.observe(probs, offset)
            del probs


def attention_probs(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor | None = None,
    attn_mask: torch.Tensor | None = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    scale: float | None = None,
    enable_gqa: bool = False,
) -> torch.Tensor:
    """The softmax ``scaled_dot_product_attention`` applies, ``[B, H_q, Q, K]`` in fp32 (dropout ignored).

    Query rows whose keys are all masked get zeros (SDPA returns NaN for them).
    """
    ((_, probs),) = iter_attention_probs(query, key, attn_mask=attn_mask, is_causal=is_causal, scale=scale)
    return probs


def iter_attention_probs(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor | None = None,
    attn_mask: torch.Tensor | None = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    scale: float | None = None,
    enable_gqa: bool = False,
    *,
    query_start: int = 0,
    chunk_bytes: int | None = None,
) -> Iterator[tuple[int, torch.Tensor]]:
    """:func:`attention_probs` of the query rows from ``query_start`` on, as ``(first row, probs)`` chunks.

    Every query row's softmax is independent of the others, so the chunks hold the rows of the whole matrix. A chunk's
    fp32 probabilities take at most ``chunk_bytes`` (at least one row; ``None``: all rows in one chunk).
    """
    n_q, n_k = query.shape[-2], key.shape[-2]
    if query_start >= n_q:
        return
    k = key.float()
    if k.shape[-3] != query.shape[-3]:
        k = k.repeat_interleave(query.shape[-3] // k.shape[-3], dim=-3)
    k = k.transpose(-1, -2)
    factor = 1.0 / math.sqrt(query.shape[-1]) if scale is None else scale
    row_bytes = 4 * n_k * math.prod(query.shape[:-2])
    step = n_q - query_start if chunk_bytes is None else max(1, chunk_bytes // max(row_bytes, 1))
    for start in range(query_start, n_q, step):
        end = min(start + step, n_q)
        logits = (query[..., start:end, :].float() @ k) * factor
        if is_causal:
            causal = torch.ones(end - start, n_k, dtype=torch.bool, device=logits.device).tril(start)
            logits = logits.masked_fill(~causal, float("-inf"))
        if attn_mask is not None:
            mask = attn_mask[..., start:end, :] if attn_mask.shape[-2] != 1 else attn_mask
            bool_mask = mask.dtype == torch.bool
            logits = logits.masked_fill(~mask, float("-inf")) if bool_mask else logits + mask.float()
        yield start, logits.softmax(-1).nan_to_num(0.0)
        del logits


def _bind(args: tuple, kwargs: dict) -> dict[str, Any]:
    bound = dict(zip(_SDPA_PARAMETERS, args))
    bound.update({name: value for name, value in kwargs.items() if name in _SDPA_PARAMETERS})
    return bound

