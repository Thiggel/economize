"""SGLang adapters of the sparse-KV self-drafters ``magicdec`` and ``triforce``: not served, with the reason.

Both draft with the target's own weights on a StreamingLLM or retrieval view of the request's KV cache and verify on
the whole cache. SGLang's speculative algorithms (EAGLE / EAGLE3, NEXTN, STANDALONE, NGRAM, DFLASH) draft with a
separate draft model or a lookup and have no hook for a custom proposer, and its attention backends read a
request's whole KV pool. The HF decoder runs both (``model.decode``).
"""

from __future__ import annotations

from .base import Role, SglangTechnique, Support, register_sglang_technique

__all__ = ["UNSUPPORTED_SPARSE_DRAFT"]

_REASON = ("self-speculation with the target's own weights on a {view} of the request's KV cache: SGLang has no "
           "proposer hook that runs the target over a sparse view of its KV pool; runs on the HF decoder")

#: ``technique -> reason``.
UNSUPPORTED_SPARSE_DRAFT: dict[str, str] = {
    "magicdec": _REASON.format(view="StreamingLLM view (attention sinks and a recent window)"),
    "triforce": _REASON.format(view="retrieval view (prompt chunks selected by the query)"),
}


def _adapter(name: str, text: str) -> type[SglangTechnique]:
    attributes = {"technique": name, "role": Role.DRAFTER, "support": Support.UNSUPPORTED, "reason": text,
                  "__doc__": f"``{name}``: not served on SGLang ({text})", "__module__": __name__}
    return type(f"Unsupported_{name}", (SglangTechnique,), attributes)


for _name, _text in UNSUPPORTED_SPARSE_DRAFT.items():
    register_sglang_technique(_adapter(_name, _text))
