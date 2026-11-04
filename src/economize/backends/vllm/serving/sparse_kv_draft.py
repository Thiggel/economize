"""vLLM adapters of the sparse-KV self-drafters ``magicdec`` and ``triforce``: not served, with the reason.

Both draft with the target's own weights on a StreamingLLM or retrieval view of the request's KV cache and verify on
the whole cache. vLLM's proposers draft with a separate draft model (or an n-gram / suffix lookup) that owns its
KV cache, and its attention backends read a request's whole paged cache; there is no proposer hook for a second
pass of the target over a sparse view of the same blocks. The HF decoder runs both (``model.decode``).
"""

from __future__ import annotations

from .base import Role, Support, VllmTechnique, register_vllm_technique

__all__ = ["UNSUPPORTED_SPARSE_DRAFT"]

_REASON = ("self-speculation with the target's own weights on a {view} of the request's KV cache: vLLM's proposers "
           "run a draft model with its own KV cache, and no proposer hook runs the target over a sparse view of its "
           "paged cache; runs on the HF decoder")

#: ``technique -> reason``.
UNSUPPORTED_SPARSE_DRAFT: dict[str, str] = {
    "magicdec": _REASON.format(view="StreamingLLM view (attention sinks and a recent window)"),
    "triforce": _REASON.format(view="retrieval view (prompt chunks selected by the query)"),
}


def _adapter(name: str, text: str) -> type[VllmTechnique]:
    attributes = {"technique": name, "role": Role.DRAFTER, "support": Support.UNSUPPORTED, "reason": text,
                  "__doc__": f"``{name}``: not served on vLLM ({text})", "__module__": __name__}
    return type(f"Unsupported_{name}", (VllmTechnique,), attributes)


for _name, _text in UNSUPPORTED_SPARSE_DRAFT.items():
    register_vllm_technique(_adapter(_name, _text))
