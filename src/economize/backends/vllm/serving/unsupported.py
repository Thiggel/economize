"""The techniques vLLM 0.30 cannot serve yet, each with the reason and the path to a vLLM version.

They are registered so that :func:`~economize.backends.vllm.serving.to_vllm` refuses them early, with the reason,
and so that :func:`~economize.backends.vllm.serving.support_table` lists every technique. A future adapter replaces
one of them by registering under the same name with ``override=True`` (or by deleting its line here once it is
implemented in this package).

Two groups (docs/VLLM_BACKEND.md):

* class B, missing in vLLM but within its plugin points: a proposer through ``speculative_config.method=
  "custom_class"`` (token-level drafters), a draft model registered with ``ModelRegistry`` (trained drafters,
  self-drafters that run a part of the target), a KV-cache dtype or attention backend (cache compression);
* class C, per-token dynamic depth inside the target's forward, which vLLM's batched, CUDA-graph-captured forward
  does not express: every token of a batch runs the same layers. (TEAL's per-token sparsity keeps every layer and is
  served by economize's linear method, :mod:`economize.backends.vllm.serving.teal`.)
"""

from __future__ import annotations

from .base import Role, Support, VllmTechnique, register_vllm_technique

__all__ = ["UNSUPPORTED"]

#: ``technique -> (role, reason)``: why vLLM cannot serve it, and what a vLLM version would take.
UNSUPPORTED: dict[str, tuple[Role, str]] = {
    # ---- drafters without an exporter ----
    "eagle": (Role.DRAFTER, "economize writes no vLLM checkpoint for an EAGLE-1/2 head yet; an official "
                            "yuhuili/EAGLE-* head runs on vLLM's own method='eagle' without economize"),
    # ---- two speculative methods in one engine ----
    "hybrid": (Role.DRAFTER, "a hybrid's fallback is a model drafter (HybridConfig refuses token-level ones). vLLM "
                             "0.30 builds one drafter per engine from speculative_config.method, and the one "
                             "drafter it lets a plugin supply (method='custom_class') receives only the token ids "
                             "of each batch row, with no request ids, KV pages or target hidden states: a draft "
                             "model fallback needs its own per-request KV cache, and a self-drafter, EAGLE-3 or MTP "
                             "fallback reads the target's hidden states. Serve the lookup or the fallback alone"),
    # ---- class B: KV-cache compression (a cache dtype or an attention backend) ----
    "kv_evict_fused": (Role.KV_EVICTION, "SnapKV / H2O-prefill with economize's fused decode kernel (the methods "
                                         "it accepts) keep a different set of entries per KV head, chosen from "
                                         "attention probabilities, while vLLM's pages, block tables and kernel "
                                         "masks belong to a request as a whole, its kernels never write out "
                                         "probabilities, and its KV cache manager keeps no per-head state; "
                                         "kv_evict serves StreamingLLM"),
    "duo_attention": (Role.KV_EVICTION, "DuoAttention gives the retrieval heads of a layer the whole cache and "
                                        "its streaming heads a StreamingLLM window, while vLLM's pages, block "
                                        "tables and KV cache spec belong to a layer as a whole: a head split "
                                        "needs a per-head (or per head group) cache layout. kv_evict serves "
                                        "StreamingLLM on every head"),
    "quest": (Role.KV_EVICTION, "Quest selects the cache pages every decoding query reads from per-page key bounds; "
                                "vLLM 0.30's attention backends read every page of a request's block table and "
                                "its KV cache manager keeps no per-page metadata, so it needs an own attention "
                                "backend that keeps the bounds and launches a page-sparse decode kernel; stays "
                                "HF-only"),
    "minference": (Role.FORWARD, "MInference fits a sparse pattern per head to every long prompt and attends the "
                                 "prefill over the indexed keys only; vLLM 0.30 ships these kernels only inside "
                                 "the Qwen2.5-1M dual-chunk attention backend, tied to that model's chunked "
                                 "positions, and has no hook for a per-head prefill pattern in its other "
                                 "backends; stays HF-only"),
    # ---- class C: per-token computation inside the target's forward ----
    "calm": (Role.FORWARD, "per-token early exit: every token of a vLLM batch runs every layer, and an exited "
                           "token's KV entries in the skipped layers must be filled in later; stays HF-only"),
    "skipdecode": (Role.FORWARD, "per-position layer skipping: a batch of requests at different positions would "
                                 "run different layers in one captured forward; stays HF-only"),
    "mod": (Role.FORWARD, "per-token routing around blocks: the routed-out tokens have no KV entries in the "
                          "skipped block, which vLLM's paged attention cannot express; stays HF-only"),
    "flexidepth": (Role.FORWARD, "per-token layer skipping with a router and adapter: skipped tokens have no KV "
                                 "entries in the skipped layers, which vLLM's paged attention cannot express; "
                                 "stays HF-only"),
    "gateskip": (Role.FORWARD, "per-token gated residual skipping: gated-out tokens have no KV entries in the skipped "
                               "blocks, which vLLM's paged attention cannot express; stays HF-only"),
    "loradrop": (Role.FORWARD, "per-token LoRA-adapter dropping inside layers: vLLM's LoRA kernels apply one "
                               "adapter set per request, not per token; stays HF-only"),
}


def _adapter(name: str, role: Role, text: str) -> type[VllmTechnique]:
    attributes = {"technique": name, "role": role, "support": Support.UNSUPPORTED, "reason": text,
                  "__doc__": f"``{name}``: not served on vLLM ({text})", "__module__": __name__}
    return type(f"Unsupported_{name}", (VllmTechnique,), attributes)


for _name, (_role, _text) in UNSUPPORTED.items():
    register_vllm_technique(_adapter(_name, _role, _text))
