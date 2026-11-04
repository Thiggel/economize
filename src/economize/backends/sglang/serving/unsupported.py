"""The techniques SGLang 0.5.21 cannot serve, each with the reason (what SGLang lacks, and where it runs instead).

They are registered so that :func:`~economize.backends.sglang.serving.to_sglang` refuses them before writing
anything. Three groups:

* drafters SGLang has no proposer for: its speculative algorithms are EAGLE / EAGLE3 (with economize's draft model
  plugged in), NEXTN, STANDALONE, NGRAM and DFLASH, with no hook for a custom proposer;
* KV-cache compression beyond an fp8 cache dtype and activation sparsity, which need an attention backend or a
  linear method SGLang does not let a plugin register;
* per-token dynamic depth inside the target's forward, which a batched, CUDA-graph-captured forward cannot express.
"""

from __future__ import annotations

from .base import Role, SglangTechnique, Support, register_sglang_technique

__all__ = ["UNSUPPORTED"]

_VLLM = "; serve it on vLLM (economize.backends.vllm.serving)"
_HF = "; stays HF-only"

#: ``technique -> (role, reason)``: why SGLang cannot serve it.
UNSUPPORTED: dict[str, tuple[Role, str]] = {
    # ---- drafters without an SGLang proposer ----
    "rest": (Role.DRAFTER, "retrieval drafting from a datastore: SGLang's NGRAM proposer matches a trie of the "
                           "request and an optional corpus, not REST's suffix-array datastore" + _VLLM),
    "suffix_decoding": (Role.DRAFTER, "SGLang has no suffix-tree proposer (vLLM's suffix method)" + _VLLM),
    "lookahead": (Role.DRAFTER, "Jacobi / lookahead decoding needs n-gram pools filled by the verifier's own "
                                "guesses; SGLang's NGRAM proposer has no such pool" + _HF),
    "kangaroo": (Role.DRAFTER, "Kangaroo drafts with its own adapter model over the target's shallow layers, a "
                               "draft model class SGLang's EAGLE worker does not provide" + _VLLM),
    "knapspec": (Role.DRAFTER, "KnapSpec picks the skipped layers per context during decoding; SGLang's draft "
                               "model has a fixed layer list" + _VLLM),
    "hidden_transfer": (Role.DRAFTER, "Hidden Transfer drafts several tokens in one target pass from transferred "
                                      "pseudo states; SGLang has no parallel-draft proposer" + _VLLM),
    "hydra": (Role.DRAFTER, "Hydra's sequential heads form a Medusa-style proposer; SGLang 0.5.21 has no Medusa "
                            "proposer" + _VLLM),
    "redrafter": (Role.DRAFTER, "ReDrafter's recurrent drafter with beam search is no SGLang speculative "
                                "algorithm" + _VLLM),
    "eagle": (Role.DRAFTER, "economize writes no SGLang checkpoint for an EAGLE-1/2 head yet; an official "
                            "yuhuili/EAGLE-* head runs on SGLang's own EAGLE algorithm without economize"),
    "medusa": (Role.DRAFTER, "SGLang 0.5.21 has no Medusa proposer (heads on the target's last state)" + _VLLM),
    "pard": (Role.DRAFTER, "PARD drafts all tokens in one parallel pass of a draft LM with mask tokens; SGLang's "
                           "STANDALONE proposer drafts autoregressively" + _VLLM),
    "hybrid": (Role.DRAFTER, "a lookup with a model fallback needs two proposers in one engine; SGLang runs one "
                             "speculative algorithm per engine. Serve the lookup or the fallback alone"),
    # ---- KV-cache compression and activation sparsity ----
    "kivi": (Role.KV_CACHE, "SGLang's KV pool stores bf16 / fp8 / fp4 entries; KIVI's 2/4-bit per-channel keys "
                            "and per-token values with a full-precision window need an own attention backend" + _HF),
    "kivi_fused": (Role.KV_CACHE, "the same 2-bit per-channel cache as kivi, so the same attention backend is "
                                  "needed" + _HF),
    "kv_evict": (Role.KV_EVICTION, "SGLang's radix KV pool has no eviction of entries inside a request "
                                   "(StreamingLLM, SnapKV, PyramidKV, H2O)" + _VLLM + " for StreamingLLM"),
    "kv_evict_fused": (Role.KV_EVICTION, "per-head KV eviction with a fused decode kernel: SGLang's pages belong "
                                         "to a request as a whole" + _HF),
    "duo_attention": (Role.KV_EVICTION, "DuoAttention keeps the whole cache for retrieval heads and a StreamingLLM "
                                        "window for streaming heads; SGLang's radix KV pool evicts nothing inside "
                                        "a request and lays out every head of a layer alike" + _HF),
    "quest": (Role.KV_EVICTION, "Quest selects the cache pages every decoding query reads from per-page key bounds; "
                                "SGLang's attention backends read every page of a request and its KV pool keeps no "
                                "per-page metadata, so it needs an own attention backend" + _HF),
    "minference": (Role.FORWARD, "MInference attends every long prompt over a sparse per-head pattern fitted at "
                                 "prefill; SGLang's attention backends have no hook for a per-head sparse prefill "
                                 "pattern" + _HF),
    "teal": (Role.FORWARD, "TEAL's activation sparsity runs as a linear method; SGLang has no plugin point for a "
                           "custom linear method" + _VLLM),
    # ---- per-token computation inside the target's forward ----
    "calm": (Role.FORWARD, "per-token early exit: every token of an SGLang batch runs every layer, and an exited "
                           "token's KV entries in the skipped layers must be filled in later" + _HF),
    "skipdecode": (Role.FORWARD, "per-position layer skipping inside one captured forward" + _HF),
    "mod": (Role.FORWARD, "per-token routing around blocks leaves routed-out tokens without KV entries in the "
                          "skipped block, which SGLang's paged attention cannot express" + _HF),
    "flexidepth": (Role.FORWARD, "per-token layer skipping with a router and adapter leaves skipped tokens without KV "
                                 "entries, which SGLang's paged attention cannot express" + _HF),
    "gateskip": (Role.FORWARD, "per-token gated residual skipping leaves gated-out tokens without KV entries, which "
                               "SGLang's paged attention cannot express" + _HF),
    "loradrop": (Role.FORWARD, "per-token LoRA-adapter dropping: SGLang's LoRA kernels apply one adapter set per "
                               "request" + _HF),
}


def _adapter(name: str, role: Role, text: str) -> type[SglangTechnique]:
    attributes = {"technique": name, "role": role, "support": Support.UNSUPPORTED, "reason": text,
                  "__doc__": f"``{name}``: not served on SGLang ({text})", "__module__": __name__}
    return type(f"Unsupported_{name}", (SglangTechnique,), attributes)


for _name, (_role, _text) in UNSUPPORTED.items():
    register_sglang_technique(_adapter(_name, _role, _text))
