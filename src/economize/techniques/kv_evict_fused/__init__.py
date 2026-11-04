"""SnapKV / prompt-time H2O with the kept entries stored compacted for the attention kernel (``kv_evict_fused``).

``kv_evict`` simulates eviction: the buffers keep every entry and attention masks the evicted ones, so a decode
kernel still reads every block of the prompt that holds one kept entry. For methods that select once after the
prompt (``"snapkv"``, ``"pyramidkv"``, ``"h2o_prefill"``) the kept prompt entries never change afterwards, and
``kv_evict_fused`` copies them, per layer, row and KV head, into compact buffers right after the prompt pass. An
attention kernel that reads the ``"gathered"`` packed format (:mod:`economize.attention.gathered`, part of
``"economize"``) then attends over the compact entries plus the dense generated entries and reads
``kept + generated`` entries per head instead of the whole cache::

    import economize
    import economize.attention
    from economize.techniques.kv_evict_fused import FusedKVEvictConfig

    economize.attention.install()
    hf_model = AutoModelForCausalLM.from_pretrained(name, attn_implementation="economize")
    model = economize.apply(hf_model, FusedKVEvictConfig(method="snapkv", heavy=1024, observation_window=32))

The selection, the keep mask and the byte accounting are those of ``kv_evict`` with the same settings, and the
kernel applies the same mask, so greedy outputs equal those of ``kv_evict`` up to the summation order of the
softmax. The dense buffers stay allocated (the prompt pass and every ``sdpa`` fallback read them), so the
technique speeds up attention without reducing the memory the process holds.
"""

from __future__ import annotations

from .fused import FusedEvictionCache, FusedKVEvictConfig, FusedKVEvictTechnique

__all__ = ["FusedEvictionCache", "FusedKVEvictConfig", "FusedKVEvictTechnique"]
