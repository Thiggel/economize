"""Quest: query-aware page selection over the KV cache at decode time (``quest``).

Every decoding step reads only the cache pages whose channel-wise key bounds promise the largest scores for its
query (Tang et al., ICML 2024). The cache keeps every entry; the saving is the time attention spends reading it,
which the ``"economize"`` attention kernel realizes by loading only the selected pages::

    import economize
    import economize.attention
    from economize.techniques.quest import QuestConfig

    economize.attention.install()
    hf_model = AutoModelForCausalLM.from_pretrained(name, attn_implementation="economize")
    model = economize.apply(hf_model, QuestConfig(page_size=16, token_budget=2048))

Prompts attend densely, as do the first ``dense_layers`` layers and sliding-window layers. Quest combines with KV
eviction (``kv_evict``, ``pyramidkv``, ``duo_attention``) and with the simulated or fp8 KV quantizers (``kv_fp8``,
``kv_fp8_fused``); it refuses
the techniques that attach their own packed K/V to the keys (``kivi_fused``, ``kv_evict_fused``).
"""

from __future__ import annotations

from .technique import QuestCache, QuestConfig, QuestTechnique, quest_layers

__all__ = ["QuestCache", "QuestConfig", "QuestTechnique", "quest_layers"]
