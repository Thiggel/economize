"""DuoAttention: retrieval heads keep the whole KV cache, streaming heads a StreamingLLM window.

    import economize
    from economize.techniques.duo_attention import DuoAttentionConfig, train_pattern
    from economize.train.data import PasskeyRetrieval

    # train the gates (the reference's gate training) and write a pattern directory
    data = PasskeyRetrieval.from_tokenizer(tokenizer, book_texts)  # synthetic multi-passkey retrieval
    trained = train_pattern(hf_model, data, "patterns/my-model")   # full_attention_heads.tsv + config.json

    # deploy it, or a pattern directory of mit-han-lab/duo-attention
    config = DuoAttentionConfig.from_pattern("patterns/my-model", sparsity=0.5)
    model = economize.apply(hf_model, config)
    result = model.decode(prompt, 64)
    model.techniques["duo_attention"].cost_summary(result)  # KV bytes held by the retrieval and streaming heads

Every head attends to the whole prompt; afterwards the streaming heads see the first ``sinks`` entries and the last
``recent`` committed ones. The cache is the positional StreamingLLM cache of :mod:`economize.techniques.kv_evict`
applied per head (CUDA-graph safe); under ``attn_implementation="economize"`` the decode kernel skips the tiles the
streaming heads no longer see. Lossy.

The pattern comes from gate training (:mod:`.objective`, objective ``duo_attention_gates``): one gate per KV head mixes
full and streaming attention, the base model stays frozen, and the gates learn to keep the full model's final states
under an L1 penalty. On models with sliding-window layers only the global layers are gated and covered.
"""

from __future__ import annotations

from . import objective as _objective  # noqa: F401  (registers the "duo_attention_gates" objective)
from .gates import (
    GatedPass,
    HeadGates,
    SinglePassCache,
    TeacherStates,
    gated_layers,
    kv_followers,
    streaming_visibility,
)
from .objective import DuoAttentionGates, TrainedPattern, train_pattern
from .pattern import (
    PATTERN_FILE,
    SETTINGS_FILE,
    PatternFiles,
    full_attention_heads,
    read_pattern,
    read_scores,
    write_pattern,
)
from .technique import DuoAttentionCache, DuoAttentionConfig, DuoAttentionTechnique, streaming_heads

__all__ = [
    "PATTERN_FILE",
    "SETTINGS_FILE",
    "DuoAttentionCache",
    "DuoAttentionConfig",
    "DuoAttentionGates",
    "DuoAttentionTechnique",
    "GatedPass",
    "HeadGates",
    "PatternFiles",
    "SinglePassCache",
    "TeacherStates",
    "TrainedPattern",
    "full_attention_heads",
    "gated_layers",
    "kv_followers",
    "read_pattern",
    "read_scores",
    "streaming_heads",
    "streaming_visibility",
    "train_pattern",
    "write_pattern",
]
