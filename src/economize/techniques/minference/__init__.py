"""MInference: sparse attention over long prompts with a per-head pattern fitted at prefill (``minference``).

Every head of a covered layer attends a long prompt over an A-shape, vertical-slash or block-sparse pattern whose
indices are estimated from the prompt itself (Jiang et al., NeurIPS 2024). Decoding is unchanged; the saving is
prefill compute, which the ``"economize"`` attention kernel realizes on CUDA by reading only the indexed keys::

    import economize
    import economize.attention
    from economize.techniques.minference import MInferenceConfig

    economize.attention.install()
    hf_model = AutoModelForCausalLM.from_pretrained(name, attn_implementation="economize")
    model = economize.apply(hf_model, MInferenceConfig(vertical=1000, slash=6096))

``pattern_file`` reads a per-head pattern file of the reference (``minference/configs/*.json``). MInference
combines with every KV-cache, weight and drafting technique.
"""

from __future__ import annotations

from .technique import MInferenceConfig, MInferenceTechnique, minference_layers

__all__ = ["MInferenceConfig", "MInferenceTechnique", "minference_layers"]
