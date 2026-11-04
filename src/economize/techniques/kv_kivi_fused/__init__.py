"""KIVI KV cache read as packed codes by the attention kernel (``kivi_fused``).

``kivi`` simulates KIVI: it writes the dequantized values back into the cache's own dtype, so attention reads two
bytes per element as before and only the byte accounting shows the compression. ``kivi_fused`` quantizes exactly
the same entries with the same codes and additionally keeps the packed codes, scales and zero points, which an
attention kernel that reads the ``"kivi"`` packed format (:mod:`economize.attention.kivi`, part of
``"economize"``) attends over directly. Attention then reads ``bits / 8`` bytes per quantized element plus the
group scales::

    import economize
    import economize.attention
    from economize.techniques.kv_kivi_fused import FusedKIVIConfig

    economize.attention.install()
    hf_model = AutoModelForCausalLM.from_pretrained(name, attn_implementation="economize")
    model = economize.apply(hf_model, FusedKIVIConfig(bits=2, group_size=32, residual_length=128))

The kernel dequantizes with the arithmetic of the simulation, so it sees the values ``kivi`` writes bit for bit
and greedy outputs equal those of ``kivi`` under the same attention implementation. The dense buffers stay
allocated (every fallback, such as prefill on ``sdpa``, reads them), so this technique speeds up attention without
reducing the memory the process holds; the byte accounting is that of ``kivi``.
"""

from __future__ import annotations

from .fused import FusedKIVICache, FusedKIVIConfig, FusedKIVITechnique

__all__ = ["FusedKIVICache", "FusedKIVIConfig", "FusedKIVITechnique"]
