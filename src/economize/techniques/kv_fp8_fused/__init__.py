"""FP8 KV cache read in place by the attention kernel (``kv_fp8_fused``).

``kv_fp8`` stores every K/V entry in 8 bits and casts the whole buffer back to the model's dtype on every read, so
each attention call reads the fp8 bytes, writes a full-precision copy and reads that copy again. ``kv_fp8_fused``
stores exactly the same bytes and hands them to an attention kernel that decodes fp8 in registers
(:class:`economize.attention.AttentionKernel` with ``fp8_kv=True``, such as ``"economize"``), so attention reads one
byte per element::

    import economize
    import economize.attention
    from economize.techniques.kv_fp8_fused import FusedFP8KVConfig

    economize.attention.install()
    hf_model = AutoModelForCausalLM.from_pretrained(name, attn_implementation="economize")
    model = economize.apply(hf_model, FusedFP8KVConfig(format="e4m3"))

The values attention sees are those of ``kv_fp8`` with ``scale="none"`` bit for bit, so outputs differ from
``kv_fp8`` only by the attention kernel's rounding. Per-token scales are not supported (the kernel takes no
scales); passes the kernel does not take (prefill, score taps) upcast inside the attention function instead.
"""

from __future__ import annotations

from .fused import FusedFP8Cache, FusedFP8KVConfig, FusedFP8KVTechnique

__all__ = ["FusedFP8Cache", "FusedFP8KVConfig", "FusedFP8KVTechnique"]
