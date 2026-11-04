"""Bridges between economize and the PyTorch / Hugging Face ecosystem.

Every submodule imports its optional dependency lazily, so importing :mod:`economize.integrations` needs nothing
beyond economize's hard dependencies:

- :mod:`.generate`: ``transformers``-compatible ``generate`` (processors, stopping criteria, streamers).
- :mod:`.hub`: ``save_pretrained`` / ``push_to_hub`` / ``from_pretrained`` through ``huggingface_hub``.
- :mod:`.peft`: PEFT bases and export of a technique's LoRA in the PEFT format.
- :mod:`.quantized`: bases loaded through a ``transformers`` quantizer (bitsandbytes, torchao, GPTQ, AWQ, ...).
- :mod:`.compressed_tensors`: export of economize's quantization codes in the compressed-tensors format (vLLM).
- :mod:`.kernels`: custom ops and layer kernels from the Hugging Face ``kernels`` hub with pure-torch fallbacks.
- :mod:`.lm_eval`: the lm-evaluation-harness model ``economize``.
- :mod:`.trl`: rollouts for online RL trainers through economize's decoding loop.
- :mod:`.trl_vllm`: speculative vLLM rollouts for TRL's trainers (colocate patch, drafter refresh, ``spec/*`` logs).

Serving engines are in :mod:`economize.backends` (the vLLM plugin and drafter export, the SGLang configuration).
See ``docs/ecosystem.md`` for the support matrix.
"""

__all__ = ["compressed_tensors", "generate", "hub", "kernels", "lm_eval", "peft", "quantized", "trl", "trl_vllm"]
