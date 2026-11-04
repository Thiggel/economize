"""economize's linear method on vLLM: TEAL activation sparsity and factored (low-rank) linear maps.

vLLM builds every linear map with one quantization method per model, so the techniques that change how a dense
linear map computes share one: the quantization config :data:`QUANT_METHOD` (``"economize"``), whose
``quantization_config`` in the served ``config.json`` holds one section per technique::

    {"quant_method": "economize",
     "teal": {"thresholds": {HF module name: t}, "factor_thresholds": {HF module name: t}, "single": "all",
              "multi": "dense"},
     "lowrank": {"ranks": {HF module name: r}},
     "kv_cache_scheme": {...}}                       # optional, calibrated fp8 KV scales

The package registers the vLLM extension ``linear`` (see :mod:`economize.backends.vllm.extensions`). Its install step
registers :class:`~economize.backends.vllm.linear.method.EconomizeLinearConfig` with vLLM's quantization registry,
together with the custom ops of the thresholded and the factored map. The method stays inert until a checkpoint names
it, which :func:`economize.backends.vllm.serving.to_vllm` writes for a model with ``teal`` or ``lowrank``. The
arithmetic lives in vLLM-free modules (:mod:`.teal`, :mod:`.factored`) that are tested on CPU; importing the package
does not import vLLM.
"""

from __future__ import annotations

from ..extensions import register_vllm_extension

__all__ = ["FACTORED_OP", "QUANT_METHOD", "TEAL_OP", "install"]

#: The ``quant_method`` of the ``quantization_config`` economize's linear method reads.
QUANT_METHOD = "economize"

#: The custom op (under ``torch.ops.vllm``) of a dense map with TEAL's thresholds.
TEAL_OP = "economize_teal_linear"

#: The custom op of a layer with factored maps (and, possibly, dense ones beside them).
FACTORED_OP = "economize_factored_linear"


@register_vllm_extension("linear")
def install() -> None:
    """Register economize's linear method and its custom ops with vLLM (idempotent)."""
    import vllm  # noqa: F401  (fails first when vLLM is missing)

    from . import method  # noqa: F401  (registers the method and the ops at import)
