"""KIVI on vLLM: an attention backend over a paged cache whose committed tokens are quantized in place.

The package registers the vLLM extension ``kivi`` (see :mod:`economize.backends.vllm.extensions`). Its install step
registers :class:`~economize.backends.vllm.kivi.backend.EconomizeKIVIBackend` as one of the backends behind vLLM's
``CUSTOM`` attention slot (:mod:`economize.backends.vllm.custom_attention`). It stays inert until an engine names the
``CUSTOM`` backend and carries the ``economize_kivi`` entry in its ``additional_config``, which
:func:`economize.backends.vllm.serving.to_vllm` writes for a model with the ``kivi`` or ``kivi_fused`` technique.

* :mod:`.layout`: the cell layout, the engine settings and the torch reference (no vLLM, no triton).
* :mod:`.kernels`: the Triton kernels (flag clearing, quantization of committed tokens, paged KIVI attention).
* :mod:`.backend`: the vLLM backend, metadata builder and attention implementation (imports vLLM).

Importing the package does not import vLLM.
"""

from __future__ import annotations

from .. import custom_attention
from ..extensions import register_vllm_extension
from .layout import CONFIG_KEY

__all__ = ["BACKEND_PATH", "CONFIG_KEY", "install"]

#: The qualified name of the KIVI backend class.
BACKEND_PATH = "economize.backends.vllm.kivi.backend.EconomizeKIVIBackend"
custom_attention.register_custom_backend(CONFIG_KEY, BACKEND_PATH)


@register_vllm_extension("kivi")
def install() -> None:
    """Register the KIVI backend behind vLLM's ``CUSTOM`` attention backend (idempotent).

    A ``CUSTOM`` backend that another package registered first is left in place; an engine that asks for KIVI then
    fails its check instead of running another backend's kernel.
    """
    import vllm  # noqa: F401  (fails first when vLLM is missing)

    custom_attention.install()
