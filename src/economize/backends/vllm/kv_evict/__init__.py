"""KV eviction on vLLM: StreamingLLM as an attention backend plus a KV cache manager that frees evicted blocks.

The package registers the vLLM extension ``kv_evict`` (see :mod:`economize.backends.vllm.extensions`). Its install
step registers :class:`~economize.backends.vllm.kv_evict.spec.StreamingSpec` with vLLM's KV cache spec registry and
:class:`~economize.backends.vllm.kv_evict.backend.EconomizeStreamingBackend` as one of the backends behind vLLM's
``CUSTOM`` attention slot (:mod:`economize.backends.vllm.custom_attention`). Both stay inert until an engine names
the ``CUSTOM`` backend and carries the ``economize_kv_evict`` entry in its ``additional_config``, which
:func:`economize.backends.vllm.serving.to_vllm` writes for a model with the ``kv_evict`` technique.
Importing the package does not import vLLM.
"""

from __future__ import annotations

from .. import custom_attention
from ..extensions import register_vllm_extension

__all__ = ["BACKEND_PATH", "CONFIG_KEY", "install"]

#: The qualified name of the streaming backend class.
BACKEND_PATH = "economize.backends.vllm.kv_evict.backend.EconomizeStreamingBackend"
#: The ``additional_config`` key that selects it (the same as ``backend.CONFIG_KEY``).
CONFIG_KEY = "economize_kv_evict"
custom_attention.register_custom_backend(CONFIG_KEY, BACKEND_PATH)


@register_vllm_extension("kv_evict")
def install() -> None:
    """Register the streaming KV cache spec and the ``CUSTOM`` attention backend with vLLM (idempotent).

    A ``CUSTOM`` backend that another package registered first is left in place; an engine that asks for
    StreamingLLM then fails its check instead of running another backend's kernel.
    """
    import vllm  # noqa: F401  (fails first when vLLM is missing)

    from . import spec  # noqa: F401  (registers StreamingSpec at import)

    custom_attention.install()
