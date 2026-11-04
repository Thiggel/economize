"""vLLM's one ``CUSTOM`` attention backend slot, shared by every economize technique that brings its own backend.

vLLM 0.30 names an out-of-tree attention backend by registering a class path for
``AttentionBackendEnum.CUSTOM``; one engine runs one attention backend, and the registry holds one path. Several
economize techniques serve through such a backend (StreamingLLM by :mod:`economize.backends.vllm.kv_evict`, KIVI by
:mod:`economize.backends.vllm.kivi`). Each registers its backend here with :func:`register_custom_backend`, under the
``additional_config`` key that the engine's plan (:func:`economize.backends.vllm.serving.to_vllm`) writes for it, and
calls :func:`install`, which points ``CUSTOM`` at :data:`DISPATCH_PATH`. When vLLM resolves that path, the module
attribute ``EconomizeCustomBackend`` is looked up through :func:`__getattr__`, which returns the backend whose key the
current engine's ``additional_config`` holds. A new backend is added by registering it; no module here changes.

vLLM caches the backend class it selects per process (``_cached_get_attn_backend`` in ``vllm.v1.attention.selector``,
keyed by the layer's selector config, which holds no ``additional_config``). One process therefore serves one
economize backend: a second engine in the same process that asks for another one would get the first one's class.
vLLM's default engine core runs in a process of its own per engine, so this matters only with
``VLLM_ENABLE_V1_MULTIPROCESSING=0`` and two economize backends in one script.

:func:`mark_draft_builders` flags the metadata builders of a drafter's own attention pass, so that a backend that
changes the cache at exact passes (KIVI's quantization of committed tokens) leaves the cache alone while a drafter
runs the target's layers.

Importing this module does not import vLLM.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from ...errors import ConfigError

__all__ = ["DISPATCH_PATH", "DRAFT_ATTRIBUTE", "CustomBackend", "install", "is_draft_builder",
           "mark_draft_builders", "register_custom_backend", "registered_backends", "resolve"]

logger = logging.getLogger(__name__)

#: The qualified name vLLM's ``CUSTOM`` attention backend resolves to.
DISPATCH_PATH = "economize.backends.vllm.custom_attention.EconomizeCustomBackend"
#: Attribute set on the metadata builders of a drafter's attention pass.
DRAFT_ATTRIBUTE = "economize_draft"


@dataclass(frozen=True)
class CustomBackend:
    """A backend that shares the ``CUSTOM`` slot.

    Attributes:
        key: The ``additional_config`` key whose presence selects it.
        path: Qualified name of the backend class (imported, with vLLM, when it is selected).
    """

    key: str
    path: str


_BACKENDS: dict[str, CustomBackend] = {}
_LAST: list[CustomBackend] = []  # the backend resolved last in this process


def register_custom_backend(key: str, path: str) -> CustomBackend:
    """Register the backend class at ``path`` for engines whose ``additional_config`` holds ``key`` (idempotent).

    Raises:
        ConfigError: ``key`` names another backend already.
    """
    backend = CustomBackend(key, path)
    known = _BACKENDS.setdefault(key, backend)
    if known != backend:
        raise ConfigError(f"the CUSTOM attention key {key!r} names {known.path} already, not {path}")
    return known


def registered_backends() -> tuple[CustomBackend, ...]:
    """The registered backends in registration order."""
    return tuple(_BACKENDS.values())


def install() -> bool:
    """Point vLLM's ``CUSTOM`` attention backend at :data:`DISPATCH_PATH` (idempotent).

    Returns:
        Whether ``CUSTOM`` resolves through economize. A ``CUSTOM`` backend that another package registered first is
        left in place (and logged); an engine that asks for an economize backend then fails its check instead of
        running another package's kernel.
    """
    from vllm.v1.attention.backends.registry import _ATTN_OVERRIDES, AttentionBackendEnum, register_backend

    current = _ATTN_OVERRIDES.get(AttentionBackendEnum.CUSTOM)
    if current not in (None, DISPATCH_PATH):
        logger.warning("economize: vLLM's CUSTOM attention backend is %s already; economize's backends (%s) stay "
                       "unavailable", current, ", ".join(sorted(_BACKENDS)))
        return False
    register_backend(AttentionBackendEnum.CUSTOM, DISPATCH_PATH)
    return True


def resolve(additional_config: dict[str, Any] | None) -> CustomBackend | None:
    """The registered backend that ``additional_config`` selects, or ``None`` when it names none.

    Raises:
        ConfigError: it names two backends (vLLM runs one attention backend per engine).
    """
    named = [backend for key, backend in _BACKENDS.items() if (additional_config or {}).get(key)]
    if len(named) > 1:
        raise ConfigError("one vLLM engine runs one attention backend, but its additional_config asks for "
                          + " and ".join(backend.key for backend in named))
    return named[0] if named else None


def _selected() -> CustomBackend:
    from vllm.config import get_current_vllm_config_or_none

    config = get_current_vllm_config_or_none()
    backend = resolve(getattr(config, "additional_config", None)) if config is not None else None
    if backend is None:
        # resolved outside an engine's config context (or for an engine with no economize entry): the backend this
        # process resolved last, else the first registered one, which leaves an engine without an entry unchanged
        backend = _LAST[0] if _LAST else next(iter(_BACKENDS.values()), None)
    if backend is None:
        raise ConfigError("vLLM asked for economize's CUSTOM attention backend, but no economize backend is "
                          "registered")
    _LAST[:] = [backend]
    return backend


def __getattr__(name: str) -> Any:
    if name != "EconomizeCustomBackend":
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from vllm.utils.import_utils import resolve_obj_by_qualname

    return resolve_obj_by_qualname(_selected().path)


def mark_draft_builders(attn_groups: Any) -> None:
    """Flag every metadata builder of ``attn_groups`` (vLLM's ``list[list[AttentionGroup]]``) as a drafter's."""
    for groups in attn_groups or ():
        for group in groups:
            for builder in getattr(group, "metadata_builders", ()) or ():
                setattr(builder, DRAFT_ATTRIBUTE, True)


def is_draft_builder(builder: Any) -> bool:
    """Whether :func:`mark_draft_builders` flagged ``builder``."""
    return bool(getattr(builder, DRAFT_ATTRIBUTE, False))
