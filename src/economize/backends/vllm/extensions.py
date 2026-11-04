"""Installation steps the vLLM plugin runs for serving techniques that are no drafters, by name in
:data:`VLLM_EXTENSIONS`.

A drafter declares what vLLM needs in :data:`economize.backends.vllm.drafters.VLLM_DRAFTERS`. A technique that
changes the target model's serving instead (an attention backend, a KV cache layout, a linear method) registers an
*extension*: a function without arguments that installs it into vLLM, for example by registering an attention
backend with vLLM's registry. The plugin (:mod:`economize.backends.vllm.plugin`) calls every registered extension in
every vLLM process, after the drafter steps, and logs a failing one instead of raising. An extension must therefore be
idempotent, must not change anything for engines that do not use its technique, and imports vLLM only inside the
function::

    from economize.backends.vllm.extensions import register_vllm_extension

    @register_vllm_extension("my_technique")
    def install() -> None:
        from vllm.v1.attention.backends.registry import AttentionBackendEnum, register_backend

        register_backend(AttentionBackendEnum.CUSTOM, "my_pkg.vllm_backend:MyBackend")

A separate package names its function in the ``economize.vllm_extensions`` entry-point group::

    [project.entry-points."economize.vllm_extensions"]
    my_technique = "my_pkg.vllm:install"
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from typing import overload

from ...registry import Registry

__all__ = ["VLLM_EXTENSIONS", "Extension", "register_vllm_extension", "registered_extensions"]

Extension = Callable[[], None]

#: The installation function of each extension, by name (entry-point group ``economize.vllm_extensions``).
VLLM_EXTENSIONS: Registry[Extension] = Registry("vLLM extension", "economize.vllm_extensions")


@overload
def register_vllm_extension(name: str, install: None = None, *,
                            override: bool = False) -> Callable[[Extension], Extension]:
    ...


@overload
def register_vllm_extension(name: str, install: Extension, *, override: bool = False) -> Extension:
    ...


def register_vllm_extension(name: str, install: Extension | None = None, *,
                            override: bool = False) -> Callable[[Extension], Extension] | Extension:
    """Register ``install`` under ``name``; without ``install``, return a decorator that registers the function.

    Raises:
        DuplicateNameError: ``name`` is registered already and ``override`` is false.
    """
    if install is None:
        return VLLM_EXTENSIONS.register(name, override=override)
    return VLLM_EXTENSIONS.register(name, install, override=override)


def registered_extensions() -> Iterator[tuple[str, Extension]]:
    """The registered extensions as ``(name, install)`` pairs in name order."""
    return ((name, VLLM_EXTENSIONS.get(name)) for name in VLLM_EXTENSIONS.names())
