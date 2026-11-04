"""The settings an economize attention backend reads from the vLLM engine it serves.

An economize attention backend (KIVI, StreamingLLM) is configured through one ``additional_config`` entry of the
engine's ``VllmConfig``, written by :func:`economize.backends.vllm.serving.to_vllm`. vLLM hands that config to most
of the backend's code: metadata builders receive it, and attention layers, with their implementations, are built
inside ``set_current_vllm_config``. A few hooks receive no config at all: ``AttentionBackend.customize_spec`` and the
block-size queries are classmethods, and the model runner asks for the KV cache specs outside the config context.

:class:`EngineSettings` answers those hooks. The settings stay a function of the config: :meth:`EngineSettings.of`
parses them, and :meth:`EngineSettings.current` returns the settings of the current config when there is one. Without
one it returns the settings of the config the backend last saw in this process (:meth:`EngineSettings.bind`, called
whenever a config is at hand). vLLM builds one engine per worker process, and that engine's layers are built (and
bind their config) before the runner asks for the KV cache specs, so the hooks read their own engine's settings.

Importing this module does not import vLLM.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any, Generic, TypeVar

__all__ = ["EngineSettings", "additional_config"]

S = TypeVar("S")


def additional_config(vllm_config: Any) -> Mapping[str, Any]:
    """The ``additional_config`` of a ``VllmConfig`` (empty if it has none); a plain mapping is taken as one."""
    if isinstance(vllm_config, Mapping):
        return vllm_config
    return getattr(vllm_config, "additional_config", None) or {}


class EngineSettings(Generic[S]):
    """The settings of one economize backend, parsed from the engine config (see the module docstring).

    Args:
        key: The ``additional_config`` key of the backend's entry.
        parse: Settings from the entry and the whole config (the config may be a plain ``additional_config``
            mapping, see :func:`additional_config`).
    """

    def __init__(self, key: str, parse: Callable[[Mapping[str, Any], Any], S]) -> None:
        self.key = key
        self._parse = parse
        self._bound: S | None = None

    def of(self, vllm_config: Any) -> S | None:
        """The settings ``vllm_config`` asks for, or ``None`` when it has no entry under :attr:`key`."""
        entry = additional_config(vllm_config).get(self.key)
        return self._parse(entry, vllm_config) if entry else None

    def bind(self, vllm_config: Any) -> S | None:
        """:meth:`of`, which :meth:`current` returns from now on outside a config context (when not ``None``)."""
        settings = self.of(vllm_config)
        if settings is not None:
            self._bound = settings
        return settings

    def current(self) -> S | None:
        """The settings of the current vLLM config (bound), else of the config last bound in this process."""
        from vllm.config import get_current_vllm_config_or_none

        config = get_current_vllm_config_or_none()
        settings = self.bind(config) if config is not None else None
        return settings if settings is not None else self._bound

    def __repr__(self) -> str:
        return f"EngineSettings({self.key!r}, bound={self._bound!r})"
