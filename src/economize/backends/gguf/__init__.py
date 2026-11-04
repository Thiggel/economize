"""llama.cpp: export a model with its techniques as GGUF files (written with the ``gguf`` package).

Install with ``pip install economize[gguf]`` (and ``economize[llama-cpp]`` for :func:`load_llama`)::

    from economize.backends.gguf import to_gguf

    export = to_gguf(model, "out/", tokenizer=tokenizer, outtype="f16")
    print(" ".join(export.command))      # llama-server -m model.gguf [--model-draft draft.gguf ...]

The runtime follows the design of the serving backends: :data:`GGUF` holds one adapter per technique (registry
``economize.gguf_techniques``) with a support level (``native``, ``densified`` or ``unsupported`` with the reason),
and :func:`economize.backend_support` reads it under the name ``"gguf"``. The architectures the writer converts are
registered in :data:`GGUF_ARCHS`. Importing this package imports neither ``gguf`` nor ``llama_cpp``.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .arch import GGUF_ARCHS, GgufArch, gguf_arch
    from .engine import GgufGreedyDraft, check_gguf, load_llama, to_gguf
    from .runtime import GGUF, GGUF_TECHNIQUES, GgufQuant
    from .writer import OUTTYPES, code_type, llama_quantize_command, pack_codes, write_gguf

__all__ = [
    "GGUF",
    "GGUF_ARCHS",
    "GGUF_TECHNIQUES",
    "OUTTYPES",
    "GgufArch",
    "GgufGreedyDraft",
    "GgufQuant",
    "check_gguf",
    "code_type",
    "gguf_arch",
    "llama_quantize_command",
    "load_llama",
    "pack_codes",
    "to_gguf",
    "write_gguf",
]

_HOME = {
    "GGUF_ARCHS": "arch", "GgufArch": "arch", "gguf_arch": "arch",
    "GgufGreedyDraft": "engine", "check_gguf": "engine", "load_llama": "engine", "to_gguf": "engine",
    "GGUF": "runtime", "GGUF_TECHNIQUES": "runtime", "GgufQuant": "runtime",
    "OUTTYPES": "writer", "code_type": "writer", "llama_quantize_command": "writer", "pack_codes": "writer",
    "write_gguf": "writer",
}


def __getattr__(name: str) -> object:
    """Import the public names on first use."""
    if name not in _HOME:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return getattr(importlib.import_module(f".{_HOME[name]}", __name__), name)
