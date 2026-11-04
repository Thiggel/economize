"""MLX (Apple silicon): export a model with its techniques as an ``mlx_lm`` model folder.

Export only: the folder is written on any platform with ``safetensors``, and ``mlx_lm`` runs it on Apple silicon
(``pip install economize[mlx]`` there)::

    from economize.backends.mlx import to_mlx

    export = to_mlx(model, "out/", tokenizer=tokenizer)
    print(" ".join(export.command))      # mlx_lm.generate --model out [--draft-model out/draft ...]

The runtime follows the design of the serving backends: :data:`MLX` holds one adapter per technique (registry
``economize.mlx_techniques``) with a support level (``native``, ``densified`` or ``unsupported`` with the reason),
and :func:`economize.backend_support` reads it under the name ``"mlx"``. Importing this package does not import
``mlx``.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .engine import check_mlx, to_mlx, write_mlx
    from .runtime import MLX, MLX_TECHNIQUES, MlxDraftModel, MlxQuant, MlxStreaming
    from .writer import MLX_BITS, MLX_GROUP_SIZES, mlx_config, mlx_reason, mlx_tensors, pack_mlx

__all__ = [
    "MLX",
    "MLX_BITS",
    "MLX_GROUP_SIZES",
    "MLX_TECHNIQUES",
    "MlxDraftModel",
    "MlxQuant",
    "MlxStreaming",
    "check_mlx",
    "mlx_config",
    "mlx_reason",
    "mlx_tensors",
    "pack_mlx",
    "to_mlx",
    "write_mlx",
]

_HOME = {
    "check_mlx": "engine", "to_mlx": "engine", "write_mlx": "engine",
    "MLX": "runtime", "MLX_TECHNIQUES": "runtime", "MlxDraftModel": "runtime", "MlxQuant": "runtime",
    "MlxStreaming": "runtime",
    "MLX_BITS": "writer", "MLX_GROUP_SIZES": "writer", "mlx_config": "writer", "mlx_reason": "writer",
    "mlx_tensors": "writer", "pack_mlx": "writer",
}


def __getattr__(name: str) -> object:
    """Import the public names on first use."""
    if name not in _HOME:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return getattr(importlib.import_module(f".{_HOME[name]}", __name__), name)
