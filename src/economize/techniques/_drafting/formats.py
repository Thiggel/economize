"""Weight-file I/O shared by the drafter checkpoint formats (``jump.pt`` files are read by
:mod:`economize.compat.tskip`)."""

from __future__ import annotations

import json
import pickle
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, TypeAlias

import torch

from ...errors import CheckpointFormatError

__all__ = ["JSONObject", "detect_config_dir", "first_file", "load_state_file", "require_weights", "state_shapes"]

#: A parsed ``config.json``. Its values are whatever JSON holds and are read by key, so they stay dynamic.
JSONObject: TypeAlias = Mapping[str, Any]

def first_file(directory: Path, names: tuple[str, ...]) -> Path | None:
    """The first of ``names`` that is a file in ``directory``, or ``None``."""
    return next((directory / name for name in names if (directory / name).is_file()), None)


def detect_config_dir(path: Path, weight_files: tuple[str, ...], accepts: Callable[[JSONObject], bool]) -> bool:
    """Whether ``path`` is a directory holding one of ``weight_files`` and a ``config.json`` that ``accepts``.

    An unreadable or malformed ``config.json`` is no match.
    """
    config = path / "config.json"
    if not path.is_dir() or not config.is_file() or first_file(path, weight_files) is None:
        return False
    try:
        return accepts(json.loads(config.read_text()))
    except (OSError, ValueError):
        return False


def require_weights(path: Path, weight_files: tuple[str, ...], what: str) -> Path:
    """The first of ``weight_files`` in ``path``, which must also hold a ``config.json``.

    Raises:
        CheckpointFormatError: One of the two is missing; ``what`` names the expected content ("Medusa heads").
    """
    weights = first_file(path, weight_files)
    if weights is None or not (path / "config.json").is_file():
        raise CheckpointFormatError(f"{path} holds no {what} (config.json and {' or '.join(weight_files)})")
    return weights


def load_state_file(path: Path, map_location: str | torch.device = "cpu") -> dict[str, torch.Tensor]:
    """A state dict from a ``.safetensors`` file or a ``torch.save`` file (loaded with ``weights_only=True``)."""
    if path.suffix == ".safetensors":
        from safetensors.torch import load_file

        return load_file(path, device=str(map_location))
    return torch.load(path, map_location=map_location, weights_only=True)


def state_shapes(path: Path) -> dict[str, tuple[int, ...]] | None:
    """Name and shape of every tensor of a ``.safetensors`` or ``torch.save`` file without reading the weights (the
    header, or a memory-mapped load), ``None`` for a file that is neither."""
    try:
        if path.suffix == ".safetensors":
            from safetensors import safe_open

            with safe_open(str(path), framework="pt") as handle:
                names = handle.keys()  # a safe_open handle is no dict and cannot be iterated
                return {key: tuple(handle.get_slice(key).get_shape()) for key in names}
        state = torch.load(path, map_location="cpu", mmap=True, weights_only=True)
    except (OSError, RuntimeError, ValueError, pickle.UnpicklingError):
        return None
    if not isinstance(state, dict):
        return None
    return {str(key): tuple(value.shape) for key, value in state.items() if isinstance(value, torch.Tensor)}
