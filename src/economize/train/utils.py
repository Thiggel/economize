"""Small helpers of training runs: autocast dtypes, seeding, random-generator state and JSON-safe values."""

from __future__ import annotations

import json
import random
from typing import Any

import numpy as np
import torch

from ..errors import ConfigError

__all__ = ["autocast_dtype", "jsonable", "rng_state", "seed_everything", "set_rng_state", "spawn_seed"]


def autocast_dtype(value: str | torch.dtype | None) -> torch.dtype | None:
    """A torch dtype from a dtype or a name (``"bfloat16"``, ``"bf16"``, ``"torch.float16"``, ...; ``None``: none)."""
    if value is None or isinstance(value, torch.dtype):
        return value
    name = str(value).removeprefix("torch.")
    dtype = getattr(torch, {"bf16": "bfloat16", "fp16": "float16", "fp32": "float32"}.get(name, name), None)
    if not isinstance(dtype, torch.dtype):
        raise ConfigError(f"unknown autocast dtype {value!r}")
    return dtype


def seed_everything(seed: int) -> None:
    """Seed ``random``, ``numpy`` and ``torch`` (numpy's legacy generator takes a seed of 64 bits as two words)."""
    random.seed(seed)
    np.random.seed(seed if seed < 2**32 else [seed & 0xFFFFFFFF, seed >> 32])
    torch.manual_seed(seed)


def spawn_seed(seed: int, *keys: int) -> int:
    """A seed for the stream ``keys`` (a process rank, an epoch, ...) of a run seeded with ``seed``, derived through
    ``numpy.random.SeedSequence``. Unlike ``seed + key``, the streams of neighbouring seeds do not overlap (the
    epoch-1 stream of seed 0 is not the epoch-0 stream of seed 1)."""
    if seed < 0 or any(key < 0 for key in keys):
        raise ConfigError(f"seeds and their keys are non-negative, got {seed} and {keys}")
    words = np.random.SeedSequence(seed, spawn_key=keys).generate_state(2, dtype=np.uint32)
    return (int(words[0]) | int(words[1]) << 32) & (2**63 - 1)


def rng_state() -> dict[str, Any]:
    """The state of every random generator a run uses (``torch``, ``numpy``, ``random`` and initialized CUDA), of
    tensors and plain Python values only, so that ``torch.load(..., weights_only=True)`` reads it back."""
    kind, key, pos, has_gauss, cached = np.random.get_state(legacy=True)
    state: dict[str, Any] = {
        "torch": torch.get_rng_state(),
        "numpy": (str(kind), torch.from_numpy(key.astype(np.int64)), int(pos), int(has_gauss), float(cached)),
        "python": random.getstate(),
    }
    if torch.cuda.is_available() and torch.cuda.is_initialized():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def set_rng_state(state: dict[str, Any]) -> None:
    """Restore a state of :func:`rng_state`."""
    torch.set_rng_state(state["torch"])
    kind, key, pos, has_gauss, cached = state["numpy"]
    np.random.set_state((kind, np.asarray(key, dtype=np.uint32), pos, has_gauss, cached))
    random.setstate(state["python"])
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def jsonable(value: Any) -> Any:
    """``value`` with everything JSON cannot hold replaced by its ``repr`` (dict keys become strings)."""
    try:
        json.dumps(value)
        return value
    except TypeError:
        if isinstance(value, dict):
            return {str(k): jsonable(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [jsonable(v) for v in value]
        return repr(value)
