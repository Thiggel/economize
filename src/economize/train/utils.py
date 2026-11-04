"""Small helpers of training runs: autocast dtypes, seeding, random-generator state and JSON-safe values."""

from __future__ import annotations

import json
import random
from typing import Any

import numpy as np
import torch

from ..errors import ConfigError

__all__ = ["autocast_dtype", "jsonable", "rng_state", "seed_everything", "set_rng_state"]


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
    """Seed ``random``, ``numpy`` and ``torch``."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def rng_state() -> dict[str, Any]:
    """The state of every random generator a run uses (``torch``, ``numpy``, ``random`` and initialized CUDA)."""
    state: dict[str, Any] = {
        "torch": torch.get_rng_state(), "numpy": np.random.get_state(), "python": random.getstate()
    }
    if torch.cuda.is_available() and torch.cuda.is_initialized():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def set_rng_state(state: dict[str, Any]) -> None:
    """Restore a state of :func:`rng_state`."""
    torch.set_rng_state(state["torch"])
    np.random.set_state(state["numpy"])
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
