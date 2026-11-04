"""Reader for Medusa checkpoints of FasterDecoding/Medusa (Medusa heads saved as ``jump.pt`` are read by
:mod:`economize.compat.tskip`).

* ``"medusa"``: a directory with ``config.json`` (``medusa_num_heads``, ``medusa_num_layers``,
  ``base_model_name_or_path``) and ``medusa_lm_head.safetensors`` or ``medusa_lm_head.pt`` holding
  ``{i}.{j}.linear.weight/bias`` (residual block ``j`` of head ``i``) and ``{i}.{num_layers}.weight`` (its
  vocabulary projection), optionally prefixed with ``medusa_head.``. It loads as ``head="own"``. Detected
  automatically. The head and block counts are read off the weights: published ``config.json`` files can disagree
  with them (``FasterDecoding/medusa-vicuna-7b-v1.3`` says 3 heads and ships 5), which warns.
* ``"tskip_medusa"`` (:class:`economize.compat.tskip.TskipMedusaFormat`): a ``jump.pt`` holding ``{"kind": "medusa",
  "K", "model", "state": {"W": [K, d, d], "b": [K, d]}}``, heads with the model's shared norm and LM head.
"""

from __future__ import annotations

import dataclasses
import json
import re
import warnings
from pathlib import Path
from typing import Any

import torch

from ...core.serialization import FORMATS, Checkpoint, CheckpointFormat
from ...errors import CheckpointFormatError
from .._drafting import detect_config_dir, load_state_file, require_weights
from .config import MedusaConfig

__all__ = ["MedusaFormat", "config_from_medusa", "head_layout", "keep_heads", "state_from_medusa"]

_HEAD_FILES = ("medusa_lm_head.safetensors", "medusa_lm_head.pt")
_KEY = re.compile(r"^(?:medusa_head\.)?(\d+)\.(\d+)\.(linear\.weight|linear\.bias|weight)$")


def config_from_medusa(config: dict[str, Any]) -> MedusaConfig:
    """The :class:`MedusaConfig` of a FasterDecoding/Medusa ``config.json``."""
    if "medusa_num_heads" not in config:
        raise CheckpointFormatError("a Medusa config.json needs 'medusa_num_heads'")
    return MedusaConfig(num_heads=int(config["medusa_num_heads"]), num_layers=int(config.get("medusa_num_layers", 1)),
                        head="own")


def head_layout(state: dict[str, torch.Tensor]) -> tuple[int, int]:
    """``(heads, residual blocks per head)`` of a FasterDecoding/Medusa head state dict."""
    matches = [_KEY.match(key) for key in state]
    if not matches or None in matches:
        bad = [key for key, match in zip(state, matches) if match is None]
        raise CheckpointFormatError(f"unexpected parameters {bad[:3]} in a Medusa checkpoint")
    heads = 1 + max(int(m[1]) for m in matches if m is not None)
    projections = {int(m[2]) for m in matches if m is not None and m[3] == "weight"}
    if len(projections) != 1:
        raise CheckpointFormatError(f"Medusa heads with differing block counts {sorted(projections)}")
    return heads, projections.pop()


def keep_heads(checkpoint: Checkpoint, count: int) -> Checkpoint:
    """``checkpoint`` with only the first ``count`` Medusa heads, e.g. for a tree that uses fewer heads than were
    trained (``mc_sim_7b_63`` reads 4 of the 5 heads of the Vicuna checkpoints)."""
    configs = [c for c in checkpoint.configs if isinstance(c, MedusaConfig)]
    if len(configs) != 1:
        raise CheckpointFormatError("keep_heads needs a checkpoint with exactly one Medusa technique")
    config = configs[0]
    if not 1 <= count <= config.num_heads:
        raise CheckpointFormatError(f"cannot keep {count} of {config.num_heads} Medusa heads")
    kept = dataclasses.replace(config, num_heads=count)
    state = {key: value for key, value in checkpoint.states["medusa"].items()
             if not key.startswith("heads.") or int(key.split(".")[1]) < count}
    return dataclasses.replace(checkpoint, configs=[kept if c is config else c for c in checkpoint.configs],
                               states={**checkpoint.states, "medusa": state})


def state_from_medusa(state: dict[str, torch.Tensor], num_layers: int) -> dict[str, torch.Tensor]:
    """Rename a FasterDecoding/Medusa head state dict to :class:`MedusaTechnique` parameters."""
    out: dict[str, torch.Tensor] = {}
    for key, value in state.items():
        match = _KEY.match(key)
        if match is None:
            raise CheckpointFormatError(f"unexpected parameter {key!r} in a Medusa checkpoint")
        head, index, leaf = int(match[1]), int(match[2]), match[3]
        if leaf == "weight" and index == num_layers:
            out[f"heads.{head}.proj.weight"] = value
        elif leaf.startswith("linear.") and index < num_layers:
            out[f"heads.{head}.blocks.{index}.{leaf}"] = value
        else:
            raise CheckpointFormatError(f"parameter {key!r} does not fit heads with {num_layers} residual blocks")
    return out


class MedusaFormat(CheckpointFormat):
    """A FasterDecoding/Medusa head directory (``config.json`` + ``medusa_lm_head.{safetensors,pt}``)."""

    def detect(self, path: Path) -> bool:
        return detect_config_dir(path, _HEAD_FILES, lambda config: "medusa_num_heads" in config)

    def read(self, path: Path, map_location: str | torch.device = "cpu") -> Checkpoint:
        weights = require_weights(path, _HEAD_FILES, "Medusa heads")
        raw = json.loads((path / "config.json").read_text())
        state = load_state_file(weights, map_location)
        heads, layers = head_layout(state)
        stated = (raw.get("medusa_num_heads"), raw.get("medusa_num_layers", 1))
        if stated != (heads, layers):
            warnings.warn(f"{path}: config.json states {stated[0]} heads with {stated[1]} blocks, the weights hold "
                          f"{heads} with {layers}; using the weights", stacklevel=2)
        config = config_from_medusa({**raw, "medusa_num_heads": heads, "medusa_num_layers": layers})
        return Checkpoint([config], {"medusa": state_from_medusa(state, config.num_layers)}, None,
                          raw.get("base_model_name_or_path"), {"source": str(path)})


FORMATS.register("medusa", MedusaFormat())
