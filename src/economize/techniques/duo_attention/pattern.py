"""DuoAttention head patterns: reading the reference's files and choosing the full-attention heads.

The reference (``mit-han-lab/duo-attention``, ``duo_attn/utils.py``) stores a trained pattern as a directory with
``full_attention_heads.tsv`` (one line per layer, one gate value per KV head) and ``config.json`` (``sink_size``,
``recent_size``). Its ``sparsify_attention_heads`` turns the gate values into a binary pattern: with a target
``sparsity`` the threshold is that quantile of all gate values (linear interpolation, as ``numpy.quantile``), and a
head with a gate value at or above the threshold keeps full attention. ``sparsity >= 1`` makes every head a
streaming head and ``sparsity <= 0`` none. The reference first adds uniform noise below ``1e-6`` to break ties;
:func:`full_attention_heads` breaks no ties, so tied gate values at the threshold all keep full attention.

:func:`write_pattern` writes a directory in the same format (the reference's ``np.savetxt`` number format, its
``config.json`` keys), plus a ``"layers"`` key listing the 1-indexed layers whose gates were trained, so that a
pattern of a model with sliding-window layers (whose rows hold untrained gates of 1) chooses its streaming heads
among the trained layers only.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from ...errors import ConfigError

__all__ = [
    "PATTERN_FILE",
    "SETTINGS_FILE",
    "PatternFiles",
    "full_attention_heads",
    "read_pattern",
    "read_scores",
    "write_pattern",
]

#: The gate values' file name inside a pattern directory.
PATTERN_FILE = "full_attention_heads.tsv"
#: The training settings' file name inside a pattern directory.
SETTINGS_FILE = "config.json"


@dataclass(frozen=True)
class PatternFiles:
    """A pattern directory of the reference.

    Attributes:
        scores: Gate values per layer and KV head, clipped to ``[0, 1]``.
        sinks: ``sink_size`` of its ``config.json`` (``None`` without one).
        recent: ``recent_size`` of its ``config.json`` (``None`` without one).
        layers: 1-indexed layers whose gates were trained (``"layers"`` of a ``config.json`` written by
            :func:`write_pattern`; empty: every layer, as in the reference's files).
    """

    scores: tuple[tuple[float, ...], ...]
    sinks: int | None = None
    recent: int | None = None
    layers: tuple[int, ...] = ()


def read_scores(path: str | Path) -> tuple[tuple[float, ...], ...]:
    """The gate values of a ``full_attention_heads.tsv`` file (or of the one inside directory ``path``), clipped to
    ``[0, 1]`` as the reference does. Lines may hold different numbers of values (layers with different KV head
    counts); blank lines are skipped.

    Raises:
        ConfigError: The file is missing, empty or holds something other than numbers.
    """
    file = Path(path)
    if file.is_dir():
        file = file / PATTERN_FILE
    if not file.is_file():
        raise ConfigError(
            f"duo_attention pattern {str(path)!r} is neither a {PATTERN_FILE} file nor a directory with one"
        )
    rows: list[tuple[float, ...]] = []
    for number, line in enumerate(file.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        try:
            rows.append(tuple(min(max(float(value), 0.0), 1.0) for value in line.split()))
        except ValueError as error:
            raise ConfigError(f"{file}:{number}: gate values must be numbers ({error})") from None
    if not rows:
        raise ConfigError(f"{file} holds no gate values")
    return tuple(rows)


def read_pattern(path: str | Path) -> PatternFiles:
    """The gate values and, if the directory has a ``config.json``, the sink and recent sizes of a pattern directory
    ``path`` (or of a bare ``full_attention_heads.tsv``)."""
    scores = read_scores(path)
    settings = Path(path) / SETTINGS_FILE
    if not settings.is_file():
        return PatternFiles(scores)
    data = json.loads(settings.read_text())
    sinks, recent = data.get("sink_size"), data.get("recent_size")
    layers = tuple(int(layer) for layer in data.get("layers") or ())
    return PatternFiles(
        scores, None if sinks is None else int(sinks), None if recent is None else int(recent), layers
    )


def write_pattern(
    path: str | Path,
    scores: Sequence[Sequence[float]],
    *,
    sinks: int,
    recent: int,
    layers: Sequence[int] = (),
    settings: Mapping[str, Any] | None = None,
) -> Path:
    """Write a pattern directory that :func:`read_pattern` and the reference read.

    Args:
        path: The directory (created if missing).
        scores: Gate values per decoder layer and KV head, clipped to ``[0, 1]`` on writing.
        sinks: ``sink_size`` of ``config.json``: the sinks the streaming heads saw in training.
        recent: ``recent_size`` of ``config.json``.
        layers: 1-indexed layers whose gates were trained (``"layers"``; empty: every layer).
        settings: Further ``config.json`` entries (the reference stores its training arguments there).

    Returns:
        ``path``.
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    lines = ["\t".join(f"{min(max(float(v), 0.0), 1.0):.18e}" for v in row) for row in scores]
    (path / PATTERN_FILE).write_text("\n".join(lines) + "\n")
    data = {**dict(settings or {}), "sink_size": int(sinks), "recent_size": int(recent),
            "layers": [int(layer) for layer in layers]}
    (path / SETTINGS_FILE).write_text(json.dumps(data))
    return path


def full_attention_heads(
    scores: Sequence[Sequence[float]], sparsity: float | None = None, threshold: float | None = None
) -> list[list[bool]]:
    """Which heads keep full attention: the reference's ``sparsify_attention_heads`` without the tie-breaking noise.

    Args:
        scores: Gate values per layer and head (rows may differ in length).
        sparsity: Target share of streaming heads over every head of ``scores``; the threshold is its quantile.
        threshold: Used when ``sparsity`` is ``None``: heads with a gate value at or above it keep full attention.

    Returns:
        ``True`` per layer and head for a full-attention head, ``False`` for a streaming head.
    """
    flat = [float(value) for row in scores for value in row]
    if sparsity is not None:
        if sparsity >= 1:
            cut = math.inf
        elif sparsity <= 0 or not flat:
            cut = -math.inf
        else:
            cut = float(torch.quantile(torch.tensor(flat, dtype=torch.float64), sparsity))
    elif threshold is not None:
        cut = threshold
    else:
        raise ConfigError("duo_attention needs a sparsity or a threshold to choose its full-attention heads")
    return [[float(value) >= cut for value in row] for row in scores]
