"""Pieces shared by the TRL integrations (:mod:`.trl` and :mod:`.trl_vllm`)."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

__all__ = ["inject_logs", "is_eval_log"]


def is_eval_log(logs: Mapping[str, Any]) -> bool:
    """Whether a ``Trainer.log`` call reports evaluation (its keys carry the ``eval_`` prefix)."""
    return any(key.startswith("eval_") for key in logs)


def inject_logs(state: Any, logs: dict[str, float], metrics: Mapping[str, float]) -> None:
    """Add ``metrics`` to the ``logs`` a callback's ``on_log`` receives and to the matching
    ``state.log_history`` entry.

    ``Trainer.log`` appends a copy of ``logs`` to the history before it calls the callbacks, so the reporting
    integrations that run after this callback see the metrics through ``logs`` and ``trainer.state.log_history``
    sees them only through the update of its last entry.
    """
    if not metrics:
        return
    logs.update(metrics)
    history = getattr(state, "log_history", None)
    if history and history[-1].get("step") == state.global_step:
        history[-1].update(metrics)
