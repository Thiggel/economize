"""What the readers of official one-layer draft heads (EAGLE-1/2, EAGLE-3, DFlash) share: the rotary fields of a
:class:`~economize.techniques._draft_layer.DraftLayerConfig` from a head's ``config.json``."""

from __future__ import annotations

from typing import TypedDict

from .._drafting.formats import JSONObject

__all__ = ["JSONObject", "RopeFields", "rope_fields"]


class RopeFields(TypedDict):
    """The rotary fields of a :class:`~economize.techniques._draft_layer.DraftLayerConfig`."""

    rope_theta: float
    rope_scaling: JSONObject | None
    partial_rotary_factor: float
    max_position_embeddings: int | None


def rope_fields(config: JSONObject) -> RopeFields:
    """The rotary fields of a :class:`~economize.techniques._draft_layer.DraftLayerConfig` (``rope_theta``,
    ``rope_scaling``, ``partial_rotary_factor``, ``max_position_embeddings``) from a ``config.json``: ``rope_theta``
    is ``10000`` when absent, as in the reference implementations, and the scaling comes from the ``rope_scaling``
    of transformers 4 or the ``rope_parameters`` of transformers 5."""
    params = config.get("rope_parameters")
    params = dict(params) if isinstance(params, dict) else {}
    scaling = dict(config.get("rope_scaling") or params)
    theta = config.get("rope_theta", params.get("rope_theta"))
    partial = config.get("partial_rotary_factor", params.get("partial_rotary_factor", 1.0))
    for key in ("rope_theta", "partial_rotary_factor"):
        scaling.pop(key, None)
    rope_type = scaling.get("rope_type", scaling.get("type", "default"))
    if "original_max_position_embeddings" in config:
        scaling.setdefault("original_max_position_embeddings", config["original_max_position_embeddings"])
    maximum = config.get("max_position_embeddings") if rope_type != "default" else None  # read by scaled types only
    return {
        "rope_theta": float(theta) if theta is not None else 10000.0,
        "rope_scaling": scaling if rope_type != "default" else None,
        "partial_rotary_factor": float(partial),
        "max_position_embeddings": int(maximum) if maximum is not None else None,
    }
