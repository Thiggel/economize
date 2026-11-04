"""Family-specific corrections to the generic config reading of :meth:`economize.bench.ModelShape.from_config`.

Two extension points, applied in this order after the generic reading:

1. :meth:`ArchAdapter.shape_fields <economize.arch.ArchAdapter.shape_fields>` of the adapter registered for the config's
   ``model_type``: a family that economize supports keeps its shape corrections on its adapter.
2. :data:`SHAPE_READERS`: a :data:`ShapeReader` registered for the ``model_type``, for model types without an economize
   adapter (priced only for comparison, such as GPT-2 or Llama-4) or to replace an adapter's reading.

Neither requires editing :mod:`economize.bench.cost_model`. The readers below are data for model types without an
adapter.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from ..arch import ARCHITECTURES
from ..arch.config import config_value
from ..registry import Registry

__all__ = ["SHAPE_READERS", "ShapeReader", "family_fields", "first_set"]

#: Reads family-specific fields: ``reader(text_config, fields)`` returns the fields of :class:`ModelShape` that
#: replace the generic reading ``fields`` for configs of one ``model_type``.
ShapeReader = Callable[[Any, Mapping[str, Any]], Mapping[str, Any]]

#: The :data:`ShapeReader` of every model type whose config the generic reading of
#: :meth:`ModelShape.from_config` gets wrong. A new family registers one (``SHAPE_READERS.register("type", fn)``,
#: or the ``economize.shape_readers`` entry point), or overrides ``shape_fields`` on its adapter, instead of editing
#: ``from_config``.
SHAPE_READERS: Registry[ShapeReader] = Registry("shape reader", "economize.shape_readers")


def first_set(config: Any, *names: str) -> Any:
    """The first of the config attributes ``names`` that is set (not ``None`` or zero), else ``None``."""
    for name in names:
        value = config_value(config, name)
        if value:
            return value
    return None


def family_fields(config: Any, fields: Mapping[str, Any]) -> dict[str, Any]:
    """``fields`` with the corrections of the config's family applied: the adapter's ``shape_fields``, then the
    registered :data:`ShapeReader` (which sees the adapter's corrections)."""
    out = dict(fields)
    model_type = getattr(config, "model_type", "")
    if model_type in ARCHITECTURES:
        out.update(ARCHITECTURES.get(model_type).shape_fields(config, out))
    if model_type in SHAPE_READERS:
        out.update(SHAPE_READERS.get(model_type)(config, out))
    return out


def _ungated(config: Any, fields: Mapping[str, Any]) -> Mapping[str, Any]:
    """An MLP of two matrices (no gate projection)."""
    return {"gated": False}


SHAPE_READERS.register("gpt2", _ungated)
SHAPE_READERS.register("gptj", _ungated)
SHAPE_READERS.register("gpt_neox", _ungated)
SHAPE_READERS.register("opt", _ungated)
SHAPE_READERS.register("phi", _ungated)
SHAPE_READERS.register("falcon", _ungated)
SHAPE_READERS.register("bloom", _ungated)
SHAPE_READERS.register("starcoder2", _ungated)
SHAPE_READERS.register("gpt_bigcode", _ungated)


@SHAPE_READERS.register("llama4_text")
def _llama4(config: Any, fields: Mapping[str, Any]) -> Mapping[str, Any]:
    """Llama-4: one ungated shared expert of width ``intermediate_size`` in every MoE layer, and dense layers of an
    interleaved model of their own width ``intermediate_size_mlp``."""
    out: dict[str, Any] = {"intermediate": first_set(config, "intermediate_size_mlp") or fields["intermediate"]}
    if fields["experts"]:
        out.update(shared_intermediate=fields["intermediate"], shared_gate=False)
    return out
