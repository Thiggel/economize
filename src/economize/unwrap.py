"""Finding the Hugging Face model inside the wrappers economize meets: an :class:`~economize.EfficientModel`, a PEFT
model, or any object that defines ``unwrap()``.

This module imports nothing from economize, so every layer (the architecture adapters, the core, the integrations)
unwraps models the same way.
"""

from __future__ import annotations

from typing import Any

__all__ = ["base_model_of", "is_peft_model", "peft_base"]


def is_peft_model(model: Any) -> bool:
    """Whether ``model`` is a PEFT model (``PeftModel`` or one of its task subclasses)."""
    return callable(getattr(type(model), "get_base_model", None)) and getattr(model, "peft_config", None) is not None


def peft_base(model: Any) -> Any:
    """The ``transformers`` model inside a PEFT model (PEFT's layers stay in it), else ``model`` itself."""
    return model.get_base_model() if is_peft_model(model) else model


def base_model_of(model: Any) -> Any:
    """The Hugging Face causal LM inside ``model``.

    ``unwrap()`` when the class defines it (an :class:`~economize.EfficientModel` returns the model it was applied to,
    possibly a PEFT model), the model itself for a ``transformers`` model (whose ``base_model`` property is the
    headless backbone), ``get_base_model()`` for a PEFT model, else the ``base_model`` attribute.
    """
    unwrap = getattr(type(model), "unwrap", None)
    if unwrap is not None:
        return unwrap(model)
    from transformers import PreTrainedModel

    if isinstance(model, PreTrainedModel):
        return model
    get_base_model = getattr(type(model), "get_base_model", None)
    if get_base_model is not None:
        return get_base_model(model)
    return getattr(model, "base_model", model)
