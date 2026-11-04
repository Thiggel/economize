"""Selection of base-model parameters for objectives that train the model itself (``ce``, ``layerskip``)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, get_args

from ...errors import ConfigError
from ..objective import ParamGroup

if TYPE_CHECKING:
    from ...core.model import EfficientModel

__all__ = ["BaseParameters", "base_parameter_groups"]

#: ``"trainable"``: the base parameters that require gradients when training starts (every weight of a freshly
#: loaded model, only the adapters of a PEFT model); ``"all"``: every base parameter; ``"none"``: no base parameter.
BaseParameters = Literal["trainable", "all", "none"]


def base_parameter_groups(model: EfficientModel, which: BaseParameters, *, techniques: bool = True) -> list[ParamGroup]:
    """A ``"backbone"`` group with the selected base parameters, and a ``"head"`` group with the techniques'
    trainable parameters (when ``techniques``). Empty groups are left out.

    Raises:
        ConfigError: No parameter is selected, or ``which`` is not a :data:`BaseParameters` value.
    """
    if which == "all":
        base = list(model.base_model.parameters())
    elif which == "trainable":
        base = [p for p in model.base_model.parameters() if p.requires_grad]
    elif which == "none":
        base = []
    else:
        raise ConfigError(f"unknown base parameter selection {which!r}; choose one of {list(get_args(BaseParameters))}")
    groups = [ParamGroup("base", base, "backbone")] if base else []
    if techniques:
        added = [p for p in model.technique_parameters() if p.requires_grad]
        if added:
            groups.append(ParamGroup("techniques", added, "head"))
    if not groups:
        raise ConfigError(
            "nothing to train: no base parameter requires gradients (pass parameters='all', or wrap the model with "
            "PEFT) and no technique adds trainable parameters"
        )
    return groups
