"""The protocol of a technique that adapts decoder layers with a :class:`~economize.modules.LayerLoRA`.

A technique opts in by defining :meth:`LoraCapable.layer_lora` (structural typing: no base class and no
registration). :mod:`economize.integrations.peft` then saves and loads that LoRA in the PEFT adapter format and checks
it against PEFT-wrapped layers, without knowing the technique. The built-in implementations are
:class:`economize.techniques.jump.JumpTechnique` (``JumpConfig(adaptation="lora")``) and
:class:`economize.techniques.early_exit.EarlyExitTechnique` (``EarlyExitConfig(adaptation="lora")``)::

    class MyTechnique(Technique):
        def layer_lora(self) -> LayerLoRA | None:
            return self.lora        # keyed by 1-indexed decoder layer, as LayerLoRA always is
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from ..modules import LayerLoRA

__all__ = ["LoraCapable"]


@runtime_checkable
class LoraCapable(Protocol):
    """A technique whose computation path adapts some decoder layers with a :class:`~economize.modules.LayerLoRA`."""

    def layer_lora(self) -> LayerLoRA | None:
        """The technique's LoRA updates, or ``None`` when this configuration of the technique trains none."""
        ...
