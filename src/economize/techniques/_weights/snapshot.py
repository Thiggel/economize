"""Undo for in-place weight transforms: a CPU copy of a module's parameters that can be put back exactly."""

from __future__ import annotations

import weakref
from dataclasses import dataclass, field

import torch
from torch import Tensor, nn

__all__ = ["ParameterSnapshot"]


@dataclass
class _Saved:
    module: nn.Module
    name: str
    value: Tensor
    original: weakref.ref[nn.Parameter]
    device: torch.device
    requires_grad: bool


@dataclass
class ParameterSnapshot:
    """The original parameters of the modules passed to :meth:`take`, kept on the CPU.

    :meth:`restore` copies the saved value back into a parameter that is still the original object (so references
    to it stay valid) and installs a fresh ``nn.Parameter`` on the original device where a transform replaced the
    parameter by another object (a kernel backend's tensor subclass). ``requires_grad`` is restored too. A parameter
    shared by several modules is saved once and shared again after the restore. Taking a module twice keeps the
    first copy.
    """

    _saved: list[_Saved] = field(default_factory=list)
    _seen: dict[int, nn.Module] = field(default_factory=dict)  # holding the module keeps its id unique
    _values: dict[int, tuple[weakref.ref[Tensor], Tensor]] = field(default_factory=dict)

    @torch.no_grad()
    def take(self, module: nn.Module) -> None:
        """Save every parameter of ``module`` and its submodules (once per module)."""
        for owner in module.modules():
            if id(owner) in self._seen:
                continue
            self._seen[id(owner)] = owner
            for name, param in owner.named_parameters(recurse=False):
                known = self._values.get(id(param))  # a weak reference: replaced weights are freed as usual
                if known is not None and known[0]() is param:
                    value = known[1]
                else:
                    value = param.detach().to("cpu", copy=True)
                    self._values[id(param)] = (weakref.ref(param), value)
                self._saved.append(
                    _Saved(owner, name, value, weakref.ref(param), param.device, param.requires_grad)
                )

    @torch.no_grad()
    def restore(self) -> None:
        """Put every saved parameter back."""
        restored: dict[int, nn.Parameter] = {}
        for saved in self._saved:
            key = id(saved.value)  # one saved value per original parameter object
            if key not in restored:
                original = saved.original()
                if original is not None and type(original) is nn.Parameter and original.shape == saved.value.shape:
                    original.data.copy_(saved.value)
                    original.requires_grad_(saved.requires_grad)
                    restored[key] = original
                else:
                    restored[key] = nn.Parameter(saved.value.to(saved.device), requires_grad=saved.requires_grad)
            if getattr(saved.module, saved.name) is not restored[key]:
                setattr(saved.module, saved.name, restored[key])

    def __len__(self) -> int:
        return len(self._saved)
