"""All-or-nothing in-place weight transforms: parameter snapshots plus a log of module swaps, undone together."""

from __future__ import annotations

from dataclasses import dataclass, field
from types import TracebackType

import torch
from torch import nn

from .snapshot import ParameterSnapshot

__all__ = ["WeightTransaction", "replace_module"]


def replace_module(root: nn.Module, name: str, new: nn.Module) -> nn.Module:
    """Replace the submodule ``name`` (dotted path inside ``root``) by ``new`` and return the old module."""
    parent, child = _parent(root, name)
    old = getattr(parent, child)
    setattr(parent, child, new)
    return old


def _device(module: nn.Module) -> torch.device | None:
    """The device of the first parameter or buffer of ``module`` (``None``: it has neither)."""
    for tensor in (*module.parameters(), *module.buffers()):
        return tensor.device
    return None


def _parent(root: nn.Module, name: str) -> tuple[nn.Module, str]:
    parent_name, _, child = name.rpartition(".")
    return (root.get_submodule(parent_name) if parent_name else root), child


@dataclass
class WeightTransaction:
    """Records everything an in-place weight transform changes so that :meth:`rollback` can undo it.

    :meth:`take` snapshots the parameters of a module before they are modified (see :class:`ParameterSnapshot`)
    and :meth:`replace_module` swaps a submodule while logging the original. :meth:`rollback` swaps the modules
    back in reverse order, then restores the parameters. Used as a context manager it rolls back when the block
    raises and lets the error propagate::

        with WeightTransaction() as tx:
            tx.take(layer)
            tx.replace_module(layer, "mlp.up_proj", factored)
    """

    snapshot: ParameterSnapshot = field(default_factory=ParameterSnapshot)
    #: Move every replaced module to the CPU until :meth:`rollback` puts it back (on its own device). Without it a
    #: transform that replaces every linear map of a model keeps the originals on the accelerator next to their
    #: replacements, so memory grows by the size of the replacements (a 31B model at 20% low-rank compression
    #: overflows an 80 GB GPU about a third of the way through). Only for transforms whose new module does not hold
    #: the old one or its tensors.
    offload_replaced: bool = False
    _swaps: list[tuple[nn.Module, str, nn.Module, torch.device | None]] = field(default_factory=list)

    def take(self, module: nn.Module) -> None:
        """Save the parameters of ``module`` (and its submodules) before they change."""
        self.snapshot.take(module)

    def replace_module(self, root: nn.Module, name: str, new: nn.Module) -> nn.Module:
        """Like :func:`replace_module`, but logged for :meth:`rollback`."""
        parent, child = _parent(root, name)
        old = getattr(parent, child)
        device = _device(old) if self.offload_replaced else None
        self._swaps.append((parent, child, old, device))
        setattr(parent, child, new)
        if device is not None and device.type != "cpu":
            old.to("cpu")
        return old

    def rollback(self) -> None:
        """Undo the module swaps (newest first), then restore the saved parameters."""
        while self._swaps:
            parent, child, old, device = self._swaps.pop()
            if device is not None and device.type != "cpu":
                old.to(device)
            setattr(parent, child, old)
        self.snapshot.restore()

    def __enter__(self) -> WeightTransaction:
        return self

    def __exit__(
        self, kind: type[BaseException] | None, error: BaseException | None, trace: TracebackType | None
    ) -> None:
        if error is not None:
            self.rollback()
