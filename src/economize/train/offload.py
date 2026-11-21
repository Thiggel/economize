"""AdamW with float32 master weights and its state in CPU memory, for trained parameters that leave no room on the
GPU for the moments."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import TYPE_CHECKING, overload

import torch

if TYPE_CHECKING:
    from torch.optim.optimizer import ParamsT, StateDict

__all__ = ["OffloadedAdamW"]

_SYNCED = ("lr", "weight_decay", "betas", "eps")


class OffloadedAdamW(torch.optim.Optimizer):
    """AdamW that keeps a float32 master copy of every parameter and both moments in CPU memory.

    ``step`` copies the gradients to the CPU (upcast to float32), runs ``torch.optim.AdamW`` on the masters and
    copies the updated weights back, rounded to the parameters' dtypes. For float32 parameters the update is the one
    ``torch.optim.AdamW`` computes on the device (up to kernel rounding). For bfloat16 parameters it is standard
    mixed-precision training: small updates accumulate in the masters instead of vanishing in the rounding. The
    device holds the parameters and their gradients only. For a trainable copy of two Gemma-4-31B layers this frees
    the 7.6 GB of moments, and storing the copy in bfloat16 (``copy_dtype="model"``) halves the rest.

    Learning rates and the other hyperparameters are read from this optimizer's ``param_groups`` at every step, so
    schedulers work on it as on any optimizer. ``state_dict`` holds the CPU optimizer's state and the masters;
    ``load_state_dict`` restores both (a state without masters refreshes them from the parameters, so load those
    first).
    """

    def __init__(self, params: ParamsT, *, betas: tuple[float, float] = (0.9, 0.999), **defaults: float) -> None:
        super().__init__(params, {"betas": betas, "lr": 1e-3, "weight_decay": 1e-2, "eps": 1e-8, **defaults})
        pin = torch.cuda.is_available()
        cpu_groups = []
        for group in self.param_groups:
            masters = [torch.empty(p.shape, dtype=_master_dtype(p), pin_memory=pin) for p in group["params"]]
            cpu_groups.append({**{k: v for k, v in group.items() if k != "params"}, "params": masters})
        self.inner = torch.optim.AdamW(cpu_groups, betas=betas)
        self._pin = pin
        self._refresh()

    def _pairs(self) -> Iterable[tuple[torch.Tensor, torch.Tensor]]:
        for group, inner in zip(self.param_groups, self.inner.param_groups):
            yield from zip(group["params"], inner["params"])

    @torch.no_grad()
    def _refresh(self) -> None:
        for param, master in self._pairs():
            master.copy_(param.detach())

    @overload
    def step(self, closure: None = None) -> None: ...

    @overload
    def step(self, closure: Callable[[], float]) -> float: ...

    @torch.no_grad()
    def step(self, closure: Callable[[], float] | None = None) -> float | None:
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group, inner in zip(self.param_groups, self.inner.param_groups):
            inner.update({key: group[key] for key in _SYNCED if key in group})
        for param, master in self._pairs():
            if param.grad is None:
                master.grad = None
                continue
            if master.grad is None:
                master.grad = torch.empty(master.shape, dtype=master.dtype, pin_memory=self._pin)
            if param.grad.dtype == master.dtype:
                master.grad.copy_(param.grad, non_blocking=True)
            else:  # upcast on the CPU: an upcast on the device would allocate a float32 gradient there
                master.grad.copy_(param.grad.to("cpu"))
        devices = self._devices()
        _synchronize(devices)  # the copies above run on every device holding a parameter, not the current one only
        self.inner.step()
        for param, master in self._pairs():
            if master.grad is not None:
                param.copy_(master, non_blocking=True)
        # the next step updates the masters in place on the CPU: the copies reading them must have finished
        _synchronize(devices)
        return loss

    def _devices(self) -> set[torch.device]:
        return {param.device for param, _ in self._pairs() if param.device.type == "cuda"}

    def state_dict(self) -> StateDict:
        return {"adamw": self.inner.state_dict(), "master": [master.detach().clone() for _, master in self._pairs()]}

    def load_state_dict(self, state_dict: StateDict) -> None:
        self.inner.load_state_dict(state_dict["adamw"])
        for group, inner in zip(self.param_groups, self.inner.param_groups):
            group.update({key: inner[key] for key in _SYNCED if key in inner})
        masters = state_dict.get("master")
        if masters is None:
            self._refresh()
            return
        with torch.no_grad():
            for (param, master), saved in zip(self._pairs(), masters, strict=True):
                master.copy_(saved)
                param.copy_(master)


def _synchronize(devices: Iterable[torch.device]) -> None:
    for device in devices:
        torch.cuda.synchronize(device)


def _master_dtype(param: torch.Tensor) -> torch.dtype:
    return torch.float32 if param.is_floating_point() and param.dtype.itemsize < 4 else param.dtype
