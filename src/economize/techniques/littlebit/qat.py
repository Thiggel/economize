"""Quantization-aware training of LittleBit factors.

The trainable form of a map is :class:`LatentLittleBitLinear`: real-valued latent factors and scales in float32, of
which the forward uses ``sign(U)`` and ``sign(V)``. The sign has the SmoothSign straight-through gradient of the
paper (Section 3.4): forward ``sign(x)``, backward the derivative of ``tanh(k x)`` with ``k = 100``.
:meth:`LatentLittleBitLinear.pack` turns the trained map into a packed :class:`~.module.LittleBitLinear`.

The paper trains the whole model end to end with ``KL(teacher || student) + 10 MSE(hidden states)`` for five
epochs. :func:`train_block` is the short-budget variant: one decoder layer at a time, all its maps jointly, with the
MSE between the layer's outputs and the original layer's outputs on the same calibration inputs (the paper's
intermediate-state term restricted to one layer, as in the block-wise reconstruction of OmniQuant and EfficientQAT).
It needs one layer's latent factors and optimizer state on the device instead of the whole model's.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .factorize import LittleBitFactors
from .module import LittleBitLinear

__all__ = ["LatentLittleBitLinear", "SmoothSign", "smooth_sign", "train_block"]


class SmoothSign(torch.autograd.Function):
    """``sign(x)`` (with ``sign(0) = +1``) whose backward is ``d/dx tanh(k x) = k (1 - tanh(k x)^2)``."""

    @staticmethod
    def forward(ctx: Any, x: Tensor, k: float) -> Tensor:  # type: ignore[override]
        ctx.save_for_backward(x)
        ctx.k = k
        return torch.where(x >= 0, 1.0, -1.0).to(x.dtype)

    @staticmethod
    def backward(ctx: Any, grad: Tensor) -> tuple[Tensor, None]:  # type: ignore[override]
        (x,) = ctx.saved_tensors
        return grad * ctx.k * (1 - torch.tanh(ctx.k * x).square()), None


def smooth_sign(x: Tensor, k: float = 100.0) -> Tensor:
    """:class:`SmoothSign` applied to ``x``."""
    return SmoothSign.apply(x, k)


class LatentLittleBitLinear(nn.Module):
    """A LittleBit map with trainable float32 latent factors (the module that QAT optimizes).

    The forward computes ``sum_p ((((x * g_p) s(V_p)^T) * l_p) s(U_p)^T) * h_p + bias`` with ``s`` the SmoothSign,
    in the dtype of ``x``; the bias stays frozen.
    """

    def __init__(self, factors: LittleBitFactors, bias: Tensor | None, smooth_k: float = 100.0) -> None:
        super().__init__()
        self.paths, self.out_features, self.rank = factors.u.shape
        self.in_features = factors.v.shape[2]
        self.smooth_k = smooth_k
        for name in ("u", "v", "h", "g", "l"):
            setattr(self, name, nn.Parameter(getattr(factors, name).detach().float().clone()))
        self.register_buffer("bias", bias.detach().clone() if bias is not None else None)

    def forward(self, x: Tensor) -> Tensor:
        dtype = x.dtype
        y = None
        for p in range(self.paths):
            v = smooth_sign(self.v[p], self.smooth_k).to(dtype)
            u = smooth_sign(self.u[p], self.smooth_k).to(dtype)
            t = F.linear(x * self.g[p].to(dtype), v) * self.l[p].to(dtype)
            part = F.linear(t, u) * self.h[p].to(dtype)
            y = part if y is None else y + part
        assert y is not None
        return y if self.bias is None else y + self.bias.to(dtype)

    def factors(self) -> LittleBitFactors:
        return LittleBitFactors(*(getattr(self, name).detach() for name in ("u", "v", "h", "g", "l")))

    def pack(self, like: Tensor, backend: str = "torch") -> LittleBitLinear:
        """The packed module on the device and dtype of ``like``."""
        return LittleBitLinear.from_factors(self.factors(), self.bias, like, backend)


def _call(layer: nn.Module, hidden: Tensor, call: Any) -> Tensor:
    out = layer(hidden, *call.args, **call.kwargs)
    return out[0] if isinstance(out, (tuple, list)) else out


def _schedule(total: int, warmup: float) -> Any:
    warm = max(1, round(warmup * total))

    def factor(step: int) -> float:
        if step < warm:
            return (step + 1) / warm
        return 0.5 * (1 + math.cos(math.pi * (step - warm) / max(1, total - warm)))

    return factor


def train_block(
    layer: nn.Module,
    latents: Mapping[str, LatentLittleBitLinear],
    inputs: Sequence[Tensor],
    calls: Sequence[Any],
    targets: Sequence[Tensor],
    device: torch.device,
    *,
    epochs: int = 2,
    lr: float = 1e-4,
    accumulate: int = 1,
    warmup: float = 0.02,
    seed: int = 0,
) -> dict[str, float]:
    """Train the latent maps inside ``layer`` (already swapped in) to reproduce ``targets`` from ``inputs``.

    Adam (betas 0.9, 0.999) with a linear warmup over ``warmup`` of the steps and a cosine decay, as in the paper;
    one optimizer step per ``accumulate`` calibration batches, batches shuffled every epoch. The other parameters of
    the layer stay frozen.

    Returns:
        ``{"init": ..., "final": ...}``: the mean output MSE over the batches before and after training.
    """
    params = [p for module in latents.values() for p in module.parameters()]
    optimizer = torch.optim.Adam(params, lr=lr, betas=(0.9, 0.999))
    steps = epochs * math.ceil(len(inputs) / accumulate)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, _schedule(steps, warmup))
    generator = torch.Generator().manual_seed(seed)

    def loss_of(b: int) -> Tensor:
        out = _call(layer, inputs[b].to(device, non_blocking=True), calls[b])
        return F.mse_loss(out.float(), targets[b].to(device, non_blocking=True).float())

    def evaluate() -> float:
        with torch.no_grad():
            return sum(loss_of(b).item() for b in range(len(inputs))) / len(inputs)

    stats = {"init": evaluate()}
    with torch.enable_grad():
        for _ in range(epochs):
            order = torch.randperm(len(inputs), generator=generator).tolist()
            for start in range(0, len(order), accumulate):
                chunk = order[start : start + accumulate]
                for b in chunk:
                    (loss_of(b) / len(chunk)).backward()
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
    stats["final"] = evaluate()
    return stats
