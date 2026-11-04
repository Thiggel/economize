"""The ``"triton"`` TEAL backend: the sparse GEMV of :mod:`.teal_gemv` in place of the maps' dense matmul.

``install`` stores the weight of every eligible map transposed (``W^T`` row-major, presented to the rest of the
model as the usual ``[out, in]`` parameter, so ``F.linear``, state dicts and other techniques see the same values and
shape; the storage is replaced, never duplicated) and replaces the map's ``forward``:

* calls that the :class:`~.sparsify.PhasePolicy` sparsifies in full (mode ``"all"``) with at most
  :attr:`TritonBackend.max_rows` rows run the Triton kernel, which thresholds the inputs and reads only the weight
  rows of the kept channels;
* every other call (more rows, mode ``"second_half"`` or ``"dense"``, autograd, a weight that another technique
  replaced since) computes exactly what the ``"dense"`` backend computes: the masked input times the dense weight.

Both paths give the numbers of the ``"dense"`` backend up to bfloat16 rounding of the accumulation order. Maps the
kernel cannot serve (a quantized or tensor-subclass weight, float32, a CPU map) get the masking pre-hook of the
``"dense"`` backend. :meth:`Installed.remove` restores the row-major storage and the original ``forward``.

With several rows a weight row is read when any row keeps its channel. Independent rows at 40-50% sparsity keep
nearly every channel from four rows on, and the kernel then runs at the speed of the dense matmul; rows with a common
magnitude pattern keep the per-token saving. The default cutoff of 16 rows (one kernel block) therefore never loses
much and wins whatever overlap the rows have; ``scripts/bench_teal_kernel.py`` measures both cases.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any, cast

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.hooks import RemovableHandle

from ...runtime.phase import current_phase
from .backends import TEAL_BACKENDS, DenseBackend, Installed, TealBackend
from .sparsify import PhasePolicy, layer_tokens, sparsify, sparsify_tokens
from .teal_gemv import MAX_ROWS, GemvConfig, teal_gemv, triton_available

__all__ = ["TritonBackend", "kernel_eligible"]

_KERNEL_DTYPES = (torch.bfloat16, torch.float16)


def kernel_eligible(module: nn.Module) -> bool:
    """Whether the kernel can serve ``module``: an ``nn.Linear`` with a plain 16-bit CUDA weight."""
    weight = getattr(module, "weight", None)
    return (
        isinstance(module, nn.Linear)
        and isinstance(weight, Tensor)
        and type(weight.data) is Tensor
        and weight.dim() == 2
        and weight.is_cuda
        and weight.dtype in _KERNEL_DTYPES
    )


def _call_mode(x: Tensor, policy: PhasePolicy) -> str:
    """The mode :func:`.sparsify.sparsify_tokens` applies to ``x`` in the current phase."""
    layout = layer_tokens()
    if layout is None and x.dim() < 2:
        return "all"
    tokens = layout[1] if layout is not None else x.shape[-2]
    return policy.mode(current_phase(), tokens)


class _KernelPatch:
    """The transposed storage and replaced ``forward`` of one map; :meth:`remove` undoes both."""

    def __init__(self, module: nn.Linear, threshold: Tensor, policy: PhasePolicy, max_rows: int) -> None:
        self.module = module
        self.threshold = threshold
        #: Largest row count this map runs the kernel for (a backend may lower it, :meth:`TritonBackend.configure`).
        self.max_rows = max_rows
        #: Launch parameters per row count; a missing row count uses :func:`.teal_gemv.default_config`.
        self.configs: dict[int, GemvConfig] = {}
        self._had_forward = "forward" in module.__dict__
        self._previous_forward = module.__dict__.get("forward")
        weight = module.weight
        weight.data = weight.data.t().contiguous().t()  # [out, in] view of a row-major [in, out] buffer
        self._storage_ptr = weight.data.data_ptr()
        module.forward = self._make_forward(module, threshold, policy)  # type: ignore[method-assign]

    def _make_forward(self, module: nn.Linear, threshold: Tensor, policy: PhasePolicy) -> Callable[..., Tensor]:
        def forward(x: Tensor) -> Tensor:
            weight, bias = module.weight, module.bias
            mode = _call_mode(x, policy)
            if mode != "all":
                return F.linear(sparsify_tokens(x, threshold, policy), weight, bias)
            weight_t = weight.t()
            rows = x.numel() // max(x.shape[-1], 1)
            grad = torch.is_grad_enabled() and (x.requires_grad or weight.requires_grad)
            if 0 < rows <= self.max_rows and not grad and x.dtype == weight.dtype and weight_t.is_contiguous():
                out = teal_gemv(x.reshape(rows, x.shape[-1]), weight_t, threshold, self.configs.get(rows))
                if bias is not None:
                    out = out + bias
                return out.reshape(*x.shape[:-1], out.shape[-1])
            return F.linear(sparsify(x, threshold), weight, bias)

        return forward

    def remove(self) -> None:
        module = self.module
        if self._had_forward:
            module.forward = self._previous_forward  # type: ignore[method-assign, assignment]
        else:
            module.__dict__.pop("forward", None)
        weight = module.weight
        if isinstance(weight, Tensor) and weight.data.data_ptr() == self._storage_ptr:
            weight.data = weight.data.contiguous()


@TEAL_BACKENDS.register("triton")
class TritonBackend(TealBackend):
    """TEAL's sparse GEMV (Triton) for decoding steps; dense masked matmul elsewhere (see the module docstring).

    Attributes:
        max_rows: Largest number of rows (tokens times sequences) of a call that runs the kernel (at most 16).
    """

    max_rows: int = MAX_ROWS

    def install(
        self, modules: Mapping[str, nn.Module], thresholds: Mapping[str, Tensor], policy: PhasePolicy | str
    ) -> Installed:
        policy = PhasePolicy.coerce(policy)
        max_rows = min(self.max_rows, MAX_ROWS)
        usable = triton_available()
        patched = {name: m for name, m in modules.items() if usable and kernel_eligible(m)}
        rest = {name: m for name, m in modules.items() if name not in patched}
        handles: list[Any] = []
        for name, module in patched.items():
            patch = _KernelPatch(cast(nn.Linear, module), thresholds[name], policy, max_rows)
            self.configure(patch)
            handles.append(patch)
        if rest:
            handles += DenseBackend().install(rest, thresholds, policy).handles
        return Installed(cast(list[RemovableHandle], handles))

    def configure(self, patch: _KernelPatch) -> None:
        """Choose the launch parameters of one installed map (:attr:`_KernelPatch.configs`,
        :attr:`_KernelPatch.max_rows`). The base backend keeps the heuristic of :func:`.teal_gemv.default_config`;
        :class:`.triton_autotune.AutotuneTritonBackend` times a grid on the device."""
