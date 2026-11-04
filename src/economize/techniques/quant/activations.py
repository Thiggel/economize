"""Activation quantization of the quantized linear maps (``QuantConfig.activations``, W8A8 and W4A4 schemes).

Weight-only formats keep the inputs of a linear map in the model dtype. A scheme registered in
:data:`ACTIVATION_SCHEMES` also quantizes those inputs, so that a kernel can multiply integer activations by integer
weights (W8A8: ``torch._int_mm`` on the torchao backend, CUTLASS ``cutlass_scaled_mm`` in vLLM). The built-in schemes
quantize dynamically per token with a symmetric absmax scale, as SmoothQuant's ``W8A8`` and vLLM's
``CompressedTensorsW8A8Int8`` do:

==========  =========================================================================================
scheme      what the input of a quantized linear map becomes
==========  =========================================================================================
``int8``    ``round(x / s).clamp(-128, 127) * s`` with ``s = max|x_token| / 127`` per token
``int4``    the same on ``[-8, 7]`` with ``s = max|x_token| / 7`` (QuaRot's W4A4; no H100 kernel)
``fp8``     ``fp8_e4m3(x / s) * s`` with ``s = max|x_token| / 448`` per token (FP8 W8A8 with fp8 weights)
==========  =========================================================================================

The ``"fake"`` and ``"compressed-tensors"`` backends simulate the scheme with a forward pre-hook (:func:`install`);
the torchao backend runs it inside its int8 tensor. Either way the module is marked with :data:`ACTIVATIONS_ATTR`,
which the compressed-tensors export reads to declare ``input_activations``. A new scheme is one registration::

    ACTIVATION_SCHEMES.register("int6", DynamicIntPerToken("int6", bits=6))
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn

from ...registry import Registry

__all__ = [
    "ACTIVATIONS_ATTR",
    "ACTIVATION_SCHEMES",
    "ActivationScheme",
    "DynamicFp8PerToken",
    "DynamicIntPerToken",
    "activation_scheme_of",
    "install",
    "mark",
    "uninstall",
]

#: Attribute naming the activation scheme of a linear map whose inputs are quantized.
ACTIVATIONS_ATTR = "economize_activations"
_HOOK_ATTR = "_economize_activation_hook"


@dataclass(frozen=True)
class ActivationScheme:
    """How the inputs of a quantized linear map are quantized.

    Attributes:
        name: Registered name (the value of ``QuantConfig.activations``).
        bits: Bits per activation code.
    """

    name: str
    bits: int

    def fake(self, x: Tensor) -> Tensor:
        """``x`` quantized and dequantized in its own dtype.

        Args:
            x: Input of a linear map, ``[..., in_features]``.

        Returns:
            A tensor of the same shape and dtype.
        """
        raise NotImplementedError

    def ct_args(self) -> dict[str, Any]:
        """Keyword arguments of the compressed-tensors ``QuantizationArgs`` that declare the scheme as
        ``input_activations``."""
        raise NotImplementedError


@dataclass(frozen=True)
class DynamicIntPerToken(ActivationScheme):
    """Symmetric integer codes with one absmax scale per token, computed at run time.

    The scale is ``max|x| / (2^(bits-1) - 1)`` (vLLM's ``scaled_int8_quant``, SmoothQuant's ``quantize_per_token``),
    the codes are rounded half to even and clamped to ``[-2^(bits-1), 2^(bits-1) - 1]``. torchao's int8 tensor
    divides by 127.5 instead, which moves a code by at most one step.
    """

    @torch.no_grad()
    def fake(self, x: Tensor) -> Tensor:
        qmax = 2 ** (self.bits - 1) - 1
        scale = x.float().abs().amax(dim=-1, keepdim=True).clamp(min=1e-8) / qmax
        codes = torch.round(x.float() / scale).clamp(-qmax - 1, qmax)
        return (codes * scale).to(x.dtype)

    def ct_args(self) -> dict[str, Any]:
        return {"num_bits": self.bits, "type": "int", "strategy": "token", "dynamic": True, "symmetric": True,
                "observer": None}


@dataclass(frozen=True)
class DynamicFp8PerToken(ActivationScheme):
    """float8 e4m3 codes with one absmax scale ``max|x| / 448`` per token, computed at run time (vLLM's
    ``CompressedTensorsW8A8Fp8`` with dynamic per-token inputs)."""

    @torch.no_grad()
    def fake(self, x: Tensor) -> Tensor:
        finfo = torch.finfo(torch.float8_e4m3fn)
        scale = x.float().abs().amax(dim=-1, keepdim=True).clamp(min=1e-12) / finfo.max
        codes = (x.float() / scale).clamp(finfo.min, finfo.max).to(torch.float8_e4m3fn)
        return (codes.float() * scale).to(x.dtype)

    def ct_args(self) -> dict[str, Any]:
        return {"num_bits": 8, "type": "float", "strategy": "token", "dynamic": True, "symmetric": True,
                "observer": None}


#: Activation schemes by name (entry-point group ``economize.activation_schemes``).
ACTIVATION_SCHEMES: Registry[ActivationScheme] = Registry("activation scheme", "economize.activation_schemes")
ACTIVATION_SCHEMES.register("int8", DynamicIntPerToken("int8", bits=8))
ACTIVATION_SCHEMES.register("int4", DynamicIntPerToken("int4", bits=4))
ACTIVATION_SCHEMES.register("fp8", DynamicFp8PerToken("fp8", bits=8))


def mark(module: nn.Module, name: str) -> None:
    """Record that ``module``'s inputs are quantized with scheme ``name`` (a kernel quantizes them itself).

    Args:
        module: A quantized linear map.
        name: A name registered in :data:`ACTIVATION_SCHEMES`.
    """
    ACTIVATION_SCHEMES.get(name)
    setattr(module, ACTIVATIONS_ATTR, name)


def install(module: nn.Module, name: str) -> None:
    """Simulate scheme ``name`` on ``module``'s inputs with a forward pre-hook (idempotent) and :func:`mark` it.

    Args:
        module: A quantized linear map whose kernel takes float inputs.
        name: A name registered in :data:`ACTIVATION_SCHEMES`.
    """
    uninstall(module)
    scheme = ACTIVATION_SCHEMES.get(name)

    def hook(_module: nn.Module, args: tuple[Any, ...]) -> tuple[Any, ...]:
        return (scheme.fake(args[0]), *args[1:])

    setattr(module, _HOOK_ATTR, module.register_forward_pre_hook(hook))
    mark(module, name)


def uninstall(module: nn.Module) -> None:
    """Remove what :func:`install` or :func:`mark` added to ``module`` (nothing if it added nothing)."""
    handle = getattr(module, _HOOK_ATTR, None)
    if handle is not None:
        handle.remove()
        delattr(module, _HOOK_ATTR)
    if hasattr(module, ACTIVATIONS_ATTR):
        delattr(module, ACTIVATIONS_ATTR)


def activation_scheme_of(module: nn.Module) -> ActivationScheme | None:
    """The activation scheme ``module`` was marked with, or ``None`` for a weight-only map."""
    name = getattr(module, ACTIVATIONS_ATTR, None)
    return ACTIVATION_SCHEMES.get(name) if isinstance(name, str) else None
