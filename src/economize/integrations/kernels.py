"""Custom ops and layer kernels from the Hugging Face ``kernels`` hub, with pure-torch fallbacks.

Two entry points, both optional (without the ``kernels`` package every op runs its pure-torch fallback):

**Ops.** :data:`KERNEL_OPS` maps an op name to a :class:`KernelOp`: a pure-torch reference implementation plus
any number of :class:`HubKernel` sources. :func:`op` returns a callable that, on its first call for a device type,
loads the first hub kernel that serves that device (``kernels.get_kernel``) and otherwise uses the fallback. A hub
kernel that fails to load logs one warning and falls back. ``ECONOMIZE_HUB_KERNELS=0`` disables hub loading. New ops
are registered, and a hub source is attached to an existing op with :func:`add_hub_kernel`::

    from economize.integrations.kernels import HubKernel, add_hub_kernel, op

    add_hub_kernel("masked_linear", HubKernel("my-org/teal-gemv", "masked_linear", version=1))
    y = op("masked_linear")(x, weight, threshold)

**Layers.** :func:`kernelize` applies the ``transformers`` kernel mapping (``from_pretrained(use_kernels=True)``)
to an :class:`~economize.EfficientModel`: the base model through ``set_use_kernels`` and every module a technique added
(trainable copies of blocks, predictors) through ``kernels.kernelize``. The jump path calls the same block modules
as the target, so drafts and verification run the same kernels and lossless decoding stays lossless with respect
to the kernelized target. Resolve ops and kernelize before capturing CUDA graphs; loading downloads from the Hub.
"""

from __future__ import annotations

import dataclasses
import functools
import logging
import operator
import os
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from ..registry import Registry
from ..unwrap import base_model_of

__all__ = [
    "ENV_VAR",
    "KERNEL_OPS",
    "HubKernel",
    "KernelOp",
    "add_hub_kernel",
    "clear_cache",
    "hub_kernels_enabled",
    "kernelize",
    "op",
    "resolve",
]

logger = logging.getLogger(__name__)

#: Debug override: set to ``0`` (or ``false`` / ``no`` / ``off``) to never load kernels from the Hub.
ENV_VAR = "ECONOMIZE_HUB_KERNELS"

Adapter = Callable[[Callable[..., Any]], Callable[..., Tensor]]


@dataclass(frozen=True)
class HubKernel:
    """One function of a kernel repository on the Hub.

    Args:
        repo_id: Kernel repository, e.g. ``"kernels-community/activation"``.
        function: Attribute of the loaded kernel module.
        version: Major version of the kernel (``kernels.get_kernel(version=...)``).
        revision: A branch, tag or commit instead of a version.
        devices: Device types the kernel runs on.
        adapter: Wraps the hub function into the op's calling convention (e.g. out-parameter kernels).
        trust_remote_code: Passed to ``kernels.get_kernel`` (repositories outside trusted organisations).
    """

    repo_id: str
    function: str
    version: int | None = None
    revision: str | None = None
    devices: tuple[str, ...] = ("cuda",)
    adapter: Adapter | None = None
    trust_remote_code: bool = False

    def load(self) -> Callable[..., Tensor]:
        import kernels

        module = kernels.get_kernel(self.repo_id, version=self.version, revision=self.revision,
                                    trust_remote_code=self.trust_remote_code)
        function = getattr(module, self.function)
        return self.adapter(function) if self.adapter is not None else function


@dataclass(frozen=True)
class KernelOp:
    """An op: the pure-torch reference implementation and the hub kernels that may replace it, in order."""

    fallback: Callable[..., Tensor]
    hub: tuple[HubKernel, ...] = field(default=())


#: Registered ops; third-party packages add ops through the ``economize.kernel_ops`` entry-point group.
KERNEL_OPS: Registry[KernelOp] = Registry("kernel op", "economize.kernel_ops")

_resolved: dict[tuple[str, str], Callable[..., Tensor]] = {}


def hub_kernels_enabled() -> bool:
    """Whether hub kernels may be loaded: the ``kernels`` package is installed and :data:`ENV_VAR` allows it."""
    if os.environ.get(ENV_VAR, "1").strip().lower() in {"0", "false", "no", "off"}:
        return False
    try:
        import kernels  # noqa: F401  (the import is the availability check)
    except ImportError:
        return False
    return True


def resolve(name: str, device: torch.device | str) -> Callable[..., Tensor]:
    """The implementation of op ``name`` on ``device`` (cached per device type until :func:`clear_cache`)."""
    device_type = torch.device(device).type
    key = (name, device_type)
    if key not in _resolved:
        _resolved[key] = _load(name, KERNEL_OPS.get(name), device_type)
    return _resolved[key]


def _load(name: str, kernel_op: KernelOp, device_type: str) -> Callable[..., Tensor]:
    candidates = [hub for hub in kernel_op.hub if device_type in hub.devices]
    if not candidates or not hub_kernels_enabled():
        return kernel_op.fallback
    for hub in candidates:
        try:
            function = hub.load()
        except Exception as error:  # a missing build, a network error, an incompatible GPU: all fall back
            logger.warning("economize: kernel %s from %s failed to load on %s (%s: %s)", name, hub.repo_id,
                           device_type, type(error).__name__, error)
            continue
        logger.info("economize: op %s on %s uses %s:%s", name, device_type, hub.repo_id, hub.function)
        return function
    logger.warning("economize: op %s on %s uses its pure-torch fallback", name, device_type)
    return kernel_op.fallback


def clear_cache() -> None:
    """Forget resolved implementations (after changing :data:`ENV_VAR` or re-registering an op)."""
    _resolved.clear()


def add_hub_kernel(name: str, hub: HubKernel, *, first: bool = True) -> None:
    """Attach a hub kernel to the registered op ``name`` (tried before the existing ones unless ``first=False``)."""
    kernel_op = KERNEL_OPS.get(name)
    sources = (hub, *kernel_op.hub) if first else (*kernel_op.hub, hub)
    KERNEL_OPS.register(name, dataclasses.replace(kernel_op, hub=sources), override=True)
    for key in [key for key in _resolved if key[0] == name]:
        del _resolved[key]


class _Op:
    """Calls op ``name`` with the implementation for the device of its first tensor argument."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.__doc__ = KERNEL_OPS.get(name).fallback.__doc__

    def __call__(self, *args: Any, **kwargs: Any) -> Tensor:
        device = next((a.device for a in (*args, *kwargs.values()) if isinstance(a, Tensor)), torch.device("cpu"))
        return resolve(self.name, device)(*args, **kwargs)

    def __repr__(self) -> str:
        return f"op({self.name!r})"


def op(name: str) -> Callable[..., Tensor]:
    """The registered op ``name``, dispatching per device type (see the module documentation)."""
    return _Op(name)


# ---- layers --------------------------------------------------------------------------------------------------------
def kernelize(model: nn.Module, *, mode: Any = None, kernel_config: Any = None, use_fallback: bool = True) -> nn.Module:
    """Replace layer forwards of ``model`` (an ``EfficientModel`` or a ``transformers`` model) by hub kernels.

    Args:
        model: The model; it is changed in place and returned.
        mode: A ``kernels.Mode``, or its name(s) such as ``"inference"`` or ``"training|torch_compile"``. Default:
            training or inference after the base model's ``training`` flag (``from_pretrained`` models are in
            eval mode).
        kernel_config: A ``transformers.KernelConfig`` with a custom mapping for the base model.
        use_fallback: Keep the original forward of layers without a kernel for the device (else raise).
    """
    try:
        import kernels
    except ImportError as error:
        raise ImportError("economize.integrations.kernels.kernelize needs the kernels package: "
                          "pip install 'economize[kernels]'") from error
    base = base_model_of(model)
    resolved = _mode(kernels.Mode, mode, base.training)
    if callable(getattr(base, "set_use_kernels", None)):
        base.set_use_kernels(True, kernel_config, mode=resolved)  # the transformers mapping, kept across train/eval
    else:
        kernels.kernelize(base, mode=resolved, use_fallback=use_fallback)
    added = getattr(model, "techniques", None) if model is not base else None
    if isinstance(added, nn.Module) and any(True for _ in added.modules()):
        device = next(base.parameters()).device
        kernels.kernelize(added, mode=resolved, device=device.type, use_fallback=use_fallback)
    return model


def _mode(mode_type: Any, mode: Any, training: bool) -> Any:
    if mode is None:
        return mode_type.TRAINING if training else mode_type.INFERENCE
    if isinstance(mode, str):
        return functools.reduce(operator.or_, (mode_type[part.strip().upper()] for part in mode.split("|")))
    return mode


# ---- built-in ops --------------------------------------------------------------------------------------------------
def _halving_out_param(function: Callable[..., Any]) -> Callable[..., Tensor]:
    """vLLM-style gated activations ``function(out, x)`` with ``out`` half as wide as ``x``."""

    def call(x: Tensor) -> Tensor:
        out = x.new_empty(*x.shape[:-1], x.shape[-1] // 2)
        function(out, x.contiguous())
        return out

    return call


def silu_and_mul(x: Tensor) -> Tensor:
    """SwiGLU gating: ``silu(x[..., :d]) * x[..., d:]`` for ``x`` of width ``2d``."""
    gate, up = x.chunk(2, dim=-1)
    return F.silu(gate) * up


def gelu_tanh_and_mul(x: Tensor) -> Tensor:
    """GeGLU gating with the tanh approximation: ``gelu_tanh(x[..., :d]) * x[..., d:]``."""
    gate, up = x.chunk(2, dim=-1)
    return F.gelu(gate, approximate="tanh") * up


def masked_linear(x: Tensor, weight: Tensor, threshold: Tensor, bias: Tensor | None = None) -> Tensor:
    """TEAL's sparse matmul: ``linear(x * 1[|x| > threshold], weight, bias)``. A sparse GEMV kernel skips the
    weight columns of zeroed inputs; this reference computes the same values densely."""
    return F.linear(torch.where(x.abs() > threshold, x, torch.zeros_like(x)), weight, bias)


_ACTIVATION = "kernels-community/activation"
_GPU = ("cuda", "rocm", "xpu")
KERNEL_OPS.register("silu_and_mul", KernelOp(silu_and_mul, (
    HubKernel(_ACTIVATION, "silu_and_mul", version=1, devices=_GPU, adapter=_halving_out_param),)))
KERNEL_OPS.register("gelu_tanh_and_mul", KernelOp(gelu_tanh_and_mul, (
    HubKernel(_ACTIVATION, "gelu_tanh_and_mul", version=1, devices=_GPU, adapter=_halving_out_param),)))
KERNEL_OPS.register("masked_linear", KernelOp(masked_linear))
