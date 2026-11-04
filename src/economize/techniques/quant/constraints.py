"""What a quantized kernel accepts: the per-format constraints that kernel backends check generically.

A backend that stores several formats keeps one :class:`KernelConstraints` per format in a registry (the torchao
backend's :data:`~.backends.TORCHAO_CONSTRAINTS`, the Marlin backend's :data:`~.marlin.MARLIN_FORMATS`) and asks
:func:`check_layout` when a config is built and :func:`check_devices` before any weight changes. A new format is
then one registration in a new module; no backend code changes::

    from economize.techniques.quant import KernelConstraints, register_torchao_format

    @register_torchao_format("int6", KernelConstraints("my int6 kernel", group_sizes=(64, 128)))
    def int6_tensor(q, dtype, config=None):
        ...

The messages name the kernel (:attr:`KernelConstraints.kernel`) and the rule it breaks, e.g. ``"tinygemm int4 needs
group_size in (32, 64, 128, 256), got 48"``.
"""

from __future__ import annotations

from collections.abc import Collection, Iterable, Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

import torch
from torch import nn

from ...errors import ConfigError, UnsupportedModelError
from ...registry import Registry
from .grid import QUANT_FORMATS, make_grid

if TYPE_CHECKING:
    from .config import QuantSpec

__all__ = ["UNCONSTRAINED", "KernelConstraints", "check_devices", "check_layout", "format_listing"]

#: Which codes need no zero point: ``"any"`` (both kinds run), ``"per_channel"`` (per-channel scales need
#: symmetric codes), ``"always"`` (every layout needs symmetric codes).
SymmetricRule = Literal["any", "per_channel", "always"]


@dataclass(frozen=True)
class KernelConstraints:
    """The layouts, dtypes and GPUs one kernel accepts for one format.

    Attributes:
        kernel: Name of the kernel in error messages (``"tinygemm int4"``, ``"Marlin fp8"``).
        group_sizes: Accepted group sizes, ``None`` among them for per-channel scales (``group_size=None``); the
            whole field ``None`` accepts any group size.
        symmetric: Where the kernel has no zero points (see :data:`SymmetricRule`).
        dtypes: Accepted dtypes of the weights it replaces (``None``: any).
        min_capability: Minimal CUDA compute capability (``None``: any device, CPU included).
        float_zeros: Whether the kernel stores real-valued zero points (tinygemm's ``(code - 8) * scale + zero``);
            integer zero-point kernels do not.
        options: Further constraints that apply when a backend option of the config is set, keyed by the option's
            name for a flag (``"fp8_activations"``) and by ``"name=value"`` for a string option
            (``"activations=int8"``); a set option missing here is rejected for this format.
    """

    kernel: str
    group_sizes: tuple[int | None, ...] | None = None
    symmetric: SymmetricRule = "any"
    dtypes: tuple[torch.dtype, ...] | None = None
    min_capability: tuple[int, int] | None = None
    float_zeros: bool = False
    options: Mapping[str, KernelConstraints] = field(default_factory=dict)

    def layout_error(self, group_size: int | None, symmetric: bool) -> str | None:
        """Why the kernel cannot store codes with ``group_size`` (``None``: per channel) and zero points
        (``symmetric=False``), or ``None`` if it can. The group size is checked first."""
        if self.group_sizes is not None and group_size not in self.group_sizes:
            return f"{self.kernel} needs group_size {_listing_of_groups(self.group_sizes)}, got {group_size}"
        if not symmetric and (self.symmetric == "always" or (self.symmetric == "per_channel" and group_size is None)):
            where = " per channel" if self.symmetric == "per_channel" else ""
            return f"{self.kernel} has no zero points{where}: use symmetric=True"
        return None

    def device_error(self, name: str, weight: torch.Tensor) -> str | None:
        """Why the kernel cannot run the weight ``name`` on its device and dtype, or ``None`` if it can. The device
        is checked first."""
        if self.min_capability is not None:
            cuda = weight.device.type == "cuda"
            if not cuda or torch.cuda.get_device_capability(weight.device) < self.min_capability:
                major, minor = self.min_capability
                return (f"{self.kernel} needs a CUDA GPU with compute capability {major}.{minor} or newer; {name} is "
                        f"on {weight.device}")
        if self.dtypes is not None and weight.dtype not in self.dtypes:
            accepted = " or ".join(str(dtype).removeprefix("torch.") for dtype in self.dtypes)
            return f"{self.kernel} needs {accepted} weights; {name} is {weight.dtype}"
        return None


#: The constraints of a format that a backend registered without any.
UNCONSTRAINED = KernelConstraints("this kernel")


def _listing_of_groups(sizes: tuple[int | None, ...]) -> str:
    groups = tuple(size for size in sizes if size is not None)
    if None not in sizes:
        return f"in {groups}"
    if not groups:
        return "None"
    return f"None or {groups[0]}" if len(groups) == 1 else f"None or one of {groups}"


def format_listing(formats: Iterable[str]) -> str:
    """``"int4, int8 and fp8"``: integer formats by bits, then floating-point formats, then unknown names."""

    def key(name: str) -> tuple[int, int, str]:
        if name not in QUANT_FORMATS:
            return 2, 0, name
        grid = make_grid(name, True)
        return int(grid.code_dtype.is_floating_point), grid.bits, name

    names = sorted(formats, key=key)
    return ", ".join(names[:-1]) + f" and {names[-1]}" if len(names) > 1 else "".join(names)


def _active(table: Registry[KernelConstraints], config: QuantSpec, options: Collection[str]) -> list[KernelConstraints]:
    """The constraints of ``config.format`` and of each backend option in ``options`` that the config sets (a
    string-valued option is looked up as ``"name=value"``)."""
    base = table.get(config.format) if config.format in table else UNCONSTRAINED
    active = [base]
    for option in sorted(options):
        value = getattr(config, option, None)
        if not value:
            continue
        key = option if value is True else f"{option}={value}"
        if key not in base.options:
            formats = [name for name in table.names() if key in table.get(name).options]
            accepted = " or ".join(repr(name) for name in formats) if formats else "none of this backend's"
            raise ConfigError(f"{key} needs format={accepted}, got {config.format!r}")
        active.append(base.options[key])
    return active


def check_layout(table: Registry[KernelConstraints], config: QuantSpec, options: Collection[str] = ()) -> None:
    """Raise :class:`ConfigError` if a kernel of ``table`` cannot store the layout of ``config``.

    Args:
        table: Constraints by format name (formats missing from it are unconstrained).
        config: The quantization config (its format, group size, grid and backend options).
        options: The backend options to check (:attr:`~.backends.QuantBackend.options`).
    """
    for constraints in _active(table, config, options):
        reason = constraints.layout_error(config.group_size, config.grid.symmetric)
        if reason is not None:
            raise ConfigError(reason)


def check_devices(table: Registry[KernelConstraints], modules: Mapping[str, nn.Linear], config: QuantSpec,
                  options: Collection[str] = ()) -> None:
    """Raise :class:`UnsupportedModelError` if a kernel of ``table`` cannot run one of ``modules`` (its device or
    dtype)."""
    active = _active(table, config, options)
    for name, module in modules.items():
        for constraints in active:
            reason = constraints.device_error(name, module.weight)
            if reason is not None:
                raise UnsupportedModelError(reason)
