"""The default quantization backend ``"auto"``: the fastest installed kernel that stores and runs a configuration.

``QuantConfig()`` and ``DraftQuantConfig()`` use ``backend="auto"``. It chooses among the real kernel backends
registered in :data:`AUTO_QUANT_CANDIDATES`, fastest first, in two steps:

* when the config is built (:meth:`AutoQuantBackend.check_config`), it keeps the candidates whose kernels store the
  layout (format, group size, zero points, backend options) and refuses a layout that none of them stores;
* before any weight changes (:meth:`AutoQuantBackend.check`), it takes the first kept candidate whose own
  :meth:`~.backends.QuantBackend.check` passes on the modules (installed package, GPU, dtype) and delegates every
  :meth:`~.backends.QuantBackend.apply` to it.

==============  ====  ======================================================================================
candidate       rank  runs
==============  ====  ======================================================================================
``machete``     40    vLLM's Machete int4 / int8 GEMM (Hopper)
``marlin``      30    vLLM's Marlin int4 / int8 / fp8 GEMM (sm80+; sm75 in fp16)
``cutlass``     20    vLLM's CUTLASS W8A8 int8 GEMM (``activations="int8"``)
``torchao``     10    torchao tensor subclasses (tinygemm int4 on sm80+ bf16, int8, fp8)
==============  ====  ======================================================================================

When no candidate fits, the error lists each candidate's reason and the packages to install. The codes do not
depend on the kernel (a backend only stores what the method chose), so every candidate yields the outputs of the
``"fake"`` reference up to the kernel's rounding. ``"fake"`` (dequantized weights, any device and format) stays
available as the explicit simulated reference. A new kernel joins the choice with one registration::

    AUTO_QUANT_CANDIDATES.register("my-kernel", AutoCandidate(rank=35, install="pip install my-kernel"))
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar

from torch import nn

from ...errors import ConfigError, UnsupportedModelError
from ...registry import Registry
from .backends import QUANT_BACKENDS, QuantBackend
from .grid import QuantizedWeight

if TYPE_CHECKING:
    from .config import QuantSpec

__all__ = ["AUTO_QUANT_CANDIDATES", "AutoCandidate", "AutoQuantBackend"]

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AutoCandidate:
    """A real kernel backend that ``backend="auto"`` may choose.

    Attributes
    ----------
    rank : int
        Preference; higher ranks are tried first.
    install : str
        What to install when the backend's package is missing (shown in the error when nothing fits).
    """

    rank: int
    install: str


#: The backends ``"auto"`` chooses from, by their name in :data:`~.backends.QUANT_BACKENDS`.
AUTO_QUANT_CANDIDATES: Registry[AutoCandidate] = Registry("automatic quantization kernel", "economize.auto_quant")
AUTO_QUANT_CANDIDATES.register("machete", AutoCandidate(40, "pip install 'economize[vllm]'"))
AUTO_QUANT_CANDIDATES.register("marlin", AutoCandidate(30, "pip install 'economize[vllm]'"))
AUTO_QUANT_CANDIDATES.register("cutlass", AutoCandidate(20, "pip install 'economize[vllm]'"))
AUTO_QUANT_CANDIDATES.register("torchao", AutoCandidate(10, "pip install 'economize[quant]'"))

_OPTIONS = ("fp8_activations", "activations")


def _ranked() -> list[str]:
    names = [name for name in AUTO_QUANT_CANDIDATES.names() if name in QUANT_BACKENDS]
    return sorted(names, key=lambda name: -AUTO_QUANT_CANDIDATES.get(name).rank)


def _layout_reason(name: str, config: QuantSpec) -> str | None:
    """Why candidate ``name`` cannot store the layout of ``config`` (``None``: it can)."""
    backend = QUANT_BACKENDS.get(name)
    for option in _OPTIONS:
        value = getattr(config, option, None)
        if value and option not in backend.options:
            return f"{name} does not read {option}={value!r}"
    try:
        backend().check_config(config)
    except ConfigError as error:
        return str(error)
    return None


def _needs_float_zeros(config: QuantSpec) -> bool:
    from .config import QuantConfig
    from .methods import QUANT_METHODS

    return isinstance(config, QuantConfig) and QUANT_METHODS.get(config.method).float_zeros(config)


def layout_candidates(config: QuantSpec) -> tuple[list[str], dict[str, str]]:
    """The candidates that store the layout of ``config``, fastest first, and why each other one does not.

    Parameters
    ----------
    config : QuantSpec
        The quantization config.

    Returns
    -------
    tuple[list[str], dict[str, str]]
        The names of the fitting candidates and, by name, the reasons of the others.
    """
    fitting: list[str] = []
    refused: dict[str, str] = {}
    for name in _ranked():
        reason = _layout_reason(name, config)
        if reason is None:
            fitting.append(name)
        else:
            refused[name] = reason
    return fitting, refused


def _layout(config: QuantSpec) -> str:
    return f"format {config.format!r} with group_size={config.group_size}, symmetric={config.symmetric}"


_FAKE_HINT = "or choose backend='fake' for the simulated reference (dequantized weights, no memory or speed saving)"


def _no_kernel(what: str, refused: Mapping[str, str], fitting: Sequence[str] = ()) -> ConfigError:
    """The error when no candidate runs ``what``; ``fitting`` names the candidates that store the layout, whose
    packages the message asks to install."""
    reasons = "; ".join(f"{name}: {reason}" for name, reason in refused.items())
    if not fitting:
        return ConfigError(f"backend='auto' found no real kernel that stores {what} ({reasons}). Change the "
                           f"format or group size, {_FAKE_HINT}")
    installs = " or ".join(sorted({AUTO_QUANT_CANDIDATES.get(name).install for name in fitting}))
    return ConfigError(f"backend='auto' found no real kernel that runs {what} ({reasons}). Install a kernel "
                       f"package ({installs}), move the model to a supported GPU and dtype, {_FAKE_HINT}")


@QUANT_BACKENDS.register("auto")
class AutoQuantBackend(QuantBackend):
    """The fastest installed real kernel for the configuration and modules (see :mod:`.auto`)."""

    options = frozenset(_OPTIONS)
    #: No candidate stores non-contiguous groups (GPTQ's plain ``act_order`` runs on ``"fake"`` only).
    supports_g_idx: ClassVar[bool] = False

    def __init__(self) -> None:
        self._chosen: tuple[str, QuantBackend] | None = None

    def check_config(self, config: QuantSpec) -> None:
        fitting, refused = layout_candidates(config)
        if not fitting:
            raise _no_kernel(_layout(config), refused)

    def stores_float_zeros(self, config: QuantSpec) -> bool:
        return any(QUANT_BACKENDS.get(name)().stores_float_zeros(config) for name in layout_candidates(config)[0])

    def unavailable(self, config: QuantSpec) -> str | None:
        fitting, refused = layout_candidates(config)
        for name in fitting:
            reason = QUANT_BACKENDS.get(name)().unavailable(config)
            if reason is None:
                return None
            refused[name] = reason
        return str(_no_kernel(_layout(config), refused, fitting))

    def check(self, modules: Mapping[str, nn.Linear], config: QuantSpec) -> None:
        fitting, refused = layout_candidates(config)
        float_zeros = _needs_float_zeros(config)
        for name in fitting:
            backend = QUANT_BACKENDS.get(name)()
            if float_zeros and not backend.stores_float_zeros(config):
                refused[name] = f"{name} rounds the real-valued zero points that method chooses"
                continue
            try:
                backend.check(modules, config)
            except (ConfigError, UnsupportedModelError) as error:
                refused[name] = str(error)
                continue
            self._chosen = (name, backend)
            logger.info("backend='auto' quantizes %d linear maps with the %s kernels", len(modules), name)
            return
        raise _no_kernel(f"{_layout(config)} on these modules", refused, fitting)

    def resolved(self, config: QuantSpec) -> str:
        return self._chosen[0] if self._chosen is not None else config.backend

    def apply(self, module: nn.Linear, qweight: QuantizedWeight, config: QuantSpec) -> None:
        if self._chosen is None:
            raise RuntimeError("AutoQuantBackend.apply before check: check() chooses the kernel")
        self._chosen[1].apply(module, qweight, config)
