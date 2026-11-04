"""Kernel backends: how a quantized linear map is stored and run.

* ``"fake"``: the dequantized weights stay in the model dtype (simulated quantization). Exact reference for the
  quality of a format and method on any device; no memory or speed gain.
* ``"torchao"``: the codes are wrapped in torchao tensor subclasses that ``F.linear`` dispatches to quantized
  kernels. The tensors are built from the codes the method chose, so GPTQ and AWQ codes run unchanged. One builder
  per format is registered in :data:`TORCHAO_TENSORS`, and what its kernel accepts (group sizes, zero points,
  dtype, GPU) in :data:`TORCHAO_CONSTRAINTS` (:func:`register_torchao_format` does both).

  ==========  ====================================  =========================================================
  format      torchao tensor                        what runs
  ==========  ====================================  =========================================================
  ``int4``    ``Int4TilePackedTo4dTensor``          tinygemm ``_weight_int4pack_mm`` (CUDA sm80+, bf16)
  ``int8``    ``Int8Tensor``                        per channel: ``x @ int8^T * scale`` (fuses under
                                                    ``torch.compile``); per group: dequantize, then matmul;
                                                    ``activations="int8"``: ``torch._int_mm`` (W8A8)
  ``fp8``     ``Float8Tensor``                      weight-only: dequantize, then matmul (memory only);
                                                    ``fp8_activations``: row-wise ``_scaled_mm`` (sm89+)
  ==========  ====================================  =========================================================

  The decoders run eagerly (under CUDA graphs, without ``torch.compile``), so torchao int8 and weight-only fp8
  save memory but decode slower than the dense model; :func:`warn_eager_dequant` says so when such a model is
  built on a GPU.

* ``"marlin"`` (:mod:`.marlin`): vLLM's Marlin weight-only GEMM for int4, int8 and fp8 (sm80+). On Ampere it is
  the fast path for int8 and fp8, which torchao runs by dequantizing the weight on every eager call.
* ``"machete"`` (:mod:`.machete`): vLLM's Machete weight-only GEMM for int4 and int8 on Hopper (sm90), the
  CUTLASS 3 kernel that vLLM prefers over Marlin on H100.
* ``"cutlass"`` (:mod:`.cutlass`): vLLM's CUTLASS int8 x int8 GEMM for W8A8 (``activations="int8"``, sm75+), with
  the per-token input quantization on the GPU, so it is captured in CUDA graphs (torchao's int8 path is not).

A new backend (bitsandbytes, a vLLM export, ...) is one :class:`QuantBackend` subclass registered in
:data:`QUANT_BACKENDS`.
"""

from __future__ import annotations

import importlib.util
import logging
from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, ClassVar, TypeVar

import torch
from torch import Tensor, nn

from ...core.metadata import Saving, Simulation
from ...errors import ConfigError
from ...registry import Registry
from . import activations
from .constraints import KernelConstraints, check_devices, check_layout, format_listing
from .grid import QuantizedWeight

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from .config import QuantSpec

__all__ = [
    "QUANT_BACKENDS",
    "TINYGEMM_GROUP_SIZES",
    "TORCHAO_CONSTRAINTS",
    "TORCHAO_EAGER_DEQUANT",
    "TORCHAO_TENSORS",
    "FakeQuantBackend",
    "QuantBackend",
    "TorchaoBackend",
    "TorchaoTensorBuilder",
    "int4_tinygemm_tensors",
    "register_torchao_format",
    "warn_eager_dequant",
]


class QuantBackend(ABC):
    """Stores a quantized linear map for inference.

    A backend validates in two places. :meth:`check_config` runs when a config is built (formats and group sizes it
    cannot store); :meth:`check` runs once per quantization call, with every module that call will quantize, before
    any weight changes (installed packages, devices, dtypes). Whole models (:func:`~.model.quantize_model`, with a
    :class:`QuantConfig`) and drafter modules (:func:`~.model.quantize_modules`, with a :class:`DraftQuantConfig`)
    go through the same :meth:`check`; both configs satisfy :class:`~.config.QuantSpec`.
    """

    #: The backend-specific fields of :class:`~.config.QuantConfig` (its ``backend_options``) this backend reads.
    options: ClassVar[frozenset[str]] = frozenset()
    #: Whether the backend stores weights whose groups are not contiguous (:attr:`QuantizedWeight.g_idx`).
    supports_g_idx: ClassVar[bool] = False

    def check_config(self, config: QuantSpec) -> None:
        """Raise :class:`ConfigError` for formats or group sizes the backend cannot store."""

    def stores_float_zeros(self, config: QuantSpec) -> bool:
        """Whether the backend stores real-valued zero points (HQQ's) for ``config``; integer-zero kernels round
        them, so a method that chooses them is refused there."""
        return False

    def check(self, modules: Mapping[str, nn.Linear], config: QuantSpec) -> None:
        """Raise before any weight is touched if the installed packages, devices or dtypes cannot run ``modules``.

        Args:
            modules: Every linear map the call will quantize, by name (``"layers.<i>.<name>"`` for decoder layers,
                with ``i`` 0-based as in the module tree).
            config: A :class:`QuantConfig` when a whole model is quantized, a :class:`DraftQuantConfig` for
                drafter-only quantization.
        """

    @abstractmethod
    def apply(self, module: nn.Linear, qweight: QuantizedWeight, config: QuantSpec) -> None:
        """Replace ``module.weight`` by its stored form. The module already holds the dequantized weight."""


QUANT_BACKENDS: Registry[type[QuantBackend]] = Registry("quantization backend", "economize.quant_backends")


@QUANT_BACKENDS.register("fake")
class FakeQuantBackend(QuantBackend):
    """Keep the dequantized weights (simulated quantization)."""

    options = frozenset({"activations"})
    supports_g_idx = True
    simulation = Simulation(Saving.WEIGHT_MEMORY | Saving.LATENCY, "keeps the dequantized weights")

    def stores_float_zeros(self, config: QuantSpec) -> bool:
        return True

    def apply(self, module: nn.Linear, qweight: QuantizedWeight, config: QuantSpec) -> None:
        if config.activations is not None:
            activations.install(module, config.activations)


#: Group sizes of torchao's tinygemm int4 kernel.
TINYGEMM_GROUP_SIZES = (32, 64, 128, 256)

#: Builds the torchao tensor of one format from the codes, the model dtype and the config.
TorchaoTensorBuilder = Callable[[QuantizedWeight, torch.dtype, "QuantSpec"], Tensor]

#: torchao tensor builders by format name; registering one makes the format available on the torchao backend.
TORCHAO_TENSORS: Registry[TorchaoTensorBuilder] = Registry("torchao tensor format", "economize.torchao_tensors")

#: What the torchao kernel of each format accepts (formats missing here are unconstrained). Registered together
#: with the builder by :func:`register_torchao_format`.
TORCHAO_CONSTRAINTS: Registry[KernelConstraints] = Registry("torchao kernel constraints")

#: Whether a config of a format runs by dequantizing the weight on every eager call (slower than the dense model
#: unless the linear maps are compiled). Registered with the builder by :func:`register_torchao_format`; formats
#: missing here run a quantized kernel eagerly.
TORCHAO_EAGER_DEQUANT: Registry[Callable[[QuantSpec], bool]] = Registry("torchao eager dequantization")

_B = TypeVar("_B", bound=TorchaoTensorBuilder)


def register_torchao_format(
    name: str,
    constraints: KernelConstraints | None = None,
    eager_dequant: Callable[[QuantSpec], bool] | None = None,
) -> Callable[[_B], _B]:
    """Decorator: register a torchao tensor builder in :data:`TORCHAO_TENSORS`, if given what its kernel accepts
    in :data:`TORCHAO_CONSTRAINTS`, and if given which of its configs dequantize the weight on every eager call
    in :data:`TORCHAO_EAGER_DEQUANT`; the format is then available on the torchao backend."""

    def decorator(builder: _B) -> _B:
        TORCHAO_TENSORS.register(name, builder)
        if constraints is not None:
            TORCHAO_CONSTRAINTS.register(name, constraints)
        if eager_dequant is not None:
            TORCHAO_EAGER_DEQUANT.register(name, eager_dequant)
        return builder

    return decorator


@QUANT_BACKENDS.register("torchao")
class TorchaoBackend(QuantBackend):
    """torchao tensor subclasses (``pip install torchao``; tested with torchao 0.18), one builder per format in
    :data:`TORCHAO_TENSORS`."""

    options = frozenset({"fp8_activations", "activations"})

    def check_config(self, config: QuantSpec) -> None:
        if config.format not in TORCHAO_TENSORS:
            raise ConfigError(
                f"the torchao backend stores {format_listing(TORCHAO_TENSORS.names())}, got format {config.format!r}"
            )
        check_layout(TORCHAO_CONSTRAINTS, config, self.options)

    def stores_float_zeros(self, config: QuantSpec) -> bool:
        return config.format in TORCHAO_CONSTRAINTS and TORCHAO_CONSTRAINTS.get(config.format).float_zeros

    def check(self, modules: Mapping[str, nn.Linear], config: QuantSpec) -> None:
        _require_torchao()
        check_devices(TORCHAO_CONSTRAINTS, modules, config, self.options)
        warn_eager_dequant(modules, config)

    def apply(self, module: nn.Linear, qweight: QuantizedWeight, config: QuantSpec) -> None:
        tensor = TORCHAO_TENSORS.get(config.format)(qweight, module.weight.dtype, config)
        module.weight = nn.Parameter(tensor, requires_grad=False)
        if config.activations is not None:  # the tensor quantizes its inputs itself
            activations.mark(module, config.activations)


def warn_eager_dequant(modules: Mapping[str, nn.Linear], config: QuantSpec) -> bool:
    """Warn (and return ``True``) when ``config`` makes torchao dequantize CUDA weights on every eager call.

    Such a model saves memory but decodes slower than the dense one (0.50x for int8 and 0.14x for weight-only fp8
    in the B8 reproduction): torchao's own speedups for these formats come from ``torch.compile`` fusing the
    dequantization into the matmul, which the eager CUDA-graph decoders do not do.
    """
    rule = TORCHAO_EAGER_DEQUANT.get(config.format) if config.format in TORCHAO_EAGER_DEQUANT else None
    if rule is None or not rule(config) or not any(m.weight.device.type == "cuda" for m in modules.values()):
        return False
    logger.warning(
        "torchao %s with these settings dequantizes the weight on every eager call: it saves memory but decodes "
        "slower than the dense model unless the linear maps are compiled (torch.compile). For speed use "
        "backend='marlin' (sm80+) or backend='machete' (int4/int8, sm90)%s.",
        config.format, ", or fp8_activations=True (sm89+)" if config.format == "fp8" else "",
    )
    return True


def _require_torchao() -> None:
    if importlib.util.find_spec("torchao") is None:
        raise ConfigError("backend 'torchao' needs the torchao package: pip install 'economize[quant]'")


@register_torchao_format(
    "int8",
    KernelConstraints("torchao's int8 kernel", symmetric="per_channel", options={
        "activations=int8": KernelConstraints(
            "int8 activations (torch._int_mm W8A8)", group_sizes=(None,), symmetric="always"
        ),
    }),
    eager_dequant=lambda config: config.activations != "int8",
)
def _int8_tensor(q: QuantizedWeight, dtype: torch.dtype, config: QuantSpec | None = None) -> Tensor:
    """torchao ``Int8Tensor`` of int8 codes; with ``activations="int8"`` it quantizes its inputs per token and
    multiplies int8 by int8 (``torch._int_mm``, CPU and CUDA), with torchao's activation scale ``max|x| / 127.5``."""
    from torchao.quantization import Int8Tensor

    assert q.zero is not None
    qdata = (q.codes.to(torch.int16) - 128).to(torch.int8)
    block = [1, q.group_size]
    scale = q.scale.to(dtype)
    if q.group_size == q.codes.shape[1]:
        if not bool((q.zero == 128).all()):
            raise ConfigError("torchao per-channel int8 needs symmetric codes (zero point 128)")
        if config is not None and config.activations == "int8":
            from torchao.quantization.granularity import PerRow
            from torchao.quantization.quantize_.workflows.int8.int8_tensor import QuantizeTensorToInt8Kwargs

            kwargs = QuantizeTensorToInt8Kwargs(granularity=PerRow())
            return Int8Tensor(qdata, scale, block, dtype, act_quant_kwargs=kwargs)
        return Int8Tensor(qdata, scale, block, dtype)
    zero = (q.zero - 128).to(torch.int8)
    return Int8Tensor(qdata, scale, block, dtype, zero_point=zero)


def int4_tinygemm_tensors(q: QuantizedWeight, dtype: torch.dtype = torch.bfloat16) -> tuple[Tensor, Tensor, Tensor]:
    """Padded ``codes [out', in']`` (int32), ``scale`` and float ``zero`` ``[out', groups']`` for tinygemm.

    tinygemm dequantizes ``(code - 8) * scale + zero_f``, so the zero point ``z`` of our grid (integer, or real as
    HQQ's) maps to ``zero_f = (8 - z) * scale``. Inputs are padded to a multiple of 1024 and outputs to a multiple
    of 8; padded groups decode to exact zeros.
    """
    assert q.zero is not None and q.grid.bits == 4
    out_features, in_features = q.codes.shape
    out_pad = -out_features % 8
    in_pad = -in_features % 1024
    codes = torch.nn.functional.pad(q.codes.to(torch.int32), (0, in_pad, 0, out_pad), value=8)
    scale = q.scale.float()
    zero = (8 - q.zero.float()) * scale
    group_pad = in_pad // q.group_size
    scale = torch.nn.functional.pad(scale, (0, group_pad, 0, out_pad), value=1.0)
    zero = torch.nn.functional.pad(zero, (0, group_pad, 0, out_pad), value=0.0)
    return codes, scale.to(dtype), zero.to(dtype)


@register_torchao_format("int4", KernelConstraints(
    "tinygemm int4", group_sizes=TINYGEMM_GROUP_SIZES, dtypes=(torch.bfloat16,), min_capability=(8, 0),
    float_zeros=True,
))
def _int4_tensor(q: QuantizedWeight, dtype: torch.dtype, config: QuantSpec | None = None) -> Tensor:
    from torchao.quantization import Int4TilePackedTo4dTensor
    from torchao.quantization.utils import pack_tinygemm_scales_and_zeros

    codes, scale, zero = int4_tinygemm_tensors(q, torch.bfloat16)
    packed = torch.ops.aten._convert_weight_to_int4pack(
        ((codes[:, ::2] << 4) | codes[:, 1::2]).to(torch.uint8).contiguous(), 8
    )
    scale_and_zero = pack_tinygemm_scales_and_zeros(scale, zero, torch.bfloat16)
    return Int4TilePackedTo4dTensor(packed, scale_and_zero, [1, q.group_size], q.codes.shape, act_pre_scale=None)


@register_torchao_format("fp8", KernelConstraints("torchao fp8", options={
    "fp8_activations": KernelConstraints(
        "fp8_activations (row-wise scaled matmuls)", group_sizes=(None,), min_capability=(8, 9)
    ),
}), eager_dequant=lambda config: not config.fp8_activations)
def _fp8_tensor(q: QuantizedWeight, dtype: torch.dtype, config: QuantSpec | None = None) -> Tensor:
    from torchao.quantization import Float8Tensor

    act_kwargs = mm_config = None
    if config is not None and config.fp8_activations:
        from torchao.quantization import Float8MMConfig
        from torchao.quantization.granularity import PerRow
        from torchao.quantization.quantize_.workflows.float8.float8_tensor import QuantizeTensorToFloat8Kwargs

        act_kwargs = QuantizeTensorToFloat8Kwargs(torch.float8_e4m3fn, PerRow())
        # the _scaled_mm settings of torchao's own Float8DynamicActivationFloat8WeightConfig (row-wise scales)
        mm_config = Float8MMConfig(use_fast_accum=True)
    return Float8Tensor(
        q.codes, q.scale.float().contiguous(), [1, q.group_size], act_quant_kwargs=act_kwargs, mm_config=mm_config,
        dtype=dtype,
    )
