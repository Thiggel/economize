"""Configurations of weight quantization (``quant``) and drafter-only quantization (``draft_quant``)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar, Protocol, runtime_checkable

from ...core.config import TechniqueConfig
from ...errors import ConfigError
from ...validation import (
    check_globs,
    check_layers,
    check_options,
    check_positive,
    check_registered,
    check_unit_interval,
)
from .grid import QUANT_FORMATS, QuantGrid, make_grid

if TYPE_CHECKING:
    from ...arch import ArchAdapter

__all__ = ["DraftQuantConfig", "QuantConfig", "QuantSpec"]


@runtime_checkable
class QuantSpec(Protocol):
    """What a kernel backend reads of a config: :class:`QuantConfig` and :class:`DraftQuantConfig` both satisfy it."""

    @property
    def format(self) -> str: ...

    @property
    def group_size(self) -> int | None: ...

    @property
    def symmetric(self) -> bool: ...

    @property
    def backend(self) -> str: ...

    @property
    def fp8_activations(self) -> bool: ...

    @property
    def activations(self) -> str | None: ...

    @property
    def grid(self) -> QuantGrid: ...


def _check_common(config: QuantConfig | DraftQuantConfig) -> None:
    from .backends import QUANT_BACKENDS

    check_registered("quantization format", config.format, QUANT_FORMATS)
    check_registered("quantization backend", config.backend, QUANT_BACKENDS)
    if config.group_size is not None:
        check_positive("group_size", config.group_size)
    config.grid.check_group_size(config.group_size)
    check_globs("targets", config.targets)
    check_globs("exclude", config.exclude, allow_empty=True)
    if config.activations is not None:
        from .activations import ACTIVATION_SCHEMES

        check_registered("activation scheme", config.activations, ACTIVATION_SCHEMES)
    backend = QUANT_BACKENDS.get(config.backend)
    check_options(config, f"backend={config.backend!r}", backend.options, type(config).backend_options)
    backend().check_config(config)


def _check_float_zeros(config: QuantConfig, float_zeros: bool) -> None:
    from .backends import QUANT_BACKENDS

    if float_zeros and not QUANT_BACKENDS.get(config.backend)().stores_float_zeros(config):
        raise ConfigError(
            f"method {config.method!r} chooses real-valued zero points, which backend {config.backend!r} cannot "
            f"store for format {config.format!r}: use backend='fake', torchao int4, or the method's integer-zero "
            f"option"
        )


@dataclass(frozen=True)
class QuantConfig(TechniqueConfig):
    """Weight-only quantization of the linear maps of the decoder layers (the LM head stays in full precision).

    Args:
        format: Registered number format: ``"int4"`` (default), ``"int8"``, ``"int3"``, ``"int2"``, ``"fp8"``
            (e4m3, absmax scale), or the FP4 microscaling formats ``"nvfp4"`` and ``"mxfp4"`` (:mod:`.fp4`).
        method: ``"rtn"`` (round to nearest, no data), ``"hqq"`` (half-quadratic zero points, no data), ``"gptq"``,
            ``"awq"`` or ``"smoothquant"`` (these need calibration data); see :mod:`.methods`.
        group_size: Input columns per scale (``None``: one scale per output channel).
        symmetric: Integer grids without a zero point.
        backend: ``"fake"`` (dequantized weights, any device), ``"torchao"`` (torchao kernels), ``"marlin"``
            (vLLM's Marlin weight-only GEMM, the fast int8 / fp8 path on A100), ``"machete"`` (vLLM's Machete
            int4 / int8 GEMM on H100), ``"cutlass"`` (vLLM's CUTLASS int8 x int8 GEMM, W8A8 with
            ``activations="int8"``) or ``"compressed-tensors"`` (dequantized weights plus the codes, for an
            export as a compressed-tensors checkpoint, :mod:`economize.integrations.compressed_tensors`); see
            :mod:`.backends`.
        targets: Globs over linear-module names inside a layer.
        exclude: Globs removed from ``targets``.
        layers: 1-indexed layers to quantize; empty means all.
        act_order: GPTQ: quantize columns by decreasing Hessian diagonal.
        static_groups: GPTQ with ``act_order`` and groups: fit the groups on contiguous columns first, so they stay
            contiguous (default, runs on every backend); ``False`` is the reference's plain ``act_order``, whose
            permuted groups only the ``"fake"`` backend stores.
        true_sequential: GPTQ: quantize the maps that share an input together, in call order.
        blocksize: GPTQ lazy-update block.
        percdamp: GPTQ relative Hessian damping.
        keep_zeros: GPTQ: keep exactly-zero (pruned) weights at zero (``None``: when at least 5% of a weight is
            exactly zero; see :mod:`.gptq`).
        awq_grid: AWQ: number of scale exponents and clipping ratios searched (20 in the paper).
        awq_clip: AWQ: search per-group clipping thresholds.
        hqq_iters: HQQ: maximal number of proximal steps (20 in the reference).
        hqq_lp_norm: HQQ: exponent ``p`` of the error norm (0.7).
        hqq_beta: HQQ: half-quadratic penalty weight (10).
        hqq_round_zero: HQQ: round the initial zero points (the reference's ``round_zero``).
        hqq_integer_zero: HQQ: round the final zero points, for backends without float zero points (Marlin,
            Machete, compressed-tensors); the reference keeps them real.
        smooth_alpha: SmoothQuant: migration strength ``alpha`` of the scales ``max|x|^alpha / max|w|^(1-alpha)``
            (0.5 in the paper, 0.8-0.9 for models with strong outliers).
        fp8_activations: torchao fp8: also quantize activations per row and run ``_scaled_mm`` (W8A8, sm89+).
        activations: Also quantize the inputs of the quantized linear maps with this registered scheme
            (:mod:`.activations`): ``"int8"`` (dynamic per token; with int8 per-channel symmetric weights this is
            W8A8, which the torchao backend runs on ``torch._int_mm`` and vLLM on its CUTLASS int8 GEMM) or
            ``"int4"`` (W4A4, simulated only). ``None`` (default): weight-only.
        offload: Keep calibration activations on the CPU between layers.

    The fields in :attr:`method_options` are read by some methods only (each method lists its own in
    :attr:`~.methods.QuantMethod.options`), and those in :attr:`backend_options` by some backends only
    (:attr:`~.backends.QuantBackend.options`). Setting one that the chosen method or backend ignores is an error.
    """

    technique: ClassVar[str] = "quant"
    method_options: ClassVar[tuple[str, ...]] = (
        "act_order",
        "static_groups",
        "true_sequential",
        "blocksize",
        "percdamp",
        "keep_zeros",
        "awq_grid",
        "awq_clip",
        "hqq_iters",
        "hqq_lp_norm",
        "hqq_beta",
        "hqq_round_zero",
        "hqq_integer_zero",
        "smooth_alpha",
    )
    backend_options: ClassVar[tuple[str, ...]] = ("fp8_activations", "activations")

    format: str = "int4"
    method: str = "rtn"
    group_size: int | None = 128
    symmetric: bool = False
    backend: str = "fake"
    targets: tuple[str, ...] = ("*",)
    exclude: tuple[str, ...] = ()
    layers: tuple[int, ...] = ()
    act_order: bool = False
    static_groups: bool = True
    true_sequential: bool = True
    blocksize: int = 128
    percdamp: float = 0.01
    keep_zeros: bool | None = None
    awq_grid: int = 20
    awq_clip: bool = True
    hqq_iters: int = 20
    hqq_lp_norm: float = 0.7
    hqq_beta: float = 10.0
    hqq_round_zero: bool = False
    hqq_integer_zero: bool = False
    smooth_alpha: float = 0.5
    fp8_activations: bool = False
    activations: str | None = None
    offload: bool = False

    def __post_init__(self) -> None:
        from .methods import QUANT_METHODS

        check_registered("quantization method", self.method, QUANT_METHODS)
        check_layers(self.layers)
        check_positive("blocksize", self.blocksize)
        check_positive("awq_grid", self.awq_grid)
        check_unit_interval("percdamp", self.percdamp, closed_high=True)
        method = QUANT_METHODS.get(self.method)
        check_options(self, f"method={self.method!r}", method.options, self.method_options)
        _check_common(self)
        method.check_config(self)
        _check_float_zeros(self, method.float_zeros(self))

    @property
    def grid(self) -> QuantGrid:
        return make_grid(self.format, self.symmetric)

    def validate(self, adapter: ArchAdapter) -> None:
        from .methods import QUANT_METHODS

        check_layers(self.layers, adapter)
        QUANT_METHODS.get(self.method).validate(self, adapter)


@dataclass(frozen=True)
class DraftQuantConfig(TechniqueConfig):
    """Round-to-nearest quantization of the drafter's own modules only (the verified base model is untouched).

    Lossless: every drafted token is still verified by the full-precision model, so only the acceptance rate
    changes. The drafter's predictor, readout and full layer copies are quantized; LoRA factors and modules whose
    weights are shared with the base model are skipped.

    Args:
        format: Registered number format (``"int8"`` by default).
        group_size: Input columns per scale (``None``: per output channel). Modules whose input size it does not
            divide are skipped.
        symmetric: Integer grids without a zero point.
        backend: ``"fake"``, ``"torchao"``, ``"marlin"`` or ``"machete"``.
        targets: Globs over module names inside the drafter (``"predictor.*"``, ``"blocks.*"``).
        exclude: Globs removed from ``targets``.
    """

    technique: ClassVar[str] = "draft_quant"
    backend_options: ClassVar[tuple[str, ...]] = ()

    format: str = "int8"
    group_size: int | None = 128
    symmetric: bool = True
    backend: str = "fake"
    targets: tuple[str, ...] = ("*",)
    exclude: tuple[str, ...] = ("*lora*",)
    #: Drafter quantization is weight-only (see :class:`QuantSpec`).
    fp8_activations: ClassVar[bool] = False
    #: Drafter quantization keeps float activations.
    activations: ClassVar[str | None] = None

    def __post_init__(self) -> None:
        _check_common(self)

    @property
    def grid(self) -> QuantGrid:
        return make_grid(self.format, self.symmetric)
