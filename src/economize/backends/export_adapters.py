"""Export adapters shared by every file-based runtime: the weight techniques a checkpoint holds as plain weights.

A depth-pruned model is a plain model, which every runtime runs (:class:`DepthPruneExport`). Pruned, low-rank and
LittleBit weights become plain weights only when they are multiplied out
(:func:`~economize.backends.runtime_export.dense_state`), which drops their whole saving, so their shared adapters
refuse them: :class:`PruneExport` exports only the explicit masked-dense reference (``semi_structured=False``,
simulated), and :class:`LowRankExport` and :class:`LittleBitExport` are ``UNSUPPORTED``. A runtime whose format
stores the factored form registers its own adapter over the shared one (``override=True``; ONNX Runtime does for
``lowrank``). Quantization depends on the runtime's own formats; :class:`QuantExport` reads that from a runtime's
subclass (:meth:`QuantExport.stores`, :meth:`QuantExport.stores_weight`). A runtime registers the shared adapters
with :func:`register_weight_adapters`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

from torch import nn

from ..core.config import TechniqueConfig
from .exporters import UnsupportedBackendError
from .runtime_export import ExportBuilder, ExportRuntime, ExportTechnique, config_of
from .vllm.serving.base import Role, Support

if TYPE_CHECKING:
    from ..techniques.quant.grid import QuantizedWeight

__all__ = [
    "PRUNE_REFERENCE_HINT",
    "DepthPruneExport",
    "LittleBitExport",
    "LowRankExport",
    "PruneExport",
    "QuantExport",
    "StreamingWindowExport",
    "float_zero_points",
    "int_bits",
    "register_weight_adapters",
]


#: How to ask for the masked dense pruning reference (the end of every pruning refusal).
PRUNE_REFERENCE_HINT = ("pass prune(semi_structured=False) to export the masked dense reference (simulated: same "
                        "outputs, no memory or speed saving)")


class PruneExport(ExportTechnique):
    """``prune``: refused, except the explicit masked-dense reference, whose weights are written dense.

    The file formats of these runtimes hold dense weights and their engines have no 2:4 sparse kernel, so a pruned
    model would run at the dense model's cost. ``semi_structured=False`` asks for that reference explicitly; it is
    exported dense and reported as simulated (``DENSIFIED``).
    """

    technique = "prune"
    role = Role.WEIGHTS
    support = Support.UNSUPPORTED
    reason = ("the runtime's file format stores dense weights and its engine has no 2:4 sparse kernel, so the pruned "
              "model would run at dense cost; " + PRUNE_REFERENCE_HINT)
    #: What the reference export writes.
    reference_reason: ClassVar[str] = ("semi_structured=False (the simulated reference): the masked weights are "
                                       "written dense")

    def level(self, config: TechniqueConfig) -> Support:
        return Support.DENSIFIED if getattr(config, "semi_structured", None) is False else Support.UNSUPPORTED

    def reason_for(self, config: TechniqueConfig) -> str:
        return self.reference_reason if self.level(config) is Support.DENSIFIED else self.reason

    def contribute(self, technique: nn.Module, builder: ExportBuilder) -> None:
        builder.note(self.technique, "pruned weights are written dense (zeros kept): same outputs, no sparse speedup")


class LowRankExport(ExportTechnique):
    """``lowrank``: refused (the runtime's format has one weight per linear map, no factored form).

    A runtime whose format stores the two factors registers its own adapter in place of this one."""

    technique = "lowrank"
    role = Role.WEIGHTS
    support = Support.UNSUPPORTED
    reason = ("the runtime's format holds one weight per linear map (no factored linear), so the factors would be "
              "multiplied out into the dense weights and the memory and compute saving lost; export lowrank to "
              "ONNX Runtime or serve it with vLLM, which run the factors")


class LittleBitExport(ExportTechnique):
    """``littlebit``: refused (no runtime stores or multiplies binary factors)."""

    technique = "littlebit"
    role = Role.WEIGHTS
    support = Support.UNSUPPORTED
    reason = ("the runtime has no format for LittleBit's binary factors and no LittleBit GEMM, so they would be "
              "multiplied out into dense weights and the whole saving lost; run littlebit with the transformers "
              "backend")


class DepthPruneExport(ExportTechnique):
    """``depth_prune``: the shallower model is a plain model."""

    technique = "depth_prune"
    role = Role.WEIGHTS
    support = Support.NATIVE
    reason = "the shallower model (layers removed, config renumbered) is a plain model"


class QuantExport(ExportTechnique):
    """``quant``: codes the runtime has a format for are stored as such; other layouts are refused.

    A runtime subclasses this with :meth:`layout_reason` (which configured layouts its file format stores) and
    :meth:`stores_weight` (whether one recorded weight fits that format). The level of a configuration is

    * ``UNSUPPORTED`` for fp8 activations (:attr:`fp8_activations_reason`) and for a layout the format does not
      store (:meth:`layout_reason`), since writing such weights dequantized would drop the whole saving;
    * ``DENSIFIED`` for the explicit simulated reference ``backend="fake"``, whose dequantized weights are written
      dense;
    * ``NATIVE`` otherwise: the codes are copied into the runtime's format. A single weight that does not fit
      (:meth:`stores_weight`, e.g. an input width the group size does not divide) is still written dequantized,
      and the export then reports the technique as ``DENSIFIED``.
    """

    technique = "quant"
    role = Role.WEIGHTS
    support = Support.UNSUPPORTED
    reason = "the runtime has no quantized weight format"
    #: Why a W8A8 configuration is refused.
    fp8_activations_reason: ClassVar[str] = ("fp8 activations (W8A8) have no export to this runtime: export the "
                                             "weight-only model")

    def layout_reason(self, config: TechniqueConfig) -> str | None:
        """Why the runtime's format cannot store the codes ``config`` produces (``None``: it can).

        Parameters
        ----------
        config : TechniqueConfig
            The ``quant`` configuration (format, group size, symmetry).

        Returns
        -------
        str or None
            The reason; the default, a runtime without a quantized format, returns :attr:`reason`.
        """
        return self.reason

    def stores(self, config: TechniqueConfig) -> bool:
        """Whether the export keeps the codes of weights quantized by ``config`` in the runtime's format.

        Parameters
        ----------
        config : TechniqueConfig
            The ``quant`` configuration.

        Returns
        -------
        bool
            ``True`` for a real kernel backend (anything but ``"fake"``) and a layout the format stores.
        """
        return getattr(config, "backend", "fake") != "fake" and self.layout_reason(config) is None

    def stores_weight(self, name: str, weight: QuantizedWeight) -> str | None:
        """Why the runtime cannot store one recorded weight in its format (``None``: it can).

        Parameters
        ----------
        name : str
            The qualified name of the linear map.
        weight : QuantizedWeight
            Its codes.

        Returns
        -------
        str or None
            The reason the weight is written dequantized instead.
        """
        return None

    def level(self, config: TechniqueConfig) -> Support:
        if getattr(config, "fp8_activations", False):
            return Support.UNSUPPORTED
        if getattr(config, "backend", "fake") == "fake":
            return Support.DENSIFIED
        return Support.NATIVE if self.layout_reason(config) is None else Support.UNSUPPORTED

    def reason_for(self, config: TechniqueConfig) -> str:
        """The reason shown for ``config``: why it is refused, or what the runtime runs.

        Parameters
        ----------
        config : TechniqueConfig
            The ``quant`` configuration.

        Returns
        -------
        str
            :attr:`fp8_activations_reason`, the :meth:`layout_reason` of a refused layout, or :attr:`reason`.
        """
        if getattr(config, "fp8_activations", False):
            return self.fp8_activations_reason
        if getattr(config, "backend", "fake") == "fake":
            return "backend='fake' (the simulated reference): the dequantized weights are written dense"
        layout = self.layout_reason(config)
        if layout is not None:
            return (f"{layout}; choose a layout the format stores or backend='fake' to export the dequantized "
                    "reference")
        return self.reason

    def contribute(self, technique: nn.Module, builder: ExportBuilder) -> None:
        from ..integrations.compressed_tensors import quantized_modules

        config = config_of(technique)
        if self.level(config) is not Support.NATIVE:
            builder.note(self.technique, f"backend {getattr(config, 'backend', '?')!r}: the dequantized weights are "
                                         "written dense (same outputs, no memory or speed saving)")
            return
        recorded = quantized_modules(builder.model)
        refused = {name: why for name, weight in recorded.items() if (why := self.stores_weight(name, weight))}
        builder.quantized.update({name: weight for name, weight in recorded.items() if name not in refused})
        if refused or not recorded:
            example = next(iter(refused.items()), ("", "no codes were recorded"))
            builder.densify(self.technique, f"{len(refused)} of {len(recorded)} quantized maps are written "
                                            f"dequantized (e.g. {example[0]}: {example[1]})")
        if builder.quantized:
            builder.note(self.technique, f"{len(builder.quantized)} linear maps keep their codes in "
                                         f"{builder.runtime.title}'s format")


def int_bits(config: TechniqueConfig) -> int | None:
    """The bit width of an integer ``quant`` format (``None`` for float formats such as ``fp8`` or ``nvfp4``).

    Parameters
    ----------
    config : TechniqueConfig
        The ``quant`` configuration.

    Returns
    -------
    int or None
        The bits per code of an ``int*`` format.
    """
    if not str(getattr(config, "format", "")).startswith("int"):
        return None
    bits = getattr(getattr(config, "grid", None), "bits", None)
    return int(bits) if bits is not None else None


def float_zero_points(config: TechniqueConfig) -> bool:
    """Whether the ``quant`` method of ``config`` chooses real-valued zero points (HQQ without integer zeros).

    Parameters
    ----------
    config : TechniqueConfig
        The ``quant`` configuration.

    Returns
    -------
    bool
        ``True`` when the codes have zero points that an integer zero-point format would round.
    """
    from ..techniques.quant.config import QuantConfig
    from ..techniques.quant.methods import QUANT_METHODS

    return isinstance(config, QuantConfig) and QUANT_METHODS.get(config.method).float_zeros(config)


#: The shared adapters every runtime registers (a runtime overrides one by registering its own with override=True).
SHARED_WEIGHT_ADAPTERS: tuple[type[ExportTechnique], ...] = (
    PruneExport, LowRankExport, LittleBitExport, DepthPruneExport,
)


def register_weight_adapters(runtime: ExportRuntime, quant: type[QuantExport] = QuantExport) -> None:
    """Register the shared weight adapters and the runtime's ``quant`` adapter.

    Parameters
    ----------
    runtime : ExportRuntime
        The runtime.
    quant : type[QuantExport]
        The runtime's quantization adapter (default: no quantized format, so only the ``"fake"`` reference is
        exported, dequantized).
    """
    for adapter in (*SHARED_WEIGHT_ADAPTERS, quant):
        runtime.register(adapter)


class StreamingWindowExport(ExportTechnique):
    """``kv_evict`` with ``method="streaming"``: the runtime's own sink-plus-window cache (StreamingLLM).

    A runtime subclasses this with :meth:`window_settings` (the settings that size its window) and, when its
    number of sink tokens is fixed, :attr:`fixed_sinks`. Other eviction methods (H2O, SnapKV, ...) score entries,
    which these caches do not, and are refused.
    """

    technique = "kv_evict"
    role = Role.KV_EVICTION
    support = Support.NATIVE
    #: The runtime's number of sink tokens when it cannot be set (``None``: any number).
    fixed_sinks: ClassVar[int | None] = None

    def sinks(self, config: TechniqueConfig) -> int:
        """The configured number of sink tokens (4 when unset, as in StreamingLLM)."""
        sinks = getattr(config, "sinks", None)
        return 4 if sinks is None else int(sinks)

    def level(self, config: TechniqueConfig) -> Support:
        streaming = getattr(config, "method", "streaming") == "streaming" and not getattr(config, "layers", ())
        fits = self.fixed_sinks is None or self.sinks(config) == self.fixed_sinks
        return Support.NATIVE if streaming and fits else Support.UNSUPPORTED

    def check(self, technique: nn.Module, builder: ExportBuilder) -> None:
        if self.level(config_of(technique)) is Support.UNSUPPORTED:
            sinks = "" if self.fixed_sinks is None else f" with sinks={self.fixed_sinks}"
            raise UnsupportedBackendError(f"{builder.runtime.title} keeps a sink-plus-window cache: only "
                                          f"kv_evict(method='streaming'{sinks}) on every layer has an equivalent")

    def window_settings(self, sinks: int, recent: int) -> dict[str, object]:
        """The runtime settings of a cache that keeps ``sinks`` first and ``recent`` last entries.

        Parameters
        ----------
        sinks : int
            Entries kept at the start.
        recent : int
            Most recent entries kept.

        Returns
        -------
        dict[str, object]
            Setting name to value.
        """
        raise NotImplementedError

    def contribute(self, technique: nn.Module, builder: ExportBuilder) -> None:
        config = config_of(technique)
        for key, value in self.window_settings(self.sinks(config), int(getattr(config, "recent", 256))).items():
            builder.set(key, value, owner=self.technique)
        builder.note(self.technique, f"{builder.runtime.title} evicts as each token is added, economize at the "
                                     "boundaries of its cache steps, so the kept window can differ by the length "
                                     "of one step")
