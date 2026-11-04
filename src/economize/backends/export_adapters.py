"""Export adapters shared by every file-based runtime: the weight techniques a checkpoint holds as plain weights.

Pruned, low-rank, LittleBit and depth-pruned models are plain models once their weights are multiplied out
(:func:`~economize.backends.runtime_export.dense_state`), so one adapter class per technique serves every runtime.
Quantization depends on the runtime's own formats; :class:`QuantExport` reads that from a runtime's subclass
(:meth:`QuantExport.stores`, :meth:`QuantExport.stores_weight`). A runtime registers the shared adapters with
:func:`register_weight_adapters`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

from torch import nn

from ..core.config import TechniqueConfig
from ..core.metadata import Saving, Simulation
from .exporters import UnsupportedBackendError
from .runtime_export import ExportBuilder, ExportRuntime, ExportTechnique, config_of
from .vllm.serving.base import Role, Support

if TYPE_CHECKING:
    from ..techniques.quant.grid import QuantizedWeight

__all__ = [
    "DepthPruneExport",
    "LittleBitExport",
    "LowRankExport",
    "PruneExport",
    "QuantExport",
    "StreamingWindowExport",
    "register_weight_adapters",
]


class _DenseWeights(ExportTechnique):
    """A weight technique the runtime runs as the dense weights the base holds (or multiplies out)."""

    role = Role.WEIGHTS
    support = Support.DENSIFIED
    message: ClassVar[str] = ""

    def contribute(self, technique: nn.Module, builder: ExportBuilder) -> None:
        if self.message:
            builder.note(self.technique, self.message)


class PruneExport(_DenseWeights):
    """``prune``: the masked weights are written dense (no file format of these runtimes stores sparsity)."""

    technique = "prune"
    reason = "the masked weights are written dense (zeros kept); the runtime has no sparse weight format"
    simulation = Simulation(Saving.WEIGHT_MEMORY | Saving.COMPUTE | Saving.LATENCY,
                            "the masked weights are stored and multiplied dense")
    message = "pruned weights are written dense (zeros kept): same outputs, no sparse speedup"

    def level(self, config: TechniqueConfig) -> Support:
        return Support.DENSIFIED if getattr(config, "semi_structured", False) else Support.NATIVE


class LowRankExport(_DenseWeights):
    """``lowrank``: the factors are multiplied out."""

    technique = "lowrank"
    reason = "the low-rank factors are multiplied out into dense weights"
    message = "low-rank factors are multiplied out: same outputs, no memory or speed saving"


class LittleBitExport(_DenseWeights):
    """``littlebit``: the binary factors are multiplied out; the latent (QAT) form is refused."""

    technique = "littlebit"
    reason = "the binary factors are multiplied out into dense weights (no runtime has a LittleBit GEMM)"
    message = "binary factors are multiplied out: same outputs, no memory or speed saving"

    def check(self, technique: nn.Module, builder: ExportBuilder) -> None:
        if any("LittleBit" in type(module).__name__ and not callable(getattr(module, "dense_weight", None))
               for module in builder.base.modules()):
            raise UnsupportedBackendError("littlebit in its latent (QAT) form has no dense weight: finish training "
                                          f"(convert to LittleBitLinear) before exporting to {builder.runtime.title}")


class DepthPruneExport(ExportTechnique):
    """``depth_prune``: the shallower model is a plain model."""

    technique = "depth_prune"
    role = Role.WEIGHTS
    support = Support.NATIVE
    reason = "the shallower model (layers removed, config renumbered) is a plain model"


class QuantExport(ExportTechnique):
    """``quant``: codes the runtime has a format for are stored as such, the rest is written dequantized.

    A runtime subclasses this with :meth:`stores` (which configurations it stores in its own format) and
    :meth:`stores_weight` (whether one recorded weight fits that format). A weight that does not fit is written
    dequantized, and the export then reports the technique as ``DENSIFIED``.
    """

    technique = "quant"
    role = Role.WEIGHTS
    support = Support.DENSIFIED
    reason = "the dequantized weights are written dense"
    #: Why a W8A8 configuration is refused.
    fp8_activations_reason: ClassVar[str] = ("fp8 activations (W8A8) have no export to this runtime: export the "
                                             "weight-only model")

    def stores(self, config: TechniqueConfig) -> bool:
        """Whether the runtime stores weights quantized by ``config`` in its own format (default: never).

        Parameters
        ----------
        config : TechniqueConfig
            The ``quant`` configuration.

        Returns
        -------
        bool
            ``True`` when the export keeps the codes.
        """
        return False

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
        return Support.NATIVE if self.stores(config) else Support.DENSIFIED

    def check(self, technique: nn.Module, builder: ExportBuilder) -> None:
        if self.level(config_of(technique)) is Support.UNSUPPORTED:
            raise UnsupportedBackendError(f"{builder.runtime.title}: {self.fp8_activations_reason}")

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
        The runtime's quantization adapter (default: everything written dequantized).
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
