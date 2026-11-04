"""Weight techniques on SGLang: the base checkpoint economize writes for vLLM, which SGLang loads as well.

SGLang reads compressed-tensors checkpoints (its wNa16 Marlin and fp8 kernels) and plain dense checkpoints, the two
formats the vLLM weight writers produce. Each adapter here therefore delegates its checks and plan entries to the
matching vLLM adapter (:mod:`economize.backends.vllm.serving.weights`) and rewords its refusals for SGLang.
"""

from __future__ import annotations

from typing import Any, ClassVar

from torch import nn

from ....core.metadata import Simulation
from ...exporters import UnsupportedBackendError
from ...vllm.serving.base import VllmTechnique
from ...vllm.serving.weights import DepthPruneVllm, LittleBitVllm, PruneVllm, QuantVllm
from .base import Role, SglangPlanBuilder, SglangTechnique, Support, register_sglang_technique

__all__ = ["DelegatedSglang", "DepthPruneSglang", "LittleBitSglang", "LowRankSglang", "PruneSglang", "QuantSglang"]


class DelegatedSglang(SglangTechnique):
    """An adapter that serves the checkpoint a vLLM adapter writes (same levels, checks and weight rewrites).

    Subclasses set :attr:`vllm` to the vLLM adapter class. Its refusals are raised again as SGLang refusals, and its
    :attr:`~SglangTechnique.simulation` becomes theirs unless they declare their own: SGLang loads the same
    checkpoint, so it realizes the same savings.
    """

    #: The vLLM adapter whose checkpoint SGLang loads.
    vllm: ClassVar[type[VllmTechnique]]

    def __init_subclass__(cls, **kwargs: object) -> None:
        super().__init_subclass__(**kwargs)
        if "simulation" not in vars(cls) and "vllm" in vars(cls):
            cls.simulation = cls.vllm.simulation

    def _inner(self) -> VllmTechnique:
        return self.vllm()

    def level(self, technique: nn.Module) -> Support:
        return self._inner().level(technique)

    def simulation_of(self, technique: nn.Module) -> Simulation | None:
        if self.simulation is not self.vllm.simulation:  # the subclass declared its own
            return self.simulation
        return self._inner().simulation_of(technique)

    def check(self, technique: nn.Module, builder: SglangPlanBuilder) -> None:
        super().check(technique, builder)
        try:
            self._inner().check(technique, builder)
        except UnsupportedBackendError as error:
            raise UnsupportedBackendError(f"SGLang loads the checkpoint economize writes for vLLM, and {error}"
                                          ) from error

    def contribute(self, technique: nn.Module, builder: SglangPlanBuilder) -> None:
        self._inner().contribute(technique, builder)


@register_sglang_technique
class QuantSglang(DelegatedSglang):
    """Quantization: a compressed-tensors checkpoint on SGLang's quantized kernels (W8A8 int8 / fp8 with
    ``activations``, checked like :class:`QuantVllm`)."""

    technique = "quant"
    role = Role.WEIGHTS
    support = Support.NATIVE
    reason = ("backends compressed-tensors, marlin, machete and torchao are written in the compressed-tensors format "
              "(SGLang's wNa16 Marlin / fp8 kernels); backend='fake' is served dense")
    vllm = QuantVllm
    #: Formats whose compressed-tensors checkpoint SGLang (0.5.21) cannot load, with the reason. A format SGLang
    #: learns to serve is removed here (or by a subclass).
    unsupported_formats: ClassVar[dict[str, str]] = {
        "nvfp4": "SGLang serves nvfp4 linear layers only as W4A4 on sm100 (CompressedTensorsW4A4Fp4); economize "
                 "writes weight-only NVFP4, which vLLM serves",
        "mxfp4": "SGLang has no mxfp4-pack-quantized scheme for linear layers (only MXFP4 MoE); vLLM serves it",
    }

    def check(self, technique: nn.Module, builder: SglangPlanBuilder) -> None:
        config: Any = technique.config
        reason = self.unsupported_formats.get(config.format)
        if reason is not None:
            raise UnsupportedBackendError(f"quant format {config.format!r} has no SGLang kernel: {reason}")
        super().check(technique, builder)

    def level(self, technique: nn.Module) -> Support:
        config: Any = technique.config
        if config.format in self.unsupported_formats:
            return Support.UNSUPPORTED
        return super().level(technique)


@register_sglang_technique
class LittleBitSglang(DelegatedSglang):
    technique = "littlebit"
    role = Role.WEIGHTS
    support = Support.DENSIFIED
    reason = "served as the dense weight of the binary factors (SGLang has no LittleBit GEMM)"
    vllm = LittleBitVllm


@register_sglang_technique
class PruneSglang(DelegatedSglang):
    technique = "prune"
    role = Role.WEIGHTS
    support = Support.NATIVE
    reason = ("the masked weights as a dense checkpoint (unstructured sparsity is dense anyway; SGLang has no 2:4 "
              "sparse checkpoint format, so 2:4 masks are DENSIFIED)")
    vllm = PruneVllm


@register_sglang_technique
class DepthPruneSglang(DelegatedSglang):
    technique = "depth_prune"
    role = Role.WEIGHTS
    support = Support.NATIVE
    reason = "the shallower model (layers removed, config renumbered) is a plain checkpoint"
    vllm = DepthPruneVllm


@register_sglang_technique
class LowRankSglang(SglangTechnique):
    """``lowrank``: SGLang has no linear-method plugin, so the factors are multiplied out into dense weights."""

    technique = "lowrank"
    role = Role.WEIGHTS
    support = Support.DENSIFIED
    reason = ("the dense product up @ down of every factored map (SGLang has no plugin hook for a factored linear "
              "method)")

    def check(self, technique: nn.Module, builder: SglangPlanBuilder) -> None:
        from ...vllm.serving.weights import dense_modules, quantized_inside

        for name, module in dense_modules(builder.base).items():
            inner = quantized_inside(module)
            if inner is not None:
                raise UnsupportedBackendError(
                    f"lowrank next to a compressed-tensors base: {name}.{inner} quantizes a factor, while SGLang "
                    "serves the factored map multiplied out. Quantize with backend='fake' (served dense) or serve "
                    "lowrank without quant"
                )

    def contribute(self, technique: nn.Module, builder: SglangPlanBuilder) -> None:
        builder.rewrite_base = True
        builder.note(self.technique, "factored maps are multiplied out: same outputs, no memory or speed saving")
