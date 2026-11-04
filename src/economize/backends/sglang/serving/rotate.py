"""SGLang adapter of ``rotate``: the vLLM checkpoint without online transforms.

SGLang (0.5.21) reads no compressed-tensors ``transform_config``, so an online rotation of a quantized map has no
SGLang path and is refused. The value rotations live in the weights, and the online rotations of dense maps are
folded back (:class:`~economize.backends.vllm.serving.rotate.RotateVllm`), so those checkpoints load unchanged.
"""

from __future__ import annotations

from torch import nn

from ...exporters import UnsupportedBackendError
from ...vllm.serving.rotate import RotateVllm, rotation_sites
from ...vllm.serving.weights import quantized_inside
from .base import Role, SglangPlanBuilder, Support, register_sglang_technique
from .weights import DelegatedSglang

__all__ = ["RotateSglang"]


@register_sglang_technique
class RotateSglang(DelegatedSglang):
    """``rotate``: value rotations in the weights; online rotations only in front of dense (unquantized) maps."""

    technique = "rotate"
    role = Role.WEIGHTS
    support = Support.NATIVE
    reason = ("value rotations are fused into the weights and online rotations of dense maps folded back; SGLang has "
              "no online Hadamard transform for a quantized map")
    vllm = RotateVllm

    def check(self, technique: nn.Module, builder: SglangPlanBuilder) -> None:
        for name in rotation_sites(builder.base):
            if quantized_inside(builder.base.get_submodule(name)) is not None:
                raise UnsupportedBackendError(
                    f"rotate rotates the input of the quantized map {name} online, and SGLang has no online Hadamard "
                    "transform (compressed-tensors transform_config): serve on vLLM, or use rotate(online=()) "
                    "(value rotations only)"
                )
        super().check(technique, builder)
