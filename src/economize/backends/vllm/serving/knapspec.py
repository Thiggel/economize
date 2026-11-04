"""vLLM adapter of ``knapspec``: the current skip set as a drafter on the target's own layers whose partial layers
run one sublayer (:mod:`economize.backends.vllm.knapspec`)."""

from __future__ import annotations

from torch import nn

from ...exporters import UnsupportedBackendError
from .base import PlanBuilder, Support, register_vllm_technique
from .drafters import JumpVllm

__all__ = ["KnapSpecVllm"]


@register_vllm_technique
class KnapSpecVllm(JumpVllm):
    """``knapspec``: the target's own layers with the skip set's attention and MLP sublayers replaced by zeros, then
    the final norm and LM head, run by the KnapSpec draft model (``EconomizeKnapSpecEagle3``)."""

    technique = "knapspec"
    support = Support.PLUGIN
    reason = ("economize's vLLM plugin: the target's own layers, whole-skipped layers as holes and partial layers "
              "with one sublayer replaced by zeros; the skip set is the one current at export (no re-search while "
              "decoding)")
    exporter = "knapspec"

    def check(self, technique: nn.Module, builder: PlanBuilder) -> None:
        from ....errors import ConfigError
        from ....techniques.knapspec.export import KnapSpecDrafterSpec

        try:
            KnapSpecDrafterSpec.of(technique)
        except ConfigError as error:
            raise UnsupportedBackendError(f"the KnapSpec drafter cannot be served on vLLM: {error}") from error

    def contribute(self, technique: nn.Module, builder: PlanBuilder) -> None:
        super().contribute(technique, builder)
        builder.note(self.technique, "vLLM serves the skip set current at export (no re-search while decoding) and "
                                     "drafts num_speculative_tokens per round in place of the searched draft length")
