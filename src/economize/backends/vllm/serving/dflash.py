"""``dflash``: vLLM's DFlash proposer with the drafter exported by
:class:`~economize.backends.vllm.dflash.DFlashVllmExporter`."""

from __future__ import annotations

from typing import Any

from torch import nn

from ...exporters import UnsupportedBackendError
from .base import PlanBuilder, Support, register_vllm_technique
from .drafters import _Exported

__all__ = ["DFlashVllm"]

#: Set when the drafter mixes sliding and full layers: vLLM 0.30's V1 runner holds one KV group per draft model.
V2_RUNNER_ENV = "VLLM_USE_V2_MODEL_RUNNER"


def _mixed(technique: Any) -> bool:
    kinds = set(technique.layer_types)
    return len(kinds) > 1


@register_vllm_technique
class DFlashVllm(_Exported):
    """``dflash``: vLLM's DFlash draft model (``method="dflash"``, the z-lab layout).

    vLLM drafts ``num_speculative_tokens`` tokens from a block of ``1 + num_speculative_tokens`` slots, so at most
    ``block_size - 1``. A drafter with mixed sliding and full layers needs vLLM's V2 model runner: the plan sets
    ``VLLM_USE_V2_MODEL_RUNNER=1`` in :attr:`VllmPlan.env` (an explicit value is kept).
    """

    technique = "dflash"
    support = Support.NATIVE
    reason = "method='dflash' (z-lab DFlashDraftModel config, target embedding and LM head shared)"
    exporter = "dflash"

    def method(self, technique: nn.Module) -> str | None:
        return "dflash"

    def check(self, technique: nn.Module, builder: PlanBuilder) -> None:
        block = int(technique.config.block_size)
        if builder.num_speculative_tokens > block - 1:
            raise UnsupportedBackendError(f"a DFlash drafter with block_size={block} drafts at most {block - 1} "
                                          f"tokens; asked for num_speculative_tokens={builder.num_speculative_tokens}")

    def contribute(self, technique: nn.Module, builder: PlanBuilder) -> None:
        super().contribute(technique, builder)
        if _mixed(technique):
            builder.require_v2_runner(self.technique, "mixed sliding/full draft layers")
            builder.note(self.technique, "mixed sliding/full draft layers: vLLM's V2 model runner "
                                         f"({V2_RUNNER_ENV}=1) serves the drafter and the target")
        bidirectional = not all(technique.causal(i) for i in range(len(technique.layers)))
        block = int(technique.config.block_size)
        if bidirectional and builder.num_speculative_tokens < block - 1:
            builder.note(self.technique, f"vLLM runs a block of {builder.num_speculative_tokens + 1} slots, the "
                                         f"drafter was trained on {block}: its bidirectional layers see fewer masks, "
                                         "so the drafts can differ from economize's (the output stays lossless)")
        cap = technique.config.logit_softcap
        if cap or (cap is None and getattr(technique.adapter, "final_logit_softcap", None)):
            builder.note(self.technique, "vLLM's DFlash head applies no logit soft cap, which changes "
                                         "probabilistic drafts only")
        if technique.config.embed_scale != 1.0:
            builder.note(self.technique, "embed_scale != 1: the export writes the scaled embedding as the drafter's "
                                         "own embed_tokens (one V x d tensor)")
