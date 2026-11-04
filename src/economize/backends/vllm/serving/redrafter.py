"""vLLM adapter of ``redrafter``: the recurrent head as an EAGLE-3 draft model run by economize's ReDrafter proposer
(:mod:`economize.backends.vllm.redrafter`)."""

from __future__ import annotations

from torch import nn

from ...exporters import UnsupportedBackendError
from .base import PlanBuilder, Support, register_vllm_technique
from .drafters import JumpVllm

__all__ = ["RedrafterVllm"]


@register_vllm_technique
class RedrafterVllm(JumpVllm):
    """``redrafter``: a recurrent state over the drafted tokens and a residual head on the target's last hidden state,
    drafted by :class:`~economize.backends.vllm.redrafter.proposer.RedrafterSpeculator`
    (``EconomizeRedrafterEagle3``)."""

    technique = "redrafter"
    support = Support.PLUGIN
    reason = ("economize's vLLM plugin (EconomizeRedrafterEagle3 through method='eagle3'): the recurrence and the "
              "head run as one draft round over the target's last hidden state; vLLM verifies a chain")
    exporter = "redrafter"

    def check(self, technique: nn.Module, builder: PlanBuilder) -> None:
        from ....errors import ConfigError
        from ....techniques.redrafter.export import RedrafterDrafterSpec

        try:
            spec = RedrafterDrafterSpec.of(technique)
        except ConfigError as error:
            raise UnsupportedBackendError(f"the ReDrafter drafter cannot be served on vLLM: {error}") from error
        if builder.num_speculative_tokens > spec.num_drafts:
            raise UnsupportedBackendError(f"ReDrafter with beam_length={spec.num_drafts + 1} drafts at most "
                                          f"{spec.num_drafts} tokens; asked for num_speculative_tokens="
                                          f"{builder.num_speculative_tokens}")

    def contribute(self, technique: nn.Module, builder: PlanBuilder) -> None:
        super().contribute(technique, builder)
        config = technique.config
        if config.beam_width > 1:
            builder.note(self.technique, f"vLLM 0.30 verifies one chain per request (no tree verification), so "
                                         f"it serves the greedy beam; the beam search over {config.beam_width} "
                                         f"beams runs in the Hugging Face backend only")
        if config.top_k:
            builder.note(self.technique, f"draft logits keep the technique's top-{config.top_k} warp")
