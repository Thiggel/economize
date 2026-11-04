"""vLLM adapter of ``dmtd``: the drafter runs the last ``L_d`` layers on the next token's embedding through
economize's vLLM plugin (:mod:`economize.backends.vllm.dmtd`)."""

from __future__ import annotations

from torch import nn

from .base import PlanBuilder, Support, register_vllm_technique
from .drafters import JumpVllm

__all__ = ["DmtdVllm"]


@register_vllm_technique
class DmtdVllm(JumpVllm):
    """``dmtd``: the late layers ``L - L_d + 1..L`` on the next token's embedding (the target's own layers, LoRA
    updates or trainable copies), then the final norm and LM head, run by economize's vLLM plugin draft model.
    vLLM verifies every draft, so the served decoding is lossless."""

    technique = "dmtd"
    support = Support.PLUGIN
    reason = ("economize's vLLM plugin: the last L_d layers on the next token's embedding (shared with the target, "
              "LoRA or copies for adapted paths); served lossless, every draft verified")
    exporter = "dmtd"

    def contribute(self, technique: nn.Module, builder: PlanBuilder) -> None:
        super().contribute(technique, builder)
        tau = technique.config.cycle_length
        builder.note(self.technique, f"vLLM verifies every draft: DMTD's lossy cycles (drafts kept, "
                                     f"DmtdTechnique.decode_config) run on the HF backend only; "
                                     f"num_speculative_tokens={max(tau - 1, 1)} drafts one cycle of tau={tau}")
