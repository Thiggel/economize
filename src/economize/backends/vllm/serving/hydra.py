"""vLLM adapter of ``hydra``: the heads (and the Hydra++ prefix layer) as an EAGLE-3 draft model run by economize's
Hydra proposer (:mod:`economize.backends.vllm.hydra`)."""

from __future__ import annotations

from torch import nn

from ...exporters import UnsupportedBackendError
from .base import PlanBuilder, Support, register_vllm_technique
from .drafters import JumpVllm

__all__ = ["HydraVllm"]


@register_vllm_technique
class HydraVllm(JumpVllm):
    """``hydra``: sequentially dependent heads over the target's last hidden state, drafted by
    :class:`~economize.backends.vllm.hydra.proposer.HydraSpeculator` (``EconomizeHydraEagle3``)."""

    technique = "hydra"
    support = Support.PLUGIN
    reason = ("economize's vLLM plugin (EconomizeHydraEagle3 through method='eagle3'): head j reads the base state "
              "and the embeddings of the tokens drafted before it; the Hydra++ prefix layer has its own KV pages; "
              "vLLM verifies the top-1 chain")
    exporter = "hydra"

    def check(self, technique: nn.Module, builder: PlanBuilder) -> None:
        from ....errors import ConfigError
        from ....techniques.hydra.export import HydraDrafterSpec

        try:
            spec = HydraDrafterSpec.of(technique)
        except ConfigError as error:
            raise UnsupportedBackendError(f"the Hydra drafter cannot be served on vLLM: {error}") from error
        if builder.num_speculative_tokens > spec.num_heads:
            raise UnsupportedBackendError(f"Hydra with {spec.num_heads} heads drafts at most {spec.num_heads} tokens; "
                                          f"asked for num_speculative_tokens={builder.num_speculative_tokens}")

    def contribute(self, technique: nn.Module, builder: PlanBuilder) -> None:
        super().contribute(technique, builder)
        builder.note(self.technique, "vLLM 0.30 verifies one chain per request (no tree verification), so "
                                     "it serves the top-1 chain of the heads")
        if technique.prefix is not None:
            builder.note(self.technique, "the Hydra++ prefix layer's attention runs in the target's dtype "
                                         "(float32 in the HF backend with param_dtype='float32')")
