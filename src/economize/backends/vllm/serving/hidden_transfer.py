"""vLLM adapter of ``hidden_transfer``: the transfer maps as an EAGLE-3 draft model run by economize's Hidden Transfer
proposer (:mod:`economize.backends.vllm.hidden_transfer`)."""

from __future__ import annotations

from torch import nn

from ...exporters import UnsupportedBackendError
from .base import PlanBuilder, Support, register_vllm_technique
from .drafters import JumpVllm

__all__ = ["HiddenTransferVllm"]


@register_vllm_technique
class HiddenTransferVllm(JumpVllm):
    """``hidden_transfer``: draft step ``i`` maps the state after layer ``t_i`` and runs the target's layers
    ``t_i + 1..L``, drafted by :class:`~economize.backends.vllm.hidden_transfer.proposer.HiddenTransferSpeculator`
    (``EconomizeHiddenTransferEagle3``)."""

    technique = "hidden_transfer"
    support = Support.PLUGIN
    reason = ("economize's vLLM plugin (EconomizeHiddenTransferEagle3 through method='eagle3'): step i maps the state "
              "after layer t_i and runs the target's own layers t_i+1..L on the paged cache, one CUDA graph per step")
    exporter = "hidden_transfer"

    def check(self, technique: nn.Module, builder: PlanBuilder) -> None:
        from ....errors import ConfigError
        from ....techniques.dynamic_depth.hidden_transfer_export import HiddenTransferDrafterSpec

        try:
            spec = HiddenTransferDrafterSpec.of(technique)
        except ConfigError as error:
            raise UnsupportedBackendError(f"the Hidden Transfer drafter cannot be served on vLLM: {error}") from error
        if builder.num_speculative_tokens > spec.num_steps:
            raise UnsupportedBackendError(f"Hidden Transfer with {spec.num_steps} transfer maps drafts at most "
                                          f"{spec.num_steps} tokens; asked for num_speculative_tokens="
                                          f"{builder.num_speculative_tokens}")

    def contribute(self, technique: nn.Module, builder: PlanBuilder) -> None:
        super().contribute(technique, builder)
        builder.note(self.technique, "the transfer maps are replicated on every tensor-parallel rank")
