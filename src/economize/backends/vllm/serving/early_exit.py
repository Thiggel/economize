"""vLLM adapters of ``early_exit``, ``free``, ``layer_skip`` and ``kangaroo``: each drafter runs the target's own
lower layers on economize's vLLM plugin (:mod:`economize.backends.vllm.early_exit`,
:mod:`economize.backends.vllm.free`, :mod:`economize.backends.vllm.layer_skip`, :mod:`economize.backends.vllm.kangaroo`,
which adds the adapter layer)."""

from __future__ import annotations

from torch import nn

from ...exporters import UnsupportedBackendError
from .base import PlanBuilder, Support, register_vllm_technique
from .drafters import JumpVllm, _ct_quantized

__all__ = ["EarlyExitVllm", "FreeVllm", "KangarooVllm", "LayerSkipVllm"]


@register_vllm_technique
class EarlyExitVllm(JumpVllm):
    """``early_exit``: layers ``1..r`` of the target, then the readout adapter (the identity for ``head="shared"``),
    final norm and LM head, run by economize's vLLM plugin draft model."""

    technique = "early_exit"
    support = Support.PLUGIN
    reason = ("economize's vLLM plugin: layers 1..r shared with the target (LoRA or copies for adapted paths), then "
              "the readout")
    exporter = "early_exit"

    def check(self, technique: nn.Module, builder: PlanBuilder) -> None:
        if technique.config.adaptation == "lora" and _ct_quantized(builder):
            raise UnsupportedBackendError(
                "an early-exit drafter with adaptation='lora' merges its updates into copies of layers 1..r, which "
                "vLLM holds as compressed-tensors codes: use backend='fake', or adaptation='frozen' or 'full' (full "
                "copies are served dense next to the quantized target)"
            )


@register_vllm_technique
class FreeVllm(EarlyExitVllm):
    """``free``: FREE's shallow model, the target's own layers ``1..L_S``, then the readout adapter (the identity for
    ``head="shared"``), final norm and LM head, run by economize's vLLM plugin draft model as the early-exit
    export. vLLM verifies every draft, so the served decoding is lossless."""

    technique = "free"
    support = Support.PLUGIN
    reason = ("economize's vLLM plugin: exported as the early-exit drafter (r=L_S, the target's own layers); served "
              "lossless, every draft verified")
    exporter = "free"

    def contribute(self, technique: nn.Module, builder: PlanBuilder) -> None:
        super().contribute(technique, builder)
        builder.note(self.technique, "vLLM verifies every draft and drafts num_speculative_tokens tokens: FREE's kept "
                                     "shallow exits and its softmax-gap stop (FreeTechnique.decode_config) run on "
                                     "the HF backend only; a model fine-tuned with the objective 'free' is served "
                                     "from its saved weights")


@register_vllm_technique
class LayerSkipVllm(JumpVllm):
    """``layer_skip``: the target's own layers outside the skip set, then the final norm and LM head, run by
    economize's vLLM plugin draft model with the skip set as holes in its path."""

    technique = "layer_skip"
    support = Support.PLUGIN
    reason = ("economize's vLLM plugin: the target's own layers and KV pages, with the skip set as holes in the "
              "draft path")
    exporter = "layer_skip"

    def check(self, technique: nn.Module, builder: PlanBuilder) -> None:
        if not technique.skip_layers:
            raise UnsupportedBackendError("the layer_skip technique has no skip set yet: attach it first")


@register_vllm_technique
class KangarooVllm(JumpVllm):
    """``kangaroo``: layers ``1..l`` of the target, then Kangaroo's adapter (one attention layer with its own KV pages)
    and the target's LM head, run by the Kangaroo draft model (``EconomizeKangarooEagle3``)."""

    technique = "kangaroo"
    support = Support.PLUGIN
    reason = ("economize's vLLM plugin: layers 1..l shared with the target, then the adapter with its own KV "
              "pages; the confidence early stop is not served (vLLM drafts k tokens)")
    exporter = "kangaroo"

    def check(self, technique: nn.Module, builder: PlanBuilder) -> None:
        from ....errors import ConfigError
        from ....techniques.kangaroo.export import KangarooDrafterSpec

        try:
            KangarooDrafterSpec.of(technique)
        except ConfigError as error:
            raise UnsupportedBackendError(f"the Kangaroo drafter cannot be served on vLLM: {error}") from error

    def contribute(self, technique: nn.Module, builder: PlanBuilder) -> None:
        super().contribute(technique, builder)
        builder.note(self.technique, "the adapter runs in the engine dtype, its attention heads sharded as the "
                                     "target's; DecodeConfig.min_confidence has no vLLM equivalent")
