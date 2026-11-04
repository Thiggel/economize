"""economize's jump-format drafters on SGLang: jump (frozen / LoRA / full), early exit, FREE, layer skipping, DMTD.

Each runs as the plugin draft model :class:`~economize.backends.sglang.model.EconomizeJumpForCausalLM` under
``speculative_algorithm=EAGLE3``: the target captures the states the drafter reads, the draft model runs its own
copies of the jump-path layers in SGLang's draft KV cache, and SGLang verifies the drafts greedily (lossless).
The draft folder is written by the SGLang exporters of :mod:`economize.backends.sglang.export` from the base as it
is served (densified weight techniques included).
"""

from __future__ import annotations

import json
from typing import Any, ClassVar

from torch import nn

from ...exporters import UnsupportedBackendError
from ...vllm.serving.drafters import DRAFTER_DIR, _ct_quantized
from ..export import (
    DmtdSglangExporter,
    EarlyExitSglangExporter,
    FreeSglangExporter,
    JumpSglangExporter,
    LayerSkipSglangExporter,
)
from .base import Role, SglangPlanBuilder, SglangTechnique, Support, register_sglang_technique

__all__ = ["DmtdSglang", "EarlyExitSglang", "FreeSglang", "JumpFormatSglang", "JumpSglang", "LayerSkipSglang"]


class JumpFormatSglang(SglangTechnique):
    """A drafter written in the jump format, served through its SGLang exporter.

    A new jump-format drafter subclasses it, names its exporter (a :class:`JumpSglangExporter` registered with
    ``register_exporter("sglang", name)``) in :attr:`exporter_class` and registers the subclass; the technique name
    is the exporter's::

        @register_sglang_technique
        class MyDrafterSglang(JumpFormatSglang):
            exporter_class = MyDrafterSglangExporter
    """

    role = Role.DRAFTER
    support = Support.PLUGIN
    reason: ClassVar[str] = ("economize's SGLang plugin draft model under speculative_algorithm EAGLE3 (aux states "
                             "captured by the target, own copies of the draft path's layers)")
    #: The exporter that writes the draft folder (it names the technique).
    exporter_class: ClassVar[type[JumpSglangExporter]]

    def __init_subclass__(cls, **kwargs: object) -> None:
        super().__init_subclass__(**kwargs)
        if "exporter_class" in cls.__dict__ and "technique" not in cls.__dict__:
            cls.technique = cls.exporter_class.technique

    def exporter(self) -> JumpSglangExporter:
        """A fresh instance of :attr:`exporter_class`."""
        return self.exporter_class()

    def check(self, technique: nn.Module, builder: SglangPlanBuilder) -> None:
        spec = self.exporter().check(technique)
        if _ct_quantized(builder) and str(spec.config.adaptation) != "full":
            raise UnsupportedBackendError(
                f"{self.technique!r} on a compressed-tensors base: SGLang's draft model loads dense copies of the "
                "draft path's layers, and the served base holds packed codes; quantize with format='fake' "
                "(densified) or serve on vLLM, whose drafter shares the quantized target layers")

    def contribute(self, technique: nn.Module, builder: SglangPlanBuilder) -> None:
        builder.set_speculative("EAGLE3", None, owner=self.technique)
        builder.note(self.technique, "the draft model holds its own copy of the draft path's layers (SGLang's draft "
                                     "worker cannot share target layers); the embedding and LM head are shared")
        if int(builder.engine.get("tp_size", 1) or 1) > 1:
            builder.note(self.technique, "tensor parallelism: the drafter's predictor and readout run replicated")

    def export(self, technique: nn.Module, builder: SglangPlanBuilder) -> None:
        from safetensors.torch import save_file

        exporter = self.exporter()
        base_state = None
        if builder.rewrite_base or builder.dense:
            from ...vllm.serving.weights import serving_state

            base_state = serving_state(builder)
        target = builder.path(DRAFTER_DIR)
        target.mkdir(parents=True, exist_ok=True)
        config: dict[str, Any] = exporter.config(technique)
        (target / exporter.config_file).write_text(json.dumps(config, indent=2, sort_keys=True, default=str) + "\n")
        save_file(exporter.tensors(technique, base_state), str(target / exporter.weights_file),
                  metadata=exporter.metadata())
        builder.set_engine("speculative_draft_model_path", str(target), owner=self.technique)


@register_sglang_technique
class JumpSglang(JumpFormatSglang):
    """``jump``: the cross-layer jump drafter (frozen, LoRA merged into the layer copies, or full copies)."""

    exporter_class = JumpSglangExporter


@register_sglang_technique
class EarlyExitSglang(JumpFormatSglang):
    """``early_exit``: layers ``1..r`` with the readout adapter (``m = u = 0``)."""

    exporter_class = EarlyExitSglangExporter


@register_sglang_technique
class FreeSglang(JumpFormatSglang):
    """``free``: FREE's shallow exit as an early-exit drafter."""

    exporter_class = FreeSglangExporter


@register_sglang_technique
class LayerSkipSglang(JumpFormatSglang):
    """``layer_skip``: a self-drafter that skips a set of layers."""

    exporter_class = LayerSkipSglangExporter


@register_sglang_technique
class DmtdSglang(JumpFormatSglang):
    """``dmtd``: DMTD's re-run of the last layers as a jump drafter."""

    exporter_class = DmtdSglangExporter
