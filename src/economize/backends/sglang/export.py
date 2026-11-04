"""SGLang exporters: drafter folders that ``speculative_algorithm=EAGLE3`` loads.

* ``"sglang/jump"`` (:class:`JumpSglangExporter`) and the drafters written in the jump format (``"sglang/early_exit"``,
  ``"sglang/free"``, ``"sglang/layer_skip"``, ``"sglang/dmtd"``): a folder for
  :class:`~economize.backends.sglang.model.EconomizeJumpForCausalLM`. The ``config.json`` is the target's config with
  ``architectures=["EconomizeJumpForCausalLM"]``, ``num_hidden_layers`` cut to the jump path, the ``eagle_config``
  aux layer ids SGLang captures, the ``economize_jump`` block (:class:`~economize.techniques.jump.export.
  JumpDrafterSpec`) and an ``economize_sglang`` block. The weights are the drafter's ``jump.*`` tensors plus copies
  of the jump-path layers under ``model.layers.<j>.*`` (LoRA updates merged, full copies as trained) and the final
  norm. SGLang's draft worker owns its layers and KV cache, so the jump path cannot reference the target's layers
  as on vLLM; the embedding and the LM head are shared with the target.
* ``"sglang/eagle3"`` (:class:`Eagle3SglangExporter`): an EAGLE-3 drafter for SGLang's ``LlamaForCausalLMEagle3``.

SGLang captures the input of its 0-indexed layer ``k + 1`` for aux id ``k``, which is ``h^{k+1}`` in economize's
notation, so its ids are the vLLM ids minus one and the state after the last layer cannot be captured.
The exporters reuse the vLLM exporters' drafter state (they import no ``vllm``).
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, ClassVar

import torch
from torch import nn

from ...errors import ConfigError
from ...techniques.jump.export import CONFIG_KEY, JumpDrafterSpec
from ..exporters import DrafterExporter, UnsupportedBackendError, register_exporter
from ..vllm.dmtd.export import DmtdVllmExporter
from ..vllm.eagle3.export import Eagle3VllmExporter
from ..vllm.early_exit.export import EarlyExitVllmExporter
from ..vllm.free.export import FreeVllmExporter
from ..vllm.jump.export import JumpVllmExporter
from ..vllm.layer_skip.export import LayerSkipVllmExporter

__all__ = [
    "DRAFTER_ARCHITECTURE",
    "LAYER_CLASSES",
    "SGLANG_CONFIG_KEY",
    "DmtdSglangExporter",
    "Eagle3SglangExporter",
    "EarlyExitSglangExporter",
    "FreeSglangExporter",
    "JumpSglangExporter",
    "LayerSkipSglangExporter",
    "check_sglang_eagle3",
    "check_sglang_jump",
    "sglang_aux_ids",
]

#: The draft architecture (registered in SGLang's ``ModelRegistry`` by :func:`economize.backends.sglang.register`).
DRAFTER_ARCHITECTURE = "EconomizeJumpForCausalLM"
#: The ``config.json`` key of the SGLang-specific block (target architecture, ``m``, ``u``, ``r``, jump layers).
SGLANG_CONFIG_KEY = "economize_sglang"
#: Target architectures whose SGLang decoder layer the draft model builds (see :mod:`.model`).
LAYER_CLASSES: dict[str, str] = {
    "Qwen2ForCausalLM": "Qwen2DecoderLayer",
    "Qwen3ForCausalLM": "Qwen3DecoderLayer",
    "LlamaForCausalLM": "LlamaDecoderLayer",
    "MistralForCausalLM": "LlamaDecoderLayer",
}
_DROPPED_KEYS = ("auto_map", "quantization_config", "compression_config", "eagle_aux_hidden_state_layer_ids")


def sglang_aux_ids(layers: list[int]) -> list[int]:
    """SGLang's ``eagle_aux_hidden_state_layer_ids`` capturing ``h^i`` for ``i`` in ``layers`` (``i >= 1``)."""
    return sorted({int(i) - 1 for i in layers})


def _architecture(technique: Any) -> str:
    adapter = technique.adapter
    names = list(getattr(adapter.model.config, "architectures", None) or getattr(adapter.config, "architectures",
                                                                                   None) or [])
    if names:
        return str(names[0])
    name = type(adapter.model).__name__  # a model built from a config, not loaded: its class names it
    return name if name.endswith("ForCausalLM") else ""


def check_sglang_jump(technique: Any, spec: JumpDrafterSpec) -> None:
    """Refuse a jump-format drafter SGLang's EAGLE-3 path cannot run.

    Raises:
        UnsupportedBackendError: Several source layers, a source layer off the jump path or after the last layer,
            a sequence predictor, an empty jump path, or a target architecture without an SGLang layer here.
    """
    config = spec.config
    m, u, r, num_layers = spec.target_layer, int(config.source_layer), spec.exit_layer, spec.num_layers
    where = f"(m={m}, u={u}, r={r})"
    if len(spec.source_layers) > 1:
        raise UnsupportedBackendError(
            f"SGLang passes one state between EAGLE draft steps: a jump drafter with several source layers "
            f"{list(spec.source_layers)} cannot run there; serve it on vLLM")
    if getattr(config, "predictor", "") == "sequence" or "sequence" in str(getattr(config, "predictor", "")):
        raise UnsupportedBackendError("SGLang's EAGLE draft step sees one state per token: a sequence predictor "
                                      "(attention over earlier states) needs its own cache; serve it on vLLM")
    if not spec.jump_layers:
        raise UnsupportedBackendError(f"the jump drafter {where} has no jump-path layer for SGLang to run")
    if m > 0 and u == 0:
        raise UnsupportedBackendError(f"SGLang cannot capture the embedding as an aux state {where}")
    if u < m or u > r or (u > m and u not in spec.jump_layers):
        raise UnsupportedBackendError(
            f"SGLang hands the next draft step a state of the draft pass: the source layer must lie on the jump path "
            f"m <= u <= r and not be skipped {where}; serve it on vLLM")
    if u == num_layers and m != u:
        raise UnsupportedBackendError(
            f"SGLang captures the input of a target layer, so h^L (after the last layer) cannot be an aux state "
            f"{where}; serve it on vLLM")
    if _architecture(technique) not in LAYER_CLASSES:
        raise UnsupportedBackendError(
            f"the SGLang jump drafter builds its layers for {sorted(LAYER_CLASSES)}, not "
            f"{_architecture(technique) or 'an unnamed architecture'}; serve it on vLLM")


def _layer_path(layer: nn.Module, target: nn.Module) -> str:
    for name, module in layer.named_modules():
        if module is target:
            return name
    raise ConfigError("a LoRA target is not part of its decoder layer")


@register_exporter("sglang", "jump")
class JumpSglangExporter(DrafterExporter):
    """A jump-format drafter as SGLang's ``EconomizeJumpForCausalLM`` draft model, see the module docstring.

    A drafter written in the jump format subclasses it and names its vLLM exporter in :attr:`vllm_exporter`, whose
    :meth:`~JumpVllmExporter.spec` and :meth:`~JumpVllmExporter.drafter_state` it reuses.
    """

    #: The vLLM exporter of the same technique (the drafter's spec and tensors).
    vllm_exporter: ClassVar[type[JumpVllmExporter]] = JumpVllmExporter

    def _vllm(self) -> JumpVllmExporter:
        return self.vllm_exporter()

    def spec(self, technique: nn.Module) -> JumpDrafterSpec:
        """The ``economize_jump`` block of ``technique`` (from :attr:`vllm_exporter`)."""
        return self._vllm().spec(technique)

    def check(self, model: Any) -> JumpDrafterSpec:
        """The spec of ``model``'s drafter after :func:`check_sglang_jump`."""
        technique: Any = self.technique_of(model)
        if self._vllm().architecture != JumpVllmExporter.architecture:
            raise UnsupportedBackendError(
                f"{self.technique!r} drafts with its own draft model on vLLM ({self._vllm().architecture}); the "
                "SGLang backend serves the jump draft step only")
        spec = self.spec(technique)
        check_sglang_jump(technique, spec)
        return spec

    def config(self, model: Any, **options: Any) -> dict[str, Any]:
        """The draft ``config.json`` (``options`` are accepted for the generic export functions and ignored)."""
        technique: Any = self.technique_of(model)
        spec = self.check(model)
        m, u, r = spec.target_layer, int(spec.config.source_layer), spec.exit_layer
        config = dict(technique.adapter.config.to_dict())
        for key in _DROPPED_KEYS:
            config.pop(key, None)
        jump_layers = [int(i) for i in spec.jump_layers]
        for key in ("layer_types",):
            if isinstance(config.get(key), list):
                config[key] = [config[key][i - 1] for i in jump_layers]
        if "max_window_layers" in config:
            config["max_window_layers"] = sum(1 for i in jump_layers if i <= int(config["max_window_layers"]))
        captured = [u] + ([m] if m >= 1 and m != u else [])
        architecture = _architecture(technique)
        config.update({
            "architectures": [DRAFTER_ARCHITECTURE],
            "num_hidden_layers": len(jump_layers),
            "target_hidden_size": int(technique.adapter.hidden_size),
            "eagle_config": {
                "eagle_aux_hidden_state_layer_ids": sglang_aux_ids(captured) if u >= 1 else [0],
                "use_aux_hidden_state": True,
            },
            CONFIG_KEY: spec.to_dict(),
            SGLANG_CONFIG_KEY: {"target_architecture": architecture, "target_layer": m, "source_layer": u,
                                "exit_layer": r, "jump_layers": jump_layers},
        })
        config.pop("draft_vocab_size", None)  # full vocabulary: SGLang shares the target's LM head
        return config

    def drafter_tensors(self, technique: nn.Module) -> dict[str, torch.Tensor]:
        """The predictor, mixer and readout tensors, keyed ``jump.<name>`` (LoRA and copies are merged elsewhere)."""
        state = self._vllm().drafter_state(technique)
        return {"jump." + name: tensor for name, tensor in state.items() if not name.startswith(("lora.", "blocks."))}

    def layer_tensors(self, technique: Any, spec: JumpDrafterSpec,
                      base_state: Mapping[str, torch.Tensor] | None = None) -> dict[str, torch.Tensor]:
        """The jump-path layers as SGLang loads them (``model.layers.<j>.<hf name>``) and the final norm.

        ``base_state`` is the target's state dict as served (:func:`economize.backends.vllm.serving.weights.
        serving_state`); by default the model's own parameters.
        """
        adapter = technique.adapter
        state = self._vllm().drafter_state(technique)
        adaptation = str(spec.config.adaptation)
        scale = float(spec.config.resolved_lora_alpha) / float(spec.config.lora_rank)
        out: dict[str, torch.Tensor] = {}
        for j, index in enumerate(spec.jump_layers):
            layer = adapter.layer(index)
            prefix = adapter.module_path(layer) + "."
            if adaptation == "full":
                copy_prefix = f"blocks.{index}."
                weights = {key.removeprefix(copy_prefix): value for key, value in state.items()
                           if key.startswith(copy_prefix)}
            elif base_state is not None:
                weights = {key.removeprefix(prefix): value for key, value in base_state.items()
                           if key.startswith(prefix)}
            else:
                weights = {key: value.detach() for key, value in layer.state_dict().items()}
            if not weights:
                raise ConfigError(f"no tensors for jump-path layer {index}")
            weights = {key: value.to("cpu", torch.float32).clone() for key, value in weights.items()}
            if adaptation == "lora":
                targets = adapter.lora_targets(layer)
                for name, target in targets.items():
                    lora_prefix = f"lora.layers.{index}.{name}."
                    a, b = state.get(lora_prefix + "lora_A"), state.get(lora_prefix + "lora_B")
                    if a is None or b is None:
                        continue
                    key = _layer_path(layer, target) + ".weight"
                    if key not in weights:
                        raise UnsupportedBackendError(
                            f"the LoRA update of layer {index} {name} adapts a map without a dense weight; SGLang "
                            "loads merged dense layers")
                    weights[key] = weights[key] + scale * (b.detach().float().cpu() @ a.detach().float().cpu())
            out.update({f"model.layers.{j}.{key}": value.contiguous() for key, value in weights.items()})
        norm = adapter.final_norm
        norm_prefix = adapter.module_path(norm) + "."
        if base_state is not None and norm_prefix + "weight" in base_state:
            out["model.norm.weight"] = base_state[norm_prefix + "weight"].to("cpu", torch.float32).contiguous()
        else:
            out["model.norm.weight"] = norm.weight.detach().to("cpu", torch.float32).contiguous()
        return out

    def tensors(self, model: Any, base_state: Mapping[str, torch.Tensor] | None = None) -> dict[str, torch.Tensor]:
        """All tensors of the draft folder in float32 on the CPU."""
        technique: Any = self.technique_of(model)
        spec = self.check(model)
        out = {name: tensor.detach().to("cpu", torch.float32).contiguous()
               for name, tensor in self.drafter_tensors(technique).items()}
        out.update(self.layer_tensors(technique, spec, base_state))
        return out

    def live_tensors(self, model: Any) -> dict[str, torch.Tensor]:
        return self.tensors(model)

    def metadata(self) -> dict[str, str]:
        return {"format": "pt", "economize": CONFIG_KEY}


@register_exporter("sglang", "early_exit")
class EarlyExitSglangExporter(JumpSglangExporter):
    """An early-exit drafter (``m = u = 0``) for SGLang."""

    vllm_exporter = EarlyExitVllmExporter


@register_exporter("sglang", "free")
class FreeSglangExporter(JumpSglangExporter):
    """A FREE shallow-exit drafter for SGLang (the early-exit layout)."""

    vllm_exporter = FreeVllmExporter


@register_exporter("sglang", "layer_skip")
class LayerSkipSglangExporter(JumpSglangExporter):
    """A layer-skipping self-drafter (skipped layers dropped from the jump path) for SGLang."""

    vllm_exporter = LayerSkipVllmExporter


@register_exporter("sglang", "dmtd")
class DmtdSglangExporter(JumpSglangExporter):
    """A DMTD drafter (the last layers re-run as a jump path) for SGLang."""

    vllm_exporter = DmtdVllmExporter


def check_sglang_eagle3(technique: Any) -> None:
    """Refuse an EAGLE-3 drafter SGLang's ``LlamaForCausalLMEagle3`` would load wrongly.

    Raises:
        UnsupportedBackendError: Biased attention, ``norm_before_residual=True`` or an aux state ``h^0``.
    """
    layer = technique.layer
    if layer.self_attn.q_proj.bias is not None:
        raise UnsupportedBackendError("SGLang's LlamaForCausalLMEagle3 builds its attention without biases: an "
                                      "EAGLE-3 drafter with attention_bias cannot load there; serve it on vLLM")
    if bool(layer.norm_before_residual):
        raise UnsupportedBackendError("SGLang's EAGLE-3 layer has no norm_before_residual option: train the drafter "
                                      "with norm_before_residual=False or serve it on vLLM")
    if any(int(i) < 1 for i in technique.aux_layers):
        raise UnsupportedBackendError("SGLang captures inputs of target layers: the embedding h^0 cannot be an "
                                      "EAGLE-3 aux state there")


@register_exporter("sglang", "eagle3")
class Eagle3SglangExporter(Eagle3VllmExporter):
    """An EAGLE-3 drafter as SGLang's ``LlamaForCausalLMEagle3`` folder (aux ids shifted to SGLang's convention)."""

    def config(self, model: Any, **options: Any) -> dict[str, Any]:
        technique: Any = self.technique_of(model)
        check_sglang_eagle3(technique)
        config = self.build_config(model, format="eagle3")
        aux = [int(i) for i in config.pop("eagle_aux_hidden_state_layer_ids")]
        config["architectures"] = ["LlamaForCausalLMEagle3"]
        config["eagle_config"] = {"eagle_aux_hidden_state_layer_ids": sglang_aux_ids(aux),
                                  "use_aux_hidden_state": True}
        config.pop("norm_before_residual", None)
        if technique.lm_head is None or int(technique.draft_vocab_size) == int(technique.vocab_size):
            config.pop("draft_vocab_size", None)  # SGLang loads the target's head only without a draft vocabulary
        if technique.lm_head is not None and "draft_vocab_size" not in config:
            config["draft_vocab_size"] = int(technique.draft_vocab_size)  # an own head is loaded from the folder
        return config
