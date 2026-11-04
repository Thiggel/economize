"""The TensorRT-LLM runtime: its technique adapters and the support level of every technique.

The export is a TensorRT-LLM checkpoint (``trtllm-build``'s input). What TensorRT-LLM realizes of economize's
techniques:

* weights: symmetric int4 codes in groups of 64 or 128 are stored as ``W4A16_AWQ``; other codes and the ``fake``
  backend are written dequantized (TensorRT-LLM's int8 and per-channel weight-only checkpoints hold float weights
  that ``trtllm-build`` quantizes itself); pruned, low-rank and LittleBit weights are written dense, and a
  depth-pruned model is a plain model;
* drafting: a separate draft model (``draft_model``, ``placed_draft_model``) is a second checkpoint in ``draft/``
  for TensorRT-LLM's draft-target speculation; Medusa heads (``head="own"``) go into the checkpoint as
  ``MedusaForCausalLM`` with the static tree as ``medusa_choices``; prompt lookup (``lookup``) is TensorRT-LLM's
  n-gram speculation;
* KV cache: an fp8 e4m3 cache cast without scales (``kv_fp8(scale="none")``) is ``kv_cache_quant_algo="FP8"``
  with unit scales; StreamingLLM (``kv_evict(method="streaming")``) is the sink-plus-window cache
  (``sink_token_length``, ``max_attention_window``);
* everything else is refused with the reason.

A technique without an adapter takes its family's refusal (:data:`TRTLLM`'s ``defaults``). Packages add adapters
through the entry-point group ``economize.trtllm_techniques`` or with ``@TRTLLM.register``.

Settings of an export: keys that start with ``--`` are ``trtllm-build`` flags; the others are keyword arguments of
TensorRT-LLM's ``LLM`` API (``speculative_config``, ``kv_cache_config``). None of this has been run through
TensorRT-LLM in economize's tests, which check the files only.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import nn

from ...core.config import TechniqueConfig
from ...registry import Registry
from ..export_adapters import QuantExport, StreamingWindowExport, register_weight_adapters
from ..exporters import UnsupportedBackendError
from ..runtime_export import ExportBuilder, ExportRuntime, ExportTechnique, ModelShape, config_of
from ..vllm.serving.base import Role, Support
from .writer import fused_mismatches, trt_reason

if TYPE_CHECKING:
    from ...techniques.quant.grid import QuantizedWeight

__all__ = ["TRTLLM", "TRTLLM_TECHNIQUES", "TrtDraftModel", "TrtKvFp8", "TrtLookup", "TrtMedusa", "TrtQuant",
           "TrtStreaming"]

#: The TensorRT-LLM adapters by technique name (the :attr:`ExportRuntime.adapters` of :data:`TRTLLM`).
TRTLLM_TECHNIQUES: Registry[type[ExportTechnique]] = Registry("TensorRT-LLM technique adapter",
                                                              "economize.trtllm_techniques")

TRTLLM = ExportRuntime(
    name="trtllm",
    title="TensorRT-LLM",
    adapters=TRTLLM_TECHNIQUES,
    defaults={
        "dynamic-depth": (Role.FORWARD, "a TensorRT-LLM engine runs every layer for every token (no per-token "
                                        "depth routing)"),
        "drafting": (Role.DRAFTER, "the TensorRT-LLM export writes draft-target, Medusa and n-gram speculation; "
                                   "this drafter has no TensorRT-LLM model"),
        "draft-compression": (Role.DRAFT_OPTION, "TensorRT-LLM runs the draft checkpoint it is given; quantize the "
                                                 "draft model itself"),
        "KV-quantization": (Role.KV_CACHE, "TensorRT-LLM's cache types are fp8 and int8 with one scale per layer; it "
                                           "has no per-token or KIVI cache"),
        "KV-eviction": (Role.KV_EVICTION, "TensorRT-LLM keeps sink tokens and a window; it has no score-based "
                                          "eviction"),
        "activation-sparsity": (Role.FORWARD, "TensorRT-LLM has no activation-sparse kernels"),
    },
    fallback="the TensorRT-LLM export has no path for this technique",
    exclusive={Role.DRAFTER: "speculative decoding mode", Role.KV_CACHE: "KV-cache type",
               Role.KV_EVICTION: "attention window"},
)


class TrtQuant(QuantExport):
    """``quant``: symmetric int4 codes in groups of 64 or 128 are stored as ``W4A16_AWQ``.

    The codes are repacked bit for bit and the scales stored in the checkpoint's float type. Other weights are
    written dequantized (the export then reports ``DENSIFIED``); a fused attention map holds codes only when its
    query, key and value maps all do.
    """

    reason = ("symmetric int4 codes of the compressed-tensors, marlin, machete and torchao backends in groups of 64 or "
              "128 are stored as W4A16_AWQ; other codes and backend='fake' are written dequantized (trtllm-build "
              "quantizes int8 / per-channel weight-only checkpoints from float weights itself)")
    fp8_activations_reason = ("the export writes TensorRT-LLM's weight-only int4 checkpoint; quantize to FP8 with "
                              "NVIDIA ModelOpt (W8A8 codes are not exported)")

    def stores(self, config: TechniqueConfig) -> bool:
        return getattr(config, "backend", "fake") != "fake" and getattr(config, "format", "") == "int4"

    def stores_weight(self, name: str, weight: QuantizedWeight) -> str | None:
        return trt_reason(weight)

    def contribute(self, technique: nn.Module, builder: ExportBuilder) -> None:
        super().contribute(technique, builder)
        mismatched = fused_mismatches(builder.quantized)
        for name in mismatched:
            del builder.quantized[name]
        if mismatched:
            example = next(iter(mismatched.items()))
            builder.densify(self.technique, f"{len(mismatched)} quantized maps are written dequantized to fit "
                                            f"TensorRT-LLM's fused maps (e.g. {example[0]}: {example[1]})")


register_weight_adapters(TRTLLM, TrtQuant)


@TRTLLM.register
class TrtDraftModel(ExportTechnique):
    """``draft_model``: the draft LM is a second checkpoint in ``draft/`` (draft-target speculation)."""

    technique = "draft_model"
    role = Role.DRAFTER
    support = Support.NATIVE
    reason = ("TensorRT-LLM draft-target speculation: the target built with --speculative_decoding_mode "
              "draft_tokens_external, the draft LM a second checkpoint in the same layout")

    def contribute(self, technique: nn.Module, builder: ExportBuilder) -> None:
        builder.drafter = getattr(technique, "draft_model")  # noqa: B009 (nn.Module types attributes as tensors)
        builder.set("--speculative_decoding_mode", "draft_tokens_external", owner=self.technique)
        builder.set("--max_draft_len", builder.num_draft_tokens, owner=self.technique)
        builder.set("speculative_config", {"decoding_type": "DraftTarget", "max_draft_len": builder.num_draft_tokens,
                                           "speculative_model_dir": "draft/engine"}, owner=self.technique)
        builder.note(self.technique, "build the draft too: trtllm-build --checkpoint_dir draft --output_dir "
                                     "draft/engine")


@TRTLLM.register
class TrtPlacedDraftModel(TrtDraftModel):
    """``placed_draft_model``: the same draft checkpoint (TensorRT-LLM places engines through its mapping)."""

    technique = "placed_draft_model"
    reason = ("TensorRT-LLM draft-target speculation with the draft checkpoint in draft/; the engines' placement is "
              "TensorRT-LLM's")


@TRTLLM.register
class TrtMedusa(ExportTechnique):
    """``medusa`` with ``head="own"``: TensorRT-LLM's ``MedusaForCausalLM`` (heads in the target's checkpoint).

    Each head is ``num_layers`` residual blocks ``x + SiLU(W x + b)`` on the post-norm state and its own vocabulary
    projection, stored as ``medusa_heads.{h}.medusa_layers.{l}.linear`` and ``medusa_heads.{h}.lm_head``. The static
    tree (or the top-1 chain of every head) becomes ``medusa_choices``.
    """

    technique = "medusa"
    role = Role.DRAFTER
    support = Support.NATIVE
    reason = ("TensorRT-LLM's MedusaForCausalLM (--speculative_decoding_mode medusa, medusa_choices from the tree); "
              "head='shared' heads (blocks before the final norm) have no TensorRT-LLM equivalent")

    def level(self, config: TechniqueConfig) -> Support:
        return Support.NATIVE if getattr(config, "head", "own") == "own" else Support.UNSUPPORTED

    def check(self, technique: nn.Module, builder: ExportBuilder) -> None:
        if self.level(config_of(technique)) is Support.UNSUPPORTED:
            raise UnsupportedBackendError("TensorRT-LLM's Medusa heads read the post-norm state and project with "
                                          "their own matrix: train with head='own' to export them")

    @staticmethod
    def choices(config: TechniqueConfig) -> list[list[int]]:
        """``medusa_choices``: the configured tree, or the top-1 chain of every head.

        Parameters
        ----------
        config : TechniqueConfig
            The ``medusa`` configuration.

        Returns
        -------
        list[list[int]]
            The tree paths in Medusa's node order.
        """
        from ...techniques.medusa.trees import resolve_tree

        tree = getattr(config, "tree", None)
        heads = int(getattr(config, "num_heads"))  # noqa: B009 (TechniqueConfig does not declare the field)
        paths = tuple((0,) * depth for depth in range(1, heads + 1)) if tree is None else resolve_tree(tree)
        return [list(path) for path in sorted(paths, key=lambda path: (len(path), path))]

    def contribute(self, technique: nn.Module, builder: ExportBuilder) -> None:
        config = config_of(technique)
        heads = list(getattr(technique, "heads"))  # noqa: B009 (nn.Module types attributes as tensors)
        choices = self.choices(config)
        for h, head in enumerate(heads):
            for i, block in enumerate(head.blocks):
                prefix = f"medusa_heads.{h}.medusa_layers.{i}.linear."
                builder.tensors[prefix + "weight"] = block.linear.weight.detach()
                builder.tensors[prefix + "bias"] = block.linear.bias.detach()
            builder.tensors[f"medusa_heads.{h}.lm_head.weight"] = head.proj.weight.detach()
        builder.header.update(architecture="MedusaForCausalLM", num_medusa_heads=len(heads),
                              num_medusa_layers=int(getattr(config, "num_layers")),  # noqa: B009
                              max_draft_len=len(choices), medusa_hidden_act="silu")
        builder.set("--speculative_decoding_mode", "medusa", owner=self.technique)
        builder.set("--max_draft_len", len(choices), owner=self.technique)
        builder.set("speculative_config", {"decoding_type": "Medusa", "max_draft_len": len(choices),
                                           "num_medusa_heads": len(heads), "medusa_choices": choices},
                    owner=self.technique)


@TRTLLM.register
class TrtLookup(ExportTechnique):
    """``lookup``: TensorRT-LLM's n-gram speculation (``NGramDecodingConfig``, the oldest match)."""

    technique = "lookup"
    role = Role.DRAFTER
    support = Support.NATIVE
    reason = ("TensorRT-LLM's NGramDecodingConfig (is_use_oldest=True, suffixes from max_ngram down to 1); "
              "match='latest' or min_ngram > 1 have no TensorRT-LLM equivalent")

    def level(self, config: TechniqueConfig) -> Support:
        earliest = getattr(config, "match", "earliest") == "earliest"
        return Support.NATIVE if earliest and getattr(config, "min_ngram", 1) == 1 else Support.UNSUPPORTED

    def check(self, technique: nn.Module, builder: ExportBuilder) -> None:
        if self.level(config_of(technique)) is Support.UNSUPPORTED:
            raise UnsupportedBackendError("TensorRT-LLM's n-gram speculation copies the oldest match of suffixes "
                                          "down to length 1: only lookup(match='earliest', min_ngram=1) maps to it")

    def contribute(self, technique: nn.Module, builder: ExportBuilder) -> None:
        config = config_of(technique)
        proposal = int(getattr(config, "max_proposal", None) or builder.num_draft_tokens)
        builder.set("speculative_config", {"decoding_type": "NGram", "max_draft_len": proposal,
                                           "max_matching_ngram_size": int(getattr(config, "max_ngram", 3)),
                                           "is_use_oldest": True, "is_keep_all": True, "is_public_pool": False},
                    owner=self.technique)


@TRTLLM.register
class TrtKvFp8(ExportTechnique):
    """``kv_fp8(format="e4m3", scale="none")``: ``kv_cache_quant_algo="FP8"`` with a unit scale per layer."""

    technique = "kv_fp8"
    role = Role.KV_CACHE
    support = Support.NATIVE
    reason = ("TensorRT-LLM's FP8 cache (kv_cache_quant_algo='FP8', kv_cache_scaling_factor 1.0) for an e4m3 cache "
              "cast without scales on every layer; it has no e5m2 or per-token cache, and one scale per layer for "
              "keys and values together")

    def level(self, config: TechniqueConfig) -> Support:
        plain = getattr(config, "format", "e4m3") == "e4m3" and getattr(config, "scale", "none") == "none"
        return Support.NATIVE if plain and not getattr(config, "layers", ()) else Support.UNSUPPORTED

    def check(self, technique: nn.Module, builder: ExportBuilder) -> None:
        if self.level(config_of(technique)) is Support.UNSUPPORTED:
            raise UnsupportedBackendError("TensorRT-LLM's FP8 cache stores e4m3 with one scale per layer: only "
                                          "kv_fp8(format='e4m3', scale='none') on every layer has an equivalent")

    def contribute(self, technique: nn.Module, builder: ExportBuilder) -> None:
        shape = ModelShape.of(builder.base, builder.runtime.title)
        builder.header["quantization"] = {"kv_cache_quant_algo": "FP8"}
        for i in range(shape.num_layers):
            builder.tensors[f"transformer.layers.{i}.attention.kv_cache_scaling_factor"] = torch.ones(1)
        builder.note(self.technique, "TensorRT-LLM's FP8 cache needs an Ada or Hopper GPU")


@TRTLLM.register
class TrtStreaming(StreamingWindowExport):
    """``kv_evict(method="streaming")``: TensorRT-LLM's sink tokens plus attention window."""

    reason = ("TensorRT-LLM's KvCacheConfig(sink_token_length=sinks, max_attention_window=[sinks + recent]) for "
              "kv_evict(method='streaming'); score-based methods are refused")

    def window_settings(self, sinks: int, recent: int) -> dict[str, object]:
        return {"kv_cache_config": {"sink_token_length": sinks, "max_attention_window": [sinks + recent]}}


#: Techniques whose family default would be too vague, with the reason TensorRT-LLM cannot run them.
_REFUSALS: dict[str, tuple[Role, str]] = {
    "eagle": (Role.DRAFTER, "TensorRT-LLM's EagleForCausalLM checkpoint layout is not written by economize; "
                            "convert the EAGLE weights with TensorRT-LLM's examples/eagle/convert_checkpoint.py"),
    "eagle3": (Role.DRAFTER, "TensorRT-LLM runs EAGLE-3 in its PyTorch backend from a Hugging Face model and draft "
                             "folder: export with economize.export(model, 'vllm', ...) and pass "
                             "EagleDecodingConfig(speculative_model_dir=...) to its LLM API"),
    "hydra": (Role.DRAFTER, "TensorRT-LLM has no Hydra heads"),
    "redrafter": (Role.DRAFTER, "TensorRT-LLM's ReDrafter checkpoint layout is not written by economize"),
    "lookahead": (Role.DRAFTER, "TensorRT-LLM's lookahead decoding keeps its own n-gram pool; it differs from "
                                "economize's Jacobi lookahead"),
    "vendor_mtp": (Role.DRAFTER, "the TensorRT-LLM checkpoint writer drops multi-token-prediction layers"),
    "frspec": (Role.DRAFT_OPTION, "TensorRT-LLM has no reduced draft vocabulary"),
    "draft_quant": (Role.DRAFT_OPTION, "quantize the draft LM itself (economize.apply(draft, QuantConfig(...)))"),
    "teal": (Role.FORWARD, "TensorRT-LLM has no activation-sparse kernels"),
}

for _name, (_role, _reason) in _REFUSALS.items():
    TRTLLM.refuse(_name, _role, _reason)
