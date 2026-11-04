"""The llama.cpp runtime: its technique adapters and the support level of every technique.

llama.cpp runs a GGUF file with its own kernels. What it realizes of economize's techniques:

* weights: ``quant`` codes that have a GGUF block type are stored in it (``int4`` as ``Q4_0`` / ``Q4_1``, ``int8``
  as ``Q8_0``); pruned, low-rank and LittleBit weights are written dense, a depth-pruned model is a plain model;
* drafting: a separate draft model (``draft_model``, ``placed_draft_model``) becomes ``llama-server --model-draft``,
  and prompt lookup (``lookup``) llama-cpp-python's ``LlamaPromptLookupDecoding``;
* everything else is refused with the reason (llama.cpp has no hook for hidden-state drafters, dynamic depth,
  activation sparsity, score-based KV eviction, fp8 or KIVI caches).

A technique without an adapter takes its family's refusal (:data:`GGUF`'s ``defaults``). Packages add adapters
through the entry-point group ``economize.gguf_techniques`` or with ``@GGUF.register``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from torch import nn

from ...core.config import TechniqueConfig
from ...registry import Registry
from ..export_adapters import QuantExport, register_weight_adapters
from ..runtime_export import ExportBuilder, ExportRuntime, ExportTechnique, config_of
from ..vllm.serving.base import Role, Support
from .writer import code_type

if TYPE_CHECKING:
    from ...techniques.quant.grid import QuantizedWeight

__all__ = ["GGUF", "GGUF_TECHNIQUES", "GgufDraftModel", "GgufLookup", "GgufQuant"]

#: The llama.cpp adapters by technique name (the :attr:`ExportRuntime.adapters` of :data:`GGUF`).
GGUF_TECHNIQUES: Registry[type[ExportTechnique]] = Registry("llama.cpp technique adapter",
                                                    "economize.gguf_techniques")

GGUF = ExportRuntime(
    name="gguf",
    title="llama.cpp",
    adapters=GGUF_TECHNIQUES,
    defaults={
        "dynamic-depth": (Role.FORWARD, "llama.cpp runs every layer for every token (no per-token depth routing)"),
        "drafting": (Role.DRAFTER, "llama.cpp drafts with a separate draft model (--model-draft) or by prompt "
                                   "lookup; it has no hook for drafters that read the target's hidden states"),
        "draft-compression": (Role.DRAFT_OPTION, "llama.cpp runs the draft model it is given; quantize the draft "
                                                 "model itself (draft_model with a quantized draft GGUF)"),
        "KV-quantization": (Role.KV_CACHE, "llama.cpp's cache types (--cache-type-k/v f16, q8_0, q4_0, ...) are "
                                           "block-wise; it has no fp8 or KIVI cache"),
        "KV-eviction": (Role.KV_EVICTION, "llama.cpp keeps or shifts the whole context (--context-shift); it has "
                                          "no score-based KV eviction"),
        "activation-sparsity": (Role.FORWARD, "llama.cpp has no activation-sparse kernels"),
    },
    fallback="llama.cpp has no path for this technique",
    exclusive={Role.DRAFTER: "draft (--model-draft or one llama-cpp-python draft_model)",
               Role.KV_CACHE: "KV-cache type"},
)


class GgufQuant(QuantExport):
    """``quant``: ``int4`` and ``int8`` codes are stored as GGUF ``Q4_0`` / ``Q4_1`` / ``Q8_0`` blocks.

    The codes are repacked bit for bit; the block scales are float16 (GGUF's), so the stored weights differ from
    economize's by the float16 rounding of the scales. Weights whose groups are not multiples of 32 input columns,
    ``act_order`` weights and other grids are written dequantized (the export then reports ``DENSIFIED``).
    """

    reason = ("int4 / int8 codes of the compressed-tensors, marlin, machete and torchao backends are stored as GGUF "
              "Q4_0, Q4_1 or Q8_0 blocks (float16 block scales); backend='fake' and other formats are written "
              "dequantized")
    fp8_activations_reason = ("llama.cpp has no fp8 activations (W8A8); export the weight-only model or let "
                              "llama-quantize make a K-quant from an f16 export")

    def stores(self, config: TechniqueConfig) -> bool:
        backend = getattr(config, "backend", "fake")
        return backend != "fake" and getattr(config, "format", "") in ("int4", "int8")

    def stores_weight(self, name: str, weight: QuantizedWeight) -> str | None:
        code, why = code_type(weight)
        return None if code is not None else why


register_weight_adapters(GGUF, GgufQuant)


@GGUF.register
class GgufDraftModel(ExportTechnique):
    """``draft_model``: the draft LM is written as ``draft.gguf`` and run with ``llama-server --model-draft``."""

    technique = "draft_model"
    role = Role.DRAFTER
    support = Support.NATIVE
    reason = "llama-server --model-draft draft.gguf (the draft LM written by the same converter)"

    def contribute(self, technique: nn.Module, builder: ExportBuilder) -> None:
        draft = getattr(technique, "draft_model")  # noqa: B009 (nn.Module types attributes as tensors)
        builder.drafter = draft
        builder.set("--model-draft", "draft.gguf", owner=self.technique)
        builder.set("--draft-max", builder.num_draft_tokens, owner=self.technique)
        builder.set("--draft-min", 0, owner=self.technique)


@GGUF.register
class GgufPlacedDraftModel(GgufDraftModel):
    """``placed_draft_model``: the same draft GGUF (llama.cpp places it with ``--device-draft``)."""

    technique = "placed_draft_model"
    reason = "llama-server --model-draft draft.gguf; llama.cpp places the draft with --device-draft"


@GGUF.register
class GgufLookup(ExportTechnique):
    """``lookup``: llama-cpp-python's ``LlamaPromptLookupDecoding`` (the earliest match of the longest n-gram)."""

    technique = "lookup"
    role = Role.DRAFTER
    support = Support.NATIVE
    reason = ("llama-cpp-python's LlamaPromptLookupDecoding (earliest match, n-grams from max_ngram down to 1); "
              "match='latest' or min_ngram > 1 have no llama.cpp equivalent")

    def level(self, config: TechniqueConfig) -> Support:
        earliest = getattr(config, "match", "earliest") == "earliest"
        return Support.NATIVE if earliest and getattr(config, "min_ngram", 1) == 1 else Support.UNSUPPORTED

    def contribute(self, technique: nn.Module, builder: ExportBuilder) -> None:
        config = config_of(technique)
        proposal = getattr(config, "max_proposal", None) or builder.num_draft_tokens
        builder.set("llama_cpp.LlamaPromptLookupDecoding",
                    {"max_ngram_size": int(getattr(config, "max_ngram", 3)), "num_pred_tokens": int(proposal)},
                    owner=self.technique)
        builder.note(self.technique, "run it with economize.backends.gguf.load_llama(export), which passes "
                                     "LlamaPromptLookupDecoding as llama_cpp.Llama(draft_model=...)")


#: Techniques whose family default would be too vague, with the reason llama.cpp cannot run them.
_REFUSALS: dict[str, tuple[Role, str]] = {
    "eagle": (Role.DRAFTER, "llama.cpp has no EAGLE draft head (it drafts with a separate model)"),
    "eagle3": (Role.DRAFTER, "llama.cpp has no EAGLE-3 draft head (it drafts with a separate model)"),
    "medusa": (Role.DRAFTER, "llama.cpp has no Medusa heads (it drafts with a separate model)"),
    "hydra": (Role.DRAFTER, "llama.cpp has no Hydra heads (it drafts with a separate model)"),
    "vendor_mtp": (Role.DRAFTER, "the GGUF writer drops multi-token-prediction layers"),
    "lookahead": (Role.DRAFTER, "llama.cpp's lookahead example is not a server option, and its n-gram pool "
                                "differs from economize's Jacobi lookahead"),
    "pard": (Role.DRAFTER, "llama.cpp drafts autoregressively; PARD's parallel mask-token drafts need a custom "
                           "forward"),
    "frspec": (Role.DRAFT_OPTION, "llama.cpp has no reduced draft vocabulary"),
    "draft_quant": (Role.DRAFT_OPTION, "quantize the draft LM's GGUF instead (llama-quantize draft.gguf)"),
    "kv_fp8": (Role.KV_CACHE, "llama.cpp has no fp8 KV cache (its cache types are f16, bf16, q8_0, q4_0, ...)"),
    "kv_evict": (Role.KV_EVICTION, "llama.cpp has no score-based KV eviction (only --context-shift)"),
    "teal": (Role.FORWARD, "llama.cpp has no activation-sparse kernels"),
}


for _name, (_role, _reason) in _REFUSALS.items():
    GGUF.refuse(_name, _role, _reason)
