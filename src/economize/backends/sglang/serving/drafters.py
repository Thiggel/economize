"""Drafters SGLang runs natively: n-gram lookup, a draft LM, EAGLE-3 (with FR-Spec), Gemma-4 assistant, vendor MTP
and DFlash.

Each adapter sets SGLang's ``speculative_*`` server arguments (a chain of ``n`` drafts, ``topk=1``, ``n + 1``
verified tokens) and writes the drafter under ``drafter/``. The draft checkpoints are the ones the vLLM layer writes
(:mod:`economize.backends.vllm.serving.drafters`) where SGLang reads the same layout, and the SGLang exporters of
:mod:`economize.backends.sglang.export` where it differs (EAGLE-3 aux layer ids and the shared head).
"""

from __future__ import annotations

import json
from typing import Any

from torch import nn

from ....errors import ConfigError
from ...exporters import UnsupportedBackendError
from ...vllm.serving.drafters import DRAFTER_DIR, MTP_COUNT_KEYS, _check_eagle3, _ct_quantized
from .base import Role, SglangPlanBuilder, SglangTechnique, Support, register_sglang_technique


def _config(technique: nn.Module) -> Any:
    """The technique's config (``nn.Module`` attributes are typed as tensors or modules)."""
    return technique.config

__all__ = [
    "DRAFTER_DIR",
    "DFlashSglang",
    "DraftModelSglang",
    "DraftQuantSglang",
    "Eagle3Sglang",
    "FRSpecSglang",
    "Gemma4AssistantSglang",
    "LookupSglang",
    "PlacedDraftModelSglang",
    "VendorMTPSglang",
]


class _Drafter(SglangTechnique):
    """A drafter: takes the ``DRAFTER`` role (one ``speculative_algorithm`` per engine)."""

    role = Role.DRAFTER


@register_sglang_technique
class LookupSglang(_Drafter):
    """``lookup`` (prompt lookup decoding): SGLang's ``NGRAM`` algorithm, a chain of ``n`` drafts from its n-gram
    trie."""

    technique = "lookup"
    support = Support.NATIVE
    reason = ("speculative_algorithm NGRAM with BFS breadth 1 (a chain); SGLang's trie proposes from the prompt and "
              "the generated text, so its drafts differ from prompt lookup, the output stays greedy-exact")

    def num_tokens(self, technique: nn.Module, builder: SglangPlanBuilder) -> int:
        """Draft length: the configured maximum proposal, else the plan's ``n``."""
        return int(_config(technique).max_proposal or builder.num_speculative_tokens)

    def contribute(self, technique: nn.Module, builder: SglangPlanBuilder) -> None:
        n = self.num_tokens(technique, builder)
        for key, value in {"speculative_algorithm": "NGRAM", "speculative_num_draft_tokens": n + 1,
                           "speculative_ngram_min_bfs_breadth": 1, "speculative_ngram_max_bfs_breadth": 1}.items():
            builder.set_engine(key, value, owner=self.technique)
        builder.note(self.technique, "SGLang's NGRAM trie matches suffixes of up to speculative_ngram_max_trie_depth "
                                     f"tokens instead of {int(_config(technique).min_ngram)}-"
                                     f"{int(_config(technique).max_ngram)}-grams at the {_config(technique).match} "
                                     "position: different drafts, the same greedy output")


@register_sglang_technique
class DraftModelSglang(_Drafter):
    """``draft_model``: a smaller LM of the same tokenizer (SGLang's ``STANDALONE`` algorithm)."""

    technique = "draft_model"
    support = Support.NATIVE
    reason = "speculative_algorithm STANDALONE with the draft LM saved to the plan folder"

    def contribute(self, technique: nn.Module, builder: SglangPlanBuilder) -> None:
        builder.set_speculative("STANDALONE", None, owner=self.technique)

    def export(self, technique: nn.Module, builder: SglangPlanBuilder) -> None:
        target = builder.path(DRAFTER_DIR)
        draft: Any = technique.draft_model
        draft.save_pretrained(target)
        builder.set_engine("speculative_draft_model_path", str(target), owner=self.technique)


@register_sglang_technique
class PlacedDraftModelSglang(DraftModelSglang):
    """``placed_draft_model``: the same draft LM; SGLang runs it on the target's GPUs."""

    technique = "placed_draft_model"
    reason = "speculative_algorithm STANDALONE; SGLang runs the draft LM on the target's GPUs"

    def contribute(self, technique: nn.Module, builder: SglangPlanBuilder) -> None:
        super().contribute(technique, builder)
        builder.note(self.technique, f"the draft LM runs on the target's GPUs, not on {_config(technique).device}")


@register_sglang_technique
class Gemma4AssistantSglang(_Drafter):
    """``gemma4_assistant``: Google's Gemma-4 assistant drafter (SGLang's ``NEXTN``, run as frozen-KV MTP)."""

    technique = "gemma4_assistant"
    support = Support.NATIVE
    reason = "speculative_algorithm NEXTN with the assistant checkpoint (SGLang's Gemma-4 assistant drafter)"

    def contribute(self, technique: nn.Module, builder: SglangPlanBuilder) -> None:
        builder.set_speculative("NEXTN", None, owner=self.technique)

    def export(self, technique: nn.Module, builder: SglangPlanBuilder) -> None:
        assistant: Any = technique.assistant
        if callable(getattr(assistant, "save_pretrained", None)):
            target = builder.path(DRAFTER_DIR)
            assistant.save_pretrained(target)
            draft = str(target)
        else:  # not a transformers model: serve the configured checkpoint as is
            draft = str(_config(technique).model)
        builder.set_engine("speculative_draft_model_path", draft, owner=self.technique)


@register_sglang_technique
class VendorMTPSglang(_Drafter):
    """``vendor_mtp``: MTP modules under their vendor names inside the target folder (SGLang's ``NEXTN``)."""

    technique = "vendor_mtp"
    support = Support.NATIVE
    reason = "speculative_algorithm NEXTN with the modules written under their vendor names into the target checkpoint"

    def check(self, technique: nn.Module, builder: SglangPlanBuilder) -> None:
        if _config(technique).head != "shared":
            raise UnsupportedBackendError("SGLang's MTP modules use the target's LM head; vendor_mtp with head='own' "
                                          "has no SGLang equivalent (train with head='shared')")
        if _ct_quantized(builder):
            raise UnsupportedBackendError("vendor_mtp next to a compressed-tensors base is not exported yet: the "
                                          "quantization config would have to exclude the MTP layers")

    def count_key(self, technique: Any, builder: SglangPlanBuilder) -> str:
        """The ``config.json`` key that counts the MTP modules (the one the base config has, else the family's)."""
        config = builder.base.config
        for key in ("num_nextn_predict_layers", "mtp_num_hidden_layers"):
            if getattr(config, key, None) is not None:
                return key
        return MTP_COUNT_KEYS.get(technique.family, "num_nextn_predict_layers")

    def contribute(self, technique: nn.Module, builder: SglangPlanBuilder) -> None:
        builder.rewrite_base = True
        builder.config_edits[self.count_key(technique, builder)] = int(_config(technique).num_modules)
        builder.set_speculative("NEXTN", None, owner=self.technique)
        builder.note(self.technique, "vendor MTP export is checked on tiny models only and was not GPU-tested on "
                                     "SGLang")

    def export(self, technique: nn.Module, builder: SglangPlanBuilder) -> None:
        from ...vllm.serving.drafters import VendorMTPVllm

        VendorMTPVllm().export(technique, builder)  # same vendor tensor names, model-mtp.safetensors
        builder.set_engine("speculative_draft_model_path", str(builder.path("model")), owner=self.technique)


@register_sglang_technique
class Eagle3Sglang(_Drafter):
    """``eagle3``: SGLang's EAGLE-3 draft model (``LlamaForCausalLMEagle3``)."""

    technique = "eagle3"
    support = Support.NATIVE
    reason = ("speculative_algorithm EAGLE3 (LlamaForCausalLMEagle3, eagle_config aux layer ids, the target's head "
              "for a shared head)")

    def check(self, technique: nn.Module, builder: SglangPlanBuilder) -> None:
        from ..export import check_sglang_eagle3

        _check_eagle3(technique)
        check_sglang_eagle3(technique)

    def contribute(self, technique: nn.Module, builder: SglangPlanBuilder) -> None:
        builder.set_speculative("EAGLE3", None, owner=self.technique)
        if technique.lm_head is None:
            builder.note(self.technique, "head='shared': SGLang applies the target's LM head without its logit "
                                         "softcapping, which changes the draft distribution only")

    def export(self, technique: nn.Module, builder: SglangPlanBuilder) -> None:
        from ..export import Eagle3SglangExporter

        target = builder.path(DRAFTER_DIR)
        Eagle3SglangExporter().export(technique, target, num_speculative_tokens=builder.num_speculative_tokens)
        builder.set_engine("speculative_draft_model_path", str(target), owner=self.technique)


@register_sglang_technique
class FRSpecSglang(_Drafter):
    """``frspec`` around ``eagle3``: the drafter's head restricted to the frequent ids (SGLang's ``hot_token_id``)."""

    technique = "frspec"
    support = Support.NATIVE
    reason = "around eagle3: EAGLE3 with draft_vocab_size, d2t (SGLang's hot_token_id) and the head rows of the ids"

    def check(self, technique: nn.Module, builder: SglangPlanBuilder) -> None:
        from ..export import check_sglang_eagle3

        inner: Any = technique.inner
        if getattr(inner, "name", None) != "eagle3":
            raise UnsupportedBackendError(f"SGLang restricts the draft vocabulary of EAGLE-3 drafters only; frspec "
                                          f"around {getattr(inner, 'name', type(inner).__name__)!r} has no SGLang "
                                          "path")
        _check_eagle3(inner)
        check_sglang_eagle3(inner)
        if int(inner.draft_vocab_size) != int(inner.vocab_size):
            raise UnsupportedBackendError("the eagle3 drafter inside frspec already has a reduced vocabulary")
        if not technique.has_vocab:
            raise ConfigError("frspec has no draft vocabulary yet: set it (set_vocab or token_ids) before export")

    def contribute(self, technique: nn.Module, builder: SglangPlanBuilder) -> None:
        builder.set_speculative("EAGLE3", None, owner=self.technique)
        builder.note(self.technique, "SGLang drafts from the hot-token head and maps the ids back (hot_token_id); "
                                     "the target's logit softcapping is dropped")

    def export(self, technique: nn.Module, builder: SglangPlanBuilder) -> None:
        from safetensors.torch import save_file

        from ...vllm.serving.drafters import eagle3_restricted
        from ..export import Eagle3SglangExporter

        inner: Any = technique.inner
        ids: Any = technique.draft_ids
        target = builder.path(DRAFTER_DIR)
        target.mkdir(parents=True, exist_ok=True)
        exporter = Eagle3SglangExporter()
        config = exporter.config(inner, num_speculative_tokens=builder.num_speculative_tokens)
        config["draft_vocab_size"] = int(ids.numel())
        tensors = eagle3_restricted(inner, ids, inner.adapter.lm_head if inner.lm_head is None else None)
        tensors.pop("t2d", None)  # SGLang derives everything from d2t
        (target / exporter.config_file).write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
        save_file(tensors, str(target / exporter.weights_file), metadata=exporter.metadata())
        builder.set_engine("speculative_draft_model_path", str(target), owner=self.technique)


@register_sglang_technique
class DFlashSglang(_Drafter):
    """``dflash``: SGLang's DFlash drafter (``speculative_algorithm DFLASH``, the z-lab layout vLLM reads as well)."""

    technique = "dflash"
    support = Support.NATIVE
    reason = "speculative_algorithm DFLASH with the z-lab DFlashDraftModel checkpoint (block of n + 1 slots)"

    def check(self, technique: nn.Module, builder: SglangPlanBuilder) -> None:
        block = int(_config(technique).block_size)
        if builder.num_speculative_tokens > block - 1:
            raise UnsupportedBackendError(f"a DFlash drafter with block_size={block} drafts at most {block - 1} "
                                          f"tokens; asked for num_speculative_tokens={builder.num_speculative_tokens}")

    def contribute(self, technique: nn.Module, builder: SglangPlanBuilder) -> None:
        n = int(builder.num_speculative_tokens)
        builder.set_speculative("DFLASH", None, owner=self.technique, speculative_num_steps=1,
                                speculative_num_draft_tokens=n + 1)
        builder.note(self.technique, "the DFlash export is checked against vLLM's loader; SGLang's DFLASH path was "
                                     "not GPU-tested")
        cap = _config(technique).logit_softcap
        if cap or (cap is None and getattr(technique.adapter, "final_logit_softcap", None)):
            builder.note(self.technique, "SGLang's DFlash head applies no logit soft cap, which changes "
                                         "probabilistic drafts only")

    def export(self, technique: nn.Module, builder: SglangPlanBuilder) -> None:
        from ...exporters import EXPORTERS, exporter_key

        target = builder.path(DRAFTER_DIR)
        EXPORTERS.get(exporter_key("vllm", "dflash"))().export(
            technique, target, num_speculative_tokens=builder.num_speculative_tokens)
        builder.set_engine("speculative_draft_model_path", str(target), owner=self.technique)


@register_sglang_technique
class DraftQuantSglang(SglangTechnique):
    """``draft_quant``: the drafter's weights are rounded in place, so the drafter export writes them as they are."""

    technique = "draft_quant"
    role = Role.DRAFT_OPTION
    support = Support.DENSIFIED
    reason = "the drafter is exported with its rounded (dequantized) weights in the dense dtype"

    def check(self, technique: nn.Module, builder: SglangPlanBuilder) -> None:
        if _config(technique).backend != "fake":
            raise UnsupportedBackendError(f"draft_quant backend {_config(technique).backend!r} has no SGLang export")

    def contribute(self, technique: nn.Module, builder: SglangPlanBuilder) -> None:
        builder.note(self.technique, "the drafter runs the rounded weights in the dense dtype: same drafts, no "
                                     "memory or speed saving")
