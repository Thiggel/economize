"""Drafters SGLang runs natively: n-gram lookup, a draft LM, EAGLE-3 (with FR-Spec), Gemma-4 assistant, vendor MTP
and DFlash.

Each adapter sets SGLang's ``speculative_*`` server arguments (a chain of ``n`` drafts, ``topk=1``, ``n + 1``
verified tokens) and writes the drafter under ``drafter/``. The draft checkpoints are the ones the vLLM layer writes
(:mod:`economize.backends.vllm.serving.drafters`) where SGLang reads the same layout, and the SGLang exporters of
:mod:`economize.backends.sglang.export` where it differs (EAGLE-3 aux layer ids and the shared head).
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, ClassVar, cast

from ....errors import ConfigError
from ...exporters import UnsupportedBackendError
from ...vllm.serving.base import Configured
from ...vllm.serving.drafters import (
    DRAFT_QUANT_FAKE,
    DRAFT_QUANT_REASON,
    DRAFTER_DIR,
    MTP_COUNT_KEYS,
    _check_eagle3,
    _ct_quantized,
)
from .base import Role, SglangPlanBuilder, SglangTechnique, Support, register_sglang_technique

if TYPE_CHECKING:
    from transformers import PreTrainedModel

    from ....core.technique import Technique
    from ....techniques.dflash.config import DFlashConfig
    from ....techniques.draft_model.technique import DraftModelTechnique
    from ....techniques.eagle3.technique import Eagle3Technique
    from ....techniques.frspec.technique import FRSpecTechnique
    from ....techniques.gemma4_assistant.config import Gemma4AssistantDraftConfig
    from ....techniques.gemma4_assistant.technique import Gemma4AssistantTechnique
    from ....techniques.lookup.config import LookupConfig
    from ....techniques.placed_draft.config import PlacedDraftModelConfig
    from ....techniques.quant.config import DraftQuantConfig
    from ....techniques.vendor_mtp.config import VendorMTPConfig
    from ....techniques.vendor_mtp.technique import VendorMTPTechnique

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

    def num_tokens(self, technique: Technique, builder: SglangPlanBuilder) -> int:
        """Draft length: the configured maximum proposal, else the plan's ``n``."""
        return int(cast("LookupConfig", technique.config).max_proposal or builder.num_speculative_tokens)

    def contribute(self, technique: Technique, builder: SglangPlanBuilder) -> None:
        n = self.num_tokens(technique, builder)
        config = cast("LookupConfig", technique.config)
        for key, value in {"speculative_algorithm": "NGRAM", "speculative_num_draft_tokens": n + 1,
                           "speculative_ngram_min_bfs_breadth": 1, "speculative_ngram_max_bfs_breadth": 1}.items():
            builder.set_engine(key, value, owner=self.technique)
        builder.note(self.technique, "SGLang's NGRAM trie matches suffixes of up to speculative_ngram_max_trie_depth "
                                     f"tokens instead of {int(config.min_ngram)}-{int(config.max_ngram)}-grams at "
                                     f"the {config.match} "
                                     "position: different drafts, the same greedy output")


@register_sglang_technique
class DraftModelSglang(_Drafter):
    """``draft_model``: a smaller LM of the same tokenizer (SGLang's ``STANDALONE`` algorithm)."""

    technique = "draft_model"
    support = Support.NATIVE
    reason = "speculative_algorithm STANDALONE with the draft LM saved to the plan folder"

    def contribute(self, technique: Technique, builder: SglangPlanBuilder) -> None:
        builder.set_speculative("STANDALONE", None, owner=self.technique)

    def export(self, technique: Technique, builder: SglangPlanBuilder) -> None:
        target = builder.path(DRAFTER_DIR)
        draft = cast("PreTrainedModel", cast("DraftModelTechnique", technique).draft_model)
        draft.save_pretrained(target)
        builder.set_engine("speculative_draft_model_path", str(target), owner=self.technique)


@register_sglang_technique
class PlacedDraftModelSglang(DraftModelSglang):
    """``placed_draft_model``: the same draft LM; SGLang runs it on the target's GPUs."""

    technique = "placed_draft_model"
    reason = "speculative_algorithm STANDALONE; SGLang runs the draft LM on the target's GPUs"

    def contribute(self, technique: Technique, builder: SglangPlanBuilder) -> None:
        super().contribute(technique, builder)
        builder.note(self.technique, f"the draft LM runs on the target's GPUs, not on "
                                     f"{cast('PlacedDraftModelConfig', technique.config).device}")


@register_sglang_technique
class Gemma4AssistantSglang(_Drafter):
    """``gemma4_assistant``: Google's Gemma-4 assistant drafter (SGLang's ``NEXTN``, run as frozen-KV MTP)."""

    technique = "gemma4_assistant"
    support = Support.NATIVE
    reason = "speculative_algorithm NEXTN with the assistant checkpoint (SGLang's Gemma-4 assistant drafter)"

    def contribute(self, technique: Technique, builder: SglangPlanBuilder) -> None:
        builder.set_speculative("NEXTN", None, owner=self.technique)

    def export(self, technique: Technique, builder: SglangPlanBuilder) -> None:
        save = getattr(cast("Gemma4AssistantTechnique", technique).assistant, "save_pretrained", None)
        if callable(save):
            target = builder.path(DRAFTER_DIR)
            save(target)
            draft = str(target)
        else:  # not a transformers model: serve the configured checkpoint as is
            draft = str(cast("Gemma4AssistantDraftConfig", technique.config).model)
        builder.set_engine("speculative_draft_model_path", draft, owner=self.technique)


@register_sglang_technique
class VendorMTPSglang(_Drafter):
    """``vendor_mtp``: MTP modules under their vendor names inside the target folder (SGLang's ``NEXTN``)."""

    technique = "vendor_mtp"
    support = Support.NATIVE
    reason = "speculative_algorithm NEXTN with the modules written under their vendor names into the target checkpoint"

    def check(self, technique: Technique, builder: SglangPlanBuilder) -> None:
        if cast("VendorMTPConfig", technique.config).head != "shared":
            raise UnsupportedBackendError("SGLang's MTP modules use the target's LM head; vendor_mtp with head='own' "
                                          "has no SGLang equivalent (train with head='shared')")
        if _ct_quantized(builder):
            raise UnsupportedBackendError("vendor_mtp next to a compressed-tensors base is not exported yet: the "
                                          "quantization config would have to exclude the MTP layers")

    def count_key(self, technique: Technique, builder: SglangPlanBuilder) -> str:
        """The ``config.json`` key that counts the MTP modules (the one the base config has, else the family's)."""
        config = builder.base.config
        for key in ("num_nextn_predict_layers", "mtp_num_hidden_layers"):
            if getattr(config, key, None) is not None:
                return key
        return MTP_COUNT_KEYS.get(cast("VendorMTPTechnique", technique).family, "num_nextn_predict_layers")

    def contribute(self, technique: Technique, builder: SglangPlanBuilder) -> None:
        builder.rewrite_base = True
        modules = int(cast("VendorMTPConfig", technique.config).num_modules)
        builder.config_edits[self.count_key(technique, builder)] = modules
        builder.set_speculative("NEXTN", None, owner=self.technique)
        builder.note(self.technique, "vendor MTP export is checked on tiny models only and was not GPU-tested on "
                                     "SGLang")

    def export(self, technique: Technique, builder: SglangPlanBuilder) -> None:
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

    def check(self, technique: Technique, builder: SglangPlanBuilder) -> None:
        from ..export import check_sglang_eagle3

        _check_eagle3(technique)
        check_sglang_eagle3(cast("Eagle3Technique", technique))

    def contribute(self, technique: Technique, builder: SglangPlanBuilder) -> None:
        builder.set_speculative("EAGLE3", None, owner=self.technique)
        if cast("Eagle3Technique", technique).lm_head is None:
            builder.note(self.technique, "head='shared': SGLang applies the target's LM head without its logit "
                                         "softcapping, which changes the draft distribution only")

    def export(self, technique: Technique, builder: SglangPlanBuilder) -> None:
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

    def check(self, technique: Technique, builder: SglangPlanBuilder) -> None:
        from ..export import check_sglang_eagle3

        frspec = cast("FRSpecTechnique", technique)
        inner = frspec.inner
        if getattr(inner, "name", None) != "eagle3":
            raise UnsupportedBackendError(f"SGLang restricts the draft vocabulary of EAGLE-3 drafters only; frspec "
                                          f"around {getattr(inner, 'name', type(inner).__name__)!r} has no SGLang "
                                          "path")
        _check_eagle3(inner)
        eagle3 = cast("Eagle3Technique", inner)
        check_sglang_eagle3(eagle3)
        if int(eagle3.draft_vocab_size) != int(eagle3.vocab_size):
            raise UnsupportedBackendError("the eagle3 drafter inside frspec already has a reduced vocabulary")
        if not frspec.has_vocab:
            raise ConfigError("frspec has no draft vocabulary yet: set it (set_vocab or token_ids) before export")

    def contribute(self, technique: Technique, builder: SglangPlanBuilder) -> None:
        builder.set_speculative("EAGLE3", None, owner=self.technique)
        builder.note(self.technique, "SGLang drafts from the hot-token head and maps the ids back (hot_token_id); "
                                     "the target's logit softcapping is dropped")

    def export(self, technique: Technique, builder: SglangPlanBuilder) -> None:
        from safetensors.torch import save_file

        from ...vllm.serving.drafters import eagle3_restricted
        from ..export import Eagle3SglangExporter

        frspec = cast("FRSpecTechnique", technique)
        inner = cast("Eagle3Technique", frspec.inner)  # check() refused any other inner drafter
        ids = frspec.draft_ids
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

    def check(self, technique: Technique, builder: SglangPlanBuilder) -> None:
        block = int(cast("DFlashConfig", technique.config).block_size)
        if builder.num_speculative_tokens > block - 1:
            raise UnsupportedBackendError(f"a DFlash drafter with block_size={block} drafts at most {block - 1} "
                                          f"tokens; asked for num_speculative_tokens={builder.num_speculative_tokens}")

    def contribute(self, technique: Technique, builder: SglangPlanBuilder) -> None:
        n = int(builder.num_speculative_tokens)
        builder.set_speculative("DFLASH", None, owner=self.technique, speculative_num_steps=1,
                                speculative_num_draft_tokens=n + 1)
        builder.note(self.technique, "the DFlash export is checked against vLLM's loader; SGLang's DFLASH path was "
                                     "not GPU-tested")
        cap = cast("DFlashConfig", technique.config).logit_softcap
        if cap or (cap is None and getattr(technique.adapter, "final_logit_softcap", None)):
            builder.note(self.technique, "SGLang's DFlash head applies no logit soft cap, which changes "
                                         "probabilistic drafts only")

    def export(self, technique: Technique, builder: SglangPlanBuilder) -> None:
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
    support = Support.UNSUPPORTED
    reason = DRAFT_QUANT_REASON.format(engine="SGLang")
    #: Quantization backend -> how SGLang serves the quantized drafter (see :class:`DraftQuantVllm`).
    levels: ClassVar[dict[str, Support]] = {"fake": Support.DENSIFIED}

    def level(self, technique: Configured) -> Support:
        return self.levels.get(cast("DraftQuantConfig", technique.config).backend, Support.UNSUPPORTED)

    def reason_of(self, technique: Configured) -> str:
        """What SGLang serves for the configured backend, or why it refuses it.

        Parameters
        ----------
        technique : Configured
            The technique, or any object with its ``config``.

        Returns
        -------
        str
            The reason shown by :func:`economize.backend_support`.
        """
        if self.level(technique) is Support.DENSIFIED:
            return DRAFT_QUANT_FAKE
        return self.reason

    def contribute(self, technique: Technique, builder: SglangPlanBuilder) -> None:
        builder.note(self.technique, "the drafter runs the rounded weights in the dense dtype: same drafts, no "
                                     "memory or speed saving")
