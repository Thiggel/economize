"""vLLM adapters of the drafters: each sets the engine's ``speculative_config`` and writes the draft folder.

=========================  ==================================================================================
economize                  vLLM 0.30 (``speculative_config``)
=========================  ==================================================================================
``lookup``                 ``method="ngram"``, ``prompt_lookup_max/min`` (earliest longest match, as vLLM's);
                           ``match="latest"``: economize's ``custom_class`` proposer
``suffix_decoding``        ``method="suffix"``, ``suffix_decoding_*`` (needs ``arctic-inference`` installed); with
                           a prebuilt ``datastore`` or ``max_spec_offset``: the ``custom_class`` proposer
``rest``                   ``method="custom_class"`` (:class:`~economize.backends.vllm.proposers.EconomizeProposer`)
``lookahead``              the same, with the n-gram pool of the committed tokens only (``pool_from_prompt=True``)
``draft_model``            ``method="draft_model"``, the draft LM saved to ``out_dir/drafter``
``placed_draft_model``     the same (vLLM places the draft LM next to the target itself)
``pard``                   ``method="draft_model"``, ``parallel_drafting=True``, ``pard_token`` in the draft config
``gemma4_assistant``       ``method="mtp"`` with the assistant folder
``vendor_mtp``             ``method="mtp"``, the MTP modules written into the target folder (vendor names)
``eagle3``                 ``method="eagle3"`` (:class:`~economize.backends.vllm.eagle3.Eagle3VllmExporter`)
``medusa``                 ``method="medusa"`` (:class:`~economize.backends.vllm.medusa.MedusaVllmExporter`)
``dflash``                 ``method="dflash"`` (:class:`~economize.backends.vllm.dflash.DFlashVllmExporter`)
``jump``                   economize's vLLM plugin (``EconomizeJumpEagle3`` through ``method="eagle3"``)
``frspec`` over ``eagle3``  ``method="eagle3"`` with ``draft_vocab_size``, ``d2t`` and the head rows of the ids
``draft_quant``            the drafter's dequantized weights (``backend="fake"``): a draft option
=========================  ==================================================================================

Model-based proposers draft by sampling when the plan samples (``draft_sample_method="probabilistic"``); the
token-level ones (``ngram``, ``suffix``, ``custom_class``, ``medusa``) propose argmax chains, which vLLM's rejection
sampler treats as
one-hot proposals (lossless at any temperature).
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any

import torch
from torch import nn

from ....core.config import TechniqueConfig
from ....errors import ConfigError
from ...exporters import UnsupportedBackendError
from ..spec import speculative_config
from .base import PlanBuilder, Role, Support, VllmTechnique, register_vllm_technique

__all__ = [
    "DraftModelVllm",
    "DraftQuantVllm",
    "Eagle3Vllm",
    "FRSpecVllm",
    "Gemma4AssistantVllm",
    "JumpVllm",
    "LookaheadVllm",
    "LookupVllm",
    "MedusaVllm",
    "PardVllm",
    "PlacedDraftModelVllm",
    "RestVllm",
    "SuffixDecodingVllm",
    "TokenProposerVllm",
    "VendorMTPVllm",
    "custom_class_proposer",
    "eagle3_restricted",
]

#: Folder of the exported drafter inside the plan's output folder.
DRAFTER_DIR = "drafter"


def _ct_quantized(builder: PlanBuilder) -> bool:
    """Whether the served base is a compressed-tensors checkpoint (vLLM holds its weights as packed codes)."""
    from .weights import QuantVllm

    quant = builder.techniques.get("quant")
    return quant is not None and QuantVllm().level(quant) is Support.NATIVE


def _model_spec(builder: PlanBuilder, method: str, drafter: str | Path | None, **extra: Any) -> dict[str, Any]:
    """The ``speculative_config`` of a model-based proposer (sampling as the plan says)."""
    spec: dict[str, Any] = {"method": method, "num_speculative_tokens": builder.num_speculative_tokens}
    if drafter is not None:
        spec["model"] = str(drafter)
    if builder.sampling:
        spec |= {"draft_sample_method": "probabilistic", "rejection_sample_method": "standard"}
    return spec | extra


class _Drafter(VllmTechnique):
    """A drafter: takes the ``DRAFTER`` role and sets :attr:`PlanBuilder.speculative` in :meth:`export`."""

    role = Role.DRAFTER


# ---- token-level proposers ----------------------------------------------------------------------------------------


def _resolved(config: TechniqueConfig) -> TechniqueConfig:
    """``config`` with an absolute ``datastore`` folder, checked to exist (vLLM's workers load it from their own
    working directory)."""
    datastore = getattr(config, "datastore", None)
    if not datastore:
        return config
    folder = Path(datastore).expanduser().resolve()
    if not folder.is_dir():
        raise UnsupportedBackendError(f"the {config.technique!r} datastore {str(folder)!r} does not exist")
    return dataclasses.replace(config, datastore=str(folder))


def custom_class_proposer(builder: PlanBuilder, owner: str, config: TechniqueConfig, num_tokens: int) -> None:
    """Serve ``config`` through economize's ``custom_class`` proposer.

    The ``speculative_config`` names :class:`~economize.backends.vllm.proposers.EconomizeProposer`, and the engine's
    ``additional_config`` holds the drafter's configuration, which every worker rebuilds.

    Args:
        builder: The plan being built.
        owner: Name of the technique (for conflicts).
        config: The drafter's configuration (a technique with a registered proposer).
        num_tokens: Draft tokens per verification.

    Raises:
        UnsupportedBackendError: The technique has no registered proposer, or its datastore does not exist.
    """
    from ..proposers import CONFIG_KEY, PROPOSER_CLASS, proposer_entry

    builder.speculative = {"method": "custom_class", "model": PROPOSER_CLASS, "num_speculative_tokens": int(num_tokens)}
    builder.set_additional(CONFIG_KEY, proposer_entry(_resolved(config)), owner=owner)


class TokenProposerVllm(_Drafter):
    """A drafter served by economize's ``custom_class`` proposer (a factory in
    :data:`~economize.backends.vllm.proposers.PROPOSERS`)."""

    support = Support.PLUGIN

    def num_tokens(self, technique: nn.Module, builder: PlanBuilder) -> int:
        """Draft tokens per verification (default: the plan's)."""
        return builder.num_speculative_tokens

    def check(self, technique: nn.Module, builder: PlanBuilder) -> None:
        super().check(technique, builder)
        if self.level(technique) is Support.PLUGIN:
            _resolved(technique.config)

    def contribute(self, technique: nn.Module, builder: PlanBuilder) -> None:
        custom_class_proposer(builder, self.technique, technique.config, self.num_tokens(technique, builder))


@register_vllm_technique
class LookupVllm(TokenProposerVllm):
    """``lookup`` (prompt lookup decoding): vLLM's n-gram proposer for the earliest match, economize's proposer for
    the latest one."""

    technique = "lookup"
    support = Support.NATIVE
    reason = ("method='ngram' with prompt_lookup_max/min (vLLM matches the longest n-gram at its earliest "
              "position); match='latest' through economize's custom_class proposer")

    def level(self, technique: nn.Module) -> Support:
        return Support.NATIVE if technique.config.match == "earliest" else Support.PLUGIN

    def num_tokens(self, technique: nn.Module, builder: PlanBuilder) -> int:
        return int(technique.config.max_proposal or builder.num_speculative_tokens)

    def contribute(self, technique: nn.Module, builder: PlanBuilder) -> None:
        if self.level(technique) is Support.PLUGIN:
            super().contribute(technique, builder)
            return
        config = technique.config
        builder.speculative = {
            "method": "ngram",
            "prompt_lookup_max": int(config.max_ngram),
            "prompt_lookup_min": int(config.min_ngram),
            "num_speculative_tokens": self.num_tokens(technique, builder),
        }


@register_vllm_technique
class SuffixDecodingVllm(TokenProposerVllm):
    """``suffix_decoding``: vLLM's suffix proposer (Arctic Inference's suffix trees) without a datastore, economize's
    proposer over the prebuilt datastore otherwise."""

    technique = "suffix_decoding"
    support = Support.NATIVE
    reason = ("method='suffix' (needs the arctic-inference package); a prebuilt datastore or max_spec_offset "
              "through economize's custom_class proposer")

    def level(self, technique: nn.Module) -> Support:
        config = technique.config
        return Support.PLUGIN if config.datastore is not None or config.max_spec_offset else Support.NATIVE

    def num_tokens(self, technique: nn.Module, builder: PlanBuilder) -> int:
        return int(technique.config.max_spec_tokens or builder.num_speculative_tokens)

    def contribute(self, technique: nn.Module, builder: PlanBuilder) -> None:
        config = technique.config
        if config.tree:
            builder.note(self.technique, "vLLM verifies the best path of the suffix tree as one chain")
        if self.level(technique) is Support.PLUGIN:
            super().contribute(technique, builder)
            return
        builder.speculative = {
            "method": "suffix",
            "num_speculative_tokens": self.num_tokens(technique, builder),
            "suffix_decoding_max_tree_depth": int(config.max_depth),
            "suffix_decoding_max_spec_factor": float(config.max_spec_factor),
            "suffix_decoding_min_token_prob": float(config.min_token_prob),
        }
        builder.note(self.technique, "vLLM also matches against the responses it served before (a global cache "
                                     "of suffix_decoding_max_cached_requests requests); drafts can differ, outputs "
                                     "stay lossless. Install arctic-inference in the serving environment.")
        if not config.local:
            builder.note(self.technique, "vLLM always matches against the request's own prompt (local=True)")


@register_vllm_technique
class RestVllm(TokenProposerVllm):
    """``rest`` (retrieval-based speculative decoding) through economize's ``custom_class`` proposer."""

    technique = "rest"
    reason = "economize's custom_class proposer over the suffix-array datastore (the best path of the trie)"

    def contribute(self, technique: nn.Module, builder: PlanBuilder) -> None:
        if technique.config.tree:
            builder.note(self.technique, "vLLM 0.30 has no tree verification (no token tree, no tree attention "
                                         "backend), so it verifies the best path of the retrieved trie as one "
                                         "chain")
        super().contribute(technique, builder)


@register_vllm_technique
class LookaheadVllm(TokenProposerVllm):
    """``lookahead`` decoding's n-gram pool, filled from the committed tokens, through the ``custom_class``
    proposer (the Jacobi branch stays HF-only)."""

    technique = "lookahead"
    reason = ("the n-gram pool of the committed tokens (pool_from_prompt=True) through economize's custom_class "
              "proposer, as one chain. The Jacobi branch stays HF-only: it appends a window of W x (N - 1) "
              "guess tokens to every target forward under lookahead's 2-D mask, while vLLM schedules 1 + k "
              "causally masked tokens per "
              "decoding request")

    def check(self, technique: nn.Module, builder: PlanBuilder) -> None:
        if not technique.config.pool_from_prompt:
            raise UnsupportedBackendError("on vLLM, lookahead drafts from the n-gram pool of the committed tokens "
                                          "only (the Jacobi window, which fills the pool otherwise, adds guess "
                                          "tokens under its own 2-D mask to the target's forward, which vLLM's "
                                          "scheduler and attention do not run): set pool_from_prompt=True")
        super().check(technique, builder)

    def num_tokens(self, technique: nn.Module, builder: PlanBuilder) -> int:
        return min(builder.num_speculative_tokens, technique.config.ngram - 1)

    def contribute(self, technique: nn.Module, builder: PlanBuilder) -> None:
        builder.note(self.technique, "vLLM serves the n-gram pool of the committed tokens without the Jacobi "
                                     "window and drafts the newest n-gram of the last token as one chain")
        super().contribute(technique, builder)


# ---- separate draft models ----------------------------------------------------------------------------------------


@register_vllm_technique
class DraftModelVllm(_Drafter):
    """``draft_model``: a smaller LM of the same tokenizer, saved to ``out_dir/drafter``."""

    technique = "draft_model"
    support = Support.NATIVE
    reason = "method='draft_model' with the draft LM saved to the plan folder"
    method = "draft_model"

    def draft_config(self, technique: nn.Module) -> dict[str, Any]:
        """Entries added to the saved draft ``config.json``."""
        return {}

    def extra(self, technique: nn.Module) -> dict[str, Any]:
        """Further ``speculative_config`` fields."""
        return {}

    def check(self, technique: nn.Module, builder: PlanBuilder) -> None:
        self.draft_config(technique)  # raises on a missing setting before anything is written

    def contribute(self, technique: nn.Module, builder: PlanBuilder) -> None:
        if builder.out_dir is not None:
            builder.drafter_dir = builder.path(DRAFTER_DIR)
        builder.speculative = _model_spec(builder, self.method, builder.drafter_dir, **self.extra(technique))

    def export(self, technique: nn.Module, builder: PlanBuilder) -> None:
        target = builder.path(DRAFTER_DIR)
        draft: Any = technique.draft_model
        draft.save_pretrained(target)
        edits = self.draft_config(technique)
        if edits:
            file = target / "config.json"
            file.write_text(json.dumps(json.loads(file.read_text()) | edits, indent=2) + "\n")


@register_vllm_technique
class PlacedDraftModelVllm(DraftModelVllm):
    """``placed_draft_model``: the same draft LM (vLLM decides where it runs; ``draft_tensor_parallel_size``)."""

    technique = "placed_draft_model"
    reason = "method='draft_model'; vLLM places the draft LM on the target's GPUs (draft_tensor_parallel_size)"

    def contribute(self, technique: nn.Module, builder: PlanBuilder) -> None:
        super().contribute(technique, builder)
        builder.note(self.technique, f"the draft LM runs on the target's GPUs, not on {technique.config.device}")


@register_vllm_technique
class PardVllm(DraftModelVllm):
    """``pard``: parallel drafting with mask tokens (vLLM's ``parallel_drafting`` of a draft model)."""

    technique = "pard"
    reason = "method='draft_model', parallel_drafting=True, pard_token in the draft config"

    def draft_config(self, technique: nn.Module) -> dict[str, Any]:
        return {"pard_token": int(technique.mask_token_id)}

    def extra(self, technique: nn.Module) -> dict[str, Any]:
        return {"parallel_drafting": True}


@register_vllm_technique
class Gemma4AssistantVllm(_Drafter):
    """``gemma4_assistant``: Google's Gemma-4 assistant drafter, which vLLM runs as ``method="mtp"``."""

    technique = "gemma4_assistant"
    support = Support.NATIVE
    reason = "method='mtp' with the assistant model (vLLM's Gemma-4 assistant proposer)"

    def contribute(self, technique: nn.Module, builder: PlanBuilder) -> None:
        if builder.out_dir is not None:
            builder.drafter_dir = builder.path(DRAFTER_DIR)
        builder.speculative = _model_spec(builder, "mtp", builder.drafter_dir)

    def export(self, technique: nn.Module, builder: PlanBuilder) -> None:
        assistant: Any = technique.assistant
        if callable(getattr(assistant, "save_pretrained", None)):
            assistant.save_pretrained(builder.path(DRAFTER_DIR))
        else:  # not a transformers model: serve the configured checkpoint as is
            builder.drafter_dir = None
            builder.speculative = _model_spec(builder, "mtp", technique.config.model)


# ---- MTP modules in the target checkpoint ---------------------------------------------------------------------


#: The ``config.json`` key that counts MTP modules, per family (the others use ``num_nextn_predict_layers``).
MTP_COUNT_KEYS: dict[str, str] = {"qwen3_5": "mtp_num_hidden_layers"}


@register_vllm_technique
class VendorMTPVllm(_Drafter):
    """``vendor_mtp``: MTP modules in the vendor layout inside the target folder, which vLLM drafts with."""

    technique = "vendor_mtp"
    support = Support.NATIVE
    reason = "method='mtp' with the modules written under their vendor names into the target checkpoint"

    def check(self, technique: nn.Module, builder: PlanBuilder) -> None:
        if technique.config.head != "shared":
            raise UnsupportedBackendError("vLLM's MTP modules use the target's LM head; vendor_mtp with head='own' "
                                          "has no vLLM equivalent (train with head='shared')")
        if _ct_quantized(builder):
            raise UnsupportedBackendError("vendor_mtp next to a compressed-tensors base is not exported yet: the "
                                          "quantization config would have to exclude the MTP layers")

    def count_key(self, technique: Any, builder: PlanBuilder) -> str:
        """The ``config.json`` key that counts the MTP modules (the one the base config has, else the family's)."""
        config = builder.base.config
        for key in ("num_nextn_predict_layers", "mtp_num_hidden_layers"):
            if getattr(config, key, None) is not None:
                return key
        return MTP_COUNT_KEYS.get(technique.family, "num_nextn_predict_layers")

    def contribute(self, technique: nn.Module, builder: PlanBuilder) -> None:
        builder.rewrite_base = True
        builder.config_edits[self.count_key(technique, builder)] = int(technique.config.num_modules)
        builder.speculative = _model_spec(builder, "mtp", None)
        builder.note(self.technique, "vendor MTP export is checked on tiny models only, not on a real checkpoint")

    def export(self, technique: nn.Module, builder: PlanBuilder) -> None:
        from safetensors.torch import save_file

        from ....techniques.vendor_mtp.formats import state_to_vendor

        state = {key: value for key, value in technique.state_dict().items() if not key.startswith("lm_head.")}
        tensors = state_to_vendor(state, technique.family, int(technique.adapter.num_layers))
        target = builder.path("model")
        file = "model-mtp.safetensors"
        save_file({key: value.detach().to("cpu").contiguous() for key, value in tensors.items()}, str(target / file),
                  metadata={"format": "pt"})
        index = target / "model.safetensors.index.json"
        if index.is_file():  # vLLM reads only the files the index names
            data = json.loads(index.read_text())
            data.setdefault("weight_map", {}).update(dict.fromkeys(tensors, file))
            index.write_text(json.dumps(data, indent=2) + "\n")


# ---- trained drafters with an exporter --------------------------------------------------------------------------


class _Exported(_Drafter):
    """A drafter written by its registered :class:`~economize.backends.exporters.DrafterExporter`."""

    exporter: str = ""

    def method(self, technique: nn.Module) -> str | None:
        """The speculative method (``None``: the one of the exported folder's architecture)."""
        return None

    def contribute(self, technique: nn.Module, builder: PlanBuilder) -> None:
        if builder.out_dir is not None:
            builder.drafter_dir = builder.path(DRAFTER_DIR)

    def write(self, technique: nn.Module, target: Path, builder: PlanBuilder) -> None:
        from ...exporters import EXPORTERS, exporter_key

        exporter = EXPORTERS.get(exporter_key("vllm", self.exporter))()
        exporter.export(technique, target, num_speculative_tokens=builder.num_speculative_tokens)

    def export(self, technique: nn.Module, builder: PlanBuilder) -> None:
        target = builder.path(DRAFTER_DIR)
        self.write(technique, target, builder)
        extra: dict[str, Any] = {}
        method = self.method(technique)
        if method is not None:
            extra["method"] = method
        builder.speculative = speculative_config(target, builder.num_speculative_tokens, sampling=builder.sampling,
                                                 **extra)


@register_vllm_technique
class Eagle3Vllm(_Exported):
    """``eagle3``: vLLM's EAGLE-3 draft model."""

    technique = "eagle3"
    support = Support.NATIVE
    reason = "method='eagle3' (flat Llama draft config, aux layer ids, d2t/t2d for a reduced vocabulary)"
    exporter = "eagle3"

    def method(self, technique: nn.Module) -> str | None:
        return "eagle3"

    def check(self, technique: nn.Module, builder: PlanBuilder) -> None:
        _check_eagle3(technique)

    def contribute(self, technique: nn.Module, builder: PlanBuilder) -> None:
        super().contribute(technique, builder)
        if technique.lm_head is None:
            builder.note(self.technique, "head='shared': vLLM applies the target's LM head without its logit "
                                         "softcapping, which changes the draft distribution only")


def _check_eagle3(technique: Any) -> None:
    if getattr(technique.config, "zero_source_first", False):
        raise UnsupportedBackendError("vLLM's EAGLE-3 reads the target's state at the first draft step; an eagle3 "
                                      "drafter trained with zero_source_first=True has no vLLM equivalent")


@register_vllm_technique
class MedusaVllm(_Exported):
    """``medusa`` (``head="own"``): vLLM's Medusa heads."""

    technique = "medusa"
    support = Support.NATIVE
    reason = "method='medusa' for head='own' heads (vLLM drafts the top-1 chain, no static tree)"
    exporter = "medusa"

    def check(self, technique: nn.Module, builder: PlanBuilder) -> None:
        heads = int(technique.config.num_heads)
        if builder.num_speculative_tokens > heads:
            raise UnsupportedBackendError(f"Medusa with {heads} heads drafts at most {heads} tokens; asked for "
                                          f"num_speculative_tokens={builder.num_speculative_tokens}")
        if technique.config.head != "own":
            raise UnsupportedBackendError("vLLM's Medusa projects each head with its own matrix after the final "
                                          "norm; head='shared' Medusa heads have no vLLM equivalent")

    def contribute(self, technique: nn.Module, builder: PlanBuilder) -> None:
        super().contribute(technique, builder)
        if technique.config.tree is not None:
            builder.note(self.technique, "vLLM verifies the top-1 chain of the heads, not the static tree")

    def export(self, technique: nn.Module, builder: PlanBuilder) -> None:
        target = builder.path(DRAFTER_DIR)
        self.write(technique, target, builder)
        builder.speculative = {"method": "medusa", "model": str(target),
                               "num_speculative_tokens": builder.num_speculative_tokens}


@register_vllm_technique
class JumpVllm(_Exported):
    """``jump``: economize's jump drafter, run by the vLLM plugin (``EconomizeJumpEagle3``)."""

    technique = "jump"
    support = Support.PLUGIN
    reason = ("economize's vLLM plugin (EconomizeJumpEagle3 through method='eagle3'): predictor, jump-path layers "
              "shared with the target, readout")
    exporter = "jump"

    def check(self, technique: nn.Module, builder: PlanBuilder) -> None:
        from ....techniques.jump.export import JumpDrafterSpec

        try:
            JumpDrafterSpec.of(technique)
        except ConfigError as error:
            raise UnsupportedBackendError(f"the jump drafter cannot be served on vLLM: {error}") from error
        adaptation = technique.config.adaptation
        if adaptation == "lora" and _ct_quantized(builder):
            raise UnsupportedBackendError(
                "a jump drafter with adaptation='lora' merges its updates into copies of the jump-path layers, "
                "which vLLM holds as compressed-tensors codes: use backend='fake', or adaptation='frozen' or 'full' "
                "(full copies are served dense next to the quantized target)"
            )

    def needs_dense_linears(self, technique: nn.Module, builder: PlanBuilder) -> str | None:
        if technique.config.adaptation == "lora":
            return "a jump drafter with adaptation='lora' merges its updates into copies of the target's dense weights"
        return None


# ---- wrappers and draft options ---------------------------------------------------------------------------------


def eagle3_restricted(inner: Any, ids: torch.Tensor, shared_head: nn.Module | None) -> dict[str, torch.Tensor]:
    """The EAGLE-3 tensors of ``inner`` restricted to the draft vocabulary ``ids`` (vLLM's ``d2t`` layout).

    Args:
        inner: The ``eagle3`` technique (full vocabulary).
        ids: Draft token ids (any order; written sorted).
        shared_head: The target's LM head when ``inner`` shares it, else ``None``.

    Returns:
        The exported tensors with ``lm_head.weight`` cut to the rows of ``ids``, ``d2t`` (``ids[i] - i``) and
        ``t2d`` (a mask over the full vocabulary).
    """
    from ..eagle3.export import Eagle3VllmExporter

    tensors = Eagle3VllmExporter().tensors(inner)
    ids = torch.sort(ids.detach().to("cpu", torch.long)).values
    head = inner.lm_head if inner.lm_head is not None else shared_head
    if head is None:
        raise ConfigError("an eagle3 drafter with a shared head needs the target's LM head")
    rows = head.weight.detach().to("cpu")[ids]
    tensors["lm_head.weight"] = rows.to(tensors["fc.weight"].dtype).contiguous()
    tensors["d2t"] = (ids - torch.arange(ids.numel())).contiguous()
    mask = torch.zeros(int(inner.vocab_size), dtype=torch.bool)
    mask[ids] = True
    tensors["t2d"] = mask
    return tensors


@register_vllm_technique
class FRSpecVllm(_Drafter):
    """``frspec`` around ``eagle3``: the drafter's head restricted to the frequent ids (vLLM's ``d2t``)."""

    technique = "frspec"
    support = Support.NATIVE
    reason = "around eagle3: method='eagle3' with draft_vocab_size, d2t/t2d and the LM-head rows of the draft ids"

    def check(self, technique: nn.Module, builder: PlanBuilder) -> None:
        inner: Any = technique.inner
        if getattr(inner, "name", None) != "eagle3":
            raise UnsupportedBackendError(f"vLLM restricts the draft vocabulary of EAGLE-3 drafters only; frspec "
                                          f"around {getattr(inner, 'name', type(inner).__name__)!r} has no vLLM path")
        _check_eagle3(inner)
        if int(inner.draft_vocab_size) != int(inner.vocab_size):
            raise UnsupportedBackendError("the eagle3 drafter inside frspec already has a reduced vocabulary")
        if not technique.has_vocab:
            raise ConfigError("frspec has no draft vocabulary yet: set it (set_vocab or token_ids) before export")

    def contribute(self, technique: nn.Module, builder: PlanBuilder) -> None:
        if builder.out_dir is not None:
            builder.drafter_dir = builder.path(DRAFTER_DIR)
        builder.note(self.technique, "vLLM's restricted head sets the other logits to -inf like economize's, and "
                                     "drops the target's logit softcapping")

    def export(self, technique: nn.Module, builder: PlanBuilder) -> None:
        from safetensors.torch import save_file

        from ..eagle3.export import Eagle3VllmExporter

        inner: Any = technique.inner
        ids = technique.draft_ids
        target = builder.path(DRAFTER_DIR)
        target.mkdir(parents=True, exist_ok=True)
        exporter = Eagle3VllmExporter()
        config = exporter.config(inner, num_speculative_tokens=builder.num_speculative_tokens)
        config["draft_vocab_size"] = int(ids.numel())
        tensors = eagle3_restricted(inner, ids, inner.adapter.lm_head if inner.lm_head is None else None)
        (target / exporter.config_file).write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
        save_file(tensors, str(target / exporter.weights_file), metadata=exporter.metadata())
        builder.speculative = speculative_config(target, builder.num_speculative_tokens, sampling=builder.sampling,
                                                 method="eagle3")


@register_vllm_technique
class DraftQuantVllm(VllmTechnique):
    """``draft_quant``: the drafter's weights are rounded in place, so the drafter export writes them as they are."""

    technique = "draft_quant"
    role = Role.DRAFT_OPTION
    support = Support.DENSIFIED
    reason = "the drafter is exported with its rounded (dequantized) weights in the dense dtype"

    def check(self, technique: nn.Module, builder: PlanBuilder) -> None:
        if technique.config.backend != "fake":
            raise UnsupportedBackendError(f"draft_quant backend {technique.config.backend!r} has no vLLM export")

    def contribute(self, technique: nn.Module, builder: PlanBuilder) -> None:
        builder.note(self.technique, "the drafter runs the rounded weights in the dense dtype: same drafts, no "
                                     "memory or speed saving")
