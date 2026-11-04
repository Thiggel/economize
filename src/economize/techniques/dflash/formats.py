"""Readers for the official DFlash drafters.

* ``"dflash"``: a directory of z-lab/dflash (``z-lab/gemma-4-31B-it-DFlash``, ``z-lab/Qwen3-8B-DFlash-b16``) with
  ``config.json`` (a Qwen3-style config with a ``dflash_config`` of ``mask_token_id`` and ``target_layer_ids``, and
  ``block_size``) and ``model.safetensors``. The reference reads target layer id ``i`` from
  ``hidden_states[i + 1]``, the output of block ``i + 1``, so the ids load shifted by one. Sliding layers attend
  causally inside the block, full layers bidirectionally (``is_causal: false`` makes the sliding layers
  bidirectional as well; ``is_causal: true``, causal full layers, is refused). ``input_embedding_scale`` loads as
  ``embed_scale`` and ``final_logit_softcapping`` as ``logit_softcap``. Detected automatically.
* ``"speculators_dflash"``: the vLLM ``speculators`` layout (``RedHatAI/gemma-4-31B-it-speculator.dflash``:
  ``DFlashDraftModel`` with ``speculators_model_type`` ``"dflash"``, the layers' fields under
  ``transformer_layer_config`` and the base model under ``speculators_config.verifier``). Its
  ``aux_hidden_state_layer_ids`` are block outputs as economize counts them and load unchanged. Sliding layers attend
  causally inside the block unless ``sliding_window_non_causal``; there is no logit soft cap. Its config also sets
  ``draft_vocab_size``, which the ``"eagle3"`` reader matches as well, so pass ``format="speculators_dflash"``.

Both layouts store ``fc.weight [d, n d]``, ``hidden_norm.weight``, ``layers.{i}.*`` (Qwen3 decoder layers with q/k
norms) and ``norm.weight``; a copy of the embedding is dropped (economize uses the model's). Variants economize does
not implement raise :class:`~economize.errors.CheckpointFormatError`: a restricted draft vocabulary or an own LM
head, a draft width other than the model's, an output multiplier other than 1, and unknown parameters.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

from ...core.serialization import FORMATS, Checkpoint, CheckpointFormat
from ...errors import CheckpointFormatError
from .._drafting import JSONObject, detect_config_dir, load_state_file, require_weights
from ..eagle3.formats import _rope, base_model_of, flat_config
from .config import DFLASH_LAYER_TYPES, DFlashConfig

__all__ = ["DFlashFormat", "SpeculatorsDFlashFormat", "config_from_dflash", "config_from_speculators",
           "draft_layer_count", "state_from_dflash"]

_WEIGHT_FILES = ("model.safetensors", "pytorch_model.bin")
_LAYER_LEAVES = frozenset({
    "input_layernorm.weight", "post_attention_layernorm.weight", "mlp.gate_proj.weight", "mlp.up_proj.weight",
    "mlp.down_proj.weight", "self_attn.q_proj.weight", "self_attn.k_proj.weight", "self_attn.v_proj.weight",
    "self_attn.o_proj.weight", "self_attn.q_proj.bias", "self_attn.k_proj.bias", "self_attn.v_proj.bias",
    "self_attn.o_proj.bias", "self_attn.q_norm.weight", "self_attn.k_norm.weight",
})


def _layer_fields(config: JSONObject, layers: int) -> dict[str, Any]:
    """The draft layers' fields of :class:`DFlashConfig` from a flat ``config.json``."""
    hidden = int(config["hidden_size"])
    heads = int(config["num_attention_heads"])
    types = config.get("layer_types")
    if types is not None:
        types = tuple(str(kind) for kind in types)[:layers]
        unknown = sorted(set(types) - set(DFLASH_LAYER_TYPES))
        if unknown:
            raise CheckpointFormatError(f"DFlash layer types {unknown} are not supported")
    window = config.get("sliding_window")
    return {
        "num_layers": layers,
        "intermediate_size": int(config["intermediate_size"]),
        "num_heads": heads,
        "num_kv_heads": int(config.get("num_key_value_heads") or heads),
        "head_dim": int(config.get("head_dim") or hidden // heads),
        "layer_types": types,
        "sliding_window": int(window) if window else 2048,
        "attention_bias": bool(config.get("attention_bias", False)),
        "rms_norm_eps": float(config.get("rms_norm_eps", 1e-6)),
        **_rope(config),
    }


def _check_width(config: JSONObject) -> None:
    target = config.get("target_hidden_size")
    if target is not None and int(target) != int(config["hidden_size"]):
        raise CheckpointFormatError(
            f"a draft width {config['hidden_size']} other than the model's ({target}) is not supported"
        )


def config_from_dflash(config: JSONObject, layers: int | None = None) -> DFlashConfig:
    """The :class:`DFlashConfig` of a z-lab/dflash ``config.json``.

    Args:
        config: The parsed ``config.json``.
        layers: The number of draft layers the weights hold (``None``: ``num_hidden_layers``).
    """
    dflash = config.get("dflash_config")
    if not isinstance(dflash, dict):
        raise CheckpointFormatError("a z-lab DFlash config.json needs a dflash_config")
    _check_width(config)
    if float(config.get("output_multiplier", 1.0)) != 1.0:
        raise CheckpointFormatError("DFlash drafters with an output_multiplier other than 1 are not supported")
    if config.get("is_causal") is True:
        raise CheckpointFormatError("DFlash drafters with causal full-attention layers (is_causal) are not supported")
    count = int(config["num_hidden_layers"]) if layers is None else layers
    ids = dflash.get("target_layer_ids")
    softcap = config.get("final_logit_softcapping")
    return DFlashConfig(
        target_layers=tuple(int(i) + 1 for i in ids) if ids else None,
        block_size=int(config.get("block_size", dflash.get("block_size", 16))),
        mask_token_id=int(dflash["mask_token_id"]),
        sliding_causal=config.get("is_causal") is not False,
        embed_scale=float(config.get("input_embedding_scale", 1.0)),
        logit_softcap=float(softcap) if softcap else 0.0,
        **_layer_fields(config, count),
    )


def config_from_speculators(config: JSONObject, layers: int | None = None) -> DFlashConfig:
    """The :class:`DFlashConfig` of a ``speculators`` DFlash ``config.json`` (``DFlashDraftModel``)."""
    flat = flat_config(config)
    _check_width(flat)
    vocab, draft_vocab = flat.get("vocab_size"), flat.get("draft_vocab_size")
    if draft_vocab is not None and vocab is not None and int(draft_vocab) < int(vocab):
        raise CheckpointFormatError("DFlash drafters with a restricted draft vocabulary are not supported")
    count = int(flat["num_hidden_layers"]) if layers is None else layers
    ids = flat.get("aux_hidden_state_layer_ids")
    return DFlashConfig(
        target_layers=tuple(int(i) for i in ids) if ids else None,
        block_size=int(flat.get("block_size", 8)),
        mask_token_id=int(flat["mask_token_id"]),
        sliding_causal=not bool(flat.get("sliding_window_non_causal", False)),
        logit_softcap=0.0,
        **_layer_fields(flat, count),
    )


def draft_layer_count(state: dict[str, torch.Tensor]) -> int:
    """How many draft layers an official DFlash state dict holds."""
    indices = {int(key.split(".")[1]) for key in state if key.startswith("layers.") and key.split(".")[1].isdigit()}
    return 1 + max(indices) if indices else 0


def state_from_dflash(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """An official DFlash state dict as :class:`~economize.techniques.dflash.DFlashTechnique` parameters (the same
    names, without the embedding copy)."""
    out: dict[str, torch.Tensor] = {}
    for key, value in state.items():
        if key.startswith("embed_tokens."):
            continue
        if key.startswith("lm_head."):
            raise CheckpointFormatError("DFlash drafters with an own LM head are not supported")
        if key in ("fc.weight", "hidden_norm.weight", "norm.weight"):
            out[key] = value
            continue
        parts = key.split(".", 2)
        if len(parts) == 3 and parts[0] == "layers" and parts[1].isdigit() and parts[2] in _LAYER_LEAVES:
            out[key] = value
            continue
        raise CheckpointFormatError(f"unexpected parameter {key!r} in a DFlash checkpoint")
    for key in ("fc.weight", "hidden_norm.weight", "norm.weight"):
        if key not in out:
            raise CheckpointFormatError(f"a DFlash checkpoint needs {key!r}")
    return out


class DFlashFormat(CheckpointFormat):
    """A z-lab/dflash drafter directory (``config.json`` with ``dflash_config`` + weights)."""

    @staticmethod
    def _is_dflash(config: JSONObject) -> bool:
        return isinstance(config.get("dflash_config"), dict) and "speculators_model_type" not in config

    def detect(self, path: Path) -> bool:
        return detect_config_dir(path, _WEIGHT_FILES, self._is_dflash)

    def read(self, path: Path, map_location: str | torch.device = "cpu") -> Checkpoint:
        weights = require_weights(path, _WEIGHT_FILES, "DFlash drafter")
        raw = json.loads((path / "config.json").read_text())
        if not self._is_dflash(raw):
            raise CheckpointFormatError(f"{path / 'config.json'} does not describe a z-lab DFlash drafter")
        state = load_state_file(weights, map_location)
        config = config_from_dflash(raw, draft_layer_count(state))
        return Checkpoint([config], {"dflash": state_from_dflash(state)}, None, base_model_of(raw),
                          {"source": str(path)})


class SpeculatorsDFlashFormat(CheckpointFormat):
    """A ``speculators`` DFlash drafter directory (``speculators_model_type`` ``"dflash"``)."""

    @staticmethod
    def _is_dflash(config: JSONObject) -> bool:
        return config.get("speculators_model_type") == "dflash"

    def detect(self, path: Path) -> bool:
        return detect_config_dir(path, _WEIGHT_FILES, self._is_dflash)

    def read(self, path: Path, map_location: str | torch.device = "cpu") -> Checkpoint:
        weights = require_weights(path, _WEIGHT_FILES, "DFlash drafter")
        raw = json.loads((path / "config.json").read_text())
        if not self._is_dflash(raw):
            raise CheckpointFormatError(f"{path / 'config.json'} does not describe a speculators DFlash drafter")
        state = load_state_file(weights, map_location)
        config = config_from_speculators(raw, draft_layer_count(state))
        return Checkpoint([config], {"dflash": state_from_dflash(state)}, None, base_model_of(raw),
                          {"source": str(path)})


FORMATS.register("dflash", DFlashFormat())
FORMATS.register("speculators_dflash", SpeculatorsDFlashFormat())
