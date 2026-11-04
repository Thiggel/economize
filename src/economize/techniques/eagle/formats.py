"""Reader for the official EAGLE-1/2 draft heads.

``"eagle"``: a directory with ``config.json`` and ``pytorch_model.bin`` or ``model.safetensors``, as published by
SafeAILab (``yuhuili/EAGLE-Vicuna-7B-v1.3``, ``yuhuili/EAGLE-llama2-chat-7B``, ``yuhuili/EAGLE-LLaMA3-Instruct-8B``,
``yuhuili/EAGLE-Qwen2-7B-Instruct``). The config is the base model's with ``num_hidden_layers = 1`` and has no field
that names EAGLE, so the weights decide: ``fc.weight [d, 2d]`` and one decoder layer ``layers.0.*`` without input
norm, and neither an own LM head nor a final norm. The copy of the embedding (``embed_tokens.weight``) and stale
rotary buffers are dropped: economize uses the model's embedding and computes the rotary table (``rope_theta`` of the
config, ``10000`` when absent, and its ``rope_scaling``). ``fc`` has a bias unless the config sets ``"bias": false``
(the weights decide), q/k/v have one when the weights hold it (``qkv_bias`` of the Qwen2 heads). The heads serve
EAGLE-1 (a chain) and EAGLE-2 (``DecodeConfig(draft_policy="dynamic_tree")``) alike. Detected automatically.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

from ...core.serialization import FORMATS, Checkpoint, CheckpointFormat
from ...errors import CheckpointFormatError
from .._draft_layer.formats import JSONObject, rope_fields
from .._drafting import detect_config_dir, first_file, load_state_file, require_weights, state_shapes
from ..eagle3.formats import base_model_of
from .config import EagleConfig

__all__ = ["EagleFormat", "config_from_eagle", "state_from_eagle"]

_WEIGHT_FILES = ("pytorch_model.bin", "model.safetensors")
#: Config fields of the other one-layer draft heads (EAGLE-3, speculators, DFlash, Medusa).
_OTHER_HEADS = ("draft_vocab_size", "speculators_model_type", "dflash_config", "medusa_num_heads", "eagle_config")
#: Parameters of the decoder layer, as named under ``layers.0``.
_LAYER = ("self_attn.q_proj.", "self_attn.k_proj.", "self_attn.v_proj.", "self_attn.o_proj.", "mlp.",
          "post_attention_layernorm.")


def _is_eagle_config(config: JSONObject) -> bool:
    return (isinstance(config, dict) and "hidden_size" in config and config.get("num_hidden_layers", 1) == 1
            and not any(key in config for key in _OTHER_HEADS)
            and not any("eagle3" in str(name).lower() for name in config.get("architectures") or []))


def _is_eagle_state(shapes: dict[str, tuple[int, ...]]) -> bool:
    fc = shapes.get("fc.weight")
    return (fc is not None and len(fc) == 2 and fc[1] == 2 * fc[0] and "layers.0.self_attn.q_proj.weight" in shapes
            and not any(key.startswith(("midlayer.", "lm_head.", "norm.")) or key in ("d2t", "t2d")
                        for key in shapes))


def state_from_eagle(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Rename an official EAGLE-1/2 state dict to :class:`~economize.techniques.eagle.EagleTechnique` parameters
    (``fc.*`` and ``layers.0.*`` to ``layer.*``).

    Raises:
        CheckpointFormatError: For variants economize does not implement (several draft layers, an input norm, a
            nonzero ``o_proj`` bias) and for unknown parameters.
    """
    out: dict[str, torch.Tensor] = {}
    for key, value in state.items():
        if key.startswith("embed_tokens.") or "rotary_emb." in key:
            continue
        if key.startswith("fc."):
            out[f"layer.{key}"] = value
        elif key.startswith("layers.0."):
            leaf = key.split(".", 2)[2]
            if leaf == "self_attn.o_proj.bias":
                if bool(value.any()):
                    raise CheckpointFormatError("EAGLE draft layers with a nonzero o_proj bias are not supported")
                continue
            if not leaf.startswith(_LAYER):
                raise CheckpointFormatError(f"EAGLE draft layers with {leaf!r} are not supported")
            out[f"layer.{leaf}"] = value
        elif key.startswith("layers."):
            raise CheckpointFormatError(f"EAGLE heads with several draft layers are not supported ({key!r})")
        else:
            raise CheckpointFormatError(f"unexpected parameter {key!r} in an EAGLE checkpoint")
    for key in ("layer.fc.weight", "layer.self_attn.q_proj.weight"):
        if key not in out:
            raise CheckpointFormatError(f"an EAGLE checkpoint needs {key.removeprefix('layer.')!r}")
    return out


def config_from_eagle(config: JSONObject, state: dict[str, torch.Tensor]) -> EagleConfig:
    """The :class:`EagleConfig` of an official EAGLE-1/2 ``config.json`` and its renamed state (see
    :func:`state_from_eagle`), which decides the biases."""
    hidden = int(config["hidden_size"])
    heads = int(config["num_attention_heads"])
    return EagleConfig(
        fc_bias="layer.fc.bias" in state,
        intermediate_size=int(config["intermediate_size"]),
        num_heads=heads,
        num_kv_heads=int(config.get("num_key_value_heads") or heads),
        head_dim=int(config.get("head_dim") or hidden // heads),
        attention_bias="layer.self_attn.q_proj.bias" in state,
        **rope_fields(config),
        rms_norm_eps=float(config.get("rms_norm_eps", 1e-6)),
    )


class EagleFormat(CheckpointFormat):
    """An official EAGLE-1/2 head directory (``config.json`` + ``pytorch_model.bin`` or ``model.safetensors``)."""

    def detect(self, path: Path) -> bool:
        if not detect_config_dir(path, _WEIGHT_FILES, _is_eagle_config):
            return False
        weights = first_file(path, _WEIGHT_FILES)
        shapes = state_shapes(weights) if weights is not None else None
        return shapes is not None and _is_eagle_state(shapes)

    def read(self, path: Path, map_location: str | torch.device = "cpu") -> Checkpoint:
        weights = require_weights(path, _WEIGHT_FILES, "EAGLE head")
        raw = json.loads((path / "config.json").read_text())
        if not _is_eagle_config(raw):
            raise CheckpointFormatError(f"{path / 'config.json'} does not describe an EAGLE-1/2 head")
        state = state_from_eagle(load_state_file(weights, map_location))
        config = config_from_eagle(raw, state)
        return Checkpoint([config], {config.technique: state}, None, base_model_of(raw), {"source": str(path)})


FORMATS.register("eagle", EagleFormat())
