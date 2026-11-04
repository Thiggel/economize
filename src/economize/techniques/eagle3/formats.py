"""Reader for the official EAGLE-3 draft heads (EAGLE-3 heads saved as ``jump.pt`` are read by
:mod:`economize.compat.tskip`).

* ``"eagle3"``: a directory with ``config.json`` (``draft_vocab_size`` or an ``*Eagle3*`` architecture) and
  ``model.safetensors`` or ``pytorch_model.bin``, as published by SafeAILab (``yuhuili/EAGLE3-LLaMA3.1-Instruct-8B``),
  SpecForge and AngelSlim (``Tengyunw/qwen3_8b_eagle3``). The state holds ``fc.weight [d, 3d]``, the draft layer
  as ``midlayer.*`` (or ``layers.0.*``), ``norm.weight``, ``lm_head.weight [K, d]``, ``d2t [K]`` (draft token ``i``
  is model token ``i + d2t[i]``) and ``t2d [V]``, sometimes a copy of the embedding (dropped: economize uses the
  model's). It loads as ``head="own"`` with the draft layer's own rotary table (``rope_theta`` of the config,
  ``10000`` when absent, as in the reference implementation, and its ``rope_scaling``, e.g. ``llama3``). The
  auxiliary layers come from ``eagle_config.eagle_aux_hidden_state_layer_ids`` when present; vLLM reads these ids as
  block outputs, as economize does (SGLang adds one). The vLLM ``speculators`` layout
  (``RedHatAI/Llama-3.1-8B-Instruct-speculator.eagle3``: ``Eagle3Speculator`` with the layer's fields under
  ``transformer_layer_config`` and the base model under ``speculators_config.verifier``) reads the same way. The number
  of draft layers comes from the weights: ``num_hidden_layers`` is not reliable (``yuhuili/EAGLE3-Vicuna1.3-13B`` copies
  the base model's 40, and the reference implementation builds one layer regardless). Detected automatically. EAGLE-1/2
  heads (``yuhuili/EAGLE-Vicuna-7B-v1.3``: no ``draft_vocab_size``, ``fc`` on ``[e, h]``) are the ``"eagle"`` format of
  :mod:`economize.techniques.eagle.formats`.
* ``"tskip_eagle3"`` (:class:`economize.compat.tskip.TskipEagle3Format`): a ``jump.pt`` whose ``cfg["pred"]`` is
  ``"eagle3"`` (EAGLE-3 head on the model's shared, frozen final norm and LM head).
"""

from __future__ import annotations

import json
from pathlib import Path
from types import ModuleType

import torch

from ..._deprecation import moved_getattr
from ...core.serialization import FORMATS, Checkpoint, CheckpointFormat
from ...errors import CheckpointFormatError
from .._draft_layer.formats import rope_fields as _rope  # also read by the DFlash format
from .._drafting import JSONObject, detect_config_dir, load_state_file, require_weights
from .config import Eagle3Config

__all__ = ["Eagle3Format", "base_model_of", "config_from_eagle3", "draft_layer_count", "flat_config",
           "state_from_eagle3"]

_WEIGHT_FILES = ("model.safetensors", "pytorch_model.bin")
#: Options of vLLM's ``llama_eagle3`` / SpecForge that change the computation and that economize does not implement.
_UNSUPPORTED_FLAGS = ("norm_before_fc", "fc_norm", "norm_output", "input_norm")


def flat_config(config: JSONObject) -> JSONObject:
    """An EAGLE-3 ``config.json`` with the fields of a ``speculators`` layout's ``transformer_layer_config`` at the
    top (the top level wins where both are set); other layouts are returned unchanged."""
    layer = config.get("transformer_layer_config")
    if not isinstance(layer, dict):
        return config
    top = {key: value for key, value in config.items() if key != "transformer_layer_config" and value is not None}
    return {**layer, **top}


def base_model_of(config: JSONObject) -> str | None:
    """The base model an EAGLE-3 ``config.json`` names, if any."""
    verifier = (config.get("speculators_config") or {}).get("verifier") or {}
    return (config.get("base_model_name_or_path") or (config.get("eagle_config") or {}).get("base_model_name_or_path")
            or verifier.get("name_or_path"))


def config_from_eagle3(config: JSONObject, draft_layers: int | None = None) -> Eagle3Config:
    """The :class:`Eagle3Config` of an official EAGLE-3 ``config.json`` (any layout :func:`flat_config` reads).

    Args:
        config: The parsed ``config.json``.
        draft_layers: The number of draft layers the weights hold. ``None`` trusts ``num_hidden_layers``, which
            published heads do not always set to the draft's (see the module docstring).

    Raises:
        CheckpointFormatError: For variants economize does not implement (several draft layers, normalization before
            or after ``fc``, a draft width other than the model's, no auxiliary states).
    """
    config = flat_config(config)
    eagle = dict(config.get("eagle_config") or {})
    layers = int(config.get("num_hidden_layers", 1)) if draft_layers is None else draft_layers
    if layers != 1:
        raise CheckpointFormatError(f"EAGLE-3 heads with {layers} draft layers are not supported (only 1)")
    for flag in _UNSUPPORTED_FLAGS:
        if config.get(flag) or eagle.get(flag):
            raise CheckpointFormatError(f"the EAGLE-3 option {flag!r} is not supported")
    if eagle.get("use_aux_hidden_state", config.get("use_aux_hidden_state", True)) is False:
        raise CheckpointFormatError("EAGLE-3 heads without auxiliary hidden states (EAGLE-1 style) are not supported")
    hidden = int(config["hidden_size"])
    target = config.get("target_hidden_size")
    if target is not None and int(target) != hidden:
        raise CheckpointFormatError(f"a draft width {hidden} other than the model's ({target}) is not supported")
    heads = int(config["num_attention_heads"])
    vocab = config.get("vocab_size")
    draft_vocab = config.get("draft_vocab_size")
    aux = eagle.get("eagle_aux_hidden_state_layer_ids") or config.get("eagle_aux_hidden_state_layer_ids")
    return Eagle3Config(
        aux_layers=tuple(int(i) for i in aux) if aux else None,
        head="own",
        draft_vocab=int(draft_vocab) if draft_vocab is not None and vocab is not None and draft_vocab < vocab
        else None,
        intermediate_size=int(config["intermediate_size"]),
        num_heads=heads,
        num_kv_heads=int(config.get("num_key_value_heads") or heads),
        head_dim=int(config.get("head_dim") or hidden // heads),
        attention_bias=bool(config.get("attention_bias", False)),
        **_rope(config),
        rms_norm_eps=float(config.get("rms_norm_eps", 1e-6)),
        norm_before_residual=bool(config.get("norm_before_residual", eagle.get("norm_before_residual", False))),
    )


def draft_layer_count(state: dict[str, torch.Tensor]) -> int:
    """How many draft layers an official EAGLE-3 state dict holds (``midlayer.*`` or ``layers.{i}.*``)."""
    if any(key.startswith("midlayer.") for key in state):
        return 1
    indices = {int(key.split(".")[1]) for key in state if key.startswith("layers.") and key.split(".")[1].isdigit()}
    return 1 + max(indices) if indices else 0


def state_from_eagle3(state: dict[str, torch.Tensor], vocab_size: int | None = None) -> dict[str, torch.Tensor]:
    """Rename an official EAGLE-3 state dict to :class:`~economize.techniques.eagle3.Eagle3Technique` parameters.

    ``t2d`` is rebuilt from ``d2t`` (``vocab_size`` entries, by default the length of the stored ``t2d``); without
    a ``d2t`` the draft vocabulary is the first ``K`` model tokens.
    """
    out: dict[str, torch.Tensor] = {}
    for key, value in state.items():
        if key.startswith(("midlayer.", "layers.0.")):
            leaf = key.split(".", 1)[1] if key.startswith("midlayer.") else key.split(".", 2)[2]
            if leaf.startswith(("self_attn.q_norm", "self_attn.k_norm")):
                raise CheckpointFormatError(f"EAGLE-3 draft layers with q/k norms are not supported ({key!r})")
            if leaf == "self_attn.o_proj.bias":
                if bool(value.any()):  # vLLM's layout has one whenever attention_bias is set; exports write zeros
                    raise CheckpointFormatError("EAGLE-3 draft layers with a nonzero o_proj bias are not supported")
                continue
            out[f"layer.{leaf}"] = value
        elif key in ("fc.weight", "norm.weight", "lm_head.weight"):
            out[key] = value
        elif key in ("d2t", "t2d") or key.startswith("embed_tokens."):
            continue
        elif key.startswith("layers."):
            raise CheckpointFormatError(f"EAGLE-3 heads with several draft layers are not supported ({key!r})")
        else:
            raise CheckpointFormatError(f"unexpected parameter {key!r} in an EAGLE-3 checkpoint")
    for key in ("fc.weight", "norm.weight", "lm_head.weight"):
        if key not in out:
            raise CheckpointFormatError(f"an EAGLE-3 checkpoint needs {key!r}")
    size = out["lm_head.weight"].shape[0]
    d2t = state.get("d2t")
    d2t = torch.zeros(size, dtype=torch.long) if d2t is None else d2t.to(torch.long)
    if d2t.shape != (size,):
        raise CheckpointFormatError(f"d2t has shape {tuple(d2t.shape)}, the draft head {size} rows")
    if vocab_size is None:
        vocab_size = int(state["t2d"].shape[0]) if "t2d" in state else size
    ids = torch.arange(size, device=d2t.device) + d2t
    if int(ids.min()) < 0 or int(ids.max()) >= vocab_size or ids.unique().numel() != size:
        raise CheckpointFormatError("d2t does not map the draft tokens to distinct model tokens")
    t2d = torch.zeros(vocab_size, dtype=torch.bool, device=d2t.device)
    t2d[ids] = True
    out["d2t"], out["t2d"] = d2t, t2d
    return out


class Eagle3Format(CheckpointFormat):
    """An official EAGLE-3 head directory (``config.json`` + ``model.safetensors`` or ``pytorch_model.bin``)."""

    @staticmethod
    def _is_eagle3(config: JSONObject) -> bool:
        kind = config.get("speculators_model_type")
        if kind is not None and str(kind).lower() != "eagle3":  # another speculators drafter (e.g. "dflash")
            return False
        architectures = config.get("architectures") or []
        return "draft_vocab_size" in config or any("eagle3" in str(a).lower() for a in architectures)

    def detect(self, path: Path) -> bool:
        return detect_config_dir(path, _WEIGHT_FILES, self._is_eagle3)

    def read(self, path: Path, map_location: str | torch.device = "cpu") -> Checkpoint:
        weights = require_weights(path, _WEIGHT_FILES, "EAGLE-3 head")
        raw = flat_config(json.loads((path / "config.json").read_text()))
        if not self._is_eagle3(raw):
            raise CheckpointFormatError(f"{path / 'config.json'} does not describe an EAGLE-3 head")
        state = load_state_file(weights, map_location)
        config = config_from_eagle3(raw, draft_layer_count(state))
        vocab = raw.get("vocab_size")
        technique_state = state_from_eagle3(state, int(vocab) if vocab is not None else None)
        return Checkpoint([config], {"eagle3": technique_state}, None, base_model_of(raw), {"source": str(path)})


FORMATS.register("eagle3", Eagle3Format())


#: Old name -> name in :mod:`economize.compat.tskip`.
_MOVED = {
    "TskipEagle3Format": "TskipEagle3Format",
    "config_from_tskip": "eagle3_config_from_tskip",
    "state_from_tskip": "eagle3_state_from_tskip",
}


def _tskip() -> ModuleType:
    from ...compat import tskip

    return tskip


__getattr__ = moved_getattr(__name__, _MOVED, "economize.compat.tskip", _tskip)
