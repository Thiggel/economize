"""The ``"vendor_mtp"`` checkpoint format: the MTP weights inside a model directory of DeepSeek-V3, GLM-4.5, MiMo,
Qwen3-Next or Qwen3.5.

Detected for a directory whose ``config.json`` (or its ``text_config``) has ``num_nextn_predict_layers > 0``
(DeepSeek-V3, GLM-4.5, MiMo) or ``mtp_num_hidden_layers > 0`` (Qwen3-Next, Qwen3.5). Only the MTP tensors are read
(through ``model.safetensors.index.json`` when the weights are sharded). Vendor names, for ``L`` decoder layers::

    DeepSeek-V3, GLM-4.5   model.layers.{L+i}.{enorm, hnorm, eh_proj, shared_head.norm, <block>}
                           (shared_head.head and embed_tokens are copies of the model's and are dropped)
    MiMo                   model.mtp_layers.{i}.{token_layernorm, hidden_layernorm, input_proj, final_layernorm,
                           <block>}
    Qwen3-Next, Qwen3.5    mtp.{pre_fc_norm_embedding, pre_fc_norm_hidden, fc, norm} (shared by all modules),
                           mtp.layers.{i}.<block>

Mixture-of-experts blocks store one tensor per expert (``mlp.experts.{e}.{gate,up,down}_proj.weight``);
transformers 5 fuses them into ``mlp.experts.gate_up_proj [E, 2I, D]`` and ``mlp.experts.down_proj [E, D, I]``,
and :func:`state_from_vendor` does the same (:func:`state_to_vendor` splits them again).

DeepSeek-V3's FP8 checkpoints store block-quantized weights with a ``weight_scale_inv`` tensor per weight (one
scale per ``128 x 128`` block, ``quantization_config.weight_block_size``); they are dequantized to bfloat16 on
reading. The tensor names of every family come from its :class:`~economize.techniques.vendor_mtp.MTPFamily`.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch

from ...core.serialization import FORMATS, Checkpoint, CheckpointFormat
from ...errors import CheckpointFormatError, ConfigError, UnknownNameError
from .config import VendorMTPConfig
from .families import MTP_FAMILIES, MTPFamily, family_for

__all__ = ["FP8_BLOCK", "VendorMTPFormat", "dequantize_fp8", "mtp_settings", "state_from_vendor", "state_to_vendor"]

_EXPERT = re.compile(r"^(?P<pre>.*\.experts)\.(?P<e>\d+)\.(?P<proj>gate_proj|up_proj|down_proj)\.weight$")
#: Block shape of DeepSeek-V3's FP8 weights when ``config.json`` does not name one.
FP8_BLOCK = (128, 128)


def _family(name: str) -> MTPFamily:
    try:
        return MTP_FAMILIES.get(name)
    except UnknownNameError as error:
        raise CheckpointFormatError(str(error)) from None


def dequantize_fp8(tensors: Mapping[str, torch.Tensor],
                   block: tuple[int, int] = FP8_BLOCK) -> dict[str, torch.Tensor]:
    """Block-dequantized copies (bfloat16) of the weights that come with a ``weight_scale_inv`` tensor (DeepSeek-V3's
    FP8 layout: ``weight = fp8 * scale`` per ``block`` of the weight); every other tensor passes through.

    Raises:
        CheckpointFormatError: A scale without its weight, a scale of the wrong shape, or a per-tensor or
            per-channel ``weight_scale`` (other quantization schemes, not supported).
    """
    out: dict[str, torch.Tensor] = {}
    for key, value in tensors.items():
        if key.endswith(".weight_scale"):
            raise CheckpointFormatError(f"quantized MTP weights with {key!r} are not supported; dequantize them")
        if key.endswith(".weight_scale_inv"):
            continue
        scale = tensors.get(key[: -len("weight")] + "weight_scale_inv") if key.endswith(".weight") else None
        if scale is None:
            out[key] = value
            continue
        rows, cols = value.shape
        expected = (-(-rows // block[0]), -(-cols // block[1]))
        if tuple(scale.shape) != expected:
            raise CheckpointFormatError(f"{key!r}: scale shape {tuple(scale.shape)}, expected {expected} for blocks "
                                        f"of {block}")
        full = scale.float().repeat_interleave(block[0], 0).repeat_interleave(block[1], 1)[:rows, :cols]
        out[key] = (value.float() * full).to(torch.bfloat16)
    orphans = [k for k in tensors if k.endswith(".weight_scale_inv") and k[: -len("_scale_inv")] not in tensors]
    if orphans:
        raise CheckpointFormatError(f"scales without their weights: {orphans[:3]}")
    return out


def _fuse_experts(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Stack per-expert ``gate_proj``/``up_proj``/``down_proj`` weights into transformers 5's fused tensors."""
    experts: dict[str, dict[int, dict[str, torch.Tensor]]] = {}
    out: dict[str, torch.Tensor] = {}
    for key, value in state.items():
        match = _EXPERT.match(key)
        if match is None:
            out[key] = value
        else:
            experts.setdefault(match["pre"], {}).setdefault(int(match["e"]), {})[match["proj"]] = value
    for prefix, table in experts.items():
        count = len(table)
        if sorted(table) != list(range(count)) or any(len(p) != 3 for p in table.values()):
            raise CheckpointFormatError(f"incomplete experts under {prefix!r}")
        out[f"{prefix}.gate_up_proj"] = torch.stack([torch.cat([table[e]["gate_proj"], table[e]["up_proj"]], 0)
                                                      for e in range(count)])
        out[f"{prefix}.down_proj"] = torch.stack([table[e]["down_proj"] for e in range(count)])
    return out


def _split_experts(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Inverse of :func:`_fuse_experts`."""
    out: dict[str, torch.Tensor] = {}
    for key, value in state.items():
        if key.endswith(".experts.gate_up_proj") and value.dim() == 3:
            prefix, half = key[: -len(".gate_up_proj")], value.shape[1] // 2
            for e in range(value.shape[0]):
                out[f"{prefix}.{e}.gate_proj.weight"] = value[e, :half]
                out[f"{prefix}.{e}.up_proj.weight"] = value[e, half:]
        elif key.endswith(".experts.down_proj") and value.dim() == 3:
            prefix = key[: -len(".down_proj")]
            for e in range(value.shape[0]):
                out[f"{prefix}.{e}.down_proj.weight"] = value[e]
        else:
            out[key] = value
    return out


def state_from_vendor(tensors: Mapping[str, torch.Tensor], family: str, num_layers: int,
                      fp8_block: tuple[int, int] = FP8_BLOCK) -> dict[str, torch.Tensor]:
    """Rename a model checkpoint's MTP tensors to :class:`~economize.techniques.vendor_mtp.VendorMTPTechnique`
    parameters (``layers.{i}.*``); tensors of the main model are ignored.

    Args:
        tensors: Checkpoint tensors by vendor name (may include the main model's).
        family: Name of a registered :class:`~economize.techniques.vendor_mtp.MTPFamily`.
        num_layers: Decoder layers of the main model (DeepSeek-V3 and GLM-4.5 number the MTP layers after them).
        fp8_block: Block shape of FP8 weights (see :func:`dequantize_fp8`).

    Raises:
        CheckpointFormatError: Unknown family, no MTP tensor, unsupported quantized weights, or incomplete experts.
    """
    spec = _family(family)
    owned = dequantize_fp8({key: value for key, value in tensors.items() if spec.owns(key, num_layers)}, fp8_block)
    out: dict[str, torch.Tensor] = {}
    shared: dict[str, torch.Tensor] = {}
    for key, value in owned.items():
        match = spec.match(key, num_layers)
        if match is not None:
            index, leaf = match
            if not leaf.startswith(spec.dropped):
                out[f"layers.{index}.{spec.economize_name(leaf)}"] = value
            continue
        shared_leaf = spec.shared(key)
        if shared_leaf is not None:
            for ours, theirs in spec.module_names.items():
                if shared_leaf.startswith(theirs + "."):
                    shared[ours + shared_leaf[len(theirs):]] = value
    if not out:
        raise CheckpointFormatError(f"no {family} MTP tensors found")
    count = 1 + max(int(key.split(".")[1]) for key in out)
    for i in range(count):
        for leaf, value in shared.items():
            out[f"layers.{i}.{leaf}"] = value
    return _fuse_experts(out)


def state_to_vendor(state: Mapping[str, torch.Tensor], family: str, num_layers: int) -> dict[str, torch.Tensor]:
    """Vendor names for a :class:`~economize.techniques.vendor_mtp.VendorMTPTechnique` state (inverse of
    :func:`state_from_vendor`; an own LM head is dropped).

    Raises:
        CheckpointFormatError: Unknown family, or modules whose shared parameters differ (the Qwen layout shares
            them).
    """
    spec = _family(family)
    out: dict[str, torch.Tensor] = {}
    for key, value in _split_experts(dict(state)).items():
        if not key.startswith("layers."):
            continue
        _, index, leaf = key.split(".", 2)
        name = spec.vendor_name(int(index), leaf, num_layers)
        if name in out and not torch.equal(out[name], value):
            raise CheckpointFormatError(f"the {family} MTP layout shares {name!r}, but the modules differ")
        out[name] = value
    return out


def _text_config(raw: dict[str, Any]) -> dict[str, Any]:
    text = raw.get("text_config")
    return {**raw, **text} if isinstance(text, dict) else raw


def mtp_settings(raw: dict[str, Any]) -> tuple[str | None, int, int]:
    """``(family, num_layers, num_modules)`` of a model ``config.json``: ``num_modules`` is ``0`` without MTP,
    ``family`` is ``None`` when no registered :class:`~economize.techniques.vendor_mtp.MTPFamily` lists the model
    type."""
    config = _text_config(raw)
    modules = int(config.get("num_nextn_predict_layers") or config.get("mtp_num_hidden_layers") or 0)
    try:
        family: str | None = family_for(str(config.get("model_type", "")))
    except ConfigError:
        family = None
    return family, int(config.get("num_hidden_layers", 0)), modules


def _fp8_block(raw: dict[str, Any]) -> tuple[int, int]:
    quant = _text_config(raw).get("quantization_config") or {}
    block = quant.get("weight_block_size") if isinstance(quant, dict) else None
    return (int(block[0]), int(block[1])) if isinstance(block, list) and len(block) == 2 else FP8_BLOCK


class VendorMTPFormat(CheckpointFormat):
    """A model directory that ships MTP modules (``config.json`` and safetensors weights)."""

    @staticmethod
    def _config(path: Path) -> dict[str, Any] | None:
        file = path / "config.json"
        if not path.is_dir() or not file.is_file():
            return None
        try:
            raw = json.loads(file.read_text())
        except (OSError, ValueError):
            return None
        return raw if isinstance(raw, dict) else None

    @staticmethod
    def _files(path: Path) -> list[Path]:
        index = path / "model.safetensors.index.json"
        if index.is_file():
            weight_map = json.loads(index.read_text()).get("weight_map", {})
            names = {f for key, f in weight_map.items() if "mtp" in key or key.startswith("model.layers.")}
            return [path / name for name in sorted(names)]
        single = path / "model.safetensors"
        return [single] if single.is_file() else []

    def detect(self, path: Path) -> bool:
        raw = self._config(path)
        return raw is not None and mtp_settings(raw)[2] > 0 and bool(self._files(path))

    def read(self, path: Path, map_location: str | torch.device = "cpu") -> Checkpoint:
        from safetensors import safe_open

        raw = self._config(path)
        if raw is None:
            raise CheckpointFormatError(f"{path} has no readable config.json")
        family, num_layers, modules = mtp_settings(raw)
        if modules < 1:
            raise CheckpointFormatError(f"{path / 'config.json'} declares no MTP module")
        if family is None:
            raise CheckpointFormatError(f"{path / 'config.json'}: no registered MTP family lists model_type "
                                        f"{_text_config(raw).get('model_type')!r}; known: {MTP_FAMILIES.names()}")
        spec = MTP_FAMILIES.get(family)
        tensors: dict[str, torch.Tensor] = {}
        for file in self._files(path):
            with safe_open(str(file), framework="pt", device=str(map_location)) as handle:
                for key in handle.keys():  # noqa: SIM118  (safetensors' safe_open handle is not iterable)
                    if spec.owns(key, num_layers):
                        tensors[key] = handle.get_tensor(key)
        state = state_from_vendor(tensors, family, num_layers, _fp8_block(raw))
        found = 1 + max(int(key.split(".")[1]) for key in state)
        if found != modules:
            raise CheckpointFormatError(f"config.json declares {modules} MTP modules, the weights hold {found}")
        config = VendorMTPConfig(family=family, num_modules=modules)
        base = raw.get("_name_or_path") or None
        return Checkpoint([config], {"vendor_mtp": state}, None, base, {"source": str(path)})


FORMATS.register("vendor_mtp", VendorMTPFormat())
