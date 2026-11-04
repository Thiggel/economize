"""TensorRT-LLM's checkpoint layout: tensor names, fused attention maps, ``W4A16_AWQ`` codes and ``config.json``.

A TensorRT-LLM checkpoint (the input of ``trtllm-build``) is a folder with ``config.json`` and one
``rank{r}.safetensors`` per tensor-parallel rank. The Llama block layout maps Hugging Face's names as follows
(:data:`LINEARS`): the query, key and value maps are fused into ``attention.qkv`` (rows of q, then k, then v),
``o_proj`` is ``attention.dense``, and the MLP's ``gate_proj``, ``up_proj`` and ``down_proj`` are ``mlp.fc``,
``mlp.gate`` and ``mlp.proj``.

Quantized weights are stored as ``W4A16_AWQ`` (TensorRT-LLM's group-wise int4 weight-only GEMM), in the layout
NVIDIA ModelOpt writes for TensorRT-LLM:

* ``p.weight``: ``int8 [out / 2, in]``, two signed int4 codes per byte, row ``2 i`` in the low nibble and row
  ``2 i + 1`` in the high nibble; the signed code is economize's code minus 8 (symmetric int4 grids only);
* ``p.weights_scaling_factor``: ``[in / group_size, out]`` in the checkpoint's float type;
* ``p.prequant_scaling_factor``: ``[in]`` ones (the AWQ smoothing factors are already folded into the weights).

``trtllm-build`` interleaves the codes for its kernels when it loads the checkpoint. This layout has not been run
through TensorRT-LLM in economize's tests (they check it against an independent unpacker only).
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping

import torch

from ...errors import ConfigError
from ...techniques.quant.grid import QuantizedWeight
from ..runtime_export import ModelShape
from .arch import TrtArch

__all__ = [
    "LINEARS",
    "QUANT_ALGO",
    "TRT_GROUP_SIZES",
    "fused_mismatches",
    "pack_int4_rows",
    "trt_config",
    "trt_reason",
    "trt_tensors",
]

#: TensorRT-LLM name of every linear map of a block, with the Hugging Face maps it concatenates (along the rows).
LINEARS: Mapping[str, tuple[str, ...]] = {
    "attention.qkv": ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"),
    "attention.dense": ("self_attn.o_proj",),
    "mlp.fc": ("mlp.gate_proj",),
    "mlp.gate": ("mlp.up_proj",),
    "mlp.proj": ("mlp.down_proj",),
}
#: TensorRT-LLM name of every norm of a block.
NORMS: Mapping[str, str] = {"input_layernorm": "input_layernorm", "post_attention_layernorm": "post_layernorm"}
#: Group sizes of TensorRT-LLM's group-wise int4 GEMM.
TRT_GROUP_SIZES = (64, 128)
#: ``quant_algo`` of the stored codes.
QUANT_ALGO = "W4A16_AWQ"


def trt_reason(weight: QuantizedWeight) -> str | None:
    """Why ``weight``'s codes have no exact ``W4A16_AWQ`` form (``None``: they have one).

    Parameters
    ----------
    weight : QuantizedWeight
        The codes of one linear map.

    Returns
    -------
    str or None
        The reason, or ``None``.
    """
    if weight.grid.name != "int4":
        return (f"{weight.grid.name} codes: TensorRT-LLM's int8 and per-channel weight-only checkpoints hold float "
                "weights that trtllm-build quantizes itself; only group-wise int4 is stored as codes")
    if weight.zero is None or not bool(torch.all(weight.zero == 8)):
        return "asymmetric int4 codes (W4A16_AWQ has no zero points)"
    if weight.g_idx is not None:
        return "GPTQ act_order groups are not contiguous (TensorRT-LLM's groups are)"
    if weight.group_size not in TRT_GROUP_SIZES or weight.codes.shape[1] % weight.group_size:
        return f"group size {weight.group_size} (TensorRT-LLM's int4 GEMM takes {TRT_GROUP_SIZES})"
    if weight.codes.shape[0] % 2:
        return "an odd number of output rows (W4A16_AWQ packs two rows per byte)"
    return None


def _members(name: str) -> tuple[str, tuple[str, ...]] | None:
    """The block prefix and the fused group of the Hugging Face map ``name`` (``None``: not a block map)."""
    for parts in LINEARS.values():
        for part in parts:
            if name.endswith("." + part):
                return name[: -len(part)], parts
    return None


def fused_mismatches(quantized: Mapping[str, QuantizedWeight]) -> dict[str, str]:
    """The stored maps that must be written dense so every TensorRT-LLM map is either all codes or all dense.

    A fused map (``attention.qkv``) holds codes only when all its parts do, with one group size; the checkpoint
    has one ``group_size`` for all maps (the most common one).

    Parameters
    ----------
    quantized : Mapping[str, QuantizedWeight]
        The maps the export would store as codes, by Hugging Face name.

    Returns
    -------
    dict[str, str]
        Maps to write dense instead, with the reason.
    """
    if not quantized:
        return {}
    group = Counter(weight.group_size for weight in quantized.values()).most_common(1)[0][0]
    out: dict[str, str] = {}
    for name, weight in quantized.items():
        found = _members(name)
        if found is None:
            out[name] = "not a map of a transformer block (TensorRT-LLM keeps the LM head dense)"
            continue
        prefix, parts = found
        names = [prefix + part for part in parts]
        if any(member not in quantized for member in names):
            out[name] = f"fused with {', '.join(parts)}, which are not all stored as codes"
        elif any(quantized[member].group_size != group for member in names):
            out[name] = f"group size {weight.group_size} (the checkpoint's group_size is {group})"
    return out


def pack_int4_rows(signed: torch.Tensor) -> torch.Tensor:
    """Pack signed int4 codes ``[out, in]`` into ``int8 [out / 2, in]``, even rows in the low nibbles.

    Parameters
    ----------
    signed : torch.Tensor
        Integer codes in ``[-8, 7]``.

    Returns
    -------
    torch.Tensor
        The packed bytes.

    Raises
    ------
    ConfigError
        The number of rows is odd.
    """
    if signed.shape[0] % 2:
        raise ConfigError(f"W4A16_AWQ packs two rows per byte; got {signed.shape[0]} rows")
    nibbles = signed.to(torch.int16) & 0xF
    packed = nibbles[0::2] | (nibbles[1::2] << 4)
    return packed.to(torch.uint8).view(torch.int8).contiguous()


def _codes(weights: list[QuantizedWeight], dtype: torch.dtype, prefix: str) -> dict[str, torch.Tensor]:
    """The ``W4A16_AWQ`` tensors of the maps ``weights`` concatenated along the rows."""
    signed = torch.cat([weight.codes.to(torch.int16) - 8 for weight in weights], 0)
    scale = torch.cat([weight.scale for weight in weights], 0)
    return {
        prefix + ".weight": pack_int4_rows(signed),
        prefix + ".weights_scaling_factor": scale.t().to(dtype).contiguous(),
        prefix + ".prequant_scaling_factor": torch.ones(signed.shape[1], dtype=dtype),
    }


def trt_tensors(
    state: Mapping[str, torch.Tensor],
    quantized: Mapping[str, QuantizedWeight],
    dtype: torch.dtype,
    shape: ModelShape,
) -> tuple[dict[str, torch.Tensor], list[str]]:
    """The checkpoint's tensors under TensorRT-LLM's names.

    Parameters
    ----------
    state : Mapping[str, torch.Tensor]
        The dense state dict under Hugging Face's names (:func:`~economize.backends.runtime_export.dense_state`).
    quantized : Mapping[str, QuantizedWeight]
        The maps stored as codes (whole fused groups, see :func:`fused_mismatches`).
    dtype : torch.dtype
        The float type of the checkpoint.
    shape : ModelShape
        The model's hyperparameters.

    Returns
    -------
    tensors : dict[str, torch.Tensor]
        The tensors.
    dense : list[str]
        The TensorRT-LLM linear maps written dense (``exclude_modules`` of a quantized checkpoint).

    Raises
    ------
    ConfigError
        ``state`` has a tensor the Llama block layout does not have (e.g. query/key norms).
    """
    used: set[str] = set()

    def take(key: str) -> torch.Tensor:
        used.add(key)
        return state[key]

    embed = take("model.embed_tokens.weight")
    out = {"transformer.vocab_embedding.weight": embed, "transformer.ln_f.weight": take("model.norm.weight"),
           "lm_head.weight": (take("lm_head.weight") if "lm_head.weight" in state else embed).clone()}
    dense = ["lm_head"]
    for i in range(shape.num_layers):
        hf, trt = f"model.layers.{i}.", f"transformer.layers.{i}."
        for source, target in NORMS.items():
            out[trt + target + ".weight"] = take(hf + source + ".weight")
        for target, parts in LINEARS.items():
            names = [hf + part for part in parts]
            weights = [quantized.get(name) for name in names]
            if all(weight is not None for weight in weights):
                out.update(_codes([weight for weight in weights if weight is not None], dtype, trt + target))
                used.update(name + ".weight" for name in names)
            else:
                out[trt + target + ".weight"] = torch.cat([take(name + ".weight") for name in names], 0)
                dense.append(trt + target)
            if all(name + ".bias" in state for name in names):
                out[trt + target + ".bias"] = torch.cat([take(name + ".bias") for name in names], 0)
    extra = sorted(set(state) - used)
    if extra:
        raise ConfigError(f"the TensorRT-LLM writer maps the Llama block layout; {len(extra)} tensors have no "
                          f"TensorRT-LLM name, e.g. {extra[:3]}")
    return out, dense


def trt_config(
    shape: ModelShape,
    arch: TrtArch,
    dtype: torch.dtype,
    *,
    group_size: int | None,
    dense: list[str],
    biases: bool,
) -> dict[str, object]:
    """TensorRT-LLM's ``config.json`` of a single-GPU checkpoint.

    Parameters
    ----------
    shape : ModelShape
        The model's hyperparameters.
    arch : TrtArch
        The architecture entry.
    dtype : torch.dtype
        The float type of the checkpoint.
    group_size : int or None
        The group size of the stored int4 codes (``None``: no codes, ``quant_algo`` is null).
    dense : list[str]
        The linear maps written dense (``exclude_modules`` when ``group_size`` is set).
    biases : bool
        The attention maps have biases.

    Returns
    -------
    dict[str, object]
        The config.
    """
    quantization: dict[str, object] = {"quant_algo": None, "kv_cache_quant_algo": None}
    if group_size is not None:
        quantization.update(quant_algo=QUANT_ALGO, group_size=group_size, has_zero_point=False,
                            pre_quant_scale=True, exclude_modules=dense)
    return {
        "architecture": arch.architecture,
        "dtype": str(dtype).removeprefix("torch."),
        "logits_dtype": "float32",
        "num_hidden_layers": shape.num_layers,
        "num_attention_heads": shape.num_heads,
        "num_key_value_heads": shape.num_kv_heads,
        "head_size": shape.head_dim,
        "hidden_size": shape.hidden_size,
        "intermediate_size": shape.intermediate_size,
        "vocab_size": shape.vocab_size,
        "max_position_embeddings": shape.max_positions,
        "hidden_act": shape.hidden_act,
        "norm_epsilon": shape.rms_norm_eps,
        "position_embedding_type": "rope_gpt_neox",
        "rotary_base": shape.rope_theta,
        "rotary_scaling": None if shape.rope_scaling is None else {"type": "linear", "factor": shape.rope_scaling},
        "attn_bias": biases,
        "mlp_bias": False,
        "use_parallel_embedding": False,
        "embedding_sharding_dim": 0,
        "share_embedding_table": False,
        "quantization": quantization,
        "mapping": {"world_size": 1, "tp_size": 1, "pp_size": 1},
        **dict(arch.fields),
    }
