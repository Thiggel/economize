"""Write a dense or economize-quantized Hugging Face model as one GGUF file (with the ``gguf`` package).

Weights are converted the way llama.cpp's ``convert_hf_to_gguf.py`` converts them: gguf-py's tensor-name map,
the query/key permutation of the ``llama`` architecture, 1-D tensors in F32. Two-dimensional tensors take the
requested ``outtype`` (gguf-py quantizes BF16, Q8_0, Q4_0, Q4_1, Q5_0 and Q5_1 in numpy; a row length that is not a
multiple of the block falls back to F16). Linear maps whose economize codes have a GGUF block type keep their codes
(:func:`code_type`): the codes are repacked bit for bit and the scales rounded to float16, which GGUF blocks store.
K-quants and i-quants are made from the F16 or F32 file by llama.cpp's ``llama-quantize``
(:func:`llama_quantize_command`).
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch
from torch import nn

from ...errors import ConfigError
from ..runtime_export import ModelShape
from .arch import GgufArch, gguf_arch

if TYPE_CHECKING:
    import gguf

    from ...techniques.quant.grid import QuantizedWeight

__all__ = ["OUTTYPES", "code_type", "llama_quantize_command", "pack_codes", "write_gguf"]

#: ``outtype -> (GGMLQuantizationType, LlamaFileType)`` names of the outtypes gguf-py writes itself.
OUTTYPES: dict[str, tuple[str, str]] = {
    "f32": ("F32", "ALL_F32"),
    "f16": ("F16", "MOSTLY_F16"),
    "bf16": ("BF16", "MOSTLY_BF16"),
    "q8_0": ("Q8_0", "MOSTLY_Q8_0"),
    "q4_0": ("Q4_0", "MOSTLY_Q4_0"),
    "q4_1": ("Q4_1", "MOSTLY_Q4_1"),
    "q5_0": ("Q5_0", "MOSTLY_Q5_0"),
    "q5_1": ("Q5_1", "MOSTLY_Q5_1"),
}
#: Values per GGUF block of the legacy quant types.
_BLOCK = 32


def code_type(weight: QuantizedWeight) -> tuple[str | None, str]:
    """The GGUF block type that holds ``weight``'s codes exactly (up to float16 scales).

    ``int4`` codes with the zero point 8 (symmetric) are ``Q4_0`` (``d * (q - 8)``), other ``int4`` codes ``Q4_1``
    (``d * q + m`` with ``m = -d * zero``) and ``int8`` codes with the zero point 128 ``Q8_0``
    (``d * (q - 128)``). Each 32-value block takes the scale of its group, so the group size must be a multiple of 32.

    Parameters
    ----------
    weight : QuantizedWeight
        The economize codes of one linear map.

    Returns
    -------
    tuple[str or None, str]
        The ``GGMLQuantizationType`` name, or ``None`` with the reason the codes have no GGUF block type.
    """
    grid = weight.grid
    if grid.name not in ("int4", "int8"):
        return None, f"{grid.name} codes have no GGUF block type"
    if weight.g_idx is not None:
        return None, "GPTQ act_order groups are not contiguous (GGUF blocks are)"
    if weight.group_size % _BLOCK or weight.codes.shape[1] % _BLOCK:
        return None, f"group size {weight.group_size} or {weight.codes.shape[1]} inputs are not multiples of 32"
    midpoint = weight.zero is not None and bool((weight.zero == 2 ** (grid.bits - 1)).all())
    if grid.name == "int8":
        return ("Q8_0", "") if midpoint else (None, "asymmetric int8 codes have no GGUF block type")
    return ("Q4_0" if midpoint else "Q4_1"), ""


def pack_codes(weight: QuantizedWeight, qtype: str) -> np.ndarray:
    """Pack economize codes into GGUF blocks of ``qtype`` (``Q4_0``, ``Q4_1`` or ``Q8_0``).

    Parameters
    ----------
    weight : QuantizedWeight
        The codes (``[out, in]``, ``uint8``).
    qtype : str
        The block type :func:`code_type` chose.

    Returns
    -------
    np.ndarray
        ``uint8`` array ``[out, in / 32 * block bytes]`` in ggml's block layout.
    """
    codes = weight.codes.to("cpu", torch.int64).numpy()
    rows, cols = codes.shape
    blocks = codes.reshape(rows, cols // _BLOCK, _BLOCK)
    group = (np.arange(cols // _BLOCK) * _BLOCK) // weight.group_size
    scale = weight.scale.to("cpu", torch.float32).numpy()[:, group]
    d = np.ascontiguousarray(scale.astype(np.float16)).view(np.uint8).reshape(rows, -1, 2)
    if qtype == "Q8_0":
        qs = (blocks - 128).astype(np.int8).view(np.uint8)
        return np.concatenate([d, qs], axis=-1).reshape(rows, -1)
    nibbles = (blocks[..., :16] | (blocks[..., 16:] << 4)).astype(np.uint8)
    if qtype == "Q4_0":
        return np.concatenate([d, nibbles], axis=-1).reshape(rows, -1)
    assert weight.zero is not None
    zero = weight.zero.to("cpu", torch.float32).numpy()[:, group]
    m = np.ascontiguousarray((-scale * zero).astype(np.float16)).view(np.uint8).reshape(rows, -1, 2)
    return np.concatenate([d, m, nibbles], axis=-1).reshape(rows, -1)


def llama_quantize_command(source: str | Path, target: str | Path, qtype: str) -> tuple[str, ...]:
    """The ``llama-quantize`` call that makes a K-quant or i-quant file from an F16 or F32 export.

    Parameters
    ----------
    source : str or Path
        The F16 or F32 GGUF file.
    target : str or Path
        The quantized file.
    qtype : str
        A llama.cpp type such as ``Q4_K_M``, ``Q5_K_M``, ``Q6_K`` or ``IQ4_XS``.

    Returns
    -------
    tuple[str, ...]
        The command line.
    """
    return ("llama-quantize", str(source), str(target), qtype)


def _permute(weight: np.ndarray, heads: int) -> np.ndarray:
    """``convert_hf_to_gguf.py``'s ``LlamaModel.permute``: rotate-half rows to interleaved pairs."""
    shape = weight.shape
    return weight.reshape(heads, 2, shape[0] // heads // 2, *shape[1:]).swapaxes(1, 2).reshape(shape)


def _quantized(array: np.ndarray, qtype: str) -> tuple[np.ndarray, str]:
    """``array`` in ``qtype``, or in F16 / F32 when gguf-py cannot quantize it to ``qtype``."""
    import gguf

    kind = gguf.GGMLQuantizationType[qtype]
    if array.ndim == 1 or qtype == "F32":
        return array.astype(np.float32), "F32"
    if qtype == "F16":
        return array.astype(np.float16), "F16"
    block = gguf.GGML_QUANT_SIZES[kind][0]
    if array.shape[-1] % block:
        return array.astype(np.float16), "F16"
    return gguf.quants.quantize(array.astype(np.float32), kind), qtype


def _pre_tokenizer(spec: object, arch: GgufArch) -> str:
    """llama.cpp's name of the pre-tokenizer: GPT-2's regex for a bare ``ByteLevel``, else the architecture's.

    A tokenizer whose only pre-tokenizer is ``ByteLevel(use_regex=True)`` splits with GPT-2's regex (digits in
    runs); the released Qwen, Llama-3 and Mistral tokenizers split with their own regex first, which
    :attr:`GgufArch.pre_tokenizer` names.
    """
    if isinstance(spec, dict) and spec.get("type") == "ByteLevel" and spec.get("use_regex", True):
        return "gpt-2"
    return arch.pre_tokenizer


def _vocabulary(writer: gguf.GGUFWriter, shape: ModelShape, arch: GgufArch, tokenizer: object | None) -> str | None:
    """Write the vocabulary; returns a note for the manifest."""
    import gguf

    if tokenizer is None:
        writer.add_tokenizer_model("llama")
        writer.add_token_list([f"<t{i}>".encode() for i in range(shape.vocab_size)])
        writer.add_token_scores([0.0] * shape.vocab_size)
        writer.add_token_types([gguf.TokenType.NORMAL] * shape.vocab_size)
        writer.add_add_bos_token(False)
        return ("no tokenizer was passed: the file has a placeholder vocabulary (<t0>, <t1>, ...), so feed token ids; "
                "pass tokenizer= to write the model's BPE vocabulary")
    backend = getattr(tokenizer, "backend_tokenizer", None)
    if backend is None:
        raise ConfigError("the GGUF writer needs a fast (tokenizers) tokenizer")
    spec = json.loads(backend.to_str())
    model = spec["model"]
    if model.get("type") != "BPE" or model.get("byte_fallback"):
        raise ConfigError("the GGUF writer writes byte-level BPE vocabularies; convert a SentencePiece model with "
                          "llama.cpp's convert_hf_to_gguf.py and pass the dense export to it")
    vocab: Mapping[str, int] = tokenizer.get_vocab()  # type: ignore[attr-defined]  # a PreTrainedTokenizerFast
    added = {token.content: token for token in getattr(tokenizer, "added_tokens_decoder", {}).values()}
    by_id = {index: token for token, index in vocab.items()}
    tokens, types = [], []
    for index in range(shape.vocab_size):
        token = by_id.get(index)
        if token is None:
            tokens.append(f"[PAD{index}]".encode())
            types.append(gguf.TokenType.UNUSED)
        else:
            tokens.append(token.encode())
            special = added.get(token)
            types.append(gguf.TokenType.NORMAL if special is None
                         else gguf.TokenType.CONTROL if special.special else gguf.TokenType.USER_DEFINED)
    merges = [merge if isinstance(merge, str) else " ".join(merge) for merge in model.get("merges", [])]
    writer.add_tokenizer_model("gpt2")
    writer.add_tokenizer_pre(_pre_tokenizer(spec.get("pre_tokenizer"), arch))
    writer.add_token_list(tokens)
    writer.add_token_types(types)
    writer.add_token_merges(merges)
    for kind in ("bos", "eos", "pad", "unk"):
        token_id = getattr(tokenizer, f"{kind}_token_id", None)
        if token_id is not None:
            getattr(writer, f"add_{kind}_token_id")(int(token_id))
    writer.add_add_bos_token(bool(getattr(tokenizer, "add_bos_token", False)))
    return None


def write_gguf(
    path: Path,
    base: nn.Module,
    state: Mapping[str, torch.Tensor],
    quantized: Mapping[str, QuantizedWeight],
    *,
    outtype: str = "f16",
    tokenizer: object | None = None,
    name: str | None = None,
) -> list[str]:
    """Write one GGUF file.

    Parameters
    ----------
    path : Path
        The file.
    base : nn.Module
        The Hugging Face model (for its config).
    state : Mapping[str, torch.Tensor]
        The dense plain-architecture state (:func:`~economize.backends.runtime_export.dense_state`).
    quantized : Mapping[str, QuantizedWeight]
        Linear maps written from their codes (each must have a :func:`code_type`).
    outtype : str
        A key of :data:`OUTTYPES` for every other 2-D tensor.
    tokenizer : PreTrainedTokenizerFast, optional
        The tokenizer whose byte-level BPE vocabulary is written (default: a placeholder vocabulary).
    name : str, optional
        ``general.name``.

    Returns
    -------
    list[str]
        Notes for the manifest.

    Raises
    ------
    ConfigError
        Unknown outtype, architecture or tensor name.
    """
    import gguf

    if outtype not in OUTTYPES:
        raise ConfigError(f"unknown GGUF outtype {outtype!r}; choose from {sorted(OUTTYPES)} (make K-quants with "
                          "llama-quantize from an f16 export)")
    shape = ModelShape.of(base, "GGUF")
    arch = gguf_arch(shape.model_type)
    kind = next(key for key, value in gguf.MODEL_ARCH_NAMES.items() if value == arch.name)
    names = gguf.get_tensor_name_map(kind, shape.num_layers)
    writer = gguf.GGUFWriter(str(path), arch.name)
    writer.add_name(name or str(getattr(base.config, "_name_or_path", "") or shape.model_type))
    writer.add_block_count(shape.num_layers)
    writer.add_context_length(shape.max_positions)
    writer.add_embedding_length(shape.hidden_size)
    writer.add_feed_forward_length(shape.intermediate_size)
    writer.add_head_count(shape.num_heads)
    writer.add_head_count_kv(shape.num_kv_heads)
    writer.add_key_length(shape.head_dim)
    writer.add_value_length(shape.head_dim)
    writer.add_layer_norm_rms_eps(shape.rms_norm_eps)
    writer.add_rope_freq_base(shape.rope_theta)
    if shape.rope_scaling is not None:
        writer.add_rope_scaling_type(gguf.RopeScalingType.LINEAR)
        writer.add_rope_scaling_factor(shape.rope_scaling)
    writer.add_file_type(gguf.LlamaFileType[OUTTYPES[outtype][1]])
    notes = [note] if (note := _vocabulary(writer, shape, arch, tokenizer)) else []
    qtype = OUTTYPES[outtype][0]
    fallback = 0
    for key, value in state.items():
        if key == "lm_head.weight" and shape.tied:
            continue
        target = names.get_name(key, try_suffixes=(".weight", ".bias"))
        if target is None:
            raise ConfigError(f"gguf-py's {arch.name} tensor map has no name for {key!r}")
        module = key.rsplit(".", 1)[0]
        heads = {"q_proj": shape.num_heads, "k_proj": shape.num_kv_heads}.get(module.rsplit(".", 1)[-1])
        if module in quantized and key.endswith(".weight"):
            code, _ = code_type(quantized[module])
            assert code is not None
            packed = pack_codes(quantized[module], code)
            if arch.permute_qk and heads:
                packed = _permute(packed, heads)
            writer.add_tensor(target, packed, raw_dtype=gguf.GGMLQuantizationType[code])
            continue
        array = value.detach().to("cpu", torch.float32).numpy()
        if arch.permute_qk and heads:
            array = _permute(array, heads)
        data, written = _quantized(array, qtype)
        fallback += written != qtype and array.ndim == 2
        writer.add_tensor(target, data, raw_dtype=gguf.GGMLQuantizationType[written])
    if fallback:
        notes.append(f"{fallback} 2-D tensors are written in F16 (their rows are not a multiple of the {qtype} block)")
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    return notes
