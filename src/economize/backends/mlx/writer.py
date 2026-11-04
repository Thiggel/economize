"""MLX's checkpoint layout: affine quantized linear maps packed into ``uint32`` words, and the ``mlx_lm`` config.

``mlx_lm`` loads a folder with ``config.json`` and ``model*.safetensors`` under the Hugging Face parameter names.
A quantized linear map ``p`` holds three tensors instead of ``p.weight`` (``mlx.nn.QuantizedLinear``):

* ``p.weight``: ``uint32 [out, in * bits / 32]``, the ``bits``-bit codes of every row as one contiguous bit stream,
  lowest bits first (for 2, 4 and 8 bits code ``k`` of a word is ``(word >> (bits * k)) & (2 ** bits - 1)``);
* ``p.scales`` and ``p.biases``: ``[out, in / group_size]`` in the model's float type, with the weight
  ``scales * code + biases``.

economize's integer grids store ``scale * (code - zero)``, so ``biases = -scale * zero`` and the codes are copied as
they are. ``config["quantization"]`` holds the ``group_size`` and ``bits`` of most maps; maps with other settings
get their own entry under their name, which ``mlx_lm``'s loader reads before the global one.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from typing import TYPE_CHECKING

import numpy as np
import torch
from torch import nn

from ...errors import ConfigError
from ..runtime_export import ModelShape

if TYPE_CHECKING:
    import numpy.typing as npt

    from ...techniques.quant.grid import QuantizedWeight

__all__ = ["MLX_BITS", "MLX_GROUP_SIZES", "mlx_config", "mlx_reason", "mlx_tensors", "pack_mlx"]

#: Code widths of MLX's affine quantization.
MLX_BITS = (2, 3, 4, 5, 6, 8)
#: Group sizes of MLX's affine quantization.
MLX_GROUP_SIZES = (32, 64, 128)


def mlx_reason(weight: QuantizedWeight) -> str | None:
    """Why ``weight``'s codes have no exact MLX form (``None``: they have one).

    Parameters
    ----------
    weight : QuantizedWeight
        The codes of one linear map.

    Returns
    -------
    str or None
        The reason, or ``None``.
    """
    bits = weight.grid.bits
    if not weight.grid.name.startswith("int") or bits not in MLX_BITS or weight.zero is None:
        return f"{weight.grid.name} codes have no MLX affine form (bits {', '.join(map(str, MLX_BITS))})"
    if weight.g_idx is not None:
        return "GPTQ act_order groups are not contiguous (MLX groups are)"
    if weight.group_size not in MLX_GROUP_SIZES or weight.codes.shape[1] % weight.group_size:
        return (f"group size {weight.group_size} is not one of MLX's {', '.join(map(str, MLX_GROUP_SIZES))} or does "
                f"not divide the {weight.codes.shape[1]} inputs")
    return None


def pack_mlx(codes: npt.NDArray[np.integer], bits: int) -> npt.NDArray[np.uint32]:
    """Pack ``bits``-bit codes of every row into one bit stream of little-endian ``uint32`` words, lowest bits first.

    Parameters
    ----------
    codes : np.ndarray
        Integer array ``[out, in]`` with values below ``2 ** bits``; ``in * bits`` is a multiple of 32.
    bits : int
        One of :data:`MLX_BITS`.

    Returns
    -------
    np.ndarray
        ``uint32 [out, in * bits / 32]``.
    """
    rows, count = codes.shape
    if (count * bits) % 32:
        raise ConfigError(f"{count} codes of {bits} bits do not fill whole 32-bit words")
    stream = np.unpackbits(codes.astype(np.uint8)[..., None], axis=-1, count=bits, bitorder="little")
    packed = np.packbits(stream.reshape(rows, count * bits), axis=-1, bitorder="little")
    return np.ascontiguousarray(packed).view("<u4").astype(np.uint32)


def mlx_tensors(
    state: Mapping[str, torch.Tensor],
    quantized: Mapping[str, QuantizedWeight],
    dtype: torch.dtype,
    *,
    tied: bool,
) -> dict[str, torch.Tensor]:
    """The tensors of an MLX checkpoint: ``state`` with the maps of ``quantized`` replaced by their packed codes.

    Parameters
    ----------
    state : Mapping[str, torch.Tensor]
        The dense state under Hugging Face names (:func:`~economize.backends.runtime_export.dense_state`).
    quantized : Mapping[str, QuantizedWeight]
        The codes to keep, by linear-map name.
    dtype : torch.dtype
        The float type of the dense tensors, the scales and the biases.
    tied : bool
        The LM head shares the embedding (``lm_head.weight`` is then left out, as ``mlx_lm`` expects).

    Returns
    -------
    dict[str, torch.Tensor]
        Contiguous CPU tensors.
    """
    out = {key: value.to(dtype) if value.is_floating_point() else value for key, value in state.items()}
    if tied:
        out.pop("lm_head.weight", None)
    for name, weight in quantized.items():
        codes = weight.codes.detach().cpu().numpy()
        bits = int(weight.grid.bits)
        scale = weight.scale.detach().cpu().float()
        zero = weight.zero if weight.zero is not None else torch.zeros_like(scale)
        out[f"{name}.weight"] = torch.from_numpy(pack_mlx(codes, bits))
        out[f"{name}.scales"] = scale.to(dtype)
        out[f"{name}.biases"] = (-scale * zero.detach().cpu().float()).to(dtype)
    return {key: value.contiguous() for key, value in sorted(out.items())}


def mlx_config(base: nn.Module, shape: ModelShape, quantized: Mapping[str, QuantizedWeight]) -> dict[str, object]:
    """The ``config.json`` ``mlx_lm`` reads: the Hugging Face config with the fields its model classes need.

    ``mlx_lm``'s model arguments read ``rope_theta`` and ``rope_scaling`` at the top level (Transformers 5 keeps
    them in ``rope_parameters``), so both are written explicitly.

    Parameters
    ----------
    base : nn.Module
        The Hugging Face model.
    shape : ModelShape
        Its hyperparameters.
    quantized : Mapping[str, QuantizedWeight]
        The maps written with codes.

    Returns
    -------
    dict[str, object]
        The JSON object.
    """
    config: dict[str, object] = dict(getattr(base, "config").to_dict())  # noqa: B009 (nn.Module attribute)
    config.pop("quantization_config", None)
    config.update(
        model_type=shape.model_type,
        num_hidden_layers=shape.num_layers,
        head_dim=shape.head_dim,
        rope_theta=shape.rope_theta,
        rope_scaling=None if shape.rope_scaling is None else {"type": "linear", "factor": shape.rope_scaling},
        tie_word_embeddings=shape.tied,
    )
    if quantized:
        settings = {name: {"group_size": int(weight.group_size), "bits": int(weight.grid.bits)}
                    for name, weight in quantized.items()}
        common = Counter((entry["group_size"], entry["bits"]) for entry in settings.values()).most_common(1)[0][0]
        quantization: dict[str, object] = {"group_size": common[0], "bits": common[1]}
        quantization.update({name: entry for name, entry in settings.items()
                             if (entry["group_size"], entry["bits"]) != common})
        config["quantization"] = quantization
    return config
