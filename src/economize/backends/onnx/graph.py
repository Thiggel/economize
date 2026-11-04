"""Trace a Hugging Face causal LM to an ONNX graph with KV-cache inputs, and store quantized maps as MatMulNBits.

The graph has the input and output names of Hugging Face Optimum's decoder exports, so ONNX Runtime loops written
for them run it unchanged:

* inputs ``input_ids`` ``[batch, seq]``, ``attention_mask`` ``[batch, past + seq]`` and per layer ``i``
  ``past_key_values.{i}.key`` / ``.value`` ``[batch, kv_heads, past, head_dim]`` (``past = 0`` on the first call);
* outputs ``logits`` ``[batch, seq, vocab]`` and ``present.{i}.key`` / ``.value`` (the cache after the call).

Positions are ``cumsum(attention_mask) - 1``, as in ``generate``, so left-padded batches decode correctly.
"""

from __future__ import annotations

import warnings
from collections.abc import Mapping
from typing import TYPE_CHECKING

import numpy as np
import torch
from torch import nn

from ...errors import ConfigError

if TYPE_CHECKING:
    import onnx

    from ...techniques.quant.grid import QuantizedWeight

__all__ = ["CACHE_INPUT", "CACHE_OUTPUT", "NBITS", "matmul_nbits", "nbits_reason", "pack_bits", "trace_onnx"]

#: Bit widths of ONNX Runtime's ``com.microsoft.MatMulNBits`` (the CPU and CUDA kernels).
NBITS = (2, 4, 8)
#: Names of the cache inputs and outputs (``.format(layer, "key" or "value")``).
CACHE_INPUT = "past_key_values.{}.{}"
CACHE_OUTPUT = "present.{}.{}"
#: Prefix of the traced model's parameter names (the wrapper's attribute).
_PREFIX = "lm."
_MS_DOMAIN = "com.microsoft"


class _CachedStep(nn.Module):
    """One forward pass with the KV cache as flat tensors (the module that :func:`trace_onnx` traces)."""

    def __init__(self, lm: nn.Module, num_layers: int) -> None:
        super().__init__()
        self.lm = lm
        self.num_layers = num_layers

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor,
                *past: torch.Tensor) -> tuple[torch.Tensor, ...]:
        from transformers.cache_utils import DynamicCache

        cache = DynamicCache(config=getattr(self.lm, "config"))  # noqa: B009 (nn.Module types attributes as tensors)
        for layer in range(self.num_layers):
            cache.update(past[2 * layer], past[2 * layer + 1], layer)
        positions = (attention_mask.cumsum(-1) - 1).clamp(min=0)[:, -input_ids.shape[1]:]
        out = self.lm(input_ids=input_ids, attention_mask=attention_mask, position_ids=positions,
                      past_key_values=cache, use_cache=True)
        present: list[torch.Tensor] = []
        for layer in out.past_key_values.layers:
            present += [layer.keys, layer.values]
        return (out.logits, *present)


def trace_onnx(lm: nn.Module) -> onnx.ModelProto:
    """Trace ``lm`` (a plain Hugging Face causal LM) with ``torch.onnx.export(dynamo=True)``.

    Linear maps stay ``Transpose(weight) -> MatMul`` on initializers named ``lm.<module>.weight`` (the graph is
    not optimized), so :func:`matmul_nbits` finds them; ONNX Runtime folds the transposes when it loads the model.

    Parameters
    ----------
    lm : nn.Module
        The model (its dtype is the graph's).

    Returns
    -------
    onnx.ModelProto
        The graph with dynamic ``batch``, ``seq``, ``past`` and ``total`` dimensions.
    """
    config = getattr(lm, "config")  # noqa: B009 (nn.Module types attributes as tensors)
    text = getattr(config, "text_config", None) or config
    layers = int(text.num_hidden_layers)
    kv_heads = int(getattr(text, "num_key_value_heads", None) or text.num_attention_heads)
    head_dim = int(getattr(text, "head_dim", None) or text.hidden_size // text.num_attention_heads)
    dtype = next(lm.parameters()).dtype
    # Batch and lengths of at least 2, so torch.export keeps the dimensions symbolic.
    batch, seq, past_len = 2, 3, 2
    ids = torch.zeros(batch, seq, dtype=torch.int64)
    mask = torch.ones(batch, past_len + seq, dtype=torch.int64)
    past = tuple(torch.zeros(batch, kv_heads, past_len, head_dim, dtype=dtype) for _ in range(2 * layers))
    dims = {name: torch.export.Dim(name) for name in ("batch", "seq", "past", "total")}
    names = [(layer, kind) for layer in range(layers) for kind in ("key", "value")]
    with warnings.catch_warnings(), torch.no_grad():
        warnings.simplefilter("ignore")
        program = torch.onnx.export(
            _CachedStep(lm.eval(), layers), (ids, mask, *past), dynamo=True, optimize=False, verbose=False,
            input_names=["input_ids", "attention_mask", *(CACHE_INPUT.format(*name) for name in names)],
            output_names=["logits", *(CACHE_OUTPUT.format(*name) for name in names)],
            dynamic_shapes={"input_ids": {0: dims["batch"], 1: dims["seq"]},
                            "attention_mask": {0: dims["batch"], 1: dims["total"]},
                            "past": tuple({0: dims["batch"], 2: dims["past"]} for _ in names)},
        )
    if program is None:  # pragma: no cover - torch returns None only for the legacy exporter
        raise ConfigError("torch.onnx.export returned no program")
    return program.model_proto


def nbits_reason(weight: QuantizedWeight) -> str | None:
    """Why ``weight``'s codes have no exact ``MatMulNBits`` form (``None``: they have one).

    ``MatMulNBits`` stores ``bits``-bit codes in contiguous blocks of ``block_size`` inputs (a power of two of at
    least 16) with one float scale and one integer zero point per block, which is economize's integer layout.

    Parameters
    ----------
    weight : QuantizedWeight
        The codes of one linear map.

    Returns
    -------
    str or None
        The reason, or ``None``.
    """
    bits = getattr(weight.grid, "bits", None)
    if not weight.grid.name.startswith("int") or bits not in NBITS:
        return f"{weight.grid.name} codes have no MatMulNBits form (bits {', '.join(map(str, NBITS))})"
    if weight.g_idx is not None:
        return "GPTQ act_order groups are not contiguous (MatMulNBits blocks are)"
    size = weight.group_size
    if size < 16 or size & (size - 1) or weight.codes.shape[1] % size:
        return f"group size {size} is not a power of two >= 16 that divides the {weight.codes.shape[1]} inputs"
    return None


def pack_bits(values: np.ndarray, bits: int) -> np.ndarray:
    """Pack integers below ``2 ** bits`` along the last axis into bytes, lowest bits first (MatMulNBits' order).

    Parameters
    ----------
    values : np.ndarray
        Integer array ``[..., n]``.
    bits : int
        2, 4 or 8.

    Returns
    -------
    np.ndarray
        ``uint8`` array ``[..., ceil(n * bits / 8)]`` (the last byte zero-padded).
    """
    per = 8 // bits
    *lead, count = values.shape
    padded = np.zeros((*lead, -(-count // per) * per), dtype=np.uint16)
    padded[..., :count] = values
    padded = padded.reshape(*lead, -1, per)
    packed = np.zeros(padded.shape[:-1], dtype=np.uint16)
    for slot in range(per):
        packed |= padded[..., slot] << (bits * slot)
    return packed.astype(np.uint8)


def _nbits_tensors(weight: QuantizedWeight, dtype: np.dtype) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``B`` ``[N, blocks, bytes]``, ``scales`` ``[N * blocks]`` and packed ``zero_points`` of one map."""
    bits = int(getattr(weight.grid, "bits"))  # noqa: B009 (checked by nbits_reason)
    codes = weight.codes.to("cpu", torch.int64).numpy()
    out_features, in_features = codes.shape
    blocks = in_features // weight.group_size
    packed = pack_bits(codes.reshape(out_features, blocks, weight.group_size), bits)
    scales = weight.scale.to("cpu", torch.float32).numpy().reshape(out_features, blocks)
    if weight.zero is None:
        zero = np.zeros((out_features, blocks), dtype=np.int64)
    else:
        zero = np.rint(weight.zero.to("cpu", torch.float64).numpy()).astype(np.int64).reshape(out_features, blocks)
    return packed, scales.astype(dtype).reshape(-1), pack_bits(zero, bits).reshape(-1)


def matmul_nbits(model: onnx.ModelProto, quantized: Mapping[str, QuantizedWeight]) -> list[str]:
    """Replace the ``MatMul`` of every map in ``quantized`` by ``com.microsoft.MatMulNBits`` on its codes.

    The scales keep the graph's float type and the zero points are integers, so the node computes the dequantized
    map of economize up to the float type of the scales.

    Parameters
    ----------
    model : onnx.ModelProto
        A graph from :func:`trace_onnx` (changed in place).
    quantized : Mapping[str, QuantizedWeight]
        The codes by qualified module name (each passed :func:`nbits_reason`).

    Returns
    -------
    list[str]
        The maps that kept their codes.

    Raises
    ------
    ConfigError
        A map's weight is not a ``Transpose -> MatMul`` input in the graph.
    """
    from onnx import helper, numpy_helper

    graph = model.graph
    inits = {init.name: init for init in graph.initializer}
    users: dict[str, list[onnx.NodeProto]] = {}
    for node in graph.node:
        for name in node.input:
            users.setdefault(name, []).append(node)
    done: list[str] = []
    for module, weight in quantized.items():
        key = f"{_PREFIX}{module}.weight"
        if key not in inits:
            raise ConfigError(f"the ONNX graph has no initializer {key!r} for the quantized map {module!r}")
        dtype = numpy_helper.to_array(inits[key]).dtype
        packed, scales, zeros = _nbits_tensors(weight, dtype)
        transposes = [node for node in users.get(key, []) if node.op_type == "Transpose"]
        matmuls = [(transpose, node) for transpose in transposes for node in users.get(transpose.output[0], [])
                   if node.op_type == "MatMul" and node.input[1] == transpose.output[0]]
        if not matmuls:
            raise ConfigError(f"the ONNX graph does not multiply by {key!r} as Transpose -> MatMul")
        stem = f"{module}.nbits"
        graph.initializer.extend([numpy_helper.from_array(packed, f"{stem}.B"),
                                  numpy_helper.from_array(scales, f"{stem}.scales"),
                                  numpy_helper.from_array(zeros, f"{stem}.zero_points")])
        out_features, in_features = weight.codes.shape
        bits = int(getattr(weight.grid, "bits"))  # noqa: B009 (checked by nbits_reason)
        for _, matmul in matmuls:
            replacement = helper.make_node(
                "MatMulNBits", [matmul.input[0], f"{stem}.B", f"{stem}.scales", f"{stem}.zero_points"],
                list(matmul.output), name=f"{stem}.{matmul.name}", domain=_MS_DOMAIN, K=in_features,
                N=out_features, bits=bits, block_size=weight.group_size)
            index = list(graph.node).index(matmul)
            graph.node.remove(matmul)
            graph.node.insert(index, replacement)
        for transpose in transposes:
            if not any(transpose.output[0] in node.input for node in graph.node):
                graph.node.remove(transpose)
        if not any(key in node.input for node in graph.node):
            graph.initializer.remove(inits[key])
        done.append(module)
    if done and not any(opset.domain == _MS_DOMAIN for opset in model.opset_import):
        model.opset_import.append(helper.make_opsetid(_MS_DOMAIN, 1))
    return done
