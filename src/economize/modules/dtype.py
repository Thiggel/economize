"""Trainable copies of decoder layers, and running them inside a model of another dtype."""

from __future__ import annotations

import copy
from collections.abc import Callable
from typing import Any

import torch
from torch import nn
from torch.func import functional_call

from ..errors import ConfigError
from ..registry import Registry
from ..validation import check_choice
from .sharding import clone_sharded, copy_tensor_parallel_forwards, is_sharded

__all__ = ["COPY_DTYPES", "DENSE_LINEARS", "DenseLinear", "InModelDtype", "copy_dtype_of", "trainable_copy"]

#: Storage dtypes of trainable layer copies: float32, or the model's own dtype (to be trained with float32 master
#: weights in the optimizer, see ``economize.train.OffloadedAdamW``).
COPY_DTYPES = ("float32", "model")


def copy_dtype_of(name: str, model_dtype: torch.dtype) -> torch.dtype:
    """The storage dtype that the name in :data:`COPY_DTYPES` stands for in a model of ``model_dtype``."""
    check_choice("copy_dtype", name, COPY_DTYPES)
    return torch.float32 if name == "float32" else model_dtype


#: Turns a quantized linear module into an equivalent float32 ``nn.Linear``, or returns ``None`` for modules of
#: other kinds.
DenseLinear = Callable[[nn.Module], nn.Linear | None]
#: Dense stand-ins for quantized linear modules whose weights are not dequantizable tensors (bitsandbytes keeps its
#: scales on the parameter or the module). :func:`trainable_copy` asks every entry; a new quantizer is supported by
#: registering one function.
DENSE_LINEARS: Registry[DenseLinear] = Registry("dense linear", "economize.dense_linears")


@DENSE_LINEARS.register("bitsandbytes")
def bitsandbytes_linear(module: nn.Module) -> nn.Linear | None:
    """A float32 ``nn.Linear`` with the dequantized weight of a bitsandbytes ``Linear4bit`` or ``Linear8bitLt``."""
    if not type(module).__module__.startswith("bitsandbytes"):
        return None
    import bitsandbytes.functional as bnb_functional

    weight: Any = module.weight
    state = getattr(module, "state", None)
    if getattr(weight, "quant_state", None) is not None:  # 4-bit NF4 / FP4
        dense = bnb_functional.dequantize_4bit(weight.data, weight.quant_state)
    else:  # LLM.int8: codes and row scales on the weight until the first forward, on the module's state after it
        codes = getattr(weight, "CB", None)
        codes = codes if codes is not None else getattr(state, "CB", None)
        scales = getattr(weight, "SCB", None)
        scales = scales if scales is not None else getattr(state, "SCB", None)
        if codes is None or scales is None:
            return None
        dense = bnb_functional.int8_vectorwise_dequant(codes, scales)
    bias: Any = module.bias
    linear = nn.Linear(dense.shape[1], dense.shape[0], bias=bias is not None, device=dense.device,
                       dtype=torch.float32)
    with torch.no_grad():
        linear.weight.copy_(dense.float())
        if bias is not None:
            linear.bias.copy_(bias.float())
    return linear


def _dense_linears(layer: nn.Module) -> dict[int, nn.Module]:
    """Dense stand-ins for the quantized linear modules of ``layer``, by ``id`` of the original module."""
    dense: dict[int, nn.Module] = {}
    converters = [DENSE_LINEARS.get(name) for name in DENSE_LINEARS.names()]
    for module in layer.modules():
        for convert in converters:
            linear = convert(module)
            if linear is not None:
                dense[id(module)] = linear
                break
    return dense


def trainable_copy(layer: nn.Module, dtype: torch.dtype | None = torch.float32,
                   layer_idx: int | None = None) -> nn.Module:
    """A copy of a decoder layer in ``dtype`` (``None``: the layer's own) with trainable dense weights (the layers
    of ``adaptation="full"``, the blocks of new drafter modules).

    With ``layer_idx`` the copy's attention modules write the cache entries of layer ``layer_idx`` (0-based, the
    index Hugging Face attention modules pass to ``past_key_values.update``) instead of the original's.

    Weights held as quantized tensor subclasses (torchao, or any tensor with a working ``dequantize()``) are copied
    in dequantized form, and quantized linear modules known to :data:`DENSE_LINEARS` (bitsandbytes 4-bit and 8-bit)
    become float32 ``nn.Linear`` modules; the quantized originals are never copied. Stand-in modules (e.g. low-rank
    factorizations) are copied as they are.

    Raises:
        ConfigError: A weight is neither a float tensor nor dequantizable.
    """
    memo: dict[int, Any] = _dense_linears(layer)
    replaced = {id(p) for module in layer.modules() if id(module) in memo for p in module.parameters()}
    for name, param in layer.named_parameters():
        if id(param) in replaced:
            continue
        if is_sharded(param):
            memo[id(param)] = nn.Parameter(clone_sharded(param), requires_grad=param.requires_grad)
            continue
        if type(param) in (nn.Parameter, torch.Tensor) and param.is_floating_point():
            continue
        dense = _dequantized(param)
        if dense is None:
            raise ConfigError(
                f"adaptation='full' trains dense copies of the jump path's layers, but weight {name!r} is a "
                f"{type(param).__name__} ({param.dtype}) that cannot be dequantized: use adaptation='lora' or 'frozen'"
            )
        memo[id(param)] = nn.Parameter(dense)
    block = copy.deepcopy(layer, memo)
    copy_tensor_parallel_forwards(layer, block)
    if layer_idx is not None:
        for module in block.modules():
            if isinstance(getattr(module, "layer_idx", None), int):
                module.layer_idx = layer_idx  # type: ignore[assignment]  # nn.Module.__setattr__ types tensors/modules
    if dtype is not None:
        block = block.to(dtype)
    # copies of a frozen base model inherit requires_grad=False; the copies are trained
    return block.requires_grad_(True)


def _dequantized(param: torch.Tensor) -> torch.Tensor | None:
    plain = type(param) in (nn.Parameter, torch.Tensor)
    if plain and not param.is_quantized:  # e.g. integer codes: their scales live elsewhere
        return None
    if not param.is_quantized and type(param).dequantize is torch.Tensor.dequantize:
        return None  # a subclass without its own dequantization (bitsandbytes parameters: see DENSE_LINEARS)
    try:
        dense = param.dequantize()
    except (RuntimeError, NotImplementedError, TypeError, AttributeError):
        return None
    if type(dense) not in (nn.Parameter, torch.Tensor) or not dense.is_floating_point():
        return None
    return dense.detach().float()


class InModelDtype:
    """Runs a trainable copy of a decoder layer in the model's dtype and returns the model's dtype.

    It has the calling convention of a decoder layer: ``hidden`` followed by the family's positional and keyword
    layer arguments (see ``ModelAdapter.layer_inputs``; Gemma-4 passes its per-layer input positionally).

    A copy stored in the model's dtype runs as it is. A float32 copy's floating-point weights are cast to the model's
    dtype once and the cast is reused until a weight changes (an optimizer step or a loaded state bumps its version),
    so the layer computes exactly what the model's own layer computes from the same weights. Casting on every call
    (as an autocast region per call does) would keep one cast of the weights per call alive for backward: a rollout
    of ``K`` steps through a 31B layer pair would then hold ``K`` bf16 copies of it.
    """

    def __init__(self, block: nn.Module, dtype: torch.dtype) -> None:
        self.block, self.dtype = block, dtype
        self._key: tuple | None = None
        self._cast: dict[str, torch.Tensor] = {}

    def __call__(self, hidden: torch.Tensor, *args: object, **kwargs: object) -> torch.Tensor:
        if self._native():
            out = self.block(hidden, *args, **kwargs)
        else:
            out = functional_call(self.block, self.cast_tensors(), (hidden, *args), kwargs, strict=False)
        out = out[0] if isinstance(out, tuple) else out
        return out.to(self.dtype)

    def _native(self) -> bool:
        weight = next((p for p in self.block.parameters() if p.is_floating_point()), None)
        return weight is None or weight.dtype == self.dtype

    def cast_tensors(self) -> dict[str, torch.Tensor]:
        """The block's floating-point parameters and buffers in the model's dtype, cast again only when one of them
        changed or gradient mode differs from the cached cast's."""
        tensors: dict[str, torch.Tensor] = dict(self.block.named_parameters())
        tensors.update(self.block.named_buffers())
        tensors = {name: t for name, t in tensors.items() if t.is_floating_point()}
        grad = torch.is_grad_enabled()
        local = {name: getattr(t, "_local_tensor", t) for name, t in tensors.items()}  # a DTensor's own shard
        key = (grad, *((name, t.data_ptr(), t._version) for name, t in local.items()))
        if key != self._key:
            self._cast = {name: t.to(self.dtype) for name, t in tensors.items()}
            self._key = key
        return self._cast
