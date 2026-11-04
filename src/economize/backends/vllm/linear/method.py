# ruff: noqa: UP045  (the custom ops' schemas are inferred from typing.Optional annotations)
"""vLLM's side of economize's linear method: the quantization config ``economize``, its linear methods and the ops.

A checkpoint served with the method carries a ``quantization_config`` with one section per technique (see
:mod:`economize.backends.vllm.linear`). vLLM asks the config for the method of every linear layer:

* a layer that no section names keeps vLLM's dense method;
* a layer whose shards TEAL thresholds and ``lowrank`` leaves dense gets :class:`TealLinearMethod`;
* a layer with a factored shard gets :class:`FactoredLinearMethod`, which stores the factors of its factored shards
  and the dense weight of the others, both partitioned for tensor parallelism like vLLM's dense weight.

Both methods compute through custom ops (``torch.ops.vllm.economize_teal_linear`` and
``torch.ops.vllm.economize_factored_linear``), opaque to ``torch.compile``. With TEAL they read the attention metadata
of the running pass to tell decoding rows from the rest, and for ``"second_half"`` the position of a row among its
request's rows (:func:`economize.backends.vllm.linear.teal.sparse_rows`). A CUDA graph freezes that choice at
capture, so when the two row classes differ the engine runs with
``cudagraph_mode="FULL_DECODE_ONLY"``: uniform decoding batches replay graphs captured with decoding metadata, and
every other batch runs eagerly.

This module imports vLLM; :mod:`economize.backends.vllm.linear` imports it from the extension's install step only. It
does not use postponed annotations, since an op's schema is inferred from the annotations of its function.
"""

from dataclasses import dataclass
from typing import Any, Optional

import torch
from torch import Tensor
from vllm.forward_context import get_forward_context, is_forward_context_available
from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    LinearBase,
    LinearMethodBase,
    MergedColumnParallelLinear,
    QKVParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
    UnquantizedLinearMethod,
)
from vllm.model_executor.layers.quantization import register_quantization_config
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig, QuantizeMethodBase
from vllm.model_executor.layers.quantization.kv_cache import BaseKVCacheMethod
from vllm.model_executor.utils import set_weight_attrs
from vllm.utils.torch_utils import direct_register_custom_op

from ....errors import ConfigError
from ...exporters import UnsupportedBackendError
from . import FACTORED_OP, QUANT_METHOD, TEAL_OP
from .factored import Unit, factored_linear, layer_units
from .teal import ALL, DENSE, SECOND_HALF, row_mode, sparse_rows, teal_linear

__all__ = ["EconomizeLinearConfig", "FactoredLinearMethod", "TealLinearMethod", "factored_linear_op", "teal_linear_op"]


def _runtime_rows(rows: int, single: int, multi: int) -> "bool | Tensor":
    """The rows of the running pass to threshold (see :func:`~economize.backends.vllm.linear.teal.sparse_rows`)."""
    single = ALL if single == SECOND_HALF else single  # a one-token request's row is its own last half
    if single == multi:
        return single == ALL
    if not is_forward_context_available():
        return multi != DENSE
    context = get_forward_context()
    mode = getattr(context, "cudagraph_runtime_mode", None)
    if mode is not None and getattr(mode, "name", "") == "PIECEWISE":
        raise UnsupportedBackendError(
            "TEAL with different modes for decoding and prefilling rows (or 'second_half') cannot run inside "
            "piecewise CUDA graphs, which freeze the choice of rows at capture; serve it with "
            "compilation_config={'cudagraph_mode': "
            "'FULL_DECODE_ONLY'} (economize's plan sets it) or with enforce_eager=True")
    return sparse_rows(context.attn_metadata, rows, single, multi)


def _rows_of(x: Tensor) -> int:
    return x.numel() // x.shape[-1] if x.shape[-1] else 0


def teal_linear_op(x: Tensor, weight: Tensor, bias: Optional[Tensor], thresholds: Tensor, sizes: list[int],
                   single: int, multi: int) -> Tensor:
    """The thresholded dense map (:func:`~economize.backends.vllm.linear.teal.teal_linear`) of the running pass."""
    return teal_linear(x, weight, bias, thresholds, sizes, _runtime_rows(_rows_of(x), single, multi))


def _teal_linear_fake(x: Tensor, weight: Tensor, bias: Optional[Tensor], thresholds: Tensor, sizes: list[int],
                      single: int, multi: int) -> Tensor:
    return x.new_empty((*x.shape[:-1], weight.shape[0]))


def factored_linear_op(x: Tensor, bias: Optional[Tensor], weight: Optional[Tensor], dense_thresholds: Tensor,
                       down: Tensor, down_thresholds: Tensor, up: Tensor, up_thresholds: Tensor, sizes: list[int],
                       ranks: list[int], single: int, multi: int) -> Tensor:
    """A layer with factored units (:func:`~economize.backends.vllm.linear.factored.factored_linear`)."""
    return factored_linear(x, bias, weight, dense_thresholds, down, down_thresholds, up, up_thresholds, sizes, ranks,
                           _runtime_rows(_rows_of(x), single, multi))


def _factored_linear_fake(x: Tensor, bias: Optional[Tensor], weight: Optional[Tensor], dense_thresholds: Tensor,
                          down: Tensor, down_thresholds: Tensor, up: Tensor, up_thresholds: Tensor, sizes: list[int],
                          ranks: list[int], single: int, multi: int) -> Tensor:
    return x.new_empty((*x.shape[:-1], sum(sizes)))


if not hasattr(torch.ops.vllm, TEAL_OP):
    direct_register_custom_op(TEAL_OP, teal_linear_op, fake_impl=_teal_linear_fake)
if not hasattr(torch.ops.vllm, FACTORED_OP):
    direct_register_custom_op(FACTORED_OP, factored_linear_op, fake_impl=_factored_linear_fake)


def _column_major(param: Tensor) -> None:
    """Store a CUDA weight column-major, the layout of TEAL's sparse GEMV (cuBLAS is as fast with it)."""
    if param.is_cuda and param.dim() == 2 and param.stride(0) != 1:
        param.data = param.data.t().contiguous().t()


class TealLinearMethod(UnquantizedLinearMethod):
    """A dense linear map whose input is thresholded per output shard, on the rows of the configured row classes.

    The weight is vLLM's dense weight; on CUDA it is stored column-major after loading.
    """

    def __init__(self, thresholds: list[float], single: int, multi: int) -> None:
        super().__init__()
        self.thresholds = thresholds
        self.single = single
        self.multi = multi

    def create_weights(self, layer: torch.nn.Module, input_size_per_partition: int,
                       output_partition_sizes: list[int], input_size: int, output_size: int,
                       params_dtype: torch.dtype, **extra_weight_attrs: Any) -> None:
        super().create_weights(layer, input_size_per_partition, output_partition_sizes, input_size, output_size,
                               params_dtype, **extra_weight_attrs)
        sizes = [int(size) for size in output_partition_sizes]
        thresholds = list(self.thresholds)
        if len(thresholds) != len(sizes):
            if len(thresholds) != 1:
                raise UnsupportedBackendError(
                    f"TEAL has {len(thresholds)} thresholds for a linear map with {len(sizes)} output shards")
            thresholds = thresholds * len(sizes)
        if len(set(thresholds)) == 1:
            sizes, thresholds = [sum(sizes)], thresholds[:1]
        layer.teal_sizes = sizes
        layer.register_buffer("teal_thresholds", torch.tensor(thresholds, dtype=torch.float32), persistent=False)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        # The dense method packs CPU weights into oneDNN's layout (and drops layer.weight); TEAL keeps the plain
        # weight on every platform.
        _column_major(layer.weight)
        layer.teal_thresholds = layer.teal_thresholds.to(layer.weight.device)

    def apply(self, layer: torch.nn.Module, x: Tensor, bias: Optional[Tensor] = None) -> Tensor:
        return getattr(torch.ops.vllm, TEAL_OP)(x, layer.weight, bias, layer.teal_thresholds, layer.teal_sizes,
                                                self.single, self.multi)


@dataclass(frozen=True)
class _Partition:
    """The rows of one output shard of a layer that this tensor-parallel rank holds."""

    local: int
    full: int
    start: int


#: The layer types whose partitioning :class:`FactoredLinearMethod` reproduces (subclasses may shard differently).
_LAYERS = (ReplicatedLinear, ColumnParallelLinear, MergedColumnParallelLinear, QKVParallelLinear, RowParallelLinear)

#: Shard ids of :class:`QKVParallelLinear`'s loader.
_QKV = {"q": 0, "k": 1, "v": 2}


def _partitions(layer: torch.nn.Module, local_sizes: list[int]) -> list[_Partition]:
    """The partition of every output shard of ``layer`` on this rank (the full HF map's rows it holds)."""
    kind = type(layer)
    if kind not in _LAYERS:
        raise UnsupportedBackendError(
            f"economize's factored maps on vLLM support {', '.join(t.__name__ for t in _LAYERS)}; "
            f"{layer.prefix!r} is a {kind.__name__}")
    if kind in (ReplicatedLinear, RowParallelLinear):
        return [_Partition(size, size, 0) for size in local_sizes]
    rank = int(layer.tp_rank)
    if kind is QKVParallelLinear:
        kv_rank = rank // int(layer.num_kv_head_replicas)
        totals = [layer.total_num_heads * layer.head_size, layer.total_num_kv_heads * layer.head_size,
                  layer.total_num_kv_heads * layer.v_head_size]
        starts = [rank * local_sizes[0], kv_rank * local_sizes[1], kv_rank * local_sizes[2]]
        return [_Partition(local, int(full), start)
                for local, full, start in zip(local_sizes, totals, starts, strict=True)]
    totals = list(layer.output_sizes) if kind is MergedColumnParallelLinear else [layer.output_size]
    return [_Partition(local, int(full), rank * local) for local, full in zip(local_sizes, totals, strict=True)]


def _gather(tensor: Tensor, partitions: list[_Partition], name: str) -> Tensor:
    """This rank's rows of a unit's full tensor, which spans ``partitions`` (in output order)."""
    if tensor.shape[0] != sum(part.full for part in partitions):
        raise ConfigError(f"{name}: {tensor.shape[0]} rows in the checkpoint, the layer expects "
                          f"{sum(part.full for part in partitions)}")
    pieces, offset = [], 0
    for part in partitions:
        pieces.append(tensor.narrow(0, offset + part.start, part.local))
        offset += part.full
    return pieces[0] if len(pieces) == 1 else torch.cat(pieces)


class FactoredLinearMethod(LinearMethodBase):
    """A linear layer with factored units (``up(down(x))``) and possibly dense ones, TEAL's masks optional.

    Parameters on the layer (local to its tensor-parallel rank, see :mod:`economize.backends.vllm.linear.factored`):
    ``lowrank_down``, ``lowrank_up`` and, with dense units, ``weight``. Each has its own weight loader, which takes
    the full HF tensor of one unit (shard id ``"q"``/``"k"``/``"v"``, an int, or ``None`` for the only unit) and
    keeps this rank's part. vLLM's bias parameter and loader stay as they are.
    """

    def __init__(self, units: list[Unit], single: int, multi: int) -> None:
        super().__init__()
        self.units = units
        sparse = any(unit.threshold > 0 or unit.factor_threshold > 0 for unit in units)
        self.single = single if sparse else DENSE
        self.multi = multi if sparse else DENSE

    def create_weights(self, layer: torch.nn.Module, input_size_per_partition: int,
                       output_partition_sizes: list[int], input_size: int, output_size: int,
                       params_dtype: torch.dtype, **extra_weight_attrs: Any) -> None:
        local_sizes = [int(size) for size in output_partition_sizes]
        partitions = _partitions(layer, local_sizes)
        if len(self.units) == len(partitions):
            spans = [[part] for part in partitions]
        elif len(self.units) == 1:
            spans = [partitions]
        else:
            raise UnsupportedBackendError(
                f"{layer.prefix!r}: {len(self.units)} factored or dense maps for {len(partitions)} output shards")
        row_parallel = type(layer) is RowParallelLinear
        if row_parallel and layer.tp_size > 1 and any(unit.factor_threshold > 0 for unit in self.units):
            raise UnsupportedBackendError(
                f"{layer.prefix!r}: TEAL on the inner factor of a row-parallel map needs the full latent, which "
                "tensor parallelism splits into partial sums; serve with tensor_parallel_size=1, or calibrate TEAL "
                "without the 'up' factors of row-parallel maps (o_proj, down_proj)")
        sizes = [sum(part.local for part in span) for span in spans]
        ranks = [unit.rank for unit in self.units]
        dense = sum(size for size, rank in zip(sizes, ranks, strict=True) if rank == 0)
        layer.factored_sizes, layer.factored_ranks = sizes, ranks
        layer.factored_spans = spans
        layer.factored_input = ((layer.tp_rank * input_size_per_partition, input_size_per_partition)
                                if row_parallel else None)
        layer.factored_loaded = set()

        def parameter(name: str, shape: tuple[int, ...]) -> None:
            param = torch.nn.Parameter(torch.empty(shape, dtype=params_dtype), requires_grad=False)
            layer.register_parameter(name, param)
            set_weight_attrs(param, {"weight_loader": self._loader(layer, name)})

        parameter("lowrank_down", (sum(ranks), input_size_per_partition))
        parameter("lowrank_up", (sum(rank * size for rank, size in zip(ranks, sizes, strict=True)),))
        if dense:
            parameter("weight", (dense, input_size_per_partition))
        else:
            layer.register_parameter("weight", None)
        thresholds = {
            "dense_thresholds": [unit.threshold for unit in self.units if not unit.factored],
            "down_thresholds": [unit.threshold for unit in self.units if unit.factored],
            "up_thresholds": [unit.factor_threshold for unit in self.units if unit.factored],
        }
        for name, values in thresholds.items():
            layer.register_buffer(f"factored_{name}", torch.tensor(values, dtype=torch.float32), persistent=False)

    def _unit(self, layer: torch.nn.Module, shard_id: Any) -> int:
        if shard_id is None:
            if len(self.units) != 1:
                raise UnsupportedBackendError(
                    f"{layer.prefix!r}: a fused checkpoint tensor for a layer whose shards are factored separately")
            return 0
        if isinstance(shard_id, str) and shard_id in _QKV:
            index = _QKV[shard_id]
        elif isinstance(shard_id, int) and not isinstance(shard_id, bool):
            index = shard_id
        else:
            raise UnsupportedBackendError(f"{layer.prefix!r}: shard id {shard_id!r} is not supported")
        if not 0 <= index < len(self.units):
            raise ConfigError(f"{layer.prefix!r}: shard {shard_id!r} out of range")
        return index

    def _loader(self, layer: torch.nn.Module, kind: str) -> Any:
        def load(param: Tensor, loaded: Tensor, shard_id: Any = None) -> None:
            index = self._unit(layer, shard_id)
            unit = self.units[index]
            name = f"{layer.prefix}.{kind}[{unit.name}]"
            if (kind == "weight") == unit.factored:
                raise ConfigError(f"{name}: the checkpoint holds a {kind} tensor for a "
                                  f"{'factored' if unit.factored else 'dense'} map")
            if kind != "lowrank_down":
                loaded = _gather(loaded, layer.factored_spans[index], name)
            if kind != "lowrank_up" and layer.factored_input is not None:
                start, size = layer.factored_input
                loaded = loaded.narrow(1, start, size)
            ranks, sizes = layer.factored_ranks, layer.factored_sizes
            if kind == "lowrank_down":
                offset = sum(ranks[:index])
                target = param.data[offset:offset + unit.rank]
            elif kind == "lowrank_up":
                offset = sum(rank * size for rank, size in zip(ranks[:index], sizes[:index], strict=True))
                target = param.data[offset:offset + unit.rank * sizes[index]].view(unit.rank, sizes[index])
                loaded = loaded.t()
            else:
                offset = sum(size for rank, size in zip(ranks[:index], sizes[:index], strict=True) if rank == 0)
                target = param.data[offset:offset + sizes[index]]
            if tuple(target.shape) != tuple(loaded.shape):
                raise ConfigError(f"{name}: shape {tuple(loaded.shape)} in the checkpoint, the layer expects "
                                  f"{tuple(target.shape)}")
            target.copy_(loaded)
            layer.factored_loaded.add((kind, index))

        return load

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        loaded = layer.factored_loaded
        if loaded:  # nothing loaded: dummy weights (load_format="dummy") or a test
            expected = {(kind, index) for index, unit in enumerate(self.units)
                        for kind in (("lowrank_down", "lowrank_up") if unit.factored else ("weight",))}
            if missing := sorted(expected - loaded):
                raise ConfigError(f"{layer.prefix!r}: the checkpoint lacks "
                                  + ", ".join(f"{kind} of {self.units[index].name}" for kind, index in missing))
        _column_major(layer.lowrank_down)
        if layer.weight is not None:
            _column_major(layer.weight)
        device = layer.lowrank_down.device
        for name in ("dense_thresholds", "down_thresholds", "up_thresholds"):
            setattr(layer, f"factored_{name}", getattr(layer, f"factored_{name}").to(device))

    def apply(self, layer: torch.nn.Module, x: Tensor, bias: Optional[Tensor] = None) -> Tensor:
        return getattr(torch.ops.vllm, FACTORED_OP)(
            x, bias, layer.weight, layer.factored_dense_thresholds, layer.lowrank_down,
            layer.factored_down_thresholds, layer.lowrank_up, layer.factored_up_thresholds, layer.factored_sizes,
            layer.factored_ranks, self.single, self.multi)


def _floats(table: Any) -> dict[str, float]:
    return {str(name): float(value) for name, value in dict(table or {}).items()}


@register_quantization_config(QUANT_METHOD)
class EconomizeLinearConfig(QuantizationConfig):
    """The ``economize`` quantization config: TEAL's thresholds and row classes, ``lowrank``'s ranks, KV scales."""

    def __init__(self, thresholds: Optional[dict[str, float]] = None,
                 factor_thresholds: Optional[dict[str, float]] = None, single: str = "dense", multi: str = "dense",
                 ranks: Optional[dict[str, int]] = None, kv_cache_scheme: Optional[dict[str, Any]] = None) -> None:
        super().__init__()
        self.thresholds = _floats(thresholds)
        self.factor_thresholds = _floats(factor_thresholds)
        self.ranks = {str(name): int(rank) for name, rank in dict(ranks or {}).items()}
        self.single_mode, self.multi_mode = single, multi
        self.single, self.multi = row_mode(single), row_mode(multi)
        self.kv_cache_scheme = kv_cache_scheme
        #: The vLLM module names that got economize's methods, with the method's kind ("teal" or "factored").
        self.matched: dict[str, str] = {}

    def __repr__(self) -> str:
        return (f"EconomizeLinearConfig({len(self.thresholds)} thresholds, {len(self.factor_thresholds)} factor "
                f"thresholds, {len(self.ranks)} ranks, single={self.single_mode!r}, multi={self.multi_mode!r}, "
                f"kv_cache_scheme={self.kv_cache_scheme is not None})")

    @classmethod
    def get_name(cls) -> str:
        return QUANT_METHOD

    @classmethod
    def get_supported_act_dtypes(cls) -> list[torch.dtype]:
        return [torch.bfloat16, torch.float16, torch.float32]

    @classmethod
    def get_min_capability(cls) -> int:
        return 70

    @staticmethod
    def get_config_filenames() -> list[str]:
        return []

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "EconomizeLinearConfig":
        teal = dict(config.get("teal") or {})
        lowrank = dict(config.get("lowrank") or {})
        default = "all" if teal else "dense"
        return cls(teal.get("thresholds"), teal.get("factor_thresholds"), str(teal.get("single", default)),
                   str(teal.get("multi", "dense")), lowrank.get("ranks"), config.get("kv_cache_scheme"))

    def apply_vllm_mapper(self, hf_to_vllm_mapper: Any) -> None:
        self.thresholds = hf_to_vllm_mapper.apply_dict(self.thresholds)
        self.factor_thresholds = hf_to_vllm_mapper.apply_dict(self.factor_thresholds)
        self.ranks = hf_to_vllm_mapper.apply_dict(self.ranks)

    def get_quant_method(self, layer: torch.nn.Module, prefix: str) -> Optional[QuantizeMethodBase]:
        if isinstance(layer, Attention):
            return BaseKVCacheMethod(self) if self.kv_cache_scheme is not None else None
        if not isinstance(layer, LinearBase):
            return None
        units = layer_units(prefix, self.packed_modules_mapping, self.ranks, self.thresholds, self.factor_thresholds)
        if units is None:
            return UnquantizedLinearMethod()
        if any(unit.factored for unit in units):
            self.matched[prefix] = "factored"
            return FactoredLinearMethod(units, self.single, self.multi)
        thresholds = [unit.threshold for unit in units]
        if not any(value > 0 for value in thresholds):
            return UnquantizedLinearMethod()
        self.matched[prefix] = "teal"
        return TealLinearMethod(thresholds, self.single, self.multi)
