"""Layer-by-layer calibration shared by the weight and activation techniques.

GPTQ (Frantar et al., 2023), SparseGPT (Frantar & Alistarh, 2023), Wanda (Sun et al., 2024), AWQ (Lin et al., 2024)
and SVD-LLM (Wang et al., 2025) all calibrate one decoder layer at a time. The inputs of layer 1 are captured on the
calibration tokens, statistics of the inputs of each linear map of the layer are collected, the layer is transformed,
and the batch is propagated to the next layer. Only one layer's activations are alive at a time, so a model whose
layers are spread over devices with a device map calibrates without ever materializing a full forward.

The engine knows nothing about pruning or quantization. Techniques use three pieces:

* :func:`token_batches` turns the calibration source of an ``AttachContext`` into token batches.
* :class:`LayerwiseCalibration` yields one :class:`CalibrationStep` per decoder layer.
* Observers (:class:`GramObserver`, :class:`MagnitudeSampleObserver` and the :class:`FeatureMeanObserver` subclasses
  :class:`SquaredNormObserver` and :class:`AbsMeanObserver`) collect input statistics through forward pre-hooks
  installed by :func:`observing`.

The engine lives in the core so that any technique package can use it; ``economize.techniques.prune.calibration``
re-exports it.
"""

from __future__ import annotations

import fnmatch
import logging
from abc import ABC, abstractmethod
from collections.abc import Callable, Hashable, Iterable, Iterator, Mapping, MutableMapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal, Protocol, TypeVar, cast

import torch
from torch import Tensor, nn

from ..errors import ConfigError

if TYPE_CHECKING:
    from ..arch import ArchAdapter

__all__ = [
    "AbsMeanObserver",
    "CalibrationStep",
    "FeatureMeanObserver",
    "GramObserver",
    "HasCalibration",
    "LayerCall",
    "LayerwiseCalibration",
    "MagnitudeSampleObserver",
    "Observer",
    "Propagation",
    "SquaredNormObserver",
    "calibration_source",
    "capture_inputs",
    "layer_device",
    "layer_types",
    "linear_modules",
    "observing",
    "resolve_layers",
    "select_modules",
    "shared_input_groups",
    "token_batches",
]

logger = logging.getLogger(__name__)

Propagation = Literal["transformed", "original"]
O = TypeVar("O", bound="Observer")
M = TypeVar("M", bound=nn.Module)


# ---- calibration data ----------------------------------------------------------------------------------------------


class HasCalibration(Protocol):
    """What :func:`calibration_source` reads of an ``AttachContext``."""

    @property
    def calibration(self) -> object: ...


def calibration_source(context: HasCalibration | None, technique: str) -> object:
    """The calibration data of an ``AttachContext``.

    Raise :class:`ConfigError` naming ``technique`` if it is missing.
    """
    source = context.calibration if context is not None else None
    if source is None:
        raise ConfigError(
            f"technique {technique!r} needs calibration data: pass calibration token batches to economize.apply "
            "(e.g. 128 sequences of 2048 tokens from C4 or WikiText-2, as in the paper)"
        )
    return source


def token_batches(source: object) -> list[Tensor]:
    """Normalize calibration data to a list of ``LongTensor [B, S]`` batches without padding.

    ``source`` may be a callable returning the data, a ``LongTensor`` (``[S]`` is one sequence, ``[N, S]`` is
    split into ``N`` batches of one sequence, as in the reference implementations), a mapping with ``input_ids``
    and optionally ``attention_mask`` (a tokenizer's output, one batch), or an iterable of tensors and mappings.

    A batch whose ``attention_mask`` marks padding (left or right) is split into one batch per sequence with the
    padding removed, so padding never reaches an observer and every architecture (also routed experts and linear
    attention, which have no notion of a padding mask in their statistics) calibrates exactly as on the unpadded
    sequences. A batch without padding stays one batch.

    Raises:
        ConfigError: The data is empty, not integer tokens, or a mask has a hole inside a sequence.
    """
    if callable(source) and not isinstance(source, Tensor):
        source = source()
    if isinstance(source, Tensor):
        source = [source] if source.dim() == 1 else list(source.unbind(0))
    elif isinstance(source, Mapping):
        source = [source]
    if not isinstance(source, Iterable):
        raise ConfigError(f"calibration data must be token tensors, got {type(source).__name__}")
    batches: list[Tensor] = []
    for batch in source:
        if isinstance(batch, Mapping):
            batches.extend(_unpadded(batch))
            continue
        batches.append(_token_tensor(batch))
    if not batches:
        raise ConfigError("calibration data is empty")
    return batches


def _token_tensor(batch: object) -> Tensor:
    if not isinstance(batch, Tensor) or batch.dtype not in (torch.int32, torch.int64):
        raise ConfigError(f"calibration batches must be integer token tensors, got {type(batch).__name__}")
    return batch.long().view(1, -1) if batch.dim() == 1 else batch.long()


def _unpadded(batch: Mapping[str, Any]) -> list[Tensor]:
    """The sequences of a tokenizer batch (``input_ids``, ``attention_mask``) without their padding."""
    if "input_ids" not in batch:
        raise ConfigError(f"a calibration mapping needs 'input_ids', got keys {sorted(batch)}")
    ids = _token_tensor(batch["input_ids"])
    mask = batch.get("attention_mask")
    if mask is None:
        return [ids]
    mask = torch.as_tensor(mask).reshape(ids.shape).bool()
    if bool(mask.all()):
        return [ids]
    rows = []
    for row in range(ids.shape[0]):
        kept = mask[row].nonzero().flatten()
        if kept.numel() == 0:
            continue
        first, last = int(kept[0]), int(kept[-1])
        if last - first + 1 != kept.numel():
            raise ConfigError(
                f"calibration attention_mask row {row} is not contiguous: only left or right padding is supported"
            )
        rows.append(ids[row, first : last + 1].view(1, -1))
    return rows


def _as_token_batches(batches: Sequence[object]) -> list[Tensor]:
    """``batches`` unchanged if they are all tensors, else normalized by :func:`token_batches`."""
    items = list(batches)
    return items if all(isinstance(b, Tensor) for b in items) else token_batches(items)  # type: ignore[return-value]


# ---- module selection ----------------------------------------------------------------------------------------------


def linear_modules(adapter: ArchAdapter, layer: nn.Module) -> dict[str, nn.Linear]:
    """Every ``nn.Linear`` of one decoder layer, keyed by its dotted name inside the layer
    (:meth:`ArchAdapter.linear_modules`; a family adapter may override what counts)."""
    return adapter.linear_modules(layer)


def select_modules(
    modules: Mapping[str, nn.Module], targets: Sequence[str], exclude: Sequence[str] = ()
) -> dict[str, nn.Module]:
    """The modules whose name matches one of the glob patterns ``targets`` and none of ``exclude``."""
    return {
        name: module
        for name, module in modules.items()
        if any(fnmatch.fnmatchcase(name, p) for p in targets) and not any(fnmatch.fnmatchcase(name, p) for p in exclude)
    }


def resolve_layers(layers: Sequence[int] | None, num_layers: int) -> list[int]:
    """The 1-indexed layers a config selects: ``layers`` in the given order, or every layer when it is empty (or
    ``None``)."""
    return list(layers or ()) or list(range(1, num_layers + 1))


# ---- observers -----------------------------------------------------------------------------------------------------


class Observer(Protocol):
    """Accumulates a statistic of the inputs ``x [..., in_features]`` of one module."""

    def update(self, x: Tensor) -> None: ...


@contextmanager
def observing(modules: Mapping[str, M], factory: Callable[[M], O]) -> Iterator[dict[str, O]]:
    """Attach one observer per module (built by ``factory``) to the module's input while the context is open."""
    observers = {name: factory(module) for name, module in modules.items()}
    handles = []
    for name, module in modules.items():
        observer = observers[name]

        def hook(_module: nn.Module, args: tuple[Any, ...], observer: Observer = observer) -> None:
            observer.update(args[0].detach())

        handles.append(module.register_forward_pre_hook(hook))
    try:
        yield observers
    finally:
        for handle in handles:
            handle.remove()


def _flat(x: Tensor) -> Tensor:
    return x.reshape(-1, x.shape[-1])


class GramObserver:
    """Running ``sum x x^T`` over tokens (the layer-wise Hessian ``H = 2 X X^T`` of GPTQ and SparseGPT, up to a factor
    that the algorithms are invariant to, and the whitening statistic of SVD-LLM)."""

    def __init__(self, module: nn.Module, dtype: torch.dtype = torch.float32) -> None:
        weight = cast(Tensor, module.weight)
        self.sum = torch.zeros(weight.shape[1], weight.shape[1], dtype=dtype, device=weight.device)
        self.tokens = 0

    def update(self, x: Tensor) -> None:
        flat = _flat(x).to(device=self.sum.device, dtype=self.sum.dtype)
        self.sum.addmm_(flat.T, flat)
        self.tokens += flat.shape[0]

    @property
    def mean(self) -> Tensor:
        """``E[x x^T]`` over the observed tokens."""
        return self.sum / max(self.tokens, 1)

    def mean_(self) -> Tensor:
        """:attr:`mean` computed in place: the running sum becomes the mean (call once, after the last update)."""
        return self.sum.div_(max(self.tokens, 1))


class FeatureMeanObserver(ABC):
    """Per input feature mean of ``reduce(x)`` over tokens, for an elementwise :meth:`reduce`."""

    def __init__(self, module: nn.Module) -> None:
        weight = cast(Tensor, module.weight)
        self.sum = torch.zeros(weight.shape[1], dtype=torch.float32, device=weight.device)
        self.tokens = 0

    @staticmethod
    @abstractmethod
    def reduce(x: Tensor) -> Tensor:
        """The elementwise statistic averaged per feature."""

    def update(self, x: Tensor) -> None:
        flat = _flat(x).to(device=self.sum.device, dtype=torch.float32)
        self.sum += self.reduce(flat).sum(0)
        self.tokens += flat.shape[0]

    @property
    def mean(self) -> Tensor:
        """The per-feature mean over the observed tokens."""
        return self.sum / max(self.tokens, 1)


class SquaredNormObserver(FeatureMeanObserver):
    """Per input feature ``sum_t x_t^2 / tokens``: the squared column norm ``||X_j||^2`` of Wanda, up to a constant."""

    @staticmethod
    def reduce(x: Tensor) -> Tensor:
        return x.square()


class AbsMeanObserver(FeatureMeanObserver):
    """Per input feature mean ``|x|`` over tokens: the activation scale AWQ searches over."""

    @staticmethod
    def reduce(x: Tensor) -> Tensor:
        return x.abs()


class MagnitudeSampleObserver:
    """A uniform random subsample of input magnitudes ``|x|`` (at most ``max_samples`` values), for quantiles.

    TEAL sets each threshold to a quantile of the dense activation magnitudes. The subsample keeps memory bounded
    on long calibration sets; with ``2^20`` samples the quantile error is about ``1e-3`` in probability.
    """

    def __init__(self, module: nn.Module | None = None, max_samples: int = 1 << 20, seed: int = 0) -> None:
        self.max_samples = max_samples
        self.samples: list[Tensor] = []
        self.count = 0
        self.generator = torch.Generator().manual_seed(seed)

    def update(self, x: Tensor) -> None:
        flat = x.detach().reshape(-1).abs().float()
        per_call = max(1, self.max_samples // 16)
        if flat.numel() > per_call:
            index = torch.randint(0, flat.numel(), (per_call,), generator=self.generator).to(flat.device)
            flat = flat[index]
        self.samples.append(flat.cpu())
        self.count += flat.numel()
        if self.count > 2 * self.max_samples:
            merged = torch.cat(self.samples)
            keep = torch.randperm(merged.numel(), generator=self.generator)[: self.max_samples]
            self.samples, self.count = [merged[keep]], self.max_samples

    def values(self) -> Tensor:
        return torch.cat(self.samples) if self.samples else torch.zeros(0)

    def quantile(self, q: float) -> float:
        """The ``q``-quantile of the observed magnitudes (lower interpolation)."""
        values = self.values()
        if values.numel() == 0:
            return 0.0
        k = min(values.numel(), max(1, round(q * values.numel())))
        return float(values.kthvalue(k).values) if q > 0 else 0.0


# ---- layer-wise engine ---------------------------------------------------------------------------------------------


@dataclass
class LayerCall:
    """Positional (after the hidden state) and keyword arguments of one decoder-layer call."""

    args: tuple[Any, ...] = ()
    kwargs: dict[str, Any] = field(default_factory=dict)

    def to(self, device: torch.device) -> LayerCall:
        return LayerCall(_to_device(self.args, device), _to_device(self.kwargs, device))


def _to_device(obj: Any, device: torch.device) -> Any:
    if isinstance(obj, Tensor):
        return obj.to(device)
    if isinstance(obj, tuple):
        return tuple(_to_device(o, device) for o in obj)
    if isinstance(obj, list):
        return [_to_device(o, device) for o in obj]
    if isinstance(obj, dict):
        return {k: _to_device(v, device) for k, v in obj.items()}
    return obj


def _move_layer_state(call: LayerCall, device: torch.device) -> None:
    """Move, in place, the tensors that layers leave in the mutable mappings of their call arguments for later layers
    (Gemma-4's ``shared_kv_states``: the K/V of a model's last layer of each attention type, one entry per batch).
    With offloading they wait on the CPU like the activations and come back for the layer call."""
    for value in call.kwargs.values():
        if isinstance(value, MutableMapping):
            for key in list(value):
                value[key] = _to_device(value[key], device)


def _first_output(output: Any) -> Tensor:
    return output[0] if isinstance(output, (tuple, list)) else output


def layer_types(adapter: ArchAdapter) -> list[Hashable]:
    """The key under which :func:`capture_inputs` stores the call arguments of each layer, by
    :meth:`ArchAdapter.layer_call_key`.

    Layers with one key receive the same mask and rotary embeddings, so their call arguments are captured once: the
    default key is the attention type (``config.layer_types``), or ``None`` for every layer of a uniform model, and
    layers that take a per-layer input of their own (Gemma-4's per-layer embeddings) get one key each.
    """
    return [adapter.layer_call_key(index) for index in range(1, adapter.num_layers + 1)]


def layer_device(adapter: ArchAdapter, index: int) -> torch.device:
    """Device of layer ``index`` (1-indexed), by :meth:`ArchAdapter.layer_device`."""
    return adapter.layer_device(index)


class _Captured(Exception):
    """Stops the model forward once every needed layer input is captured."""


@torch.no_grad()
def capture_inputs(
    adapter: ArchAdapter, batches: Sequence[Tensor], *, offload: bool = False
) -> tuple[list[Tensor], list[dict[object, LayerCall]]]:
    """Inputs of layer 1 and the call arguments of every layer key (see :func:`layer_types`), one entry per batch.

    The model runs with ``use_cache=False`` and stops at the first layer of the last key. Padded tokenizer batches
    are split into their sequences first (see :func:`token_batches`).
    """
    batches = _as_token_batches(batches)
    types = layer_types(adapter)
    first: dict[object, int] = {}
    for i, kind in enumerate(types):
        first.setdefault(kind, i)
    stop_at = max(first.values())
    hidden: list[Tensor] = []
    calls: list[dict[object, LayerCall]] = []

    def make_hook(i: int) -> Callable[..., None]:
        def hook(_module: nn.Module, args: tuple[Any, ...], kwargs: dict[str, Any]) -> None:
            state = args[0] if args else kwargs["hidden_states"]
            rest = {k: v for k, v in kwargs.items() if k != "hidden_states"}
            if i == 0:
                hidden.append(state.detach().cpu() if offload else state.detach())
            calls[-1][types[i]] = LayerCall(tuple(args[1:]), rest)
            if i == stop_at:
                raise _Captured

        return hook

    layers = adapter.layers
    handles = [layers[i].register_forward_pre_hook(make_hook(i), with_kwargs=True) for i in sorted(set(first.values()))]
    try:
        for batch in batches:
            calls.append({})
            try:
                adapter.model(input_ids=batch.to(adapter.device), use_cache=False)
            except _Captured:
                pass
            else:  # pragma: no cover - a model that never reaches its layers
                raise RuntimeError("the model forward did not reach the decoder layers")
    finally:
        for handle in handles:
            handle.remove()
    return hidden, calls


@dataclass
class CalibrationStep:
    """One decoder layer during calibration: its index, module and the inputs of every calibration batch.

    :attr:`first_outputs` holds the untransformed layer's outputs. With ``propagate="original"`` the engine records
    them before the step is yielded, so they are right even if the caller transforms the layer first.
    """

    index: int
    layer: nn.Module
    inputs: list[Tensor]
    calls: list[LayerCall]
    device: torch.device
    offload: bool = False
    first_outputs: list[Tensor] | None = None

    @torch.no_grad()
    def forward(self, batches: Iterable[int] | None = None) -> list[Tensor]:
        """Run the layer on the calibration inputs (all batches, or the given batch indices) and return the outputs.

        Observers attached with :func:`observing` see the inputs of the layer's modules during this call. Unless
        :class:`LayerwiseCalibration` recorded them before yielding the step, the outputs of the first full call are
        kept as :attr:`first_outputs`.
        """
        indices = range(len(self.inputs)) if batches is None else list(batches)
        outputs = []
        for b in indices:
            call = self.calls[b]
            if self.offload:
                _move_layer_state(call, self.device)
            out = _first_output(self.layer(self.inputs[b].to(self.device), *call.args, **call.kwargs))
            if self.offload:
                _move_layer_state(call, torch.device("cpu"))
            outputs.append(out.cpu() if self.offload else out)
        if batches is None and self.first_outputs is None:
            self.first_outputs = outputs
        return outputs

    def run_module(self, module: nn.Module, inputs: Tensor, batch: int, *, with_layer_kwargs: bool) -> Tensor:
        """Call a submodule of the layer (AWQ inspects attention blocks with the layer's call arguments)."""
        kwargs = self.calls[batch].kwargs if with_layer_kwargs else {}
        return _first_output(module(inputs, **kwargs))


class LayerwiseCalibration:
    """Iterate over decoder layers with their calibration inputs, propagating the batch between layers.

    Args:
        adapter: The architecture adapter of the model being calibrated.
        batches: Calibration token batches (see :func:`token_batches`; padded tokenizer batches are split into their
            sequences).
        propagate: ``"transformed"`` feeds the next layer the outputs of the transformed layer (GPTQ, SparseGPT and
            Wanda), ``"original"`` the outputs of the untransformed layer (AWQ, SVD-LLM and TEAL profile the original
            model). For ``"original"`` the engine runs each layer once before yielding its step and keeps the outputs
            as :attr:`CalibrationStep.first_outputs`, so a caller's own full forward is one extra layer pass.
        layers: 1-indexed layers to yield (default: all). Iteration stops after the last one.
        offload: Keep the captured activations on the CPU between layers (for models larger than one device).
    """

    def __init__(
        self,
        adapter: ArchAdapter,
        batches: Sequence[Tensor],
        *,
        propagate: Propagation = "transformed",
        layers: Sequence[int] | None = None,
        offload: bool = False,
    ) -> None:
        if propagate not in ("transformed", "original"):
            raise ConfigError(f"propagate must be 'transformed' or 'original', got {propagate!r}")
        self.adapter = adapter
        self.batches = _as_token_batches(batches)
        self.propagate = propagate
        self.layers = sorted(set(resolve_layers(layers, adapter.num_layers)))
        self.offload = offload

    def __iter__(self) -> Iterator[CalibrationStep]:
        if not self.layers:
            return
        hidden, calls = capture_inputs(self.adapter, self.batches, offload=self.offload)
        types = layer_types(self.adapter)
        wanted = set(self.layers)
        for index in range(1, self.layers[-1] + 1):
            step = CalibrationStep(
                index=index,
                layer=self.adapter.layer(index),
                inputs=hidden,
                calls=[c[types[index - 1]].to(self.adapter.layer_device(index)) for c in calls],
                device=self.adapter.layer_device(index),
                offload=self.offload,
            )
            if self.propagate == "original":
                step.forward()  # records first_outputs before the caller can transform the layer
            if index in wanted:
                yield step
            if index == self.layers[-1]:
                return
            if self.propagate == "original":
                assert step.first_outputs is not None
                hidden = step.first_outputs
            else:
                hidden = step.forward()


@torch.no_grad()
def shared_input_groups(step: CalibrationStep, modules: Mapping[str, nn.Module]) -> list[list[str]]:
    """Group modules that receive the same input tensor, in call order (GPTQ's "true sequential" order).

    For a Llama-style block this gives ``[[q, k, v], [o], [gate, up], [down]]`` without naming any module. Modules
    that the forward never calls are reported last, one group each, with a warning.
    """
    seen: list[tuple[Tensor, list[str]]] = []
    handles = []
    for name, module in modules.items():

        def hook(_module: nn.Module, args: tuple[Any, ...], name: str = name) -> None:
            for tensor, names in seen:
                if tensor is args[0]:
                    if name not in names:
                        names.append(name)
                    return
            if not any(name in names for _, names in seen):
                seen.append((args[0], [name]))

        handles.append(module.register_forward_pre_hook(hook))
    try:
        step.forward(batches=[0])
    finally:
        for handle in handles:
            handle.remove()
    groups = [names for _, names in seen]
    called = {n for g in groups for n in g}
    missing = [n for n in modules if n not in called]
    if missing:
        logger.warning("layer %d: modules %s were not called on the calibration data", step.index, missing)
    return groups + [[n] for n in missing]
