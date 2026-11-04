"""AWQ: activation-aware weight quantization.

Lin, Tang, Tang, Yang, Chen, Wang, Xiao, Dang, Gan & Han, "AWQ: Activation-aware Weight Quantization for On-Device
LLM Compression and Acceleration", MLSys 2024 (https://github.com/mit-han-lab/llm-awq). Port of ``auto_scale_block``,
``apply_scale``, ``auto_clip_block`` and ``apply_clip``:

1. **Scale search.** For every scale group (a previous op followed by the linear maps that read its output) the
   per-input-channel scales ``s = mean|x|^alpha / sqrt(max * min)`` are tried for ``alpha`` in ``0, 1/20, ..,
   19/20``. The linear maps are replaced by ``Q(W diag(s)) diag(s)^-1`` and the alpha with the smallest mean squared
   error of the inspected block's output (attention, MLP or the linear map itself) is kept. All searches of a layer
   see the unmodified layer; the scales are applied afterwards by folding ``1/s`` into the previous op (an RMSNorm
   weight or the output rows of a linear map) and ``s`` into the input columns of the linear maps, which leaves the
   float model unchanged.
2. **Clip search.** For every linear map except the query and key projections, the per-group clipping threshold
   ``max|w| * (1 - i/20)``, ``i < 10``, that minimizes the error of the group's partial dot products on 512 sampled
   tokens is kept and the weights are clamped.
3. The scaled and clipped weights are quantized with round to nearest.

Calibration activations come from the original model (the reference propagates the unquantized layers).

Scale groups are declared per decoder-layer layout in :data:`AWQ_SCALE_GROUPS`: a family whose adapter declares
``layer_layout = "llama"`` gets the Llama groups, and a family may register its own groups under its
``config.model_type`` (or a new layout name). Neither needs an edit of this module. A family with no registered groups
gets them derived per layer by :func:`derive_scale_groups`, which probes every candidate fold on the calibration batch
(Gemma 4, whose layers differ in their projections and normalize the values, uses this path). Previous ops must be a
linear map or a norm whose output is ``weight * normalized(x) (+ bias)``; Gemma 2/3's ``(1 + weight)`` norms are not of
that form, and the derivation rejects them.

Deviation from the reference: the quantizer is the format's grid (:mod:`.grid`), whose range always contains zero,
where llm-awq fits the exact min and max of a group. The search, scaling and clipping are unchanged.
"""

from __future__ import annotations

import inspect as _inspect
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch
from torch import Tensor, nn

from ...core.calibration import CalibrationStep
from ...errors import ConfigError
from ...registry import Registry
from .grid import QuantGrid, quantize_rtn

if TYPE_CHECKING:
    from ...arch import ArchAdapter

__all__ = [
    "AWQ_SCALE_GROUPS",
    "InputCache",
    "ScaleGroup",
    "apply_clip",
    "apply_scale",
    "awq_clip_thresholds",
    "awq_scale_layer",
    "derive_scale_groups",
    "registered_scale_groups",
    "scale_groups_for",
    "search_scale",
]


@dataclass(frozen=True)
class ScaleGroup:
    """Linear maps that read the output of ``prev`` and can absorb a per-channel scale from it.

    Attributes:
        prev: Name of the previous op inside the decoder layer (a norm or a linear map).
        linears: Names of the linear maps that read ``prev``'s output.
        inspect: Module whose output error ranks the scales (default: the single linear map).
        layer_kwargs: Call ``inspect`` with the decoder layer's call arguments (attention needs the mask and the
            rotary embeddings).
        no_clip: Linear maps of the group the clip search leaves alone. llm-awq skips the query and key projections
            ("due to qk bmm, it is hard to clip precisely"). ``None`` finds them by name (``q_``, ``k_``, ``query``,
            ``key``, ``Wqkv``); a family whose names do not follow that pattern lists them.
    """

    prev: str
    linears: tuple[str, ...]
    inspect: str | None = None
    layer_kwargs: bool = False
    no_clip: tuple[str, ...] | None = None

    def unclipped(self) -> tuple[str, ...]:
        """The linear maps of the group that are not clipped."""
        if self.no_clip is not None:
            return self.no_clip
        return tuple(name for name in self.linears if _named_query_or_key(name))


_NO_CLIP = ("q_", "k_", "query", "key", "Wqkv")  # llm-awq's name test for the query and key projections


def _named_query_or_key(name: str) -> bool:
    return any(key in name for key in _NO_CLIP)


_LLAMA_GROUPS = (
    ScaleGroup(
        "input_layernorm", ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"), "self_attn", True,
        no_clip=("self_attn.q_proj", "self_attn.k_proj"),
    ),
    # only when v_proj's outputs are o_proj's inputs one to one (no grouped-query attention); llm-awq PR #67
    ScaleGroup("self_attn.v_proj", ("self_attn.o_proj",)),
    ScaleGroup("post_attention_layernorm", ("mlp.gate_proj", "mlp.up_proj"), "mlp"),
    ScaleGroup("mlp.up_proj", ("mlp.down_proj",)),
)

#: Scale groups per decoder-layer layout (:attr:`ArchAdapter.layer_layout <economize.arch.ArchAdapter.layer_layout>`)
#: or per ``config.model_type``; an entry for the model type wins over one for its layout.
AWQ_SCALE_GROUPS: Registry[tuple[ScaleGroup, ...]] = Registry("AWQ scale-group set", "economize.awq_scale_groups")
AWQ_SCALE_GROUPS.register("llama", _LLAMA_GROUPS)


def scale_groups_for(adapter: ArchAdapter) -> tuple[ScaleGroup, ...]:
    """The AWQ scale groups registered for the adapter's ``config.model_type``, else for its ``layer_layout``.

    Raises:
        ConfigError: Neither key is registered.
    """
    model_type = getattr(adapter.config, "model_type", None)
    layout = getattr(adapter, "layer_layout", None)
    for key in (model_type, layout):
        if key is not None and key in AWQ_SCALE_GROUPS:
            return AWQ_SCALE_GROUPS.get(key)
    raise ConfigError(
        f"AWQ has no scale groups for model_type {model_type!r} (layer layout {layout!r}; registered: "
        f"{AWQ_SCALE_GROUPS.names()}); register them with economize.techniques.quant.AWQ_SCALE_GROUPS.register("
        "model_type, (ScaleGroup(...), ...)), or set layer_layout on the family's ArchAdapter"
    )


def registered_scale_groups(adapter: ArchAdapter) -> tuple[ScaleGroup, ...] | None:
    """Like :func:`scale_groups_for`, but ``None`` (derive the groups per layer) when neither key is registered."""
    try:
        return scale_groups_for(adapter)
    except ConfigError:
        return None


def _foldable(module: nn.Module) -> int | None:
    """Output width of a module that :func:`apply_scale` can fold scales into (a linear map or a module with a
    per-channel weight, such as a norm), else ``None``."""
    if isinstance(module, nn.Linear):
        return module.out_features
    weight = getattr(module, "weight", None)
    if isinstance(weight, nn.Parameter) and weight.dim() == 1 and not any(True for _ in module.children()):
        return weight.numel()
    return None


_PROBE_TOKENS = 256  # tokens of each linear input the derivation compares (bounds its memory on large models)


class _Trace:
    """The first input tokens of every linear map, and the identity of every module's first input and output,
    during one call."""

    def __init__(self, layer: nn.Module) -> None:
        self.layer = layer
        self.inputs: dict[str, Tensor] = {}
        self.input_ids: dict[str, int] = {}
        self.output_ids: dict[str, int] = {}

    def run(self, step: CalibrationStep) -> Tensor:
        def pre(name: str) -> Any:
            def hook(module: nn.Module, args: tuple[Any, ...], kwargs: dict[str, Any]) -> None:
                x = args[0] if args else next(iter(kwargs.values()), None)
                if isinstance(x, Tensor):
                    self.input_ids.setdefault(name, id(x))
                    if isinstance(module, nn.Linear) and name not in self.inputs:
                        self.inputs[name] = x.detach().reshape(-1, x.shape[-1])[:_PROBE_TOKENS].clone()

            return hook

        def post(name: str) -> Any:
            def hook(_module: nn.Module, _args: Any, out: Any) -> None:
                if isinstance(out, Tensor):
                    self.output_ids.setdefault(name, id(out))

            return hook

        handles = []
        for name, module in self.layer.named_modules():
            if name:
                handles.append(module.register_forward_pre_hook(pre(name), with_kwargs=True))
                handles.append(module.register_forward_hook(post(name)))
        try:
            return step.forward([0])[0]
        finally:
            for handle in handles:
                handle.remove()


def _needs_layer_kwargs(module: nn.Module, step: CalibrationStep) -> bool:
    """Whether ``module.forward`` takes the layer's keyword arguments (attention reads the mask and rotary
    embeddings): it accepts ``**kwargs`` or names one of them after its input."""
    params = list(_inspect.signature(module.forward).parameters.values())[1:]
    names = set(step.calls[0].kwargs) if step.calls else set()
    return any(p.kind is p.VAR_KEYWORD or p.name in names for p in params)


def _inspected(layer: nn.Module, prev: str, linears: Sequence[str], trace: _Trace) -> str | None:
    """The smallest module containing all of ``linears`` whose input is ``prev``'s output (llm-awq inspects the
    attention or MLP block), or ``None`` (the single linear map) when there is none or the group has one map."""
    if len(linears) == 1:
        return None
    parts = [name.split(".") for name in linears]
    common = []
    for level in zip(*parts):
        if len(set(level)) != 1:
            break
        common.append(level[0])
    produced = trace.output_ids.get(prev)
    while common:
        name = ".".join(common)
        if produced is not None and trace.input_ids.get(name) == produced:
            return name
        common.pop()
    return None


@torch.no_grad()
def derive_scale_groups(step: CalibrationStep, *, rtol: float = 1e-3) -> tuple[ScaleGroup, ...]:
    """Find the AWQ scale groups of ``step.layer`` by probing every candidate fold on the first calibration batch.

    For every module ``prev`` that :func:`apply_scale` can fold into (a linear map or a per-channel-weight norm), the
    probe folds ``1 / s`` into it with random per-channel powers of two ``s`` (exact in every float format) and keeps
    the linear maps whose inputs become exactly ``x / s``. The group is accepted only if folding ``s`` into those maps
    as well leaves the layer's output unchanged, so a fold the layer does not commute with is never applied: Gemma 4's
    value norm rejects ``v_proj -> o_proj``, a ``(1 + weight)`` norm or an output that also feeds the residual stream
    rejects its group. Weights are restored bit for bit. ``inspect`` is the smallest module that holds the group and
    reads ``prev``'s output (attention, MLP), called with the layer's arguments when its forward needs more than
    its input.

    Args:
        step: The calibration step of the layer (its weights still untransformed).
        rtol: Relative tolerance of the input and output comparisons (kernels may reorder reductions).
    """
    layer = step.layer
    trace = _Trace(layer)
    reference = trace.run(step).float()
    linears = {name: m for name, m in layer.named_modules() if isinstance(m, nn.Linear) and name in trace.inputs}
    generator = torch.Generator().manual_seed(0)

    def same(a: Tensor, b: Tensor) -> bool:
        a, b = a.float(), b.float().to(a.device)
        return bool(torch.allclose(a, b, rtol=rtol, atol=rtol * float(b.abs().max().clamp(min=1e-30))))

    groups: list[ScaleGroup] = []
    claimed: set[str] = set()
    for prev_name, prev in layer.named_modules():
        width = _foldable(prev) if prev_name else None
        if width is None or prev_name not in trace.output_ids:
            continue
        candidates = [n for n, fc in linears.items() if n != prev_name and n not in claimed and fc.in_features == width]
        if not candidates:
            continue
        params = [t for t in (getattr(prev, "weight", None), getattr(prev, "bias", None)) if isinstance(t, Tensor)]
        saved = [t.data.clone() for t in params]
        scales = torch.pow(2.0, torch.randint(0, 2, (width,), generator=generator).float() * 2 - 1)
        try:
            apply_scale(prev, [], scales)
        except ConfigError:
            continue
        probe = _Trace(layer)
        probe.run(step)
        members = []
        for name in candidates:
            x, ref = probe.inputs.get(name), trace.inputs[name]
            s = scales.to(ref.device).view(1, -1)
            if x is not None and same(x.float() * s, ref) and not same(x, ref):
                members.append(name)
        if members:
            originals = [linears[name].weight.data.clone() for name in members]
            _scale_columns([linears[name] for name in members], scales)
            kept = same(step.forward([0])[0], reference)
            for name, original in zip(members, originals):
                linears[name].weight.data.copy_(original)
            if kept:
                inspect = _inspected(layer, prev_name, members, trace)
                module = layer.get_submodule(inspect) if inspect else None
                needs_kwargs = module is not None and _needs_layer_kwargs(module, step)
                groups.append(ScaleGroup(prev_name, tuple(members), inspect, needs_kwargs))
                claimed.update(members)
        for tensor, original in zip(params, saved):
            tensor.data.copy_(original)
    return tuple(groups)


class InputCache:
    """Observer that keeps the inputs of a linear map, one tensor per calibration batch."""

    def __init__(self, module: nn.Module, offload: bool = False) -> None:
        self.inputs: list[Tensor] = []
        self.offload = offload

    def update(self, x: Tensor) -> None:
        self.inputs.append(x.detach().cpu() if self.offload else x.detach())

    def channel_abs_mean(self) -> Tensor:
        """``mean |x|`` per input channel over every token (llm-awq ``get_act_scale``).

        Raises:
            ConfigError: No calibration batch reached the layer.
        """
        total = None
        count = 0
        for x in self.inputs:
            flat = x.reshape(-1, x.shape[-1]).float()
            part = flat.abs().sum(0)
            total = part if total is None else total + part.to(total.device)
            count += flat.shape[0]
        if total is None:
            raise ConfigError("AWQ observed no calibration inputs at this layer: pass a non-empty calibration set")
        return total / count

    def tokens(self) -> Tensor:
        """All observed inputs as one ``[tokens, in]`` float32 matrix."""
        return torch.cat([x.reshape(-1, x.shape[-1]).float() for x in self.inputs])


def _fake(weight: Tensor, grid: QuantGrid, group_size: int | None) -> Tensor:
    return quantize_rtn(weight, grid, group_size).dequantize(weight.dtype)


@torch.no_grad()
def search_scale(
    step: CalibrationStep,
    inspect: nn.Module,
    linears: Sequence[nn.Linear],
    cache: InputCache,
    grid: QuantGrid,
    group_size: int | None,
    *,
    layer_kwargs: bool = False,
    n_grid: int = 20,
) -> Tensor:
    """The best per-input-channel scales of ``linears`` (float32, ``[in]``)."""
    device = linears[0].weight.device
    x_mean = cache.channel_abs_mean().to(device)

    def run() -> list[Tensor]:
        return [
            step.run_module(inspect, x.to(device), b, with_layer_kwargs=layer_kwargs).float()
            for b, x in enumerate(cache.inputs)
        ]

    reference = run()
    numel = sum(o.numel() for o in reference)
    originals = [fc.weight.data.clone() for fc in linears]
    best_error, best_scales = math.inf, None
    for r in range(n_grid):
        ratio = r / n_grid
        scales = x_mean.pow(ratio).clamp(min=1e-4).view(-1)
        scales = scales / (scales.max() * scales.min()).sqrt()
        for fc in linears:
            scaled = fc.weight.data.float() * scales.view(1, -1)
            fc.weight.data.copy_((_fake(scaled, grid, group_size).float() / scales.view(1, -1)).to(fc.weight.dtype))
        error = sum(float((o - ref.to(o.device)).pow(2).sum()) for o, ref in zip(run(), reference)) / numel
        if error < best_error:
            best_error, best_scales = error, scales.clone()
        for fc, original in zip(linears, originals):
            fc.weight.data.copy_(original)
    if best_scales is None:  # every candidate gave a non-finite error
        raise FloatingPointError(f"AWQ scale search produced no finite error for {len(linears)} linear maps")
    return best_scales


@torch.no_grad()
def apply_scale(prev: nn.Module, linears: Sequence[nn.Linear], scales: Tensor) -> None:
    """Fold ``1 / scales`` into ``prev`` and ``scales`` into the input columns of ``linears`` (llm-awq
    ``scale_ln_fcs`` / ``scale_fc_fc``)."""
    s = scales.float()
    if isinstance(prev, nn.Linear):
        rows = prev.weight.data[-s.numel() :]
        rows.copy_((rows.float() / s.to(rows.device).view(-1, 1)).to(rows.dtype))
    else:
        weight = getattr(prev, "weight", None)
        if not isinstance(weight, Tensor) or weight.shape != s.shape:
            raise ConfigError(f"AWQ cannot fold scales into {type(prev).__name__}: it needs a per-channel weight")
        weight.data.copy_((weight.data.float() / s.to(weight.device)).to(weight.dtype))
    bias = getattr(prev, "bias", None)
    if isinstance(bias, Tensor):
        tail = bias.data[-s.numel() :]
        tail.copy_((tail.float() / s.to(tail.device)).to(tail.dtype))
    _scale_columns(linears, s)


def _scale_columns(linears: Sequence[nn.Linear], scales: Tensor) -> None:
    """Multiply the input columns of ``linears`` by ``scales``."""
    for fc in linears:
        fc.weight.data.copy_((fc.weight.data.float() * scales.to(fc.weight.device).view(1, -1)).to(fc.weight.dtype))


@torch.no_grad()
def awq_clip_thresholds(
    weight: Tensor,
    inputs: Tensor,
    grid: QuantGrid,
    group_size: int | None,
    *,
    n_grid: int = 20,
    max_shrink: float = 0.5,
    n_sample_token: int = 512,
) -> Tensor:
    """Per-group clipping thresholds ``[out, groups]`` (llm-awq ``auto_clip_layer``).

    Args:
        weight: ``[out, in]`` weight (already scaled).
        inputs: ``[tokens, in]`` inputs of the linear map (already divided by the scales).
    """
    w = weight.float()
    out_features, in_features = w.shape
    size = group_size or in_features
    x = inputs.to(w.device).float()
    x = x[:: max(1, x.shape[0] // n_sample_token)]
    x = x.reshape(x.shape[0], -1, size)  # tokens, groups, size
    w = w.reshape(out_features, -1, size)  # out, groups, size
    batch = 256 if out_features % 256 == 0 else 64 if out_features % 64 == 0 else out_features
    best_all = []
    for start in range(0, out_features, batch):
        wb = w[start : start + batch]
        org_max = wb.abs().amax(-1, keepdim=True)  # batch, groups, 1
        best = org_max.clone()
        min_errs = torch.full_like(org_max, math.inf)
        org_out = torch.einsum("tgk,ogk->otg", x, wb)
        for i_s in range(int(max_shrink * n_grid)):
            max_val = org_max * (1 - i_s / n_grid)
            clipped = torch.maximum(torch.minimum(wb, max_val), -max_val)
            q = grid.fake(clipped.reshape(-1, size)).reshape(clipped.shape)
            err = (torch.einsum("tgk,ogk->otg", x, q) - org_out).pow(2).mean(1).unsqueeze(-1)
            better = err < min_errs
            min_errs[better] = err[better]
            best[better] = max_val[better]
        best_all.append(best.squeeze(-1))
    return torch.cat(best_all)


@torch.no_grad()
def apply_clip(linear: nn.Linear, thresholds: Tensor) -> None:
    """Clamp every weight group to ``[-threshold, threshold]`` (llm-awq ``apply_clip``)."""
    w = linear.weight.data
    shaped = w.float().reshape(thresholds.shape[0], thresholds.shape[1], -1)
    limit = thresholds.to(w.device).unsqueeze(-1)
    w.copy_(torch.maximum(torch.minimum(shaped, limit), -limit).reshape(w.shape).to(w.dtype))


@torch.no_grad()
def awq_scale_layer(
    step: CalibrationStep,
    groups: Sequence[ScaleGroup] | None,
    targets: Mapping[str, nn.Linear],
    grid: QuantGrid,
    group_size: int | None,
    *,
    n_grid: int = 20,
    clip: bool = True,
    offload: bool = False,
) -> dict[str, Any]:
    """Search and apply the AWQ scales and clipping thresholds of one decoder layer (weights stay in float).

    Runs the layer once on the calibration batch, caching the inputs of every linear map. Scale groups whose linear
    maps are not all in ``targets``, or whose previous linear map's outputs do not line up with the inputs, are
    skipped. ``groups=None`` derives the layer's groups with :func:`derive_scale_groups`. Returns
    ``{"scaled": [group prev names], "clipped": [linear names]}``.
    """
    if groups is None:
        groups = derive_scale_groups(step)
    layer = step.layer
    linears = {name: m for name, m in layer.named_modules() if isinstance(m, nn.Linear)}
    caches = {name: InputCache(m, offload) for name, m in linears.items()}
    handles = [
        m.register_forward_pre_hook(lambda _m, args, cache=caches[name]: cache.update(args[0]))
        for name, m in linears.items()
    ]
    try:
        step.forward()
    finally:
        for handle in handles:
            handle.remove()

    plans = []
    for group in groups:
        if not all(name in targets for name in group.linears):
            continue
        prev = layer.get_submodule(group.prev)
        fcs = [linears[name] for name in group.linears]
        if isinstance(prev, nn.Linear) and any(prev.out_features != fc.in_features for fc in fcs):
            continue
        inspect = layer.get_submodule(group.inspect) if group.inspect else fcs[0]
        cache = caches[group.linears[0]]
        scales = search_scale(
            step, inspect, fcs, cache, grid, group_size, layer_kwargs=group.layer_kwargs, n_grid=n_grid
        )
        plans.append((group, prev, fcs, scales))

    for group, prev, fcs, scales in plans:
        apply_scale(prev, fcs, scales)
        for name in group.linears:
            cache = caches[name]
            cache.inputs = [x / scales.to(device=x.device, dtype=x.dtype) for x in cache.inputs]

    clipped = []
    if clip:
        grouped = {name for group in groups for name in group.linears}
        unclipped = {name for group in groups for name in group.unclipped()}
        for name, fc in targets.items():
            if name in unclipped or (name not in grouped and _named_query_or_key(name)):
                continue
            thresholds = awq_clip_thresholds(fc.weight.data, caches[name].tokens(), grid, group_size, n_grid=n_grid)
            apply_clip(fc, thresholds)
            clipped.append(name)
    return {"scaled": [group.prev for group, *_ in plans], "clipped": clipped}
