"""``lowrank``: factored linear maps on vLLM, through the ``lowrank`` section of economize's linear method.

=================================  ==============================================================================
economize                          vLLM 0.30 with economize's plugin
=================================  ==============================================================================
``LowRankLinear`` ``up(down(x))``  the map's tensors ``<name>.lowrank_down`` ``[r, in]``, ``<name>.lowrank_up``
                                   ``[out, r]`` and ``<name>.bias`` in the served checkpoint, its rank in the
                                   ``quantization_config`` ``{"quant_method": "economize", "lowrank": {"ranks":
                                   {HF module name: r}}}``; vLLM's fused maps (``qkv_proj``, ``gate_up_proj``)
                                   hold factored and dense shards side by side, each with its own rank
maps ``lowrank`` keeps dense       ``<name>.weight`` as in the plain checkpoint
tensor parallelism                 column-parallel layers split ``up`` by output rows, row-parallel layers split
                                   ``down`` by input columns (the partial sums are reduced as for a dense map)
=================================  ==============================================================================

The factored layers compute ``x @ down^T`` and then ``. @ up^T``, so the memory of the factors and their FLOPs are
what vLLM runs. With ``teal`` the thresholds of the two factors become the layer's input and latent thresholds
(:mod:`economize.backends.vllm.serving.teal`), and passes of at most 16 rows run TEAL's sparse GEMV on both.

A technique that needs the target's linear maps in vLLM's plain layout
(:meth:`~economize.backends.vllm.serving.base.VllmTechnique.needs_dense_linears`: a compressed-tensors base, a
``lora`` jump drafter, fake quantization's native backends) makes ``lowrank`` fall back to the dense product
``up @ down`` (``DENSIFIED``). A factor that the compressed-tensors backend quantized is refused there: its codes
belong to the factor, and vLLM, which loads one quantization method per model, would serve the factored map as one
dense map. A map whose factors are not plain ``nn.Linear`` maps, and a map of an MoE expert (vLLM's fused MoE layers
have no linear method), are multiplied out on their own.
"""

from __future__ import annotations

from torch import nn

from ...exporters import UnsupportedBackendError
from .base import PlanBuilder, Role, Support, VllmTechnique, adapter_for, register_vllm_technique

__all__ = ["LowRankVllm", "dense_reasons", "factored_modules"]

#: The section of economize's linear method that holds the ranks.
SECTION = "lowrank"


def _servable(name: str, module: nn.Module) -> bool:
    """Whether vLLM can serve ``module`` factored: plain ``nn.Linear`` factors, outside the MoE experts."""
    plain = all(type(getattr(module, factor, None)) is nn.Linear for factor in ("down", "up"))
    return plain and ".experts." not in f".{name}."


def dense_reasons(builder: PlanBuilder) -> list[str]:
    """Why ``lowrank`` is served multiplied out next to the other techniques of ``builder`` (empty: factored)."""
    reasons = []
    for name, technique in builder.techniques.items():
        if name == LowRankVllm.technique:
            continue
        reason = adapter_for(name).needs_dense_linears(technique, builder)
        if reason:
            reasons.append(reason)
    return reasons


def factored_modules(builder: PlanBuilder) -> dict[str, nn.Module]:
    """The ``LowRankLinear`` modules that vLLM serves factored, by module name (outermost only).

    Empty when ``lowrank`` is not attached or falls back to the dense product (:func:`dense_reasons`). A module whose
    factors are not plain ``nn.Linear`` maps, or that belongs to an MoE expert, is served multiplied out on its own.
    """
    from ....techniques.lowrank.module import LowRankLinear
    from .weights import dense_modules

    if LowRankVllm.technique not in builder.techniques or dense_reasons(builder):
        return {}
    return {name: module for name, module in dense_modules(builder.base).items()
            if isinstance(module, LowRankLinear) and _servable(name, module)}


@register_vllm_technique
class LowRankVllm(VllmTechnique):
    """``lowrank``: the factors as economize's factored linear layers (the dense product next to a quantized base)."""

    technique = "lowrank"
    role = Role.WEIGHTS
    support = Support.PLUGIN
    reason = ("economize's linear method: each factored map runs x @ down^T @ up^T on its two factors (fused maps "
              "hold factored and dense shards side by side); the dense product up @ down next to a quantized base")

    def contribute(self, technique: nn.Module, builder: PlanBuilder) -> None:
        from .weights import _dense_dtype, dense_modules, quantized_inside

        builder.rewrite_base = True
        modules = factored_modules(builder)
        if not modules:
            for name, module in dense_modules(builder.base).items():
                inner = quantized_inside(module)
                if inner is not None:
                    raise UnsupportedBackendError(
                        f"lowrank next to a compressed-tensors base: {name}.{inner} quantizes a factor, while vLLM "
                        "loads one quantization method per model and would serve the factored map as one dense "
                        "map. Quantize with backend='fake' (served dense) or serve lowrank without quant")
            builder.supports[self.technique] = Support.DENSIFIED
            reasons = dense_reasons(builder) or ["no factored map has plain nn.Linear factors outside MoE experts"]
            builder.note(self.technique, f"factored maps are multiplied out ({'; '.join(reasons)}): same outputs, "
                                         "no memory or speed saving")
            return
        dtype = _dense_dtype(builder.base)
        for name, module in modules.items():
            tensors = {"lowrank_down": module.down.weight, "lowrank_up": module.up.weight}
            if module.up.bias is not None:
                tensors["bias"] = module.up.bias
            builder.plugin_tensors[name] = {key: value.detach().to(dtype) for key, value in tensors.items()}
        builder.add_linear_section(SECTION, {"ranks": {name: int(module.rank) for name, module in modules.items()}},
                                   owner=self.technique)
        params = sum(module.rank * (module.in_features + module.out_features) for module in modules.values())
        dense = sum(module.in_features * module.out_features for module in modules.values())
        others = len(dense_modules(builder.base)) - len(modules)
        rest = f"; {others} other factored maps (MoE experts, wrapped factors) are multiplied out" if others else ""
        builder.note(self.technique, f"{len(modules)} maps served factored ({params / dense:.0%} of their dense "
                                     f"parameters){rest}; the written checkpoint loads in vLLM with economize's "
                                     "plugin only")
