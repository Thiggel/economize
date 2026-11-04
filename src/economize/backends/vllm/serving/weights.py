"""vLLM adapters of the weight techniques, and the base state dict they write.

vLLM loads the target as a ``transformers`` checkpoint of the plain architecture (the ``config.json`` names the
model class, the tensors follow its parameter names). The weight techniques change that checkpoint:

* ``quant`` with ``backend="compressed-tensors"``, ``"marlin"``, ``"machete"``, ``"cutlass"`` or ``"torchao"``: written
  in the compressed-tensors format, which vLLM's ``compressed_tensors`` quantization runs with its own weight-only
  kernels (Marlin / Machete for ``wNa16``, the fp8 GEMMs for ``fp8``). The kernel backends' codes are read back from
  their packed weights (:mod:`economize.techniques.quant.readback`). ``backend="fake"`` holds dequantized weights, so
  vLLM serves the same numbers in the dense dtype (``DENSIFIED``).
* ``prune`` (unstructured or 2:4 masks): the masked dense weights. vLLM 0.30 removed sparse checkpoints from its
  compressed-tensors support and has no 2:4 GEMM left, so the 2:4 masks run dense (``DENSIFIED`` for
  ``semi_structured``).
* ``depth_prune``: the model with the removed layers (the technique edits the model and its config in place), which
  vLLM runs natively since it is just a shallower model of the same architecture.
* ``littlebit``: the factored modules multiplied out (LittleBit's dense weight), ``DENSIFIED``: the outputs equal
  the HF runtime's to rounding, the memory and speed savings are lost. ``lowrank`` serves its factors through
  economize's linear method (:mod:`economize.backends.vllm.serving.lowrank`) and is multiplied out the same way only
  next to a quantized base.

:func:`serving_state` builds the state dict all of them write: the base state dict with every module that has a
``dense_weight()`` replaced by ``<name>.weight`` / ``<name>.bias`` (or by the tensors of
:attr:`~economize.backends.vllm.serving.base.PlanBuilder.plugin_tensors` for a module economize's linear method
serves), sparse tensors made dense, and the keys reduced to those of the plain architecture (checked on a
meta-device copy, so a renamed or missing tensor fails here).
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, ClassVar

import torch
from torch import nn

from ....core.metadata import Saving, Simulation
from ....errors import ConfigError
from ...exporters import UnsupportedBackendError
from .base import PlanBuilder, Role, Support, VllmTechnique, register_vllm_technique

__all__ = [
    "DepthPruneVllm",
    "LittleBitVllm",
    "PruneVllm",
    "QuantVllm",
    "dense_modules",
    "quantized_inside",
    "serving_state",
]

#: Attribute the compressed-tensors quant backend sets on a quantized module (kept here so this module does not
#: import compressed_tensors).
_QUANTIZED_ATTR = "economize_quantized_weight"


def dense_modules(base: nn.Module) -> dict[str, nn.Module]:
    """The modules of ``base`` that stand for one linear map through a ``dense_weight()`` method (outermost only)."""
    found: dict[str, nn.Module] = {}
    for name, module in base.named_modules():
        if not callable(getattr(module, "dense_weight", None)):
            continue
        if any(name.startswith(outer + ".") for outer in found):
            continue
        found[name] = module
    return found


def _dense_dtype(base: nn.Module) -> torch.dtype:
    embedding = base.get_input_embeddings() if hasattr(base, "get_input_embeddings") else None
    weight = getattr(embedding, "weight", None)
    if isinstance(weight, torch.Tensor) and weight.is_floating_point():
        return weight.dtype
    return next(p.dtype for p in base.parameters() if p.is_floating_point())


def _bias_of(module: nn.Module) -> torch.Tensor | None:
    bias = getattr(module, "bias", None)
    if isinstance(bias, torch.Tensor):
        return bias
    up = getattr(module, "up", None)  # LowRankLinear keeps the original bias on its second factor
    bias = getattr(up, "bias", None)
    return bias if isinstance(bias, torch.Tensor) else None


def _plain(tensor: torch.Tensor) -> torch.Tensor:
    """A dense, detached tensor (2:4 semi-structured and torch sparse tensors are expanded)."""
    tensor = tensor.detach()
    if tensor.is_sparse or "SparseSemiStructured" in type(tensor).__name__:
        tensor = tensor.to_dense()
    return tensor


def _reference_keys(base: nn.Module, edits: dict[str, Any]) -> set[str]:
    """The state-dict keys of the plain architecture for ``base.config`` with ``edits`` (built on the meta device)."""
    config = copy.deepcopy(base.config)
    for key, value in edits.items():
        setattr(config, key, value)
    with torch.device("meta"):
        reference = type(base)(config)
    return set(reference.state_dict())


def _kernel_weight(module: nn.Module) -> bool:
    """Whether ``module`` holds a kernel weight whose codes a reader recovers (Marlin, Machete, torchao)."""
    from ....techniques.quant.readback import reader_for

    return reader_for(module) is not None


def quantized_inside(module: nn.Module) -> str | None:
    """The name of a map inside ``module`` that holds compressed-tensors codes or a kernel weight, else ``None``."""
    for name, inner in module.named_modules():
        if getattr(inner, _QUANTIZED_ATTR, None) is not None or _kernel_weight(inner):
            return name
    return None


@torch.no_grad()
def serving_state(builder: PlanBuilder) -> dict[str, torch.Tensor]:
    """The base state dict vLLM loads: densified factored modules (or the tensors economize's linear method loads
    for them), dense sparse tensors, plain parameter names.

    Raises:
        UnsupportedBackendError: A compressed-tensors quantized map lies inside a densified module (its codes would
            quantize a factor, while vLLM sees one dense map).
        ConfigError: A tensor of the plain architecture is missing from the model.
    """
    base = builder.base
    dtype = _dense_dtype(base)
    state = {key: _plain(value) for key, value in base.state_dict().items()}
    plugin: dict[str, torch.Tensor] = {}
    for name, tensors in builder.plugin_tensors.items():
        prefix = name + "."
        for key in [key for key in state if key.startswith(prefix)]:
            del state[key]
        plugin.update({prefix + key: value.detach() for key, value in tensors.items()})
    for name, module in dense_modules(base).items():
        if name in builder.plugin_tensors:
            continue
        inner_name = quantized_inside(module)
        if inner_name is not None:
            raise UnsupportedBackendError(
                f"{name}.{inner_name} is quantized for compressed-tensors inside the factored map {name}, which "
                "vLLM serves multiplied out: quantize with backend='fake' (served dense)"
            )
        prefix = name + "."
        for key in [key for key in state if key.startswith(prefix)]:
            del state[key]
        state[prefix + "weight"] = module.dense_weight().detach().to(dtype)
        bias = _bias_of(module)
        if bias is not None:
            state[prefix + "bias"] = bias.detach().to(dtype)
    for name, tensors in builder.dense.items():
        state.update({f"{name}.{key}": value for key, value in tensors.items()})
    expected = _reference_keys(base, builder.config_edits)
    replaced = tuple(name + "." for name in builder.plugin_tensors)
    missing = sorted(key for key in expected - set(state) if not key.startswith(replaced))
    if missing:
        raise ConfigError(f"the served checkpoint lacks {len(missing)} tensors of the plain architecture, e.g. "
                          f"{missing[:3]}: a technique changed the base in a way no vLLM adapter undoes")
    state = {key: value for key, value in state.items() if key in expected}
    state.update(plugin)
    return {key: value.contiguous() for key, value in state.items()}


# ---- adapters ---------------------------------------------------------------------------------------------------


def _write_compressed_tensors(builder: PlanBuilder, target: Path, state: dict[str, torch.Tensor]) -> None:
    from ....integrations.compressed_tensors import export_compressed_tensors

    export_compressed_tensors(builder.base, target, state_dict=state)


@register_vllm_technique
class QuantVllm(VllmTechnique):
    """Quantization: compressed-tensors checkpoints run on vLLM's quantized kernels.

    Weight-only formats run on vLLM's wNa16 (Marlin / Machete) or fp8 kernels. With ``activations`` (W8A8) the
    checkpoint declares dynamic per-token ``input_activations`` and runs on the kernel :attr:`activation_kernels`
    names for the (format, activations) pair; another pair, a group-wise or asymmetric weight, or ``backend='fake'``
    (served dense, which would drop the activation quantization) is refused.
    """

    technique = "quant"
    role = Role.WEIGHTS
    support = Support.NATIVE
    reason = ("backends compressed-tensors, marlin, machete, cutlass and torchao are written in the compressed-tensors "
              "format (vLLM's wNa16 / fp8 kernels); backend='fake' is served dense")
    #: Quantization backend -> how vLLM serves it; a new backend registers its level here or subclasses. A NATIVE
    #: backend's modules must yield their codes to ``compressed_tensors.quantized_modules`` (recorded, or through a
    #: reader in ``economize.techniques.quant.readback.CODE_READERS``).
    levels: ClassVar[dict[str, Support]] = {
        "compressed-tensors": Support.NATIVE,
        "marlin": Support.NATIVE,
        "machete": Support.NATIVE,
        "cutlass": Support.NATIVE,
        "torchao": Support.NATIVE,
        "fake": Support.DENSIFIED,
    }
    #: (weight format, activation scheme) -> the vLLM kernel that serves the W8A8 checkpoint (per-channel symmetric
    #: weights, dynamic per-token inputs). A new pair registers its kernel here.
    activation_kernels: ClassVar[dict[tuple[str, str], str]] = {
        ("int8", "int8"): "CompressedTensorsW8A8Int8 (CUTLASS int8 scaled_mm)",
        ("fp8", "fp8"): "CompressedTensorsW8A8Fp8 (CUTLASS fp8 scaled_mm)",
    }
    #: Weight format -> the vLLM kernel of its weight-only checkpoint, where the plan names it.
    format_kernels: ClassVar[dict[str, str]] = {
        "nvfp4": "CompressedTensorsW4A4Fp4 as W4A16 (MarlinNvFp4LinearKernel before sm100)",
        "mxfp4": "CompressedTensorsW4A4Mxfp4 as W4A16 (MarlinMxFp4LinearKernel before sm100)",
    }
    #: Weight format -> what vLLM does not realize for it on the GPUs economize is tested on. FP4 tensor cores exist
    #: from Blackwell (sm100) on; before that Marlin dequantizes FP4 to 16 bits, so the weight memory is saved and
    #: the timing is not FP4's (economize never reports FP4 timings measured before Blackwell).
    format_simulations: ClassVar[dict[str, Simulation]] = {
        fmt: Simulation(Saving.LATENCY, f"{fmt} runs on Marlin W4A16 before Blackwell (sm100): FP4 latency is "
                                         "only real on FP4 tensor cores")
        for fmt in ("nvfp4", "mxfp4")
    }

    def level(self, technique: nn.Module) -> Support:
        config: Any = technique.config
        return self.level_of(config.backend)

    def simulation_of(self, technique: nn.Module) -> Simulation | None:
        config: Any = technique.config
        return self.format_simulations.get(config.format, self.simulation)

    def check(self, technique: nn.Module, builder: PlanBuilder) -> None:
        config = technique.config
        if getattr(config, "fp8_activations", False):
            raise UnsupportedBackendError(
                "quant with fp8_activations (W8A8) has no economize export to vLLM: quantize with "
                "activations='fp8' and backend='compressed-tensors', use vLLM's own fp8 quantization "
                "(quantization='fp8') or export the weight-only model"
            )
        if self.level(technique) is Support.UNSUPPORTED:
            raise UnsupportedBackendError(
                f"quant backend {config.backend!r} has no vLLM export: quantize with backend "
                "'compressed-tensors', 'marlin', 'machete' or 'torchao' (exported as compressed-tensors; vLLM picks "
                "its Marlin or Machete kernels for such a checkpoint) or 'fake' (served dense)"
            )
        activations = getattr(config, "activations", None)
        if activations is not None:
            self._check_activations(config, activations)

    def _check_activations(self, config: Any, activations: str) -> None:
        """Raise :class:`UnsupportedBackendError` unless vLLM has a W8A8 kernel for the config."""
        if self.level_of(config.backend) is not Support.NATIVE:
            raise UnsupportedBackendError(
                f"quant backend {config.backend!r} with activations={activations!r} would be served dense without "
                "activation quantization: quantize with backend='compressed-tensors' (or 'torchao' for int8)"
            )
        kernel = self.activation_kernels.get((config.format, activations))
        if kernel is None:
            pairs = ", ".join(f"format={f!r} with activations={a!r}" for f, a in sorted(self.activation_kernels))
            raise UnsupportedBackendError(
                f"vLLM has no GPU kernel for format={config.format!r} with activations={activations!r}; "
                f"served: {pairs}"
            )
        if config.group_size is not None or (config.format.startswith("int") and not config.symmetric):
            raise UnsupportedBackendError(
                f"vLLM's {kernel} needs symmetric per-channel weights: set group_size=None"
                + (" and symmetric=True" if config.format.startswith("int") else "")
            )

    def level_of(self, backend: str) -> Support:
        """How vLLM serves the quantization backend ``backend``."""
        return self.levels.get(backend, Support.UNSUPPORTED)

    def needs_dense_linears(self, technique: nn.Module, builder: PlanBuilder) -> str | None:
        if self.level(technique) is Support.NATIVE:
            return f"quant backend {technique.config.backend!r} writes the base in the compressed-tensors format"
        return None

    def contribute(self, technique: nn.Module, builder: PlanBuilder) -> None:
        config: Any = technique.config
        backend = config.backend
        if self.level(technique) is Support.NATIVE:
            builder.set_writer(_write_compressed_tensors, owner=self.technique)
            source = "" if backend == "compressed-tensors" else f" (codes read back from the {backend} weights)"
            activations = getattr(config, "activations", None)
            kernel = self.activation_kernels.get((config.format, activations or ""))
            served = f", {activations} per-token activations on {kernel}" if kernel else ""
            if not kernel and config.format in self.format_kernels:
                served = f" on {self.format_kernels[config.format]}"
            builder.note(self.technique, f"{config.format} weights in the compressed-tensors format{source}{served}")
        else:
            builder.rewrite_base = True
            builder.note(self.technique, "fake quantization: the dequantized weights are served in the dense dtype "
                                         "(same outputs, no memory or speed saving)")


class _DenseWeights(VllmTechnique):
    """A weight technique that vLLM serves as the dense weights the base already holds (or multiplies out)."""

    role = Role.WEIGHTS
    support = Support.DENSIFIED
    message: str = ""

    def contribute(self, technique: nn.Module, builder: PlanBuilder) -> None:
        builder.rewrite_base = True
        if self.message:
            builder.note(self.technique, self.message)


@register_vllm_technique
class LittleBitVllm(_DenseWeights):
    technique = "littlebit"
    reason = "served as the dense weight of the binary factors (vLLM has no LittleBit GEMM)"
    message = "binary factors are multiplied out: same outputs, no memory or speed saving"

    def check(self, technique: nn.Module, builder: PlanBuilder) -> None:
        if any("LittleBit" in type(module).__name__ and not callable(getattr(module, "dense_weight", None))
               for module in builder.base.modules()):
            raise UnsupportedBackendError("littlebit in its latent (QAT) form has no dense weight: finish "
                                          "training (convert to LittleBitLinear) before serving")


@register_vllm_technique
class PruneVllm(_DenseWeights):
    technique = "prune"
    reason = "the masked dense weights (vLLM 0.30 has no sparse checkpoint format and no 2:4 GEMM)"
    simulation = Simulation(Saving.WEIGHT_MEMORY | Saving.COMPUTE | Saving.LATENCY,
                            "the masked weights are served dense")

    def level(self, technique: nn.Module) -> Support:
        return Support.DENSIFIED if technique.config.semi_structured else Support.NATIVE

    def contribute(self, technique: nn.Module, builder: PlanBuilder) -> None:
        builder.rewrite_base = True
        builder.note(self.technique, "pruned weights are served dense (zeros kept): same outputs, no sparse speedup")


@register_vllm_technique
class DepthPruneVllm(VllmTechnique):
    technique = "depth_prune"
    role = Role.WEIGHTS
    support = Support.NATIVE
    reason = "the shallower model (layers removed, config renumbered) is a plain checkpoint"

    def contribute(self, technique: nn.Module, builder: PlanBuilder) -> None:
        builder.rewrite_base = True
