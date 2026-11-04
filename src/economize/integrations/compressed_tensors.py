"""Export the base-model weights of economize's ``quant`` technique as a compressed-tensors checkpoint.

compressed-tensors is the checkpoint format of ``llm-compressor``; vLLM, SGLang and ``transformers`` load it. A model
quantized by economize (RTN, GPTQ or AWQ codes, int2-8 or fp8, any group size) can be written in that format, so the
same codes run on vLLM's kernels. economize needs the codes for that. The ``"compressed-tensors"`` backend keeps the
dequantized weights (numerically the ``"fake"`` backend) and remembers each module's codes, scales and zero points; the
kernel backends (``"marlin"``, ``"machete"``, ``"cutlass"``, ``"torchao"``) keep only their packed weight, from which
the readers of :mod:`economize.techniques.quant.readback` recover the codes (scales rounded to the kernel's 16-bit
compute dtype). The ``"fake"`` backend keeps neither, so a fake-quantized model cannot be exported. An installed
economize finds the backend through its ``economize.quant_backends`` entry point; importing this module also registers
it (for a source checkout without installed metadata)::

    import economize
    import economize.integrations.compressed_tensors as ct

    model = economize.apply(base, QuantConfig(format="int4", group_size=128, backend="compressed-tensors"))
    ct.export_compressed_tensors(model, "qwen-int4-ct")   # then: vllm serve qwen-int4-ct

The export covers the base model's weights only (all WEIGHTS techniques: pruned zeros, removed layers and
low-rank factors are in those weights). Drafters and other techniques are saved with
:func:`economize.integrations.hub.save_pretrained`. Scales are stored in the model's dtype (``scale_dtype``), so a
float32 model reloads bit for bit and a bfloat16 model up to the rounding of its scales. compressed-tensors no
longer has a sparse storage format, so 2:4-pruned weights are exported dense.

How each economize format is stored is registered in :data:`CT_SCHEMES` (the built-in integer formats as packed
integers, ``fp8`` as float8, ``nvfp4`` / ``mxfp4`` as packed FP4 with their own scale dtypes). A new format registers
its scheme there (or through the ``economize.ct_schemes`` entry-point group); a format without one is rejected when
the config is built, so nothing is exported wrongly::

    from economize.integrations.compressed_tensors import CT_SCHEMES, CtScheme

    CT_SCHEMES.register("int6", CtScheme(type="int", format="pack-quantized"))
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import torch
from torch import Tensor, nn

from ..core.metadata import Saving, Simulation
from ..errors import ConfigError
from ..registry import Registry
from ..techniques.quant import activations
from ..techniques.quant.backends import QUANT_BACKENDS, QuantBackend
from ..techniques.quant.grid import QuantizedWeight
from ..techniques.quant.readback import read_quantized_weight
from ..unwrap import base_model_of

if TYPE_CHECKING:
    from ..techniques.quant.config import QuantSpec

__all__ = [
    "CT_SCHEMES",
    "QUANTIZED_WEIGHT_ATTR",
    "CompressedTensorsBackend",
    "CtScheme",
    "Fp4CtScheme",
    "export_compressed_tensors",
    "quantized_modules",
]

#: Attribute under which :class:`CompressedTensorsBackend` keeps a module's :class:`QuantizedWeight`.
QUANTIZED_WEIGHT_ATTR = "economize_quantized_weight"


@dataclass(frozen=True)
class CtScheme:
    """How compressed-tensors stores the codes of one economize format.

    Attributes:
        type: The ``QuantizationArgs.type`` of the codes (``"int"`` or ``"float"``).
        format: The compressed-tensors storage format (``"pack-quantized"``, ``"float-quantized"``).
        activation_formats: The storage format by activation scheme (``QuantConfig.activations``) for weights whose
            inputs are quantized too (int8 W8A8: ``"int-quantized"``); a scheme missing here is not exported for
            this format.
    """

    type: Literal["int", "float"]
    format: str
    activation_formats: Mapping[str, str] = field(default_factory=dict)

    def weight_args(self, qweight: QuantizedWeight) -> dict[str, Any]:
        """The ``QuantizationArgs`` fields of one weight (per-channel or group-wise scales, symmetric or not).

        Args:
            qweight: The weight's codes and scales.

        Returns:
            Keyword arguments of ``compressed_tensors.quantization.QuantizationArgs``.
        """
        per_channel = qweight.scale.shape[1] == 1
        symmetric = qweight.zero is None or bool((qweight.zero == 2 ** (qweight.grid.bits - 1)).all())
        return {
            "num_bits": qweight.grid.bits,
            "type": self.type,
            "symmetric": symmetric,
            "strategy": "channel" if per_channel else "group",
            "group_size": None if per_channel else qweight.group_size,
            "observer": None,
            "dynamic": False,
        }

    def local_state(self, qweight: QuantizedWeight, scale_dtype: torch.dtype) -> dict[str, Tensor]:
        """The per-module state the compressor re-quantizes: a float32 weight that lies exactly on the grid of the
        stored scales, so that its rounding reproduces economize's codes.

        Args:
            qweight: The weight's codes and scales.
            scale_dtype: dtype of the stored scales.

        Returns:
            ``weight``, ``weight_scale`` and, for integer codes with zero points, ``weight_zero_point``.

        Raises:
            ConfigError: Real-valued zero points (HQQ without ``hqq_integer_zero``).
        """
        return _local_state(qweight, scale_dtype)


@dataclass(frozen=True)
class Fp4CtScheme(CtScheme):
    """FP4 E2M1 codes packed two per byte (``nvfp4-pack-quantized``, ``mxfp4-pack-quantized``) with the format's own
    scale dtype (float8 e4m3 group scales and a float32 global scale for NVFP4, E8M0 exponents for MXFP4); the
    ``scale_dtype`` of the export does not apply.

    Attributes:
        strategy: ``"tensor_group"`` (NVFP4: group scales times a per-tensor scale) or ``"group"`` (MXFP4).
        stored_scale_dtype: The ``QuantizationArgs.scale_dtype`` of the format.
    """

    type: Literal["int", "float"] = "float"
    format: str = "nvfp4-pack-quantized"
    strategy: str = "tensor_group"
    stored_scale_dtype: torch.dtype = torch.float8_e4m3fn

    def weight_args(self, qweight: QuantizedWeight) -> dict[str, Any]:
        return {
            "num_bits": 4,
            "type": "float",
            "symmetric": True,
            "strategy": self.strategy,
            "group_size": qweight.group_size,
            "observer": None,
            "dynamic": False,
            "scale_dtype": self.stored_scale_dtype,
            "zp_dtype": self.stored_scale_dtype,
        }

    def local_state(self, qweight: QuantizedWeight, scale_dtype: torch.dtype) -> dict[str, Tensor]:
        """``weight`` on the FP4 grid, the group scales before their storage cast and (NVFP4) the global scale.

        Raises:
            ConfigError: Act-order codes (``g_idx``), which the packed FP4 formats cannot store.
        """
        if qweight.g_idx is not None:
            raise ConfigError(f"{self.format} stores contiguous groups: quantize {qweight.grid.name} without act_order")
        state = {"weight": qweight.dequantize(torch.float32)}
        global_scale = getattr(qweight.grid, "global_scale", None)
        if global_scale is None:  # MXFP4: the power-of-two scales, which the compressor stores as E8M0 exponents
            state["weight_scale"] = qweight.scale.float()
            return state
        state["weight_scale"] = qweight.grid.local_scale(qweight.scale)  # type: ignore[attr-defined]
        state["weight_global_scale"] = torch.tensor([global_scale], dtype=torch.float32, device=qweight.scale.device)
        return state


#: The compressed-tensors scheme of each exportable format, by economize format name (entry-point group
#: ``economize.ct_schemes``).
CT_SCHEMES: Registry[CtScheme] = Registry("compressed-tensors scheme", "economize.ct_schemes")
for _bits in (2, 3, 4):
    CT_SCHEMES.register(f"int{_bits}", CtScheme("int", "pack-quantized"))
CT_SCHEMES.register("int8", CtScheme("int", "pack-quantized", {"int8": "int-quantized"}))
CT_SCHEMES.register("fp8", CtScheme("float", "float-quantized", {"fp8": "float-quantized"}))
CT_SCHEMES.register("nvfp4", Fp4CtScheme())
CT_SCHEMES.register(
    "mxfp4", Fp4CtScheme(format="mxfp4-pack-quantized", strategy="group", stored_scale_dtype=torch.uint8)
)


@QUANT_BACKENDS.register("compressed-tensors")
class CompressedTensorsBackend(QuantBackend):
    """Dequantized weights (as ``"fake"``) plus the codes, so that :func:`export_compressed_tensors` can write
    them without re-quantizing. With ``activations`` it simulates the activation scheme as ``"fake"`` does and
    exports it as the checkpoint's ``input_activations``."""

    options = frozenset({"activations"})
    simulation = Simulation(Saving.WEIGHT_MEMORY | Saving.LATENCY,
                            "keeps the dequantized weights (the codes are for export_compressed_tensors)")

    def check_config(self, config: QuantSpec) -> None:
        """Raise :class:`ConfigError` for a format without a compressed-tensors scheme in :data:`CT_SCHEMES`, or an
        activation scheme the format's scheme cannot export."""
        if config.format not in CT_SCHEMES:
            raise ConfigError(f"format {config.format!r} has no compressed-tensors scheme; registered: "
                              f"{', '.join(CT_SCHEMES.names())} (register one in CT_SCHEMES)")
        scheme = CT_SCHEMES.get(config.format)
        if config.activations is not None and config.activations not in scheme.activation_formats:
            accepted = ", ".join(sorted(scheme.activation_formats)) or "none"
            raise ConfigError(f"compressed-tensors exports format {config.format!r} with activations {accepted}, "
                              f"got activations={config.activations!r}")

    def apply(self, module: nn.Linear, qweight: QuantizedWeight, config: QuantSpec) -> None:
        setattr(module, QUANTIZED_WEIGHT_ATTR, qweight)
        if config.activations is not None:
            activations.install(module, config.activations)


def quantized_modules(model: Any) -> dict[str, QuantizedWeight]:
    """The base model's quantized modules by qualified name: the codes the ``"compressed-tensors"`` backend recorded,
    or those read back from a kernel weight (:func:`~economize.techniques.quant.readback.read_quantized_weight`)."""
    base = base_model_of(model)
    found: dict[str, QuantizedWeight] = {}
    for name, module in base.named_modules():
        recorded = getattr(module, QUANTIZED_WEIGHT_ATTR, None)
        qweight = recorded if isinstance(recorded, QuantizedWeight) else read_quantized_weight(module)
        if qweight is not None:
            found[name] = qweight
    return found


def export_compressed_tensors(
    model: Any,
    save_directory: str | Path,
    *,
    scale_dtype: torch.dtype | None = None,
    state_dict: dict[str, Tensor] | None = None,
    **save_kwargs: Any,
) -> Path:
    """Write ``model``'s base model with its quantized linear maps in the compressed-tensors format.

    Args:
        model: An :class:`~economize.EfficientModel` (or a ``transformers`` model) quantized by the ``quant``
            technique with ``backend="compressed-tensors"``, ``"marlin"``, ``"machete"``, ``"cutlass"`` or
            ``"torchao"``.
        save_directory: Output folder; it receives ``config.json`` (with ``quantization_config``), the
            safetensors weights and the generation config, as ``save_pretrained`` writes them.
        scale_dtype: dtype of the stored scales (default: the dtype of each quantized weight).
        state_dict: The base state dict to write (default: ``base.state_dict()``); the quantized modules'
            ``weight`` entries in it are replaced by their codes. The vLLM backend passes its densified state.
        save_kwargs: Passed to ``PreTrainedModel.save_pretrained`` (``max_shard_size``, ...).

    Returns:
        The output folder.

    Raises:
        ConfigError: no module carries recorded codes.
    """
    from compressed_tensors.compressors import BaseCompressor
    from compressed_tensors.compressors.model_compressors import ModelCompressor
    from compressed_tensors.quantization import QuantizationConfig, QuantizationScheme

    base = base_model_of(model)
    records = quantized_modules(base)
    if not records:
        raise ConfigError(
            "no quantized codes found: attach the 'quant' technique with backend='compressed-tensors' (import "
            "economize.integrations.compressed_tensors first) or a kernel backend ('marlin', 'machete', 'torchao')"
        )
    modules = dict(base.named_modules())
    state = dict((state_dict if state_dict is not None else base.state_dict()).items())
    groups: dict[str, list[str]] = defaultdict(list)
    schemes: dict[str, Any] = {}
    for name, qweight in records.items():
        weights, fmt = _weight_args(qweight)
        inputs, fmt = _input_args(qweight, modules[name], fmt)
        key = f"{fmt}:{weights.model_dump_json()}:{inputs.model_dump_json() if inputs is not None else ''}"
        groups[key].append(name)
        scheme = QuantizationScheme(targets=[name], weights=weights, input_activations=inputs, format=fmt)
        dtype = scale_dtype or modules[name].weight.dtype
        compressed = BaseCompressor.get_value_from_registry(fmt).compress(
            CT_SCHEMES.get(qweight.grid.name).local_state(qweight, dtype), scheme
        )
        del state[f"{name}.weight"]
        state.update({f"{name}.{key_}": value.contiguous() for key_, value in compressed.items()})
        schemes[key] = weights, inputs, fmt
    config_groups = {
        f"group_{i}": QuantizationScheme(
            targets=sorted(names), weights=schemes[key][0], input_activations=schemes[key][1], format=schemes[key][2]
        )
        for i, (key, names) in enumerate(sorted(groups.items()))
    }
    formats = {fmt for *_, fmt in schemes.values()}
    qconfig = QuantizationConfig(
        config_groups=config_groups,
        format=formats.pop() if len(formats) == 1 else "mixed-precision",
        quantization_status="compressed",
    )
    out = Path(save_directory)
    base.save_pretrained(out, state_dict=state, **save_kwargs)
    ModelCompressor(quantization_config=qconfig).update_config(str(out))
    return out


def _weight_args(qweight: QuantizedWeight) -> tuple[Any, str]:
    """compressed-tensors ``QuantizationArgs`` and storage format of one quantized weight."""
    from compressed_tensors.quantization import QuantizationArgs

    scheme = CT_SCHEMES.get(qweight.grid.name)
    return QuantizationArgs(**scheme.weight_args(qweight)), scheme.format


def _input_args(qweight: QuantizedWeight, module: nn.Module, fmt: str) -> tuple[Any, str]:
    """``(input_activations, storage format)`` of a module: ``(None, fmt)`` for a weight-only map.

    Raises:
        ConfigError: The module's activation scheme has no storage format for its weight format.
    """
    from compressed_tensors.quantization import QuantizationArgs

    scheme = activations.activation_scheme_of(module)
    if scheme is None:
        return None, fmt
    formats = CT_SCHEMES.get(qweight.grid.name).activation_formats
    if scheme.name not in formats:
        raise ConfigError(f"compressed-tensors cannot store {qweight.grid.name} weights with {scheme.name} "
                          f"activations (CtScheme.activation_formats: {sorted(formats)})")
    return QuantizationArgs(**scheme.ct_args()), formats[scheme.name]


def _local_state(qweight: QuantizedWeight, scale_dtype: torch.dtype) -> dict[str, Tensor]:
    """The per-module state compressed-tensors re-quantizes: a float32 weight that lies exactly on the grid of the
    stored scales, so its rounding reproduces economize's codes."""
    scale = qweight.scale.to(scale_dtype)
    expanded = scale.float().repeat_interleave(qweight.group_size, dim=1)
    if qweight.zero is None:  # fp8: codes are the scaled values
        return {"weight": qweight.codes.float() * expanded, "weight_scale": scale}
    if not bool((qweight.zero == qweight.zero.round()).all()):
        raise ConfigError(
            "compressed-tensors stores integer zero points; these codes have real-valued ones (HQQ): quantize with "
            "hqq_integer_zero=True to export them"
        )
    offset = 2 ** (qweight.grid.bits - 1)  # economize codes are unsigned, compressed-tensors codes signed
    zero = qweight.zero.repeat_interleave(qweight.group_size, dim=1)
    weight = (qweight.codes.float() - zero) * expanded
    return {"weight": weight, "weight_scale": scale, "weight_zero_point": (qweight.zero - offset).to(torch.int8)}
