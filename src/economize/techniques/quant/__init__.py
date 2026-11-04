"""Weight quantization (RTN, HQQ, GPTQ, AWQ, SmoothQuant; int2-8, fp8 e4m3, NVFP4 and MXFP4), optional per-token
activation quantization (W8A8) and drafter-only quantization.

Methods, number formats, kernel backends and AWQ scale groups are registries, so a new one is added by
registration: :data:`QUANT_METHODS`, :data:`QUANT_FORMATS`, :data:`QUANT_BACKENDS`, :data:`AWQ_SCALE_GROUPS`,
:data:`TORCHAO_TENSORS` (with :data:`TORCHAO_CONSTRAINTS`), :data:`MARLIN_FORMATS`. What a kernel accepts for a format
(group sizes, zero points, dtypes, GPUs) is a :class:`KernelConstraints` registered with the format, which the
backends check generically. A method or backend lists the config fields it reads (``options``); the others must keep
their defaults.
"""

from .activations import ACTIVATION_SCHEMES, ActivationScheme, DynamicFp8PerToken, DynamicIntPerToken
from .awq import AWQ_SCALE_GROUPS, ScaleGroup, awq_clip_thresholds, awq_scale_layer, search_scale
from .backends import (
    QUANT_BACKENDS,
    TORCHAO_CONSTRAINTS,
    TORCHAO_TENSORS,
    QuantBackend,
    int4_tinygemm_tensors,
    register_torchao_format,
)
from .config import DraftQuantConfig, QuantConfig, QuantSpec
from .constraints import KernelConstraints
from .cutlass import CutlassBackend, CutlassInt8Weight, cutlass_available
from .draft import DraftSnapshot, draft_modules, quantize_draft, quantized_copy, restore_draft
from .fp4 import E2M1_VALUES, Fp4Grid, Mxfp4Grid, Nvfp4Grid
from .gptq import gptq_quantize
from .grid import QUANT_FORMATS, Fp8Grid, IntGrid, QParams, QuantGrid, QuantizedWeight, make_grid, quantize_rtn
from .hqq import HQQMethod, hqq_quantize
from .machete import MacheteBackend, MacheteWeight, machete_available
from .marlin import MARLIN_FORMATS, MarlinBackend, MarlinFormat, MarlinWeight, marlin_available, marlin_layout_error
from .methods import QUANT_METHODS, QuantMethod
from .model import quantizable, quantize_model, quantize_modules
from .readback import CODE_READERS, CodeReader, read_quantized_weight
from .smoothquant import SmoothQuantMethod, smoothquant_scales
from .technique import DraftQuantTechnique, QuantTechnique

__all__ = [
    "ACTIVATION_SCHEMES",
    "AWQ_SCALE_GROUPS",
    "CODE_READERS",
    "E2M1_VALUES",
    "MARLIN_FORMATS",
    "QUANT_BACKENDS",
    "QUANT_FORMATS",
    "QUANT_METHODS",
    "TORCHAO_CONSTRAINTS",
    "TORCHAO_TENSORS",
    "ActivationScheme",
    "CodeReader",
    "CutlassBackend",
    "CutlassInt8Weight",
    "DraftQuantConfig",
    "DraftQuantTechnique",
    "DraftSnapshot",
    "DynamicFp8PerToken",
    "DynamicIntPerToken",
    "Fp4Grid",
    "Fp8Grid",
    "HQQMethod",
    "IntGrid",
    "KernelConstraints",
    "MacheteBackend",
    "MacheteWeight",
    "MarlinBackend",
    "MarlinFormat",
    "MarlinWeight",
    "Mxfp4Grid",
    "Nvfp4Grid",
    "QParams",
    "QuantBackend",
    "QuantConfig",
    "QuantGrid",
    "QuantMethod",
    "QuantSpec",
    "QuantTechnique",
    "QuantizedWeight",
    "ScaleGroup",
    "SmoothQuantMethod",
    "awq_clip_thresholds",
    "awq_scale_layer",
    "cutlass_available",
    "draft_modules",
    "gptq_quantize",
    "hqq_quantize",
    "int4_tinygemm_tensors",
    "machete_available",
    "make_grid",
    "marlin_available",
    "marlin_layout_error",
    "quantizable",
    "quantize_draft",
    "quantize_model",
    "quantize_modules",
    "quantize_rtn",
    "quantized_copy",
    "read_quantized_weight",
    "register_torchao_format",
    "restore_draft",
    "search_scale",
    "smoothquant_scales",
]
