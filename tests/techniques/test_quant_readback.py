"""Reading the codes back out of kernel weights (:mod:`economize.techniques.quant.readback`).

A reader must return the codes and zero points the backend was given exactly, and the scales up to the rounding to the
kernel's 16-bit compute dtype, for every layout the backend stores.
"""

from __future__ import annotations

import importlib.util

import pytest
import torch
from torch import nn

from economize.techniques.quant import QuantConfig, make_grid, quantize_model, quantize_rtn
from economize.techniques.quant.grid import QuantizedWeight
from economize.techniques.quant.readback import CODE_READERS, read_quantized_weight, reader_for

HAS_CUDA = torch.cuda.is_available()


def _marlin_ready() -> bool:
    from economize.techniques.quant.marlin import marlin_available

    return HAS_CUDA and torch.cuda.get_device_capability() >= (8, 0) and marlin_available()


def _machete_ready() -> bool:
    from economize.techniques.quant.machete import machete_available

    return HAS_CUDA and torch.cuda.get_device_capability() == (9, 0) and machete_available()


requires_marlin = pytest.mark.skipif(not _marlin_ready(), reason="needs vLLM's Marlin kernels and an sm80+ GPU")
requires_machete = pytest.mark.skipif(not _machete_ready(), reason="needs vLLM's Machete kernels and a Hopper GPU")
requires_torchao = pytest.mark.skipif(importlib.util.find_spec("torchao") is None, reason="needs torchao")

#: Marlin layouts (format, symmetric, group_size), as in test_quant_marlin.py.
MARLIN_LAYOUTS = [
    ("int8", True, None),
    ("int8", True, 128),
    ("int4", True, 128),
    ("int4", False, 128),
    ("int4", False, 64),
    ("int4", True, None),
    ("int4", False, None),
    ("fp8", True, None),
    ("fp8", True, 128),
]

#: Machete layouts, as in test_quant_machete.py.
MACHETE_LAYOUTS = [
    ("int4", True, 128),
    ("int4", True, 64),
    ("int4", True, None),
    ("int4", False, 128),
    ("int4", False, 64),
    ("int8", True, None),
    ("int8", True, 128),
]


def _assert_same_codes(read: QuantizedWeight, given: QuantizedWeight, dtype: torch.dtype = torch.bfloat16) -> None:
    """Codes and zero points exact (an fp8 code -0 may come back as +0), scales equal to the given ones rounded to
    ``dtype`` (the kernel's storage dtype of the scales)."""
    assert read.grid.name == given.grid.name and read.grid.symmetric == given.grid.symmetric
    assert read.group_size == given.group_size
    assert read.codes.shape == given.codes.shape and read.codes.dtype == given.codes.dtype
    assert torch.equal(read.codes.float().cpu(), given.codes.float().cpu())
    torch.testing.assert_close(read.scale.cpu(), given.scale.to(dtype).float().cpu(), atol=0, rtol=0)
    if given.zero is None:
        assert read.zero is None
    else:
        torch.testing.assert_close(read.zero.float().cpu(), given.zero.float().cpu(), atol=0, rtol=0)


def _weight(n: int, k: int) -> torch.Tensor:
    gen = torch.Generator(device="cuda").manual_seed(0)
    return torch.randn(n, k, device="cuda", generator=gen) * 0.05


# ---- no kernels ---------------------------------------------------------------------------------------------------


def test_plain_weights_have_no_reader() -> None:
    linear = nn.Linear(8, 4)
    assert reader_for(linear) is None and read_quantized_weight(linear) is None
    assert read_quantized_weight(torch.zeros(4, 8)) is None
    assert read_quantized_weight(nn.ReLU()) is None


def test_the_kernel_readers_are_registered() -> None:
    assert {"marlin", "machete", "torchao-int8", "torchao-fp8", "torchao-int4"} <= set(CODE_READERS.names())


# ---- kernels ------------------------------------------------------------------------------------------------------


@requires_marlin
@pytest.mark.parametrize(("fmt", "symmetric", "group"), MARLIN_LAYOUTS)
@pytest.mark.parametrize(("n", "k"), [(512, 1024), (200, 384)])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_marlin_round_trip(fmt: str, symmetric: bool, group: int | None, n: int, k: int, dtype: torch.dtype) -> None:
    from economize.techniques.quant.marlin import marlin_weight

    given = quantize_rtn(_weight(n, k), make_grid(fmt, symmetric), group)
    _assert_same_codes(read_quantized_weight(marlin_weight(given, dtype)), given, dtype)


@requires_machete
@pytest.mark.parametrize(("fmt", "symmetric", "group"), MACHETE_LAYOUTS)
@pytest.mark.parametrize(("n", "k"), [(512, 1024), (200, 384)])
def test_machete_round_trip(fmt: str, symmetric: bool, group: int | None, n: int, k: int) -> None:
    from economize.techniques.quant.machete import machete_weight

    given = quantize_rtn(_weight(n, k), make_grid(fmt, symmetric), group)
    _assert_same_codes(read_quantized_weight(machete_weight(given, torch.bfloat16)), given)


def _model_codes(tiny_factory, adapter_of, backend: str, device: str, **kwargs) -> dict[str, QuantizedWeight]:
    from economize.integrations.compressed_tensors import quantized_modules

    torch.manual_seed(0)
    model = tiny_factory(num_layers=2).to(device, torch.bfloat16)
    quantize_model(adapter_of(model), QuantConfig(backend=backend, **kwargs))
    return quantized_modules(model)


def _assert_backend_matches_recorded(
    tiny_factory, adapter_of, backend: str, device: str, scale_dtype: torch.dtype = torch.bfloat16, **kwargs
) -> None:
    """A kernel backend yields the same codes as the ``compressed-tensors`` backend records for the same model."""
    pytest.importorskip("compressed_tensors")
    recorded = _model_codes(tiny_factory, adapter_of, "compressed-tensors", device, **kwargs)
    read = _model_codes(tiny_factory, adapter_of, backend, device, **kwargs)
    assert recorded and set(read) == set(recorded)
    for name, given in recorded.items():
        _assert_same_codes(read[name], given, scale_dtype)


@requires_marlin
@pytest.mark.parametrize(
    "kwargs",
    [
        {"format": "int4", "group_size": 32},
        {"format": "int4", "symmetric": True, "group_size": 64},
        {"format": "int8", "symmetric": True, "group_size": None},
        {"format": "fp8", "group_size": None},
    ],
)
def test_marlin_model_yields_the_recorded_codes(tiny_factory, adapter_of, kwargs) -> None:
    _assert_backend_matches_recorded(tiny_factory, adapter_of, "marlin", "cuda", **kwargs)


@requires_machete
@pytest.mark.parametrize(
    "kwargs", [{"format": "int4", "group_size": 64}, {"format": "int8", "symmetric": True, "group_size": None}]
)
def test_machete_model_yields_the_recorded_codes(tiny_factory, adapter_of, kwargs) -> None:
    _assert_backend_matches_recorded(tiny_factory, adapter_of, "machete", "cuda", **kwargs)


@requires_torchao
@pytest.mark.parametrize(
    "kwargs",
    [
        {"format": "int8", "symmetric": True, "group_size": None},
        {"format": "int8", "group_size": 32},
        {"format": "fp8", "group_size": None},
    ],
)
def test_torchao_model_yields_the_recorded_codes(tiny_factory, adapter_of, kwargs) -> None:
    scale_dtype = torch.float32 if kwargs["format"] == "fp8" else torch.bfloat16  # Float8Tensor keeps fp32 scales
    device = "cuda" if HAS_CUDA else "cpu"
    _assert_backend_matches_recorded(tiny_factory, adapter_of, "torchao", device, scale_dtype, **kwargs)


@requires_torchao
@pytest.mark.skipif(not HAS_CUDA or torch.cuda.get_device_capability() < (8, 0), reason="tinygemm needs sm80+")
@pytest.mark.parametrize("kwargs", [{"format": "int4", "group_size": 32}, {"format": "int4", "group_size": 64}])
def test_torchao_int4_model_yields_the_recorded_codes(tiny_factory, adapter_of, kwargs) -> None:
    _assert_backend_matches_recorded(tiny_factory, adapter_of, "torchao", "cuda", **kwargs)
