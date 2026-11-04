"""Bases loaded through ``transformers`` quantizers, and the compressed-tensors export of economize's codes.

torchao int8 weight-only runs on the CPU; bitsandbytes 4-bit and the CUDA torchao path need a GPU and are skipped
without one.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any

import pytest
import torch
from safetensors import safe_open

from helpers import make_tiny_model, reference_greedy

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning", "ignore::FutureWarning")

needs_torchao = pytest.mark.skipif(importlib.util.find_spec("torchao") is None, reason="needs torchao")
needs_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")
needs_bnb = pytest.mark.skipif(importlib.util.find_spec("bitsandbytes") is None, reason="needs bitsandbytes")
needs_ct = pytest.mark.skipif(importlib.util.find_spec("compressed_tensors") is None, reason="needs compressed-tensors")


@pytest.fixture(scope="module")
def tiny_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp("tiny_base")
    make_tiny_model().save_pretrained(path)
    return path


def _load(path: Path, quantization_config: Any, **kwargs: Any) -> Any:
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(path, quantization_config=quantization_config, **kwargs).eval()
    model.requires_grad_(False)
    model.generation_config.eos_token_id = None
    return model


def _torchao_int8() -> Any:
    from torchao.quantization import Int8WeightOnlyConfig
    from transformers import TorchAoConfig

    return TorchAoConfig(Int8WeightOnlyConfig())


# ---- describing a quantized base ---------------------------------------------------------------------------------
def test_unquantized_base_is_described_as_none() -> None:
    from economize.integrations.quantized import describe_quantization, is_quantized

    assert describe_quantization(make_tiny_model()) is None
    assert not is_quantized(make_tiny_model())


@needs_torchao
def test_torchao_int8_base_is_described(tiny_dir: Path) -> None:
    from economize.integrations.quantized import describe_quantization, same_quantization

    info = describe_quantization(_load(tiny_dir, _torchao_int8(), dtype=torch.float32))
    assert info is not None
    assert info["method"] == "torchao" and info["quantizer"] == "TorchAoHfQuantizer" and not info["pre_quantized"]
    assert info["config"]["quant_type"]["default"]["_type"] == "Int8WeightOnlyConfig"
    json.dumps(info)  # stored in economize_config.json
    assert same_quantization(info, json.loads(json.dumps(info)))
    assert not same_quantization(info, None) and same_quantization(None, None)


@needs_torchao
def test_jump_attaches_to_torchao_base_and_hub_records_it(tiny_dir: Path, applied_jump: Any, tmp_path: Path) -> None:
    from economize.integrations import hub

    base = _load(tiny_dir, _torchao_int8(), dtype=torch.float32)
    model = applied_jump(base, adaptation="lora")
    meta = hub.base_metadata(model)
    assert meta["base_quantization"]["method"] == "torchao"
    assert meta["teacher_fingerprint"] != hub.teacher_fingerprint(make_tiny_model())
    hub.save_pretrained(model, tmp_path)
    saved = json.loads((tmp_path / "economize_config.json").read_text())
    assert saved["base_quantization"]["method"] == "torchao"


@needs_torchao
def test_lossless_decoding_on_torchao_base_matches_its_greedy(tiny_dir: Path, jump_model: Any) -> None:
    import economize

    base = _load(tiny_dir, _torchao_int8(), dtype=torch.float32)
    model = economize.apply(base, economize.JumpConfig(target_layer=2, source_layer=4),
                         decoding=economize.DecodeConfig(cycle_length=4))
    prompt = torch.arange(1, 13)
    result = model.decode(prompt, 16)
    assert torch.equal(torch.as_tensor(getattr(result, "tokens", result)).view(-1), reference_greedy(base, prompt, 16))


@needs_cuda
@needs_torchao
def test_torchao_int8_cuda_base(tiny_dir: Path, applied_jump: Any) -> None:
    from economize.integrations.quantized import describe_quantization

    base = _load(tiny_dir, _torchao_int8(), dtype=torch.bfloat16, device_map="cuda")
    model = applied_jump(base)
    assert describe_quantization(model)["method"] == "torchao"
    with torch.no_grad():
        assert torch.isfinite(model(torch.arange(1, 9, device="cuda")[None]).logits).all()


@needs_cuda
@needs_bnb
def test_bitsandbytes_4bit_base(tiny_dir: Path, applied_jump: Any) -> None:
    from transformers import BitsAndBytesConfig

    from economize.integrations import hub
    from economize.integrations.quantized import describe_quantization

    config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_quant_type="nf4")
    base = _load(tiny_dir, config, dtype=torch.bfloat16, device_map="cuda")
    model = applied_jump(base, adaptation="lora")
    info = describe_quantization(model)
    assert info["method"] == "bitsandbytes" and info["config"]["load_in_4bit"]
    assert hub.teacher_fingerprint(model).startswith("sha256:")
    with torch.no_grad():
        assert torch.isfinite(model(torch.arange(1, 9, device="cuda")[None]).logits).all()


# ---- compressed-tensors export -----------------------------------------------------------------------------------
def _quantize_layers(model: Any, **config: Any) -> Any:
    """RTN-quantize every linear map of the decoder layers with the recording backend (no technique needed)."""
    import economize.integrations.compressed_tensors  # noqa: F401 - registers the backend
    from economize.techniques.quant.config import DraftQuantConfig
    from economize.techniques.quant.model import quantize_modules

    linears = {n: m for n, m in model.named_modules() if isinstance(m, torch.nn.Linear) and ".layers." in n}
    quantize_modules(linears, DraftQuantConfig(backend="compressed-tensors", **config))
    return model


FORMATS = [
    ({"format": "int4", "group_size": 32, "symmetric": False}, "pack-quantized"),
    ({"format": "int4", "group_size": 16, "symmetric": True}, "pack-quantized"),
    ({"format": "int8", "group_size": None, "symmetric": True}, "pack-quantized"),
    ({"format": "fp8", "group_size": None}, "float-quantized"),
]


@needs_ct
@pytest.mark.parametrize(("config", "fmt"), FORMATS, ids=[c["format"] + str(c["group_size"]) for c, _ in FORMATS])
def test_export_reloads_bit_exact_in_transformers(config: dict[str, Any], fmt: str, tmp_path: Path) -> None:
    from transformers import AutoModelForCausalLM

    from economize.integrations.compressed_tensors import export_compressed_tensors, quantized_modules
    from economize.integrations.quantized import describe_quantization

    model = _quantize_layers(make_tiny_model(), **config)
    assert len(quantized_modules(model)) == 6 * 7
    export_compressed_tensors(model, tmp_path)
    qconfig = json.loads((tmp_path / "config.json").read_text())["quantization_config"]
    assert qconfig["quant_method"] == "compressed-tensors" and qconfig["format"] == fmt
    assert qconfig["quantization_status"] == "compressed"
    name = "model.layers.0.mlp.down_proj"
    with safe_open(str(tmp_path / "model.safetensors"), "pt") as f:
        stored = {key: f.get_tensor(key) for key in f.keys() if key.startswith(name)}  # noqa: SIM118  (safe_open is not iterable)
    assert f"{name}.weight_scale" in stored
    if fmt == "pack-quantized":
        assert f"{name}.weight" not in stored and stored[f"{name}.weight_packed"].dtype == torch.int32
    else:
        assert stored[f"{name}.weight"].dtype == torch.float8_e4m3fn
    reloaded = AutoModelForCausalLM.from_pretrained(tmp_path, dtype=torch.float32).eval()
    assert describe_quantization(reloaded)["method"] == "compressed-tensors"
    ids = torch.arange(1, 20)[None]
    with torch.no_grad():
        assert torch.equal(model(ids).logits, reloaded(ids).logits)
        assert not torch.equal(make_tiny_model()(ids).logits, reloaded(ids).logits)


@needs_ct
def test_mixed_formats_export_as_mixed_precision(tmp_path: Path) -> None:
    from economize.integrations.compressed_tensors import QUANTIZED_WEIGHT_ATTR, export_compressed_tensors

    model = _quantize_layers(make_tiny_model(), format="int4", group_size=32, symmetric=False)
    fp8 = _quantize_layers(make_tiny_model(), format="fp8", group_size=None)
    for (_, ours), (_, theirs) in zip(model.model.layers[0].named_modules(), fp8.model.layers[0].named_modules()):
        if hasattr(theirs, QUANTIZED_WEIGHT_ATTR):
            ours.weight.data.copy_(theirs.weight.data)
            setattr(ours, QUANTIZED_WEIGHT_ATTR, getattr(theirs, QUANTIZED_WEIGHT_ATTR))
    export_compressed_tensors(model, tmp_path)
    qconfig = json.loads((tmp_path / "config.json").read_text())["quantization_config"]
    assert qconfig["format"] == "mixed-precision" and len(qconfig["config_groups"]) == 2


@needs_ct
def test_export_without_recorded_codes_raises(tmp_path: Path) -> None:
    from economize.integrations.compressed_tensors import export_compressed_tensors

    with pytest.raises(ValueError, match="backend='compressed-tensors'"):
        export_compressed_tensors(make_tiny_model(), tmp_path)


def test_format_without_a_compressed_tensors_scheme_is_rejected() -> None:
    """The recording backend accepts only formats with a scheme in CT_SCHEMES, so a new economize format is never
    exported under a wrong compressed-tensors type."""
    from economize.errors import ConfigError
    from economize.integrations.compressed_tensors import CT_SCHEMES
    from economize.techniques.quant.config import DraftQuantConfig

    scheme = CT_SCHEMES.unregister("int3")
    try:
        with pytest.raises(ConfigError, match="no compressed-tensors scheme"):
            DraftQuantConfig(format="int3", backend="compressed-tensors")
    finally:
        CT_SCHEMES.register("int3", scheme)
    DraftQuantConfig(format="int3", backend="compressed-tensors")


@needs_ct
def test_quant_technique_with_recording_backend(tmp_path: Path) -> None:
    import economize
    from economize.integrations.compressed_tensors import export_compressed_tensors, quantized_modules

    if "quant" not in economize.TECHNIQUES:
        pytest.skip("the 'quant' technique is not registered yet (needs Capability.WEIGHTS in the core)")
    from economize.techniques.quant import QuantConfig

    model = economize.apply(make_tiny_model(), QuantConfig(format="int4", group_size=32, backend="compressed-tensors"))
    assert len(quantized_modules(model)) == 6 * 7
    export_compressed_tensors(model, tmp_path)
    assert (tmp_path / "config.json").is_file()
