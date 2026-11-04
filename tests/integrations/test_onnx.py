"""The ONNX Runtime export (:mod:`economize.backends.onnx`): support levels, refusals, the graph and its outputs.

The support levels, the refusals and the MatMulNBits packing need neither ``onnx`` nor ``onnxruntime``. The round
trips trace two-layer models (an export takes about 20 s on a CPU) and run them in ONNX Runtime: the dense float32
graph reproduces the Hugging Face float32 logits and greedy decoding with the cache, and a graph with int4 codes in
``MatMulNBits`` reproduces the logits of the quantized economize model (up to float32 accumulation order).
"""

from __future__ import annotations

import dataclasses
import json
import warnings
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

import economize
from economize.backends.exporters import UnsupportedBackendError
from economize.backends.onnx import ONNX, OnnxQuant, check_onnx, load_onnx, nbits_reason, pack_bits, to_onnx
from economize.backends.runtime_export import RuntimeExport
from economize.backends.vllm.serving.base import Support
from economize.core.technique import listed_techniques
from economize.errors import ConfigError, SimulatedPathWarning
from economize.techniques.draft_model.config import DraftModelConfig
from economize.techniques.kv_quant.fp8 import FP8KVConfig
from economize.techniques.lookup.config import LookupConfig
from economize.techniques.quant import QuantConfig
from helpers import make_family_model, make_tiny_model, reference_greedy

BATCH = [list(range(3, 20)), list(range(23, 40))]
PROMPT = [5, 9, 11, 3, 7, 21, 8]


def _quant(fmt: str = "int4", symmetric: bool = False, backend: str = "compressed-tensors") -> QuantConfig:
    return QuantConfig(format=fmt, group_size=32, symmetric=symmetric, backend=backend)


def _logits(model: nn.Module) -> np.ndarray:
    with torch.no_grad():
        return model(torch.tensor(BATCH)).logits.float().numpy()


def _onnx_logits(export: RuntimeExport) -> np.ndarray:
    lm = load_onnx(export)
    ids = np.array(BATCH, dtype=np.int64)
    logits, _ = lm.step(ids, np.ones(ids.shape, dtype=np.int64), lm.empty_cache(len(BATCH)))
    return np.asarray(logits)


def _greedy(model: nn.Module, max_new: int = 10) -> list[int]:
    with torch.no_grad():
        return reference_greedy(model, torch.tensor(PROMPT), max_new).tolist()


# ---- support levels and refusals (no onnx needed) ------------------------------------------------------------------
def test_every_technique_has_an_onnx_status() -> None:
    rows = {name: (role, support, reason) for name, role, support, reason in ONNX.support_table()}
    assert set(listed_techniques()) <= set(rows)
    assert all(reason for _, _, reason in rows.values())
    assert rows["quant"][1] == Support.DENSIFIED.value  # the default backend is "fake"
    assert rows["depth_prune"][1] == Support.NATIVE.value
    assert rows["lowrank"][1] == Support.DENSIFIED.value
    for refused in ("draft_model", "lookup", "eagle", "medusa", "kv_fp8", "kv_evict", "teal", "calm"):
        assert rows[refused][1] == Support.UNSUPPORTED.value, refused
    assert "jump" not in rows and ONNX.support_table(["jump"])[0][2] == Support.UNSUPPORTED.value  # unlisted
    assert "speculative" in rows["draft_model"][2] and "fp8" in rows["kv_fp8"][2]


def test_backend_support_reports_onnx() -> None:
    assert economize.backend_support("draft_model")["onnx"].status.value == "unsupported"
    assert "ONNX Runtime" in economize.backend_support("lookup")["onnx"].detail
    assert "onnx" in economize.backend_support("quant")


@pytest.mark.parametrize("config", [FP8KVConfig(), LookupConfig()], ids=["kv_fp8", "lookup"])
def test_refused_techniques_fail_before_writing(tmp_path: Path, config: object) -> None:
    model = economize.apply(make_tiny_model(num_layers=2), config)  # type: ignore[arg-type]  # a TechniqueConfig
    with pytest.raises(UnsupportedBackendError, match="ONNX Runtime"):
        check_onnx(model)
    with pytest.raises(UnsupportedBackendError):
        to_onnx(model, tmp_path / "out")
    assert not (tmp_path / "out").exists()


def test_draft_model_is_refused(tmp_path: Path) -> None:
    make_tiny_model(num_layers=1, seed=1).save_pretrained(tmp_path / "draft_lm")
    model = economize.apply(make_tiny_model(num_layers=2), DraftModelConfig(model=str(tmp_path / "draft_lm")))
    with pytest.raises(UnsupportedBackendError, match="draft LM as its own ONNX graph"):
        check_onnx(model)


def test_quant_levels() -> None:
    adapter = OnnxQuant()
    assert adapter.level(_quant("int4")) is Support.NATIVE
    assert adapter.level(_quant("int8", symmetric=True, backend="marlin")) is Support.NATIVE
    assert adapter.level(_quant("int4", backend="fake")) is Support.DENSIFIED
    assert adapter.level(_quant("fp8", backend="compressed-tensors")) is Support.DENSIFIED
    w8a8 = QuantConfig(format="fp8", group_size=None, backend="torchao", fp8_activations=True)
    assert adapter.level(w8a8) is Support.UNSUPPORTED


def test_codes_without_a_matmul_nbits_form_are_densified() -> None:
    model = economize.apply(make_tiny_model(num_layers=2), _quant("int3"))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", SimulatedPathWarning)
        builder = check_onnx(model)
    assert builder.supports["quant"] is Support.DENSIFIED
    assert not builder.quantized


def test_nbits_reason() -> None:
    from economize.integrations.compressed_tensors import quantized_modules

    model = economize.apply(make_tiny_model(num_layers=2), _quant("int4"))
    weight = quantized_modules(model)["model.layers.0.mlp.down_proj"]
    assert nbits_reason(weight) is None
    assert "act_order" in str(nbits_reason(dataclasses.replace(weight, g_idx=torch.zeros(1, dtype=torch.int64))))
    assert "group size 24" in str(nbits_reason(dataclasses.replace(weight, group_size=24)))
    assert "group size 8" in str(nbits_reason(dataclasses.replace(weight, group_size=8)))
    int3 = quantized_modules(economize.apply(make_tiny_model(num_layers=2), _quant("int3")))
    assert "int3" in str(nbits_reason(int3["model.layers.0.mlp.down_proj"]))


@pytest.mark.parametrize("bits", [2, 4, 8])
def test_pack_bits_is_lsb_first(bits: int) -> None:
    rng = np.random.default_rng(bits)
    values = rng.integers(0, 2**bits, size=(3, 5, 33))
    packed = pack_bits(values, bits)
    per = 8 // bits
    assert packed.dtype == np.uint8 and packed.shape == (3, 5, -(-33 // per))
    unpacked = np.stack([(packed >> (bits * slot)) & (2**bits - 1) for slot in range(per)], axis=-1)
    assert np.array_equal(unpacked.reshape(3, 5, -1)[..., :33], values)
    assert not unpacked.reshape(3, 5, -1)[..., 33:].any()


def test_unknown_dtype_is_refused(tmp_path: Path) -> None:
    pytest.importorskip("onnx")
    with pytest.raises(ConfigError, match="dtype"):
        to_onnx(make_tiny_model(num_layers=2), tmp_path, dtype="bfloat16")


# ---- round trips in ONNX Runtime -----------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def dense(tmp_path_factory: pytest.TempPathFactory) -> tuple[nn.Module, RuntimeExport]:
    pytest.importorskip("onnxruntime")
    pytest.importorskip("onnxscript")
    model = make_tiny_model(num_layers=2)
    return model, to_onnx(model, tmp_path_factory.mktemp("onnx_dense"))


def test_dense_graph_matches_hugging_face(dense: tuple[nn.Module, RuntimeExport]) -> None:
    model, export = dense
    assert export.supports == {} and not export.simulated
    folder = export.path.parent
    assert {path.name for path in export.files} >= {"model.onnx", "config.json"}
    assert json.loads((folder / "economize_onnx.json").read_text())["runtime"] == "onnx"
    lm = load_onnx(folder)
    names = [arg.name for arg in lm.session.get_inputs()]
    assert names[:4] == ["input_ids", "attention_mask", "past_key_values.0.key", "past_key_values.0.value"]
    assert [arg.name for arg in lm.session.get_outputs()][:2] == ["logits", "present.0.key"]
    assert np.abs(_onnx_logits(export) - _logits(model)).max() < 1e-4


def test_dense_graph_decodes_greedily_with_its_cache(dense: tuple[nn.Module, RuntimeExport]) -> None:
    model, export = dense
    tokens = load_onnx(export).generate(np.array([PROMPT]), 10)
    assert tokens[0, len(PROMPT):].tolist() == _greedy(model)


def test_batched_cache_steps_match_one_pass(dense: tuple[nn.Module, RuntimeExport]) -> None:
    _, export = dense
    lm = load_onnx(export)
    ids = np.array(BATCH, dtype=np.int64)
    full, _ = lm.step(ids, np.ones(ids.shape, dtype=np.int64), lm.empty_cache(2))
    first, cache = lm.step(ids[:, :10], np.ones((2, 10), dtype=np.int64), lm.empty_cache(2))
    rest, _ = lm.step(ids[:, 10:], np.ones(ids.shape, dtype=np.int64), cache)
    assert np.abs(np.concatenate([first, rest], axis=1) - full).max() < 1e-4


@pytest.mark.parametrize("fmt", ["int4", "int8"])
def test_quant_codes_run_as_matmul_nbits(tmp_path: Path, fmt: str) -> None:
    pytest.importorskip("onnxruntime")
    pytest.importorskip("onnxscript")
    import onnx

    model = economize.apply(make_tiny_model(num_layers=2), _quant(fmt))
    with warnings.catch_warnings():
        warnings.simplefilter("error", SimulatedPathWarning)
        export = to_onnx(model, tmp_path)
    assert export.supports == {"quant": "native"} and not export.simulated
    graph = onnx.load(str(export.path)).graph
    nodes = [node for node in graph.node if node.op_type == "MatMulNBits"]
    assert len(nodes) == 14  # q, k, v, o, gate, up, down of two layers
    bits = {attr.i for node in nodes for attr in node.attribute if attr.name == "bits"}
    assert bits == {int(fmt[3:])}
    assert "lm.model.layers.0.mlp.down_proj.weight" not in {init.name for init in graph.initializer}
    assert np.abs(_onnx_logits(export) - _logits(model)).max() < 1e-4
    tokens = load_onnx(export).generate(np.array([PROMPT]), 10)
    assert tokens[0, len(PROMPT):].tolist() == _greedy(model)


@pytest.mark.parametrize("family", ["llama", "qwen3"])
def test_families_round_trip(tmp_path: Path, family: str) -> None:
    pytest.importorskip("onnxruntime")
    pytest.importorskip("onnxscript")
    model = make_family_model(family)
    export = to_onnx(model, tmp_path)
    tokens = load_onnx(export).generate(np.array([PROMPT]), 8)
    assert tokens[0, len(PROMPT):].tolist() == _greedy(model, 8)
