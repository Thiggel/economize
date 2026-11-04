"""TEAL on vLLM (:mod:`economize.backends.vllm.linear`): the written config, the shard thresholds, the row classes and
the thresholded linear map, checked on CPU.

The GPU test in ``test_vllm_serving_gpu.py`` decodes with the plugin and compares it with the economize HF runtime.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch

pytest.importorskip("vllm")

import economize
from economize.backends import UnsupportedBackendError
from economize.backends.vllm import serving
from economize.backends.vllm.extensions import VLLM_EXTENSIONS
from economize.backends.vllm.linear import QUANT_METHOD
from economize.backends.vllm.linear.teal import (
    ALL,
    DENSE,
    SECOND_HALF,
    row_mode,
    second_half_rows,
    shard_thresholds,
    single_rows,
    sparse_rows,
    teal_linear,
)
from economize.backends.vllm.serving import Role, check_servable, support_table, to_vllm
from economize.runtime.phase import phase
from economize.techniques.kv_quant.fp8 import FP8KVConfig
from economize.techniques.lookup.config import LookupConfig
from economize.techniques.quant.config import QuantConfig
from economize.techniques.sparsity.config import TealConfig
from economize.techniques.sparsity.sparsify import PhasePolicy, sparsify, sparsify_tokens
from helpers import VOCAB, make_tiny_model

CALIBRATION = torch.randint(0, VOCAB, (8, 32), generator=torch.Generator().manual_seed(0))
PACKED = {"qkv_proj": ["q_proj", "k_proj", "v_proj"], "gate_up_proj": ["gate_proj", "up_proj"]}


@pytest.fixture
def tiny_dir(tmp_path: Path) -> Path:
    folder = tmp_path / "tiny"
    make_tiny_model(seed=0).save_pretrained(folder)
    return folder


def _base(folder: Path) -> Any:
    from transformers import AutoModelForCausalLM

    return AutoModelForCausalLM.from_pretrained(folder, dtype=torch.float32).eval()


def _teal(folder: Path, *others: Any, **config: Any) -> Any:
    options = {"sparsity": 0.5, "allocation": "uniform", **config}
    return economize.apply(_base(folder), *others, TealConfig(**options), calibration=CALIBRATION)


def _quantization(plan: Any) -> dict[str, Any]:
    return json.loads((Path(plan.model) / "config.json").read_text())["quantization_config"]


# ---- the plan --------------------------------------------------------------------------------------------------


def test_thresholds_are_written_by_hf_module_name(tiny_dir: Path, tmp_path: Path) -> None:
    model = _teal(tiny_dir)
    plan = to_vllm(model, tmp_path / "out")
    quantization = _quantization(plan)
    technique = model.techniques["teal"]
    expected = {f"model.layers.{int(key.split('.', 1)[0]) - 1}.{key.split('.', 1)[1]}": value
                for key, value in technique.threshold_map().items()}
    assert quantization == {"quant_method": QUANT_METHOD, "teal": {"thresholds": expected, "factor_thresholds": {},
                                                                    "single": "all", "multi": "dense"}}
    assert plan.techniques == {"teal": "plugin"}
    assert plan.engine["compilation_config"] == {"cudagraph_mode": "FULL_DECODE_ONLY"}
    assert plan.engine["enable_prefix_caching"] is False
    weights = _base(Path(plan.model)).state_dict()
    plain = _base(tiny_dir).state_dict()
    assert set(weights) == set(plain) and all(torch.equal(weights[key], plain[key]) for key in plain)


def test_one_mode_for_every_row_keeps_cuda_graphs(tiny_dir: Path, tmp_path: Path) -> None:
    plan = to_vllm(_teal(tiny_dir, prefill="all"), tmp_path / "out")
    assert {key: _quantization(plan)["teal"][key] for key in ("single", "multi")} == {"single": "all", "multi": "all"}
    assert "compilation_config" not in plan.engine and "enable_prefix_caching" not in plan.engine


def test_a_drafter_puts_every_row_in_the_refresh_mode(tiny_dir: Path, tmp_path: Path) -> None:
    model = _teal(tiny_dir, LookupConfig(max_ngram=3, min_ngram=1), prefill="all")
    plan = to_vllm(model, tmp_path / "out")
    assert {key: _quantization(plan)["teal"][key] for key in ("single", "multi")} == {"single": "all", "multi": "all"}
    assert plan.speculative["method"] == "ngram"
    assert any("refresh" in note for note in plan.notes["teal"])


def test_refusals(tiny_dir: Path) -> None:
    model = _teal(tiny_dir)
    model.techniques["teal"].thresholds.fill_(float("nan"))
    with pytest.raises(UnsupportedBackendError, match="no thresholds"):
        check_servable(model)
    split = PhasePolicy(prefill="all", refresh="dense")
    with pytest.raises(UnsupportedBackendError, match="same mode"):
        check_servable(_teal(tiny_dir, LookupConfig(max_ngram=3, min_ngram=1), prefill=split))
    with pytest.raises(UnsupportedBackendError, match="drafter's steps"):
        check_servable(_teal(tiny_dir, LookupConfig(max_ngram=3, min_ngram=1)))
    with pytest.raises(UnsupportedBackendError, match="unchanged"):
        check_servable(_teal(tiny_dir, prefill=PhasePolicy(decode="dense")))
    with pytest.raises(UnsupportedBackendError, match="sparsity 0"):
        check_servable(_teal(tiny_dir, sparsity=0.0))


def test_second_half_prompts_run_in_one_pass(tiny_dir: Path, tmp_path: Path) -> None:
    plan = to_vllm(_teal(tiny_dir, prefill="second_half"), tmp_path / "out")
    assert {key: _quantization(plan)["teal"][key] for key in ("single", "multi")} == {
        "single": "all", "multi": "second_half"}
    assert plan.engine["enable_chunked_prefill"] is False
    assert plan.engine["compilation_config"] == {"cudagraph_mode": "FULL_DECODE_ONLY"}
    assert any("max(1, n // 2)" in note for note in plan.notes["teal"])


def test_second_half_next_to_a_drafter(tiny_dir: Path, tmp_path: Path) -> None:
    model = _teal(tiny_dir, LookupConfig(max_ngram=3, min_ngram=1), prefill="second_half")
    plan = to_vllm(model, tmp_path / "out")
    assert {key: _quantization(plan)["teal"][key] for key in ("single", "multi")} == {
        "single": "all", "multi": "second_half"}
    assert plan.engine["enable_chunked_prefill"] is False


def test_a_quantized_base_is_refused(tiny_dir: Path) -> None:
    pytest.importorskip("compressed_tensors")
    model = _teal(tiny_dir, QuantConfig(format="int4", group_size=32, backend="compressed-tensors"))
    with pytest.raises(UnsupportedBackendError, match="one quantization method"):
        check_servable(model)


def test_fake_quantization_is_served_with_teal(tiny_dir: Path, tmp_path: Path) -> None:
    model = _teal(tiny_dir, QuantConfig(format="int4", group_size=32))
    plan = to_vllm(model, tmp_path / "out")
    assert plan.techniques == {"quant": "densified", "teal": "plugin"}
    assert _quantization(plan)["quant_method"] == QUANT_METHOD


def test_calibrated_kv_scales_join_the_linear_config(tiny_dir: Path, tmp_path: Path) -> None:
    model = _teal(tiny_dir, FP8KVConfig(scale="tensor"))
    builder, _ = serving.base._build(model, tmp_path / "out", 4, True)
    folder = Path(serving.base._write_base(builder))
    quantization = json.loads((folder / "config.json").read_text())["quantization_config"]
    assert quantization["quant_method"] == QUANT_METHOD
    assert quantization["kv_cache_scheme"]["strategy"] == "tensor" and quantization["teal"]["thresholds"]


def test_support_table_lists_teal_as_a_plugin() -> None:
    assert support_table(["teal"])[0][1:3] == (Role.FORWARD.value, "plugin")


# ---- shards, modes and rows ---------------------------------------------------------------------------------------


def test_shard_thresholds() -> None:
    thresholds = {"model.layers.0.self_attn.q_proj": 0.3, "model.layers.0.self_attn.v_proj": 0.1,
                  "model.layers.0.mlp.down_proj": 0.2, "model.layers.1.mlp.down_proj": 0.0}
    assert shard_thresholds(thresholds, "model.layers.0.self_attn.qkv_proj", PACKED) == [0.3, 0.0, 0.1]
    assert shard_thresholds(thresholds, "model.layers.0.mlp.down_proj", PACKED) == [0.2]
    assert shard_thresholds(thresholds, "model.layers.0.mlp.gate_up_proj", PACKED) is None
    assert shard_thresholds(thresholds, "model.layers.1.mlp.down_proj", PACKED) is None
    assert shard_thresholds(thresholds, "model.layers.0.self_attn.o_proj", PACKED) is None


def test_modes() -> None:
    assert (row_mode("dense"), row_mode("all"), row_mode("second_half")) == (DENSE, ALL, SECOND_HALF)
    with pytest.raises(UnsupportedBackendError, match="'second_half', got 'first_half'"):
        row_mode("first_half")


def _offsets(*lengths: int) -> SimpleNamespace:
    starts = torch.tensor([0, *lengths]).cumsum(0).to(torch.int32)
    return SimpleNamespace(query_start_loc=starts, max_query_len=max(lengths))


def test_single_rows_from_query_offsets() -> None:
    assert single_rows(_offsets(1, 1, 1), 4) is True  # a padded CUDA-graph batch of decodes
    rows = single_rows(_offsets(1, 3, 1), 6)  # the padding row takes the class of the last request
    assert rows.tolist() == [True, False, False, False, True, True]


def test_single_rows_from_decode_counts() -> None:
    decodes = SimpleNamespace(num_decodes=2, num_decode_tokens=2, num_prefills=1)
    assert single_rows(decodes, 5).tolist() == [True, True, False, False, False]
    assert single_rows(SimpleNamespace(num_decodes=3, num_decode_tokens=3, num_prefills=0), 3) is True
    assert single_rows(SimpleNamespace(num_decodes=0, num_decode_tokens=0, num_prefills=2), 7) is False
    with pytest.raises(UnsupportedBackendError, match="several tokens"):
        single_rows(SimpleNamespace(num_decodes=2, num_decode_tokens=4, num_prefills=1), 6)
    with pytest.raises(UnsupportedBackendError, match="cannot tell"):
        single_rows(SimpleNamespace(), 3)


def test_sparse_rows() -> None:
    metadata = {"layer.0": _offsets(1, 2)}
    assert sparse_rows(metadata, 3, ALL, DENSE).tolist() == [True, False, False]
    assert sparse_rows(metadata, 3, DENSE, ALL).tolist() == [False, True, True]
    assert sparse_rows(metadata, 3, ALL, ALL) is True
    assert sparse_rows(metadata, 3, SECOND_HALF, ALL) is True  # a one-token request is its own last half
    assert sparse_rows(None, 3, ALL, DENSE) is False  # profiling runs count as prompts
    assert sparse_rows(None, 3, DENSE, SECOND_HALF) is True
    with pytest.raises(UnsupportedBackendError, match="micro-batching"):
        sparse_rows([metadata, metadata], 3, ALL, DENSE)


def test_second_half_rows() -> None:
    #           request of 5 rows       1     4 rows             1 (other)  padding (class of the last request)
    expected = [0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1]
    rows = sparse_rows({"layer.0": _offsets(5, 1, 4, 1)}, 12, ALL, SECOND_HALF)
    assert rows.tolist() == [bool(x) for x in expected]
    rows = sparse_rows({"layer.0": _offsets(5, 1, 4, 1)}, 12, DENSE, SECOND_HALF)
    assert rows.tolist() == [bool(x) for x in [0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0]]
    assert second_half_rows(_offsets(1, 1), 4, False) is False  # a batch of decodes takes the single mode
    assert sparse_rows({"layer.0": _offsets(2, 3)}, 5, SECOND_HALF, SECOND_HALF).tolist() == [
        False, True, False, False, True]  # max(1, 3 // 2) = 1
    with pytest.raises(UnsupportedBackendError, match="query offsets"):
        sparse_rows({"layer.0": SimpleNamespace(num_decodes=1)}, 3, DENSE, SECOND_HALF)


@pytest.mark.parametrize("lengths", [(7,), (6, 1, 3), (1, 2, 9)])
def test_second_half_matches_the_hf_call(lengths: tuple[int, ...]) -> None:
    """The rows thresholded in one vLLM pass equal TEAL's HF ``second_half`` call on each request alone."""
    generator = torch.Generator().manual_seed(sum(lengths))
    x = torch.randn(sum(lengths), 16, generator=generator)
    weight = torch.randn(8, 16, generator=generator)
    threshold = torch.tensor([0.7])
    rows = sparse_rows({"layer.0": _offsets(*lengths)}, x.shape[0], ALL, SECOND_HALF)
    served = teal_linear(x, weight, None, threshold, [8], rows)
    policy = PhasePolicy(prefill="second_half")
    expected = []
    for part in x.split(list(lengths)):
        with phase("prefill"):
            expected.append(torch.nn.functional.linear(sparsify_tokens(part, 0.7, policy), weight))
    torch.testing.assert_close(served, torch.cat(expected))


def test_teal_linear_matches_the_unfused_maps() -> None:
    generator = torch.Generator().manual_seed(0)
    x = torch.randn(2, 3, 16, generator=generator)
    weight, bias = torch.randn(12, 16, generator=generator), torch.randn(12, generator=generator)
    thresholds, sizes = torch.tensor([0.5, 0.0, 1.0]), [4, 4, 4]
    expected = torch.cat([torch.nn.functional.linear(sparsify(x, t), weight[a:a + 4], bias[a:a + 4])
                          for t, a in zip(thresholds, (0, 4, 8))], dim=-1)
    torch.testing.assert_close(teal_linear(x, weight, bias, thresholds, sizes, True), expected)
    dense = torch.nn.functional.linear(x, weight, bias)
    torch.testing.assert_close(teal_linear(x, weight, bias, thresholds, sizes, False), dense)
    rows = torch.tensor([True, False, True, False, False, True])
    mixed = teal_linear(x, weight, None, thresholds[:1], [12], rows).reshape(6, 12)
    flat = x.reshape(6, 16)
    for row in range(6):
        inputs = sparsify(flat[row], thresholds[0]) if rows[row] else flat[row]
        torch.testing.assert_close(mixed[row], inputs @ weight.t())


# ---- the vLLM side ------------------------------------------------------------------------------------------------


def _install() -> None:
    VLLM_EXTENSIONS.get("linear")()


def test_install_registers_the_quantization_method() -> None:
    from vllm.model_executor.layers.quantization import QUANTIZATION_METHODS, get_quantization_config

    _install()
    _install()  # idempotent
    assert QUANT_METHOD in QUANTIZATION_METHODS
    from economize.backends.vllm.linear.method import EconomizeLinearConfig

    assert get_quantization_config(QUANT_METHOD) is EconomizeLinearConfig
    assert hasattr(torch.ops.vllm, "economize_teal_linear")


def test_quant_method_per_layer() -> None:
    _install()
    from vllm.model_executor.layers.attention import Attention
    from vllm.model_executor.layers.linear import QKVParallelLinear, RowParallelLinear, UnquantizedLinearMethod
    from vllm.model_executor.layers.quantization.kv_cache import BaseKVCacheMethod

    from economize.backends.vllm.linear.method import EconomizeLinearConfig, TealLinearMethod

    config = EconomizeLinearConfig.from_config({"quant_method": QUANT_METHOD, "teal": {
        "single": "all", "multi": "dense", "thresholds": {"model.layers.0.self_attn.k_proj": 0.25}}})
    config.packed_modules_mapping = PACKED
    qkv = config.get_quant_method(object.__new__(QKVParallelLinear), "model.layers.0.self_attn.qkv_proj")
    assert isinstance(qkv, TealLinearMethod) and qkv.thresholds == [0.0, 0.25, 0.0]
    assert (qkv.single, qkv.multi) == (ALL, DENSE)
    dense = config.get_quant_method(object.__new__(RowParallelLinear), "model.layers.0.self_attn.o_proj")
    assert type(dense) is UnquantizedLinearMethod
    assert config.get_quant_method(object.__new__(Attention), "model.layers.0.self_attn.attn") is None
    assert config.matched == {"model.layers.0.self_attn.qkv_proj": "teal"}
    config.kv_cache_scheme = {"num_bits": 8, "type": "float", "strategy": "tensor"}
    method = config.get_quant_method(object.__new__(Attention), "model.layers.0.self_attn.attn")
    assert isinstance(method, BaseKVCacheMethod)


def test_linear_method_weights_and_output(monkeypatch: pytest.MonkeyPatch) -> None:
    _install()
    import vllm.model_executor.parameter as parameter

    monkeypatch.setattr(parameter, "get_tensor_model_parallel_rank", lambda: 0)  # one process, no TP group
    monkeypatch.setattr(parameter, "get_tensor_model_parallel_world_size", lambda: 1)
    from economize.backends.vllm.linear.method import TealLinearMethod

    layer = torch.nn.Module()
    method = TealLinearMethod([0.5, 0.5], single=ALL, multi=ALL)
    method.create_weights(layer, 16, [8, 4], 16, 12, torch.float32, weight_loader=lambda *args: None)
    assert layer.teal_sizes == [12] and layer.teal_thresholds.tolist() == [0.5]  # equal shards collapse to one
    with torch.no_grad():
        layer.weight.copy_(torch.randn(12, 16, generator=torch.Generator().manual_seed(1)))
    method.process_weights_after_loading(layer)
    x = torch.randn(3, 16, generator=torch.Generator().manual_seed(2))
    torch.testing.assert_close(method.apply(layer, x), sparsify(x, 0.5) @ layer.weight.t())
    with pytest.raises(UnsupportedBackendError, match="2 thresholds"):
        TealLinearMethod([0.1, 0.2], ALL, ALL).create_weights(torch.nn.Module(), 16, [4, 4, 4], 16, 12,
                                                                torch.float32, weight_loader=lambda *args: None)
