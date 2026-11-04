"""``lowrank`` on vLLM (:mod:`economize.backends.vllm.serving.lowrank`): the written checkpoint and config, the
fallback to the dense product, and economize's factored linear method on vLLM's own layers (simulated tensor
parallelism), checked on CPU against the HF ``up(down(x))`` with TEAL's masks.

The GPU test in ``test_vllm_serving_gpu.py`` decodes a factored checkpoint with the plugin.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import torch
from safetensors.torch import load_file

pytest.importorskip("vllm")

import economize
from economize.backends import UnsupportedBackendError
from economize.backends.vllm.extensions import VLLM_EXTENSIONS
from economize.backends.vllm.linear import QUANT_METHOD
from economize.backends.vllm.linear.factored import Unit, factored_linear, layer_units
from economize.backends.vllm.serving import Support, check_servable, to_vllm
from economize.errors import ConfigError
from economize.techniques.jump import JumpConfig
from economize.techniques.lowrank.config import LowRankConfig
from economize.techniques.lowrank.module import LowRankLinear
from economize.techniques.quant.config import QuantConfig
from economize.techniques.sparsity.config import TealConfig
from economize.techniques.sparsity.sparsify import sparsify
from economize.unwrap import base_model_of
from helpers import VOCAB, make_tiny_model

TOKENS = torch.randint(0, VOCAB, (2, 16), generator=torch.Generator().manual_seed(0))
CALIBRATION = torch.randint(0, VOCAB, (8, 32), generator=torch.Generator().manual_seed(1))
PACKED = {"qkv_proj": ["q_proj", "k_proj", "v_proj"], "gate_up_proj": ["gate_proj", "up_proj"]}


@pytest.fixture
def tiny_dir(tmp_path: Path) -> Path:
    folder = tmp_path / "tiny"
    make_tiny_model(seed=0).save_pretrained(folder)
    return folder


def _base(folder: Path) -> Any:
    from transformers import AutoModelForCausalLM

    return AutoModelForCausalLM.from_pretrained(folder, dtype=torch.float32).eval()


def _lowrank(folder: Path, *others: Any, **teal: Any) -> Any:
    configs: list[Any] = [LowRankConfig(method="svd", compression=0.5), *others]
    if teal:
        configs.append(TealConfig(**{"sparsity": 0.5, "allocation": "uniform", **teal}))
    return economize.apply(_base(folder), *configs, calibration=CALIBRATION)


def _factored(model: Any) -> dict[str, LowRankLinear]:
    return {name: module for name, module in base_model_of(model).named_modules()
            if isinstance(module, LowRankLinear)}


def _quantization(plan: Any) -> dict[str, Any]:
    return json.loads((Path(plan.model) / "config.json").read_text())["quantization_config"]


# ---- the plan ------------------------------------------------------------------------------------------------------


def test_the_factors_are_written_with_their_ranks(tiny_dir: Path, tmp_path: Path) -> None:
    model = _lowrank(tiny_dir)
    factored = _factored(model)
    assert factored
    plan = to_vllm(model, tmp_path / "out")
    assert plan.techniques == {"lowrank": "plugin"}
    assert _quantization(plan) == {"quant_method": QUANT_METHOD,
                                   "lowrank": {"ranks": {name: m.rank for name, m in factored.items()}}}
    state = load_file(Path(plan.model) / "model.safetensors")
    plain = make_tiny_model().state_dict()
    for name, module in factored.items():
        assert f"{name}.weight" not in state
        assert torch.equal(state[f"{name}.lowrank_down"], module.down.weight)
        assert torch.equal(state[f"{name}.lowrank_up"], module.up.weight)
        assert (f"{name}.bias" in state) == (module.up.bias is not None)
    rest = {key for key in plain if not any(key.startswith(name + ".") for name in factored)}
    assert rest - {"lm_head.weight"} <= set(state)  # the tiny model ties its head to the embedding
    # the written factors give the HF model's logits once multiplied out
    dense = {key: value for key, value in state.items() if key in plain}
    for name in factored:
        dense[f"{name}.weight"] = state[f"{name}.lowrank_up"] @ state[f"{name}.lowrank_down"]
    dense["lm_head.weight"] = state["model.embed_tokens.weight"]
    reference = make_tiny_model()
    reference.load_state_dict(dense, strict=True)
    with torch.no_grad():
        torch.testing.assert_close(reference.eval()(TOKENS).logits, model(TOKENS).logits, atol=1e-4, rtol=1e-4)


def test_teal_thresholds_the_two_factors(tiny_dir: Path, tmp_path: Path) -> None:
    model = _lowrank(tiny_dir, prefill="all")
    plan = to_vllm(model, tmp_path / "out")
    assert plan.techniques == {"lowrank": "plugin", "teal": "plugin"}
    teal = _quantization(plan)["teal"]
    factored = _factored(model)
    technique = model.techniques["teal"]
    names = {id(module): name for name, module in base_model_of(model).named_modules()}
    from economize.techniques.sparsity.model import teal_modules

    values = technique.threshold_map()
    by_name = {names[id(module)]: float(values[key]) for key, module in teal_modules(technique.adapter,
                                                                                     technique.config).items()}
    for name in factored:
        assert teal["thresholds"][name] == by_name[f"{name}.down"]
        assert teal["factor_thresholds"][name] == by_name[f"{name}.up"]
    assert set(teal["thresholds"]) == set(factored) | {n for n in by_name if not n.endswith((".down", ".up"))}
    assert (teal["single"], teal["multi"]) == ("all", "all")
    assert set(_quantization(plan)) == {"quant_method", "teal", "lowrank"}


def test_a_lora_jump_drafter_multiplies_the_factors_out(tiny_dir: Path, tmp_path: Path) -> None:
    jump = JumpConfig(target_layer=1, source_layer=2, exit_layer=4, adaptation="lora", lora_rank=4)
    model = _lowrank(tiny_dir, jump)
    assert check_servable(model)["lowrank"] is Support.DENSIFIED
    plan = to_vllm(model, tmp_path / "out")
    assert plan.techniques["lowrank"] == "densified"
    assert any("lora" in note for note in plan.notes["lowrank"])
    assert "quantization_config" not in json.loads((Path(plan.model) / "config.json").read_text())
    reloaded = _base(Path(plan.model))
    assert set(reloaded.state_dict()) == set(make_tiny_model().state_dict())
    with torch.no_grad():
        torch.testing.assert_close(reloaded(TOKENS).logits, base_model_of(model)(TOKENS).logits, atol=1e-4, rtol=1e-4)


def test_a_compressed_tensors_base_is_refused(tiny_dir: Path, tmp_path: Path) -> None:
    model = _lowrank(tiny_dir, QuantConfig(format="int4", group_size=32, backend="compressed-tensors"))
    with pytest.raises(UnsupportedBackendError, match="one quantization method per model"):
        check_servable(model)


# ---- the factored arithmetic ---------------------------------------------------------------------------------------


def test_layer_units() -> None:
    ranks = {"model.layers.0.self_attn.q_proj": 3, "model.layers.0.mlp.down_proj": 2}
    thresholds = {"model.layers.0.self_attn.k_proj": 0.5, "model.layers.0.self_attn.q_proj": 0.25}
    factor = {"model.layers.0.self_attn.q_proj": 0.1, "model.layers.0.self_attn.k_proj": 0.7}
    units = layer_units("model.layers.0.self_attn.qkv_proj", PACKED, ranks, thresholds, factor)
    assert units == [Unit("model.layers.0.self_attn.q_proj", 3, 0.25, 0.1),
                     Unit("model.layers.0.self_attn.k_proj", 0, 0.5, 0.0),  # a dense unit has no latent
                     Unit("model.layers.0.self_attn.v_proj")]
    assert layer_units("model.layers.0.mlp.down_proj", PACKED, ranks, {}, {}) == [
        Unit("model.layers.0.mlp.down_proj", 2)]
    assert layer_units("model.layers.0.mlp.gate_up_proj", PACKED, ranks, thresholds, factor) is None


def _reference(x: torch.Tensor, units: list[Unit], tensors: dict[str, dict[str, torch.Tensor]],
               bias: torch.Tensor | None) -> torch.Tensor:
    parts = []
    for unit in units:
        inputs = sparsify(x, unit.threshold) if unit.threshold else x
        if unit.factored:
            latent = inputs @ tensors[unit.name]["down"].t()
            latent = sparsify(latent, unit.factor_threshold) if unit.factor_threshold else latent
            parts.append(latent @ tensors[unit.name]["up"].t())
        else:
            parts.append(inputs @ tensors[unit.name]["weight"].t())
    out = torch.cat(parts, dim=-1)
    return out + bias if bias is not None else out


def _tensors(units: list[Unit], sizes: list[int], width: int, seed: int) -> dict[str, dict[str, torch.Tensor]]:
    generator = torch.Generator().manual_seed(seed)
    found = {}
    for unit, size in zip(units, sizes, strict=True):
        if unit.factored:
            found[unit.name] = {"down": torch.randn(unit.rank, width, generator=generator),
                                "up": torch.randn(size, unit.rank, generator=generator)}
        else:
            found[unit.name] = {"weight": torch.randn(size, width, generator=generator)}
    return found


def test_factored_linear_matches_the_hf_factors() -> None:
    units = [Unit("q", 3, 0.5, 0.25), Unit("k", 0, 0.3), Unit("v", 2)]
    sizes = [8, 4, 4]
    tensors = _tensors(units, sizes, 16, seed=2)
    bias = torch.randn(16, generator=torch.Generator().manual_seed(3))
    x = torch.randn(2, 5, 16, generator=torch.Generator().manual_seed(4))
    down = torch.cat([tensors[u.name]["down"] for u in units if u.factored])
    up = torch.cat([tensors[u.name]["up"].t().reshape(-1) for u in units if u.factored])
    args = (bias, tensors["k"]["weight"], torch.tensor([0.3]), down, torch.tensor([0.5, 0.0]), up,
            torch.tensor([0.25, 0.0]), sizes, [3, 0, 2])
    torch.testing.assert_close(factored_linear(x, *args, True), _reference(x, units, tensors, bias))
    dense = [Unit(u.name, u.rank) for u in units]
    torch.testing.assert_close(factored_linear(x, *args, False), _reference(x, dense, tensors, bias))
    with pytest.raises(UnsupportedBackendError, match="dense units"):
        factored_linear(x, bias, None, *args[2:], True)


# ---- vLLM's layers -------------------------------------------------------------------------------------------------


def _config(**sections: Any) -> Any:
    VLLM_EXTENSIONS.get("linear")()
    from economize.backends.vllm.linear.method import EconomizeLinearConfig

    config = EconomizeLinearConfig.from_config({"quant_method": QUANT_METHOD, **sections})
    config.packed_modules_mapping = PACKED
    return config


def _tp(monkeypatch: pytest.MonkeyPatch, rank: int, size: int) -> None:
    import vllm.model_executor.layers.linear as linear
    import vllm.model_executor.parameter as parameter

    for module in (linear, parameter):
        monkeypatch.setattr(module, "get_tensor_model_parallel_rank", lambda: rank)
        monkeypatch.setattr(module, "get_tensor_model_parallel_world_size", lambda: size)


QKV = "model.layers.0.self_attn.qkv_proj"
NAMES = ["model.layers.0.self_attn.q_proj", "model.layers.0.self_attn.k_proj", "model.layers.0.self_attn.v_proj"]


def _qkv(config: Any) -> Any:
    from vllm.model_executor.layers.linear import QKVParallelLinear

    return QKVParallelLinear(16, 4, 4, 2, bias=True, params_dtype=torch.float32, quant_config=config, prefix=QKV)


def _load(layer: Any, units: list[Unit], tensors: dict[str, dict[str, torch.Tensor]], shard_ids: list[Any]) -> None:
    for unit, shard_id in zip(units, shard_ids, strict=True):
        for kind, key in (("lowrank_down", "down"), ("lowrank_up", "up")) if unit.factored else (("weight", "weight"),):
            param = getattr(layer, kind)
            param.weight_loader(param, tensors[unit.name][key], shard_id)


@pytest.mark.parametrize("tp", [1, 2, 4])
def test_qkv_layer_with_factored_and_dense_shards(monkeypatch: pytest.MonkeyPatch, tp: int) -> None:
    from economize.backends.vllm.linear.method import FactoredLinearMethod

    ranks = {NAMES[0]: 3, NAMES[2]: 2}
    thresholds = {NAMES[0]: 0.5, NAMES[1]: 0.3}
    factor = {NAMES[0]: 0.25}
    units = layer_units(QKV, PACKED, ranks, thresholds, factor)
    sizes = [16, 8, 8]  # 4 heads, 2 KV heads of size 4
    tensors = _tensors(units, sizes, 16, seed=5)
    bias = torch.randn(32, generator=torch.Generator().manual_seed(6))
    x = torch.randn(3, 16, generator=torch.Generator().manual_seed(7))
    outputs = []
    for rank in range(tp):
        _tp(monkeypatch, rank, tp)
        config = _config(teal={"thresholds": thresholds, "factor_thresholds": factor, "single": "all",
                               "multi": "all"}, lowrank={"ranks": ranks})
        layer = _qkv(config)
        assert isinstance(layer.quant_method, FactoredLinearMethod) and config.matched == {QKV: "factored"}
        _load(layer, units, tensors, ["q", "k", "v"])
        for shard_id, (start, end) in zip("qkv", ((0, 16), (16, 24), (24, 32)), strict=True):
            layer.weight_loader(layer.bias, bias[start:end], shard_id)
        layer.quant_method.process_weights_after_loading(layer)
        outputs.append(layer.quant_method.apply(layer, x, layer.bias))
    expected = _reference(x, units, tensors, bias)
    replicas = max(tp // 2, 1)  # KV heads are replicated beyond 2 ranks
    q, kv = 16 // tp, 8 // tp * replicas
    gathered = [torch.cat([out[:, :q] for out in outputs], dim=-1)]
    for shard in range(2):
        start = q + shard * kv
        gathered.append(torch.cat([out[:, start:start + kv] for out in outputs[::replicas]], dim=-1))
    torch.testing.assert_close(torch.cat(gathered, dim=-1), expected)


@pytest.mark.parametrize("tp", [1, 2])
def test_row_parallel_factored_layer(monkeypatch: pytest.MonkeyPatch, tp: int) -> None:
    from vllm.model_executor.layers.linear import RowParallelLinear

    name = "model.layers.0.mlp.down_proj"
    unit = Unit(name, 3, 0.4)
    tensors = _tensors([unit], [12], 16, seed=8)
    bias = torch.randn(12, generator=torch.Generator().manual_seed(9))
    x = torch.randn(4, 16, generator=torch.Generator().manual_seed(10))
    total = torch.zeros(4, 12)
    for rank in range(tp):
        _tp(monkeypatch, rank, tp)
        layer = RowParallelLinear(16, 12, bias=True, params_dtype=torch.float32, prefix=name,
                                  quant_config=_config(teal={"thresholds": {name: 0.4}, "multi": "all"},
                                                       lowrank={"ranks": {name: 3}}))
        _load(layer, [unit], tensors, [None])
        layer.weight_loader(layer.bias, bias)
        layer.quant_method.process_weights_after_loading(layer)
        width = 16 // tp
        part = x[:, rank * width:(rank + 1) * width]
        total += layer.quant_method.apply(layer, part, layer.bias if rank == 0 else None)
    torch.testing.assert_close(total, _reference(x, [unit], tensors, bias))
    if tp > 1:
        with pytest.raises(UnsupportedBackendError, match="row-parallel"):
            RowParallelLinear(16, 12, params_dtype=torch.float32, prefix=name, quant_config=_config(
                teal={"factor_thresholds": {name: 0.1}}, lowrank={"ranks": {name: 3}}))


def test_loader_refusals(monkeypatch: pytest.MonkeyPatch) -> None:
    _tp(monkeypatch, 0, 1)
    ranks = {NAMES[0]: 3}
    layer = _qkv(_config(lowrank={"ranks": ranks}))
    units = layer_units(QKV, PACKED, ranks, {}, {})
    tensors = _tensors(units, [16, 8, 8], 16, seed=11)
    with pytest.raises(ConfigError, match="weight tensor for a factored map"):
        layer.weight.weight_loader(layer.weight, tensors[NAMES[1]]["weight"], "q")
    with pytest.raises(UnsupportedBackendError, match="fused checkpoint tensor"):
        layer.lowrank_down.weight_loader(layer.lowrank_down, tensors[NAMES[0]]["down"])
    with pytest.raises(ConfigError, match="shape"):
        layer.lowrank_down.weight_loader(layer.lowrank_down, torch.zeros(2, 16), "q")
    layer.lowrank_down.weight_loader(layer.lowrank_down, tensors[NAMES[0]]["down"], "q")
    with pytest.raises(ConfigError, match="lacks"):
        layer.quant_method.process_weights_after_loading(layer)
