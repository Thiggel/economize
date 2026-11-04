"""``economize.backends.vllm.serving``: plans for technique combinations, checked on CPU without vLLM.

The written folders are reloaded with ``transformers`` and compared with the economize HF runtime; the engine
arguments and ``speculative_config`` fields are checked against vLLM 0.30's names (the GPU tests in
``test_vllm_serving_gpu.py`` run vLLM on them).
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from safetensors.torch import load_file

import economize
from economize.backends import UnsupportedBackendError
from economize.backends.vllm import serving
from economize.backends.vllm.proposers import CONFIG_KEY, PROPOSER_CLASS
from economize.backends.vllm.serving import Role, Support, VllmPlan, check_servable, to_vllm
from economize.backends.vllm.serving.kv import kv_cache_settings
from economize.errors import ConfigError, IncompatibleTechniquesError
from economize.techniques.draft_model.config import DraftModelConfig
from economize.techniques.eagle3 import Eagle3Config
from economize.techniques.frspec import FRSpecConfig
from economize.techniques.jump import JumpConfig
from economize.techniques.kv_quant.fp8 import FP8KVConfig
from economize.techniques.lookup.config import LookupConfig
from economize.techniques.medusa import MedusaConfig
from economize.techniques.prune.config import DepthPruneConfig, PruneConfig
from economize.techniques.quant.config import DraftQuantConfig, QuantConfig
from economize.techniques.suffix_decoding.config import SuffixDecodingConfig
from helpers import VOCAB, make_tiny_model, perturb

needs_ct = pytest.mark.skipif(importlib.util.find_spec("compressed_tensors") is None, reason="needs compressed-tensors")
JUMP = JumpConfig(target_layer=2, source_layer=4)
TOKENS = torch.randint(0, VOCAB, (2, 16), generator=torch.Generator().manual_seed(0))


@pytest.fixture
def tiny_dir(tmp_path: Path) -> Path:
    """A saved tiny model, so the base has a ``name_or_path`` to serve."""
    folder = tmp_path / "tiny"
    make_tiny_model(seed=0).save_pretrained(folder)
    return folder


def _base(folder: Path) -> Any:
    from transformers import AutoModelForCausalLM

    return AutoModelForCausalLM.from_pretrained(folder, dtype=torch.float32).eval()


def _logits(model: Any) -> torch.Tensor:
    with torch.no_grad():
        return economize.unwrap.base_model_of(model)(TOKENS).logits


# ---- registry and coverage ----------------------------------------------------------------------------------------


def test_every_technique_has_an_adapter() -> None:
    from economize.core.technique import TECHNIQUES

    assert set(TECHNIQUES.names()) <= set(serving.VLLM_TECHNIQUES.names())
    rows = {row[0]: row for row in serving.support_table()}
    assert rows["jump"][2] == "plugin" and rows["calm"][2] == "unsupported" and rows["kv_fp8"][1] == "kv_cache"
    assert all(row[3] for row in rows.values())  # every adapter says what vLLM runs, or why it cannot


def test_a_plain_model_serves_its_folder(tiny_dir: Path, tmp_path: Path) -> None:
    plan = to_vllm(_base(tiny_dir), tmp_path / "out")
    assert plan.model == str(tiny_dir) and plan.speculative is None and plan.tokenizer is None
    assert plan.engine == {"dtype": "float32"}
    assert VllmPlan.load(tmp_path / "out") == plan


# ---- refusals before anything is written ------------------------------------------------------------------------


def test_unsupported_technique_is_refused_with_its_reason(tiny_dir: Path, tmp_path: Path) -> None:
    from economize.techniques.kv_evict.technique import KVEvictConfig

    model = economize.apply(_base(tiny_dir), KVEvictConfig(method="snapkv"))
    with pytest.raises(UnsupportedBackendError, match=r"kv_evict.*per KV head"):
        check_servable(model)
    with pytest.raises(UnsupportedBackendError):
        to_vllm(model, tmp_path / "out")
    assert not (tmp_path / "out").exists()


def test_two_kv_cache_techniques_are_refused(tiny_dir: Path) -> None:
    builder = serving.PlanBuilder(_base(tiny_dir), None)
    builder.claim(Role.KV_CACHE, "kv_fp8")
    with pytest.raises(IncompatibleTechniquesError, match=r"kv_fp8.*kivi|kivi.*kv_fp8"):
        builder.claim(Role.KV_CACHE, "kivi")
    builder.set_engine("dtype", "float32", owner="a")
    with pytest.raises(IncompatibleTechniquesError):
        builder.set_engine("dtype", "bfloat16", owner="b")


def test_draft_quant_without_a_drafter_is_refused(tiny_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    model = economize.apply(_base(tiny_dir), JUMP, DraftQuantConfig())
    assert check_servable(model) == {"jump": Support.PLUGIN, "draft_quant": Support.DENSIFIED}
    techniques = dict(model.techniques)
    techniques.pop("jump")
    monkeypatch.setattr(serving.PlanBuilder, "techniques", property(lambda self: techniques))
    with pytest.raises(UnsupportedBackendError, match="none is attached"):
        check_servable(model)


def test_quant_levels_and_refusals(tiny_dir: Path) -> None:
    adapter = serving.base.adapter_for("quant")
    fake = economize.apply(_base(tiny_dir), QuantConfig(format="int4", group_size=32)).techniques["quant"]
    assert adapter.level(fake) is Support.DENSIFIED
    for backend in ("compressed-tensors", "marlin", "machete", "torchao"):  # codes recorded or read back
        fake.config = QuantConfig(format="int4", group_size=64, backend=backend)
        assert adapter.level(fake) is Support.NATIVE, backend
    # a plugin backend without a registered vLLM level (and without readable codes) is refused
    fake.config = SimpleNamespace(backend="my-kernel", format="int4", fp8_activations=False)
    with pytest.raises(UnsupportedBackendError, match="compressed-tensors"):
        adapter.check(fake, serving.PlanBuilder(None, None))


# ---- weights ------------------------------------------------------------------------------------------------------


def test_fake_quant_writes_the_dequantized_base(tiny_dir: Path, tmp_path: Path) -> None:
    model = economize.apply(_base(tiny_dir), QuantConfig(format="int4", group_size=32))
    plan = to_vllm(model, tmp_path / "out")
    assert plan.model == str(tmp_path / "out" / "model") and plan.tokenizer == str(tiny_dir)
    assert plan.techniques == {"quant": "densified"}
    torch.testing.assert_close(_base(Path(plan.model))(TOKENS).logits, _logits(model), atol=1e-5, rtol=1e-5)


def test_unstructured_prune_is_native_and_reloads(tiny_dir: Path, tmp_path: Path) -> None:
    model = economize.apply(_base(tiny_dir), PruneConfig(method="magnitude", sparsity=0.5))
    plan = to_vllm(model, tmp_path / "out")
    assert plan.techniques == {"prune": "native"}
    torch.testing.assert_close(_base(Path(plan.model))(TOKENS).logits, _logits(model), atol=1e-5, rtol=1e-5)


def test_depth_prune_writes_the_shallower_model(tiny_dir: Path, tmp_path: Path) -> None:
    model = economize.apply(_base(tiny_dir), DepthPruneConfig(method="shortgpt", num_blocks=2), calibration=[TOKENS])
    plan = to_vllm(model, tmp_path / "out")
    config = json.loads((Path(plan.model) / "config.json").read_text())
    assert config["num_hidden_layers"] == 4
    torch.testing.assert_close(_base(Path(plan.model))(TOKENS).logits, _logits(model), atol=1e-5, rtol=1e-5)


@needs_ct
def test_compressed_tensors_base_with_jump_and_fp8_kv(tiny_dir: Path, tmp_path: Path) -> None:
    """The composition of the docs: int4 weights, a jump drafter, an fp8 KV cache in one plan."""
    model = economize.apply(_base(tiny_dir), QuantConfig(format="int4", group_size=32, backend="compressed-tensors"),
                            FP8KVConfig(), JUMP)
    perturb(model.techniques["jump"], 0.02, seed=1)
    plan = to_vllm(model, tmp_path / "out", num_speculative_tokens=3)
    config = json.loads((Path(plan.model) / "config.json").read_text())
    assert config["quantization_config"]["quant_method"] == "compressed-tensors"
    assert plan.engine["kv_cache_dtype"] == "fp8_e4m3"
    assert plan.speculative is not None and plan.speculative["method"] == "eagle3"
    assert plan.speculative["model"] == str(tmp_path / "out" / "drafter")
    assert plan.speculative["num_speculative_tokens"] == 3
    assert plan.techniques == {"quant": "native", "kv_fp8": "native", "jump": "plugin"}
    drafter = json.loads((tmp_path / "out" / "drafter" / "config.json").read_text())
    assert drafter["architectures"] == ["EconomizeJumpEagle3"]


@needs_ct
def test_jump_lora_on_a_compressed_tensors_base_is_refused(tiny_dir: Path) -> None:
    model = economize.apply(_base(tiny_dir), QuantConfig(format="int4", group_size=32, backend="compressed-tensors"),
                            JumpConfig(target_layer=2, source_layer=4, adaptation="lora"))
    with pytest.raises(UnsupportedBackendError, match="adaptation='lora'"):
        check_servable(model)


@needs_ct
def test_jump_copy_on_a_compressed_tensors_base(tmp_path: Path, tiny_dir: Path) -> None:
    """A ``full`` jump path next to an int4 target: the copies are served dense (``_dense_linears`` in vLLM)."""
    model = economize.apply(_base(tiny_dir), QuantConfig(format="int4", group_size=32, backend="compressed-tensors"),
                            JumpConfig(target_layer=2, source_layer=4, adaptation="full"))
    check_servable(model)
    plan = to_vllm(model, tmp_path / "out", num_speculative_tokens=3)
    assert plan.techniques == {"quant": "native", "jump": "plugin"}


# ---- KV cache -----------------------------------------------------------------------------------------------------


def test_kv_cache_settings() -> None:
    assert kv_cache_settings("e5m2", "none", (), 6) == {"kv_cache_dtype": "fp8_e5m2"}
    assert kv_cache_settings("e4m3", "token", (), 6) == {"kv_cache_dtype": "fp8_per_token_head"}
    assert kv_cache_settings("e4m3", "none", (1, 2, 6), 6)["kv_cache_dtype_skip_layers"] == ["2", "3", "4"]
    with pytest.raises(UnsupportedBackendError, match="e4m3 only"):
        kv_cache_settings("e5m2", "token", (), 6)


def test_calibrated_kv_scales_settings() -> None:
    assert kv_cache_settings("e4m3", "tensor", (), 6) == {"kv_cache_dtype": "fp8_e4m3"}
    with pytest.raises(UnsupportedBackendError, match="fp8_e5m2"):
        kv_cache_settings("e5m2", "tensor", (), 6)


def _saved(folder: Path) -> dict[str, torch.Tensor]:
    state: dict[str, torch.Tensor] = {}
    for shard in folder.glob("*.safetensors"):
        state.update(load_file(str(shard)))
    return state


def _scales(folder: Path) -> dict[str, torch.Tensor]:
    return {key: value for key, value in _saved(folder).items() if key.endswith(("k_scale", "v_scale"))}


def test_calibrated_kv_scales_are_written_with_a_kv_scheme(tiny_dir: Path, tmp_path: Path) -> None:
    model = economize.apply(_base(tiny_dir), FP8KVConfig(scale="tensor", layers=(1, 3)), calibration=TOKENS)
    plan = to_vllm(model, tmp_path / "out")
    folder = Path(plan.model)
    quantization = json.loads((folder / "config.json").read_text())["quantization_config"]
    assert quantization["quant_method"] == "compressed-tensors" and quantization["config_groups"] == {}
    assert quantization["kv_cache_scheme"] == {"num_bits": 8, "type": "float", "strategy": "tensor",
                                                "dynamic": False, "symmetric": True}
    assert plan.engine["kv_cache_dtype"] == "fp8_e4m3"
    assert plan.engine["kv_cache_dtype_skip_layers"] == ["1", "3", "4", "5"]
    technique = model.techniques["kv_fp8"]
    scales = _scales(folder)
    assert sorted(scales) == sorted(f"model.layers.{i}.self_attn.{kind}_scale" for i in range(6) for kind in "kv")
    assert torch.equal(scales["model.layers.2.self_attn.k_scale"], technique.key_scale[2:3])
    assert torch.equal(scales["model.layers.0.self_attn.v_scale"], technique.value_scale[0:1])
    weights = {key: value for key, value in _saved(folder).items() if key not in scales}
    plain = _base(tiny_dir).state_dict()
    assert set(weights) <= set(plain) and all(torch.equal(value, plain[key]) for key, value in weights.items())


@needs_ct
def test_calibrated_kv_scales_join_compressed_tensors_weights(tiny_dir: Path, tmp_path: Path) -> None:
    model = economize.apply(_base(tiny_dir), QuantConfig(format="int4", group_size=32, backend="compressed-tensors"),
                            FP8KVConfig(scale="tensor"), calibration=TOKENS)
    folder = Path(to_vllm(model, tmp_path / "out").model)
    quantization = json.loads((folder / "config.json").read_text())["quantization_config"]
    assert quantization["config_groups"] and quantization["kv_cache_scheme"]["strategy"] == "tensor"
    assert len(_scales(folder)) == 2 * model.adapter.num_layers


def test_uncalibrated_kv_scales_are_refused(tiny_dir: Path) -> None:
    model = economize.apply(_base(tiny_dir), FP8KVConfig(scale="tensor"), calibration=TOKENS)
    model.techniques["kv_fp8"].key_scale.fill_(float("nan"))
    with pytest.raises(ConfigError, match="not calibrated"):
        to_vllm(model, "/nonexistent-economize-out")


# ---- drafters -----------------------------------------------------------------------------------------------------


def test_lookup_maps_to_ngram(tiny_dir: Path, tmp_path: Path) -> None:
    model = economize.apply(_base(tiny_dir), LookupConfig(max_ngram=4, min_ngram=2))
    plan = to_vllm(model, tmp_path / "out", num_speculative_tokens=5)
    assert plan.speculative == {"method": "ngram", "prompt_lookup_max": 4, "prompt_lookup_min": 2,
                                "num_speculative_tokens": 5}
    model = economize.apply(_base(tiny_dir), LookupConfig(match="latest", max_proposal=3))
    assert check_servable(model) == {"lookup": Support.PLUGIN}
    plan = to_vllm(model, tmp_path / "latest")
    assert plan.speculative == {"method": "custom_class", "model": PROPOSER_CLASS, "num_speculative_tokens": 3}
    assert plan.engine["additional_config"][CONFIG_KEY]["technique"] == "lookup"
    assert plan.engine["additional_config"][CONFIG_KEY]["config"]["match"] == "latest"


def test_suffix_decoding_maps_to_suffix(tiny_dir: Path, tmp_path: Path) -> None:
    model = economize.apply(_base(tiny_dir), SuffixDecodingConfig(max_depth=32, max_spec_tokens=12))
    plan = to_vllm(model, tmp_path / "out")
    assert plan.speculative == {"method": "suffix", "num_speculative_tokens": 12,
                                "suffix_decoding_max_tree_depth": 32, "suffix_decoding_max_spec_factor": 1.0,
                                "suffix_decoding_min_token_prob": 0.1}
    assert any("arctic-inference" in note for note in plan.notes["suffix_decoding"])
    model = economize.apply(_base(tiny_dir), SuffixDecodingConfig(max_spec_offset=1.0))
    assert check_servable(model) == {"suffix_decoding": Support.PLUGIN}
    assert to_vllm(model, tmp_path / "offset").speculative["method"] == "custom_class"


def test_draft_model_is_saved(tiny_dir: Path, tmp_path: Path) -> None:
    draft = tmp_path / "draft"
    make_tiny_model(seed=3, num_layers=2).save_pretrained(draft)
    model = economize.apply(_base(tiny_dir), DraftModelConfig(model=str(draft), dtype="float32"))
    plan = to_vllm(model, tmp_path / "out", sampling=False)
    assert plan.speculative == {"method": "draft_model", "model": str(tmp_path / "out" / "drafter"),
                                "num_speculative_tokens": 4}
    assert json.loads((tmp_path / "out" / "drafter" / "config.json").read_text())["num_hidden_layers"] == 2


def test_pard_writes_its_mask_token(tiny_dir: Path, tmp_path: Path) -> None:
    from economize.techniques.pard.config import PardConfig

    draft = tmp_path / "draft"
    make_tiny_model(seed=3, num_layers=2).save_pretrained(draft)
    model = economize.apply(_base(tiny_dir), PardConfig(model=str(draft), dtype="float32", mask_token_id=VOCAB - 1))
    plan = to_vllm(model, tmp_path / "out")
    assert plan.speculative is not None and plan.speculative["parallel_drafting"] is True
    assert plan.speculative["draft_sample_method"] == "probabilistic"
    config = json.loads((tmp_path / "out" / "drafter" / "config.json").read_text())
    assert config["pard_token"] == VOCAB - 1


def test_medusa_drafts_at_most_its_heads(tiny_dir: Path, tmp_path: Path) -> None:
    model = economize.apply(_base(tiny_dir), MedusaConfig(num_heads=3, num_layers=1))
    with pytest.raises(UnsupportedBackendError, match="at most 3"):
        check_servable(model, num_speculative_tokens=4)
    plan = to_vllm(model, tmp_path / "out", num_speculative_tokens=3)
    assert plan.speculative == {"method": "medusa", "model": str(tmp_path / "out" / "drafter"),
                                "num_speculative_tokens": 3}


def test_eagle3_zero_source_first_is_refused(tiny_dir: Path) -> None:
    model = economize.apply(_base(tiny_dir), Eagle3Config(zero_source_first=True))
    with pytest.raises(UnsupportedBackendError, match="zero_source_first"):
        check_servable(model)


@pytest.mark.parametrize("types", [("sliding_attention",) * 2, ("sliding_attention", "full_attention")])
def test_dflash_is_served_natively(types: tuple[str, ...], tiny_dir: Path, tmp_path: Path) -> None:
    from economize.techniques.dflash import DFlashConfig

    config = DFlashConfig(num_layers=2, layer_types=types, block_size=5, sliding_window=4, mask_token_id=4,
                          target_layers=(1, 3))
    model = economize.apply(_base(tiny_dir), config)
    with pytest.raises(UnsupportedBackendError, match="at most 4"):
        check_servable(model, num_speculative_tokens=5)
    plan = to_vllm(model, tmp_path / "out", num_speculative_tokens=4, sampling=False)
    assert plan.techniques["dflash"] == Support.NATIVE.value
    assert plan.speculative == {"method": "dflash", "model": str(tmp_path / "out" / "drafter"),
                                "num_speculative_tokens": 4}
    assert json.loads((tmp_path / "out" / "drafter" / "config.json").read_text())["block_size"] == 5
    assert ("VLLM_USE_V2_MODEL_RUNNER" in plan.env) == (len(set(types)) > 1)


@pytest.mark.parametrize("head", ["own", "shared"])
def test_frspec_restricts_the_eagle3_head(head: str, tiny_dir: Path, tmp_path: Path) -> None:
    ids = torch.randperm(VOCAB, generator=torch.Generator().manual_seed(4))[:40]
    model = economize.apply(_base(tiny_dir), FRSpecConfig(drafter=Eagle3Config(head=head), vocab_size=40,
                                                          token_ids=tuple(ids.tolist())))
    technique = model.techniques["frspec"]
    perturb(technique.inner, 0.05, seed=1)
    plan = to_vllm(model, tmp_path / "out", num_speculative_tokens=2)
    assert plan.speculative is not None and plan.speculative["method"] == "eagle3"
    folder = tmp_path / "out" / "drafter"
    config = json.loads((folder / "config.json").read_text())
    tensors = load_file(str(folder / "model.safetensors"))
    assert config["draft_vocab_size"] == 40 and config["vocab_size"] == VOCAB
    targets = torch.arange(40) + tensors["d2t"]  # vllm llama_eagle3 compute_logits
    assert targets.tolist() == sorted(ids.tolist())
    assert int(tensors["t2d"].sum()) == 40 and bool(tensors["t2d"][targets].all())
    inner = technique.inner
    head_weight = (inner.lm_head if inner.lm_head is not None else inner.adapter.lm_head).weight.detach()
    torch.testing.assert_close(tensors["lm_head.weight"], head_weight[targets].to(tensors["lm_head.weight"]))


def test_frspec_around_jump_is_refused(tiny_dir: Path) -> None:
    model = economize.apply(_base(tiny_dir), FRSpecConfig(drafter=JUMP, vocab_size=40,
                                                          token_ids=tuple(range(40))))
    with pytest.raises(UnsupportedBackendError, match="EAGLE-3"):
        check_servable(model)


def test_jump_with_draft_quant(tiny_dir: Path, tmp_path: Path) -> None:
    model = economize.apply(_base(tiny_dir), JUMP, DraftQuantConfig(format="int8"))
    plan = to_vllm(model, tmp_path / "out")
    assert plan.model == str(tiny_dir)  # the base is unchanged, vLLM loads it from its folder
    assert plan.techniques == {"jump": "plugin", "draft_quant": "densified"}
    tensors = load_file(str(tmp_path / "out" / "drafter" / "model.safetensors"))
    state = model.techniques["jump"].state_dict()
    for key, value in tensors.items():
        torch.testing.assert_close(value, state[key.removeprefix("jump.")].float())


# ---- the plan -----------------------------------------------------------------------------------------------------


def test_plan_round_trip_and_serve_args(tiny_dir: Path, tmp_path: Path) -> None:
    model = economize.apply(_base(tiny_dir), FP8KVConfig(layers=(1, 2)), LookupConfig())
    plan = to_vllm(model, tmp_path / "out")
    assert VllmPlan.load(tmp_path / "out") == plan
    args = plan.serve_args()
    assert args[:2] == ["serve", str(tiny_dir)]
    skip = args.index("--kv-cache-dtype-skip-layers")
    assert args[skip + 1 : skip + 5] == ["2", "3", "4", "5"]
    spec = json.loads(args[args.index("--speculative-config") + 1])
    assert spec["method"] == "ngram"
    kwargs = plan.llm_kwargs(gpu_memory_utilization=0.5)
    assert kwargs["kv_cache_dtype"] == "fp8_e4m3" and kwargs["gpu_memory_utilization"] == 0.5


@pytest.mark.parametrize("plugin", [True, False])
def test_llm_installs_the_plugin_when_needed(monkeypatch: pytest.MonkeyPatch, plugin: bool) -> None:
    import types

    from economize.backends.vllm import plugin as vllm_plugin

    calls: list[str] = []
    monkeypatch.setattr(vllm_plugin, "register", lambda: calls.append("register"))
    monkeypatch.setitem(sys.modules, "vllm", types.SimpleNamespace(LLM=lambda **kwargs: calls.append("llm") or kwargs))
    techniques = {"jump": "plugin"} if plugin else {"lookup": "native"}
    plan = VllmPlan(model="m", engine={"dtype": "float32"}, techniques=techniques)
    assert plan.llm(seed=0) == {"model": "m", "dtype": "float32", "seed": 0}
    assert calls == (["register", "llm"] if plugin else ["llm"])


@pytest.mark.parametrize("preset", [None, "fork"])
def test_llm_spawns_the_engine_unless_told_otherwise(monkeypatch: pytest.MonkeyPatch, preset: str | None) -> None:
    import os
    import types

    if preset is None:
        monkeypatch.delenv("VLLM_WORKER_MULTIPROC_METHOD", raising=False)
    else:
        monkeypatch.setenv("VLLM_WORKER_MULTIPROC_METHOD", preset)
    monkeypatch.setitem(sys.modules, "vllm", types.SimpleNamespace(LLM=lambda **kwargs: kwargs))
    VllmPlan(model="m").llm()
    assert os.environ["VLLM_WORKER_MULTIPROC_METHOD"] == (preset or "spawn")


SPEC = {"method": "eagle3", "model": "d", "num_speculative_tokens": 3}


@pytest.mark.parametrize(("engine", "techniques", "overrides", "backend"), [
    ({"kv_cache_dtype": "fp8_e4m3"}, {"kv_fp8": "native", "jump": "plugin"}, {}, "TRITON_ATTN"),
    ({"kv_cache_dtype": "fp8_e4m3"}, {"kv_fp8": "native", "jump": "plugin"},
     {"attention_config": {"backend": "FLASHINFER"}}, "FLASHINFER"),
    ({"kv_cache_dtype": "fp8_e4m3"}, {"kv_fp8": "native", "jump": "plugin"},
     {"attention_backend": "FLASH_ATTN"}, None),
    ({"kv_cache_dtype": "fp8_e4m3"}, {"kv_fp8": "native", "eagle3": "native"}, {}, None),
    ({}, {"jump": "plugin"}, {}, None),
])
def test_fp8_cache_with_a_plugin_drafter_picks_triton_attention(engine: dict[str, Any], techniques: dict[str, str],
                                                                 overrides: dict[str, Any], backend: str | None,
                                                                 ) -> None:
    plan = VllmPlan(model="m", engine=engine, speculative=SPEC, techniques=techniques)
    kwargs = plan.llm_kwargs(**overrides)
    assert (kwargs.get("attention_config") or {}).get("backend") == backend
    # serve_args sees the plan only: it names Triton whenever the plan names no backend
    assert ("--attention-config" in plan.serve_args()) == ("jump" in techniques and "kv_cache_dtype" in engine)


def test_config_edits_leave_the_model_unchanged(tiny_dir: Path, tmp_path: Path) -> None:
    base = _base(tiny_dir)
    builder = serving.PlanBuilder(base, tmp_path)
    builder.config_edits["num_nextn_predict_layers"] = 1
    serving.base._write_base(builder)
    assert not hasattr(base.config, "num_nextn_predict_layers")
    assert json.loads((tmp_path / "model" / "config.json").read_text())["num_nextn_predict_layers"] == 1


def test_a_registered_adapter_extends_the_backend(tiny_dir: Path, tmp_path: Path) -> None:
    """Open-closed: a KV technique of another package is served by registering one class."""

    class MyKV(serving.VllmTechnique):
        technique = "kv_fp8"
        role = Role.KV_CACHE
        support = Support.NATIVE

        def contribute(self, technique: Any, builder: serving.PlanBuilder) -> None:
            builder.set_engine("kv_cache_dtype", "fp8_e5m2", owner=self.technique)

    original = serving.VLLM_TECHNIQUES.get("kv_fp8")
    serving.register_vllm_technique(MyKV, override=True)
    try:
        model = economize.apply(_base(tiny_dir), FP8KVConfig())
        assert to_vllm(model, tmp_path / "out").engine["kv_cache_dtype"] == "fp8_e5m2"
    finally:
        serving.register_vllm_technique(original, override=True)


def test_plan_path_raises_while_checking() -> None:
    with pytest.raises(ConfigError, match="only checks"):
        serving.PlanBuilder(None, None).path("model")
