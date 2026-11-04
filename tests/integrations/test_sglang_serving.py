"""``economize.backends.sglang``: plans and draft checkpoints for SGLang, checked on CPU without SGLang.

The written folders are reloaded with ``transformers`` (base checkpoints) or ``safetensors`` (draft checkpoints) and
compared with the economize HF runtime; the server argument names are SGLang 0.5's. The GPU tests in
``test_sglang_serving_gpu.py`` run SGLang on these folders.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
import torch
from safetensors.torch import load_file

import economize
from economize.backends import UnsupportedBackendError
from economize.backends.exporters import EXPORTERS
from economize.backends.sglang import serving
from economize.backends.sglang.export import (
    DRAFTER_ARCHITECTURE,
    SGLANG_CONFIG_KEY,
    JumpSglangExporter,
    sglang_aux_ids,
)
from economize.backends.sglang.serving import SglangPlan, Support, check_servable, to_sglang
from economize.errors import IncompatibleTechniquesError
from economize.techniques.draft_model.config import DraftModelConfig
from economize.techniques.eagle3 import Eagle3Config
from economize.techniques.early_exit.config import EarlyExitConfig
from economize.techniques.jump import JumpConfig
from economize.techniques.kv_quant.fp8 import FP8KVConfig
from economize.techniques.lookup.config import LookupConfig
from economize.techniques.quant.config import QuantConfig
from helpers import VOCAB, make_tiny_model, perturb

needs_ct = pytest.mark.skipif(importlib.util.find_spec("compressed_tensors") is None, reason="needs compressed-tensors")
JUMP = JumpConfig(target_layer=2, source_layer=4)
TOKENS = torch.randint(0, VOCAB, (2, 16), generator=torch.Generator().manual_seed(0))


@pytest.fixture
def tiny_dir(tmp_path: Path) -> Path:
    folder = tmp_path / "tiny"
    make_tiny_model(seed=0).save_pretrained(folder)
    return folder


def _base(folder: Path) -> Any:
    from transformers import AutoModelForCausalLM

    return AutoModelForCausalLM.from_pretrained(folder, dtype=torch.float32).eval()


def _logits(model: Any) -> torch.Tensor:
    with torch.no_grad():
        return economize.unwrap.base_model_of(model)(TOKENS).logits


def _jump(tiny_dir: Path, **options: Any) -> Any:
    model = economize.apply(_base(tiny_dir), JumpConfig(target_layer=2, source_layer=4, **options))
    perturb(model.techniques["jump"], 0.02, seed=1)
    return model


# ---- registry and coverage ----------------------------------------------------------------------------------------


def test_every_technique_has_an_adapter() -> None:
    from economize.core.technique import TECHNIQUES

    assert set(TECHNIQUES.names()) <= set(serving.SGLANG_TECHNIQUES.names())
    rows = {row[0]: row for row in serving.support_table()}
    for name in ("jump", "early_exit", "layer_skip", "free", "dmtd"):
        assert rows[name][2] == "plugin", name
    for name in ("eagle3", "lookup", "draft_model", "kv_fp8", "quant", "vendor_mtp", "gemma4_assistant"):
        assert rows[name][2] == "native", name
    for name in ("kivi", "kv_evict", "medusa", "hydra", "calm", "teal", "rest"):
        assert rows[name][2] == "unsupported", name
    assert all(row[3] for row in rows.values())  # every adapter says what SGLang runs, or why it cannot


def test_importing_the_backend_imports_no_engine() -> None:
    code = ("import sys, economize.backends.sglang, economize.backends.sglang.serving; "
            "print(sorted(m for m in sys.modules if m.split('.')[0] in ('sglang', 'vllm')))")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout
    assert out.strip() == "[]"


def test_a_plain_model_serves_its_folder(tiny_dir: Path, tmp_path: Path) -> None:
    plan = to_sglang(_base(tiny_dir), tmp_path / "out")
    assert plan.model_path == str(tiny_dir) and plan.tokenizer is None
    assert plan.server_args == {"dtype": "float32"} and not plan.needs_plugin
    assert SglangPlan.load(tmp_path / "out") == plan
    args = plan.launch_args(port=30001)
    assert args[:3] == ["python", "-m", "sglang.launch_server"] and "--model-path" in args and "30001" in args


# ---- refusals -----------------------------------------------------------------------------------------------------


def test_unsupported_technique_is_refused_with_its_reason(tiny_dir: Path, tmp_path: Path) -> None:
    from economize.techniques.kv_evict.technique import KVEvictConfig

    model = economize.apply(_base(tiny_dir), KVEvictConfig(method="snapkv"))
    with pytest.raises(UnsupportedBackendError, match="kv_evict"):
        check_servable(model)
    with pytest.raises(UnsupportedBackendError):
        to_sglang(model, tmp_path / "out")
    assert not (tmp_path / "out").exists()


def test_kivi_is_refused(tiny_dir: Path) -> None:
    from economize.techniques.kv_quant.kivi import KIVIConfig

    with pytest.raises(UnsupportedBackendError, match="kivi"):
        check_servable(economize.apply(_base(tiny_dir), KIVIConfig(bits=4)))


def test_two_engine_values_for_one_argument_are_refused(tiny_dir: Path) -> None:
    builder = serving.SglangPlanBuilder(_base(tiny_dir), None, 4, sampling=False)
    builder.set_engine("kv_cache_dtype", "fp8_e4m3", owner="kv_fp8")
    with pytest.raises(IncompatibleTechniquesError):
        builder.set_engine("kv_cache_dtype", "fp8_e5m2", owner="other")


def test_jump_shapes_sglang_cannot_capture_are_refused(tiny_dir: Path) -> None:
    exporter = JumpSglangExporter()
    # u = L: SGLang captures the input of a layer, so the final-layer state is out of reach (unless m == u)
    model = economize.apply(_base(tiny_dir), JumpConfig(target_layer=2, source_layer=6))
    with pytest.raises(UnsupportedBackendError, match=r"h\^L.*vLLM"):
        exporter.check(model)


@needs_ct
def test_jump_lora_on_a_compressed_tensors_base_is_refused(tiny_dir: Path) -> None:
    model = economize.apply(_base(tiny_dir), QuantConfig(format="int4", group_size=32, backend="compressed-tensors"),
                            JumpConfig(target_layer=2, source_layer=4, adaptation="lora"))
    with pytest.raises(UnsupportedBackendError, match="compressed-tensors base"):
        check_servable(model)


# ---- the jump drafter ---------------------------------------------------------------------------------------------


def test_aux_ids_are_sglang_layer_inputs() -> None:
    assert sglang_aux_ids([4, 2]) == [1, 3]  # h^k is the input of 0-indexed layer k, captured by SGLang id k - 1


@pytest.mark.parametrize("adaptation", ["frozen", "lora", "full"])
def test_jump_exports_an_eagle3_draft_model(tiny_dir: Path, tmp_path: Path, adaptation: str) -> None:
    model = _jump(tiny_dir, adaptation=adaptation)
    plan = to_sglang(model, tmp_path / "out", num_speculative_tokens=3)
    drafter = tmp_path / "out" / "drafter"
    assert plan.techniques == {"jump": "plugin"} and plan.needs_plugin
    assert plan.server_args["speculative_algorithm"] == "EAGLE3"
    assert plan.server_args["speculative_draft_model_path"] == str(drafter)
    assert plan.server_args["speculative_num_steps"] == 3 and plan.server_args["speculative_num_draft_tokens"] == 4
    assert plan.server_args["speculative_eagle_topk"] == 1
    config = json.loads((drafter / "config.json").read_text())
    assert config["architectures"] == [DRAFTER_ARCHITECTURE]
    assert config["eagle_config"]["eagle_aux_hidden_state_layer_ids"] == [1, 3]
    block = config[SGLANG_CONFIG_KEY]
    assert block["jump_layers"] == [3, 4, 5, 6] and config["num_hidden_layers"] == 4  # blocks m + 1 .. L
    assert block["target_architecture"] == "Qwen2ForCausalLM" or block["target_architecture"].endswith("ForCausalLM")
    assert block["target_layer"] == 2 and block["source_layer"] == 4
    tensors = load_file(drafter / "model.safetensors")
    assert any(name.startswith("jump.") for name in tensors)
    assert "model.norm.weight" in tensors and not any(".lora_" in name for name in tensors)
    layers = {name.split(".")[2] for name in tensors if name.startswith("model.layers.")}
    assert len(layers) == len(block["jump_layers"])


def test_frozen_jump_layers_equal_the_base(tiny_dir: Path) -> None:
    model = _jump(tiny_dir)
    exporter = JumpSglangExporter()
    tensors = exporter.tensors(model)
    base = economize.unwrap.base_model_of(model).state_dict()
    layers = exporter.spec(model.techniques["jump"]).jump_layers
    for j, layer in enumerate(layers):
        name = "self_attn.q_proj.weight"
        torch.testing.assert_close(tensors[f"model.layers.{j}.{name}"], base[f"model.layers.{layer - 1}.{name}"])


def test_lora_updates_are_merged(tiny_dir: Path) -> None:
    model = _jump(tiny_dir, adaptation="lora")
    technique = model.techniques["jump"]
    exporter = JumpSglangExporter()
    tensors = exporter.tensors(model)
    base = economize.unwrap.base_model_of(model).state_dict()
    layers = exporter.spec(model.techniques["jump"]).jump_layers
    differs = False
    for j, layer in enumerate(layers):
        for name in ("self_attn.q_proj.weight", "mlp.down_proj.weight"):
            merged, original = tensors[f"model.layers.{j}.{name}"], base[f"model.layers.{layer - 1}.{name}"]
            assert merged.shape == original.shape
            differs |= not torch.equal(merged, original)
    assert differs  # the perturbed LoRA factors moved the weights
    assert technique.layer_lora() is not None


def test_early_exit_exports_through_its_own_exporter(tiny_dir: Path, tmp_path: Path) -> None:
    model = economize.apply(_base(tiny_dir), EarlyExitConfig(exit_layer=4))
    perturb(model.techniques["early_exit"], 0.02, seed=1)
    plan = to_sglang(model, tmp_path / "out")
    assert plan.techniques == {"early_exit": "plugin"}
    config = json.loads((tmp_path / "out" / "drafter" / "config.json").read_text())
    assert config["architectures"] == [DRAFTER_ARCHITECTURE]
    assert EXPORTERS.get("sglang/early_exit") is not None


# ---- native drafters and KV ---------------------------------------------------------------------------------------


def test_lookup_maps_to_ngram(tiny_dir: Path, tmp_path: Path) -> None:
    model = economize.apply(_base(tiny_dir), LookupConfig(max_ngram=4, min_ngram=2))
    plan = to_sglang(model, tmp_path / "out", num_speculative_tokens=5)
    assert plan.server_args["speculative_algorithm"] == "NGRAM"
    assert plan.server_args["speculative_num_draft_tokens"] == 6 and plan.techniques == {"lookup": "native"}


def test_draft_model_maps_to_standalone(tiny_dir: Path, tmp_path: Path) -> None:
    model = economize.apply(_base(tiny_dir), DraftModelConfig(model=str(tiny_dir), dtype="float32"))
    plan = to_sglang(model, tmp_path / "out")
    assert plan.server_args["speculative_algorithm"] == "STANDALONE"
    assert Path(plan.server_args["speculative_draft_model_path"]).exists()


def test_eagle3_head_with_q_bias_is_refused(tiny_dir: Path) -> None:
    model = economize.apply(_base(tiny_dir), Eagle3Config(aux_layers=(2, 3, 4)))  # Qwen2: q/k/v biases
    with pytest.raises(UnsupportedBackendError, match="attention_bias"):
        check_servable(model)


def test_eagle3_head_exports_sglang_ids(tiny_dir: Path, tmp_path: Path) -> None:
    model = economize.apply(_base(tiny_dir), Eagle3Config(aux_layers=(2, 3, 4), attention_bias=False))
    plan = to_sglang(model, tmp_path / "out")
    assert plan.server_args["speculative_algorithm"] == "EAGLE3" and plan.techniques == {"eagle3": "native"}
    config = json.loads((Path(plan.server_args["speculative_draft_model_path"]) / "config.json").read_text())
    assert config["architectures"] == ["LlamaForCausalLMEagle3"]
    assert config["eagle_config"]["eagle_aux_hidden_state_layer_ids"] == [1, 2, 3]


def test_fp8_kv_cache(tiny_dir: Path, tmp_path: Path) -> None:
    plan = to_sglang(economize.apply(_base(tiny_dir), FP8KVConfig()), tmp_path / "out")
    assert plan.server_args["kv_cache_dtype"] == "fp8_e4m3" and plan.techniques == {"kv_fp8": "native"}


def test_fake_quant_with_jump_writes_the_dequantized_base(tiny_dir: Path, tmp_path: Path) -> None:
    model = economize.apply(_base(tiny_dir), QuantConfig(format="int4", group_size=32), JUMP)
    perturb(model.techniques["jump"], 0.02, seed=1)
    plan = to_sglang(model, tmp_path / "out")
    assert plan.model_path == str(tmp_path / "out" / "model") and plan.tokenizer == str(tiny_dir)
    assert plan.techniques == {"quant": "densified", "jump": "plugin"}
    torch.testing.assert_close(_base(Path(plan.model_path))(TOKENS).logits, _logits(model), atol=1e-5, rtol=1e-5)
    # the drafter's layer copies are the served (dequantized) weights
    tensors = load_file(tmp_path / "out" / "drafter" / "model.safetensors")
    served = load_file(next((tmp_path / "out" / "model").glob("*.safetensors")))
    layer = JumpSglangExporter().spec(model.techniques["jump"]).jump_layers[0]
    torch.testing.assert_close(tensors["model.layers.0.mlp.up_proj.weight"],
                               served[f"model.layers.{layer - 1}.mlp.up_proj.weight"])


# ---- open-closed --------------------------------------------------------------------------------------------------


def test_a_new_adapter_is_added_by_registration(tiny_dir: Path) -> None:
    previous = serving.base.adapter_for("kv_evict")

    @serving.register_sglang_technique(override=True)
    class MyEvict(serving.SglangTechnique):
        technique = "kv_evict"
        role = serving.Role.KV_CACHE
        support = Support.NATIVE
        reason = "a hypothetical SGLang eviction policy"

        def contribute(self, technique: Any, builder: Any) -> None:
            builder.set_engine("kv_cache_dtype", "auto", owner=self.technique)

    try:
        from economize.techniques.kv_evict.technique import KVEvictConfig

        model = economize.apply(_base(tiny_dir), KVEvictConfig(method="snapkv"))
        assert check_servable(model) == {"kv_evict": Support.NATIVE}
    finally:
        serving.register_sglang_technique(type(previous), override=True)
