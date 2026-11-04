"""The ``"vllm/eagle3"`` exporter, checked on CPU: the folder is the official EAGLE-3 layout vLLM loads, and reading
it back with economize's own EAGLE-3 reader gives the exported drafter (configuration, tensors, rotary table)."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from safetensors.torch import load_file

import economize
from economize.backends import export_drafter, exporter_for
from economize.backends.vllm.eagle3 import Eagle3VllmExporter
from economize.backends.vllm.eagle3.export import _rope_parameters
from economize.errors import ConfigError
from economize.techniques.eagle3 import Eagle3Config
from economize.techniques.eagle3.formats import Eagle3Format, flat_config
from helpers import VOCAB, make_tiny_model, perturb

DRAFT_VOCAB = 48
OFFICIAL = {"fc.weight", "norm.weight", "lm_head.weight", "d2t", "t2d"}


def _model(**options: Any) -> Any:
    model = economize.apply(make_tiny_model(seed=0), Eagle3Config(**{"draft_vocab": DRAFT_VOCAB, **options}))
    technique = model.techniques["eagle3"]
    perturb(technique, 0.05, seed=1)
    technique.set_draft_vocab(torch.randperm(VOCAB, generator=torch.Generator().manual_seed(0))[:DRAFT_VOCAB])
    return model


@pytest.fixture
def exported(tmp_path: Path) -> tuple[Any, Path]:
    model = _model()
    return model, export_drafter(model, tmp_path / "eagle3", backend="vllm")


def test_the_registered_exporter_is_picked() -> None:
    assert isinstance(exporter_for(_model(), "vllm"), Eagle3VllmExporter)


def test_official_tensor_names(exported: tuple[Any, Path]) -> None:
    model, folder = exported
    tensors = load_file(str(folder / "model.safetensors"))
    state = model.techniques["eagle3"].state_dict()
    layer = {name.removeprefix("layer.") for name in state if name.startswith("layer.")}
    assert set(tensors) == OFFICIAL | {"midlayer." + name for name in layer} | {"midlayer.self_attn.o_proj.bias"}
    assert not any(name.startswith(("embed_tokens", "model.", "layers.")) for name in tensors)
    assert tensors["d2t"].dtype == torch.long and tensors["t2d"].dtype == torch.bool
    assert tensors["lm_head.weight"].shape == (DRAFT_VOCAB, 64) and tensors["t2d"].shape == (VOCAB,)
    assert int(tensors["t2d"].sum()) == DRAFT_VOCAB


@pytest.mark.parametrize("bias", [True, False])
def test_o_proj_bias_is_zero_when_vllm_allocates_one(bias: bool, tmp_path: Path) -> None:
    # vLLM's llama layer gives o_proj a bias whenever attention_bias is set, allocated with torch.empty
    model = _model(attention_bias=bias)
    folder = export_drafter(model, tmp_path / "eagle3", backend="vllm")
    tensors = load_file(str(folder / "model.safetensors"))
    assert json.loads((folder / "config.json").read_text())["attention_bias"] is bias
    if bias:
        assert torch.equal(tensors["midlayer.self_attn.o_proj.bias"], torch.zeros(64, dtype=tensors["fc.weight"].dtype))
    else:
        assert not any(name.endswith(".bias") for name in tensors)


def test_config_is_a_one_layer_llama(exported: tuple[Any, Path]) -> None:
    model, folder = exported
    technique = model.techniques["eagle3"]
    config = json.loads((folder / "config.json").read_text())
    assert config["architectures"] == ["LlamaForCausalLM"] and config["num_hidden_layers"] == 1
    assert (config["num_attention_heads"], config["num_key_value_heads"], config["head_dim"]) == (4, 2, 16)
    assert config["eagle_aux_hidden_state_layer_ids"] == list(technique.aux_layers)
    assert (config["vocab_size"], config["draft_vocab_size"], config["target_hidden_size"]) == (VOCAB, DRAFT_VOCAB, 64)
    assert config["rope_parameters"]["rope_theta"] == model.config.rope_parameters["rope_theta"]
    from transformers import AutoConfig

    hf = AutoConfig.from_pretrained(folder)
    assert hf.model_type == "llama" and hf.head_dim == 16 and hf.intermediate_size == 128


@pytest.mark.parametrize("fmt", ["eagle3", "speculators"])
def test_round_trip_through_the_official_reader(fmt: str, tmp_path: Path) -> None:
    model = _model()
    technique = model.techniques["eagle3"]
    folder = export_drafter(model, tmp_path / fmt, backend="vllm", format=fmt, verifier="tiny/qwen2")
    checkpoint = Eagle3Format().read(folder)
    (config,) = checkpoint.configs
    assert (config.num_heads, config.num_kv_heads, config.head_dim) == (4, 2, 16)
    assert config.aux_layers == technique.aux_layers and config.draft_vocab == DRAFT_VOCAB
    assert config.intermediate_size == 128 and config.head == "own"
    state = checkpoint.states["eagle3"]
    original = technique.state_dict()
    assert set(state) == set(original)
    for name, tensor in original.items():
        assert torch.equal(state[name].to(tensor.device), tensor), name

    # the reader gives the draft layer its own rotary table: it must equal the model's table the drafter used
    reloaded = economize.apply(make_tiny_model(seed=0), config).techniques["eagle3"]
    reloaded.load_state_dict(state)
    positions = torch.arange(9).view(1, -1)
    for got, want in zip(reloaded.rotary_table(positions), technique.rotary_table(positions), strict=True):
        torch.testing.assert_close(got, want, rtol=1e-6, atol=1e-6)


def test_speculators_layout(tmp_path: Path) -> None:
    model = _model()
    folder = export_drafter(model, tmp_path / "spec", backend="vllm", format="speculators", verifier="tiny/qwen2",
                            num_speculative_tokens=3)
    config = json.loads((folder / "config.json").read_text())
    assert config["architectures"] == ["Eagle3Speculator"] and config["speculators_model_type"] == "eagle3"
    assert config["speculators_config"]["verifier"]["name_or_path"] == "tiny/qwen2"
    assert config["speculators_config"]["proposal_methods"][0]["speculative_tokens"] == 3
    flat = json.loads((export_drafter(model, tmp_path / "flat", backend="vllm") / "config.json").read_text())
    flat.pop("architectures")  # the top level names the speculators architecture
    assert {k: v for k, v in flat_config(config).items() if k in flat} == flat


def test_unknown_format_is_refused(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="unknown export format"):
        export_drafter(_model(), tmp_path / "y", backend="vllm", format="nope")


def test_shared_head_exports_the_final_norm_and_no_lm_head(tmp_path: Path) -> None:
    """head="shared": vLLM shares the target's LM head (no lm_head, no d2t in the folder) and applies the exported
    plain RMSNorm, which must equal the model's final norm, so vLLM's draft logits are the technique's."""
    model = economize.apply(make_tiny_model(seed=0), Eagle3Config(head="shared"))
    technique = model.techniques["eagle3"]
    perturb(technique, 0.05, seed=1)
    with torch.no_grad():  # a non-trivial final-norm gain
        technique.adapter.final_norm.weight.mul_(torch.linspace(0.5, 1.5, technique.adapter.hidden_size))
    folder = export_drafter(model, tmp_path / "shared", backend="vllm")
    config = json.loads((folder / "config.json").read_text())
    tensors = load_file(str(folder / "model.safetensors"))
    assert config["draft_vocab_size"] == config["vocab_size"] == VOCAB
    assert "lm_head.weight" not in tensors and "d2t" not in tensors and "norm.weight" in tensors
    hidden = torch.randn(3, technique.adapter.hidden_size, generator=torch.Generator().manual_seed(2))
    eps = config["rms_norm_eps"]
    normed = tensors["norm.weight"] * hidden * torch.rsqrt(hidden.pow(2).mean(-1, keepdim=True) + eps)
    vllm_logits = normed @ technique.adapter.lm_head.weight.detach().float().T
    torch.testing.assert_close(vllm_logits, technique.head_logits(hidden), rtol=1e-5, atol=1e-5)


def _gemma4_rope() -> dict[str, Any]:
    return {"full_attention": {"partial_rotary_factor": 0.25, "rope_theta": 1000000.0, "rope_type": "proportional"},
            "sliding_attention": {"rope_theta": 10000.0, "rope_type": "default"}}


def test_rope_of_a_model_with_one_table_per_layer_type() -> None:
    """Gemma 4: the draft layer uses the full-attention table (partial, proportional), as vLLM's get_rope reads it."""
    technique = SimpleNamespace(rotary=None, rope_table="full_attention", config=None,
                                adapter=SimpleNamespace(config=SimpleNamespace(rope_parameters=_gemma4_rope())))
    assert _rope_parameters(technique) == _gemma4_rope()["full_attention"]


def test_rope_of_the_cached_gemma4_31b_config() -> None:
    from huggingface_hub import try_to_load_from_cache

    path = try_to_load_from_cache("google/gemma-4-31B-it", "config.json")
    if not isinstance(path, str):
        pytest.skip("google/gemma-4-31B-it config.json is not cached")
    text = json.loads(Path(path).read_text()).get("text_config") or {}
    technique = SimpleNamespace(rotary=None, rope_table="full_attention", config=None,
                                adapter=SimpleNamespace(config=SimpleNamespace(**text)))
    rope = _rope_parameters(technique)
    assert rope["rope_type"] == "proportional" and rope["partial_rotary_factor"] == 0.25
    assert rope["rope_theta"] == 1e6


def test_own_rope_is_exported() -> None:
    technique = SimpleNamespace(rotary=object(), config=SimpleNamespace(
        rope_theta=500000.0, partial_rotary_factor=1.0,
        rope_scaling={"rope_type": "llama3", "factor": 8.0, "low_freq_factor": 1.0, "high_freq_factor": 4.0,
                      "original_max_position_embeddings": 8192}))
    assert _rope_parameters(technique) == {"rope_type": "llama3", "factor": 8.0, "low_freq_factor": 1.0,
                                           "high_freq_factor": 4.0, "original_max_position_embeddings": 8192,
                                           "rope_theta": 500000.0}


def test_full_vocabulary_own_head_exports_no_d2t(tmp_path: Path) -> None:
    """vLLM 0.30 allocates draft_id_to_target_id only when draft_vocab_size < vocab_size and refuses a d2t else."""
    model = economize.apply(make_tiny_model(seed=0), Eagle3Config())
    folder = export_drafter(model, tmp_path / "full", backend="vllm")
    config = json.loads((folder / "config.json").read_text())
    tensors = load_file(str(folder / "model.safetensors"))
    assert config["draft_vocab_size"] == VOCAB
    assert "lm_head.weight" in tensors and "d2t" not in tensors and "t2d" not in tensors
