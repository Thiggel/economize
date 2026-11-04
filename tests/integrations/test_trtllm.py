"""The TensorRT-LLM export (:mod:`economize.backends.trtllm`): support levels, refusals and the checkpoint layout.

TensorRT-LLM needs an NVIDIA GPU, so these tests check the files: the checkpoint is read back with
``safetensors``, the fused maps are compared with the Hugging Face weights, and the ``W4A16_AWQ`` codes are
unpacked by an independent reference (nibble arithmetic on the bytes) and compared with economize's dequantized
weights.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pytest
import torch

import economize
from economize.backends.exporters import UnsupportedBackendError
from economize.backends.trtllm import TRTLLM, TrtMedusa, check_trtllm, pack_int4_rows, to_trtllm, trt_reason
from economize.backends.trtllm.writer import fused_mismatches
from economize.backends.vllm.serving.base import Role, Support
from economize.core.technique import listed_techniques
from economize.errors import ConfigError, IncompatibleTechniquesError, SimulatedPathWarning
from economize.techniques.draft_model.config import DraftModelConfig
from economize.techniques.kv_evict.technique import KVEvictConfig
from economize.techniques.kv_quant.fp8 import FP8KVConfig
from economize.techniques.lookup.config import LookupConfig
from economize.techniques.medusa.config import MedusaConfig
from economize.techniques.quant import QuantConfig
from helpers import make_family_model, make_tiny_model

safetensors = pytest.importorskip("safetensors.numpy")


def _quant(fmt: str = "int4", symmetric: bool = True, group_size: int = 64, backend: str = "compressed-tensors",
           exclude: tuple[str, ...] = ()) -> QuantConfig:
    return QuantConfig(format=fmt, group_size=group_size, symmetric=symmetric, backend=backend, exclude=exclude)


def _unpack(packed: np.ndarray) -> np.ndarray:
    """Reference unpacker: byte ``[i, c]`` holds row ``2 i`` in its low and row ``2 i + 1`` in its high nibble."""
    raw = packed.view(np.uint8).astype(np.int64)
    out = np.zeros((2 * raw.shape[0], raw.shape[1]), dtype=np.int64)
    out[0::2], out[1::2] = raw & 0xF, raw >> 4
    return np.where(out >= 8, out - 16, out)


def _load(path: Path) -> dict[str, np.ndarray]:
    return dict(safetensors.load_file(str(path)))


def _config(folder: Path) -> dict[str, object]:
    config: dict[str, object] = json.loads((folder / "config.json").read_text())
    return config


# ---- support levels and refusals ------------------------------------------------------------------------------------
def test_every_technique_has_a_trtllm_status() -> None:
    rows = {name: (role, support, reason) for name, role, support, reason in TRTLLM.support_table()}
    assert set(listed_techniques()) <= set(rows)
    assert all(reason for _, _, reason in rows.values())
    for native in ("draft_model", "placed_draft_model", "medusa", "lookup", "kv_fp8", "kv_evict", "depth_prune"):
        assert rows[native][1] == Support.NATIVE.value, native
    assert rows["quant"][1] == rows["lowrank"][1] == rows["prune"][1] == Support.DENSIFIED.value
    for refused in ("eagle", "eagle3", "hydra", "redrafter", "lookahead", "teal", "calm", "frspec"):
        assert rows[refused][1] == Support.UNSUPPORTED.value, refused
    assert "jump" not in rows and TRTLLM.support_table(["jump"])[0][2] == Support.UNSUPPORTED.value  # unlisted
    assert "convert_checkpoint" in rows["eagle"][2]


def test_backend_support_reports_trtllm() -> None:
    assert economize.backend_support("draft_model")["trtllm"].status.value == "real"
    assert economize.backend_support("eagle3")["trtllm"].status.value == "unsupported"
    assert "EAGLE-3" in economize.backend_support("eagle3")["trtllm"].detail


def test_levels_follow_the_configuration() -> None:
    quant = TRTLLM.adapter_for("quant")
    assert quant.level(_quant()) is Support.NATIVE
    assert quant.level(_quant(symmetric=False)) is Support.NATIVE  # decided per map (asymmetric maps are densified)
    assert quant.level(_quant("int8")) is Support.DENSIFIED
    assert quant.level(_quant(backend="fake")) is Support.DENSIFIED
    w8a8 = QuantConfig(format="fp8", backend="torchao", fp8_activations=True, group_size=None)
    assert quant.level(w8a8) is Support.UNSUPPORTED
    medusa = TRTLLM.adapter_for("medusa")
    assert medusa.level(MedusaConfig()) is Support.NATIVE
    assert medusa.level(MedusaConfig(head="shared")) is Support.UNSUPPORTED
    lookup = TRTLLM.adapter_for("lookup")
    assert lookup.level(LookupConfig()) is Support.NATIVE
    assert lookup.level(LookupConfig(match="latest")) is Support.UNSUPPORTED
    assert lookup.level(LookupConfig(min_ngram=2)) is Support.UNSUPPORTED
    fp8 = TRTLLM.adapter_for("kv_fp8")
    assert fp8.level(FP8KVConfig()) is Support.NATIVE
    for refused in (FP8KVConfig(format="e5m2"), FP8KVConfig(scale="token"), FP8KVConfig(layers=(1,))):
        assert fp8.level(refused) is Support.UNSUPPORTED, refused
    evict = TRTLLM.adapter_for("kv_evict")
    assert evict.level(KVEvictConfig(sinks=8)) is Support.NATIVE
    assert evict.level(KVEvictConfig(method="h2o")) is Support.UNSUPPORTED


def test_unsupported_techniques_are_refused_before_writing(tmp_path: Path) -> None:
    with pytest.raises(UnsupportedBackendError, match="e4m3"):
        to_trtllm(economize.apply(make_tiny_model(), FP8KVConfig(format="e5m2")), tmp_path / "a")
    with pytest.raises(UnsupportedBackendError, match="head='own'"):
        check_trtllm(economize.apply(make_tiny_model(), MedusaConfig(num_heads=2, head="shared")))
    with pytest.raises(UnsupportedBackendError, match="oldest match"):
        check_trtllm(economize.apply(make_tiny_model(), LookupConfig(match="latest")))
    with pytest.raises(UnsupportedBackendError, match="sink-plus-window"):
        check_trtllm(economize.apply(make_tiny_model(), KVEvictConfig(method="h2o")))
    assert not (tmp_path / "a").exists()


def test_one_speculation_mode_cache_type_and_window_per_export() -> None:
    builder = check_trtllm(make_tiny_model())
    for role, label in ((Role.DRAFTER, "speculative decoding mode"), (Role.KV_CACHE, "KV-cache type"),
                        (Role.KV_EVICTION, "attention window")):
        builder.claim(role, "first")
        with pytest.raises(IncompatibleTechniquesError, match=label):
            builder.claim(role, "second")


def test_unknown_dtype_architecture_and_sliding_window_are_refused(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="dtype"):
        to_trtllm(make_tiny_model(), tmp_path / "a", dtype="int8")
    with pytest.raises(ConfigError, match="qwen3"):
        to_trtllm(make_family_model("qwen3"), tmp_path / "b")
    with pytest.raises(ConfigError, match="sliding window"):
        to_trtllm(make_family_model("mistral"), tmp_path / "c")


# ---- int4 packing ---------------------------------------------------------------------------------------------------
def test_pack_matches_the_reference_unpacker() -> None:
    signed = torch.randint(-8, 8, (6, 40), generator=torch.Generator().manual_seed(0))
    packed = pack_int4_rows(signed)
    assert packed.dtype == torch.int8 and packed.shape == (3, 40)
    assert np.array_equal(_unpack(packed.numpy()), signed.numpy())


def test_pack_puts_even_rows_in_the_low_nibble() -> None:
    packed = pack_int4_rows(torch.tensor([[3], [-2]]))
    assert int(packed.view(torch.uint8)[0, 0]) == 3 | (0xE << 4)


def test_pack_refuses_an_odd_number_of_rows() -> None:
    with pytest.raises(ConfigError, match="two rows per byte"):
        pack_int4_rows(torch.zeros(3, 4, dtype=torch.int64))


# ---- the exported checkpoint ----------------------------------------------------------------------------------------
def test_dense_export_layout(tmp_path: Path) -> None:
    model = make_tiny_model()
    export = to_trtllm(model, tmp_path, dtype="float32")
    assert export.path == tmp_path / "rank0.safetensors" and not export.simulated
    assert export.command == ("trtllm-build", "--checkpoint_dir", str(tmp_path), "--output_dir",
                              str(tmp_path / "engine"))
    assert json.loads((tmp_path / "economize_trtllm.json").read_text())["runtime"] == "trtllm"
    tensors, state = _load(export.path), model.state_dict()
    layer, hf = "transformer.layers.2.", "model.layers.2."
    for part in ("weight", "bias"):
        fused = torch.cat([state[f"{hf}self_attn.{p}_proj.{part}"] for p in "qkv"], 0).numpy()
        assert np.array_equal(tensors[f"{layer}attention.qkv.{part}"], fused), part
    pairs = {"attention.dense": "self_attn.o_proj", "mlp.fc": "mlp.gate_proj", "mlp.gate": "mlp.up_proj",
             "mlp.proj": "mlp.down_proj", "input_layernorm": "input_layernorm",
             "post_layernorm": "post_attention_layernorm"}
    for trt, name in pairs.items():
        assert np.array_equal(tensors[f"{layer}{trt}.weight"], state[f"{hf}{name}.weight"].numpy()), trt
    embed = state["model.embed_tokens.weight"].numpy()
    assert np.array_equal(tensors["transformer.vocab_embedding.weight"], embed)
    assert np.array_equal(tensors["lm_head.weight"], embed)  # tied: TensorRT-LLM gets its own copy
    assert np.array_equal(tensors["transformer.ln_f.weight"], state["model.norm.weight"].numpy())
    config = _config(tmp_path)
    assert config["architecture"] == "QWenForCausalLM" and config["qwen_type"] == "qwen2"
    assert config["num_hidden_layers"] == 6 and config["head_size"] == 16 and config["num_key_value_heads"] == 2
    assert config["attn_bias"] is True and config["dtype"] == "float32"
    assert config["quantization"] == {"quant_algo": None, "kv_cache_quant_algo": None}


def test_llama_writes_llama_for_causal_lm(tmp_path: Path) -> None:
    to_trtllm(make_family_model("llama"), tmp_path)
    config = _config(tmp_path)
    assert config["architecture"] == "LlamaForCausalLM" and "qwen_type" not in config


def test_float16_is_the_default_type(tmp_path: Path) -> None:
    tensors = _load(to_trtllm(make_tiny_model(), tmp_path).path)
    assert tensors["transformer.layers.0.mlp.gate.weight"].dtype == np.float16
    assert _config(tmp_path)["dtype"] == "float16"


def test_int4_codes_dequantize_to_economize_weights(tmp_path: Path) -> None:
    from economize.integrations.compressed_tensors import quantized_modules

    model = economize.apply(make_tiny_model(), _quant())
    with warnings.catch_warnings():
        warnings.simplefilter("error", SimulatedPathWarning)
        export = to_trtllm(model, tmp_path, dtype="float32")
    assert export.supports == {"quant": "native"}
    recorded = quantized_modules(model)
    tensors = _load(export.path)
    groups = {"attention.qkv": ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"),
              "mlp.proj": ("mlp.down_proj",)}
    for i in (0, 5):
        for trt, parts in groups.items():
            weights = [recorded[f"model.layers.{i}.{part}"] for part in parts]
            prefix = f"transformer.layers.{i}.{trt}."
            codes = _unpack(tensors[prefix + "weight"])
            assert np.array_equal(codes, torch.cat([w.codes.long() - 8 for w in weights], 0).numpy()), prefix
            scales = tensors[prefix + "weights_scaling_factor"]
            assert scales.shape == (codes.shape[1] // 64, codes.shape[0])
            stored = codes * np.repeat(scales.T, 64, axis=1)
            expected = torch.cat([w.scale.repeat_interleave(64, dim=1) * (w.codes.float() - 8) for w in weights], 0)
            np.testing.assert_allclose(stored, expected.numpy(), rtol=1e-6, atol=1e-6)
            assert np.array_equal(tensors[prefix + "prequant_scaling_factor"], np.ones(codes.shape[1], np.float32))
    quantization = _config(tmp_path)["quantization"]
    assert quantization == {"quant_algo": "W4A16_AWQ", "kv_cache_quant_algo": None, "group_size": 64,
                            "has_zero_point": False, "pre_quant_scale": True, "exclude_modules": ["lm_head"]}


@pytest.mark.parametrize(("config", "reason"), [
    (_quant(symmetric=False), "asymmetric"),
    (_quant(group_size=32), "group size 32"),
    (_quant("int8"), "dense equivalent"),
    (_quant(backend="fake"), "dense equivalent"),
])
def test_codes_without_a_w4a16_form_are_written_dense(tmp_path: Path, config: QuantConfig, reason: str) -> None:
    model = economize.apply(make_tiny_model(), config)
    with pytest.warns(SimulatedPathWarning, match="quant"):
        export = to_trtllm(model, tmp_path, dtype="float32")
    assert export.supports == {"quant": "densified"} and export.simulated
    assert reason in export.simulations["quant"]
    tensors = _load(export.path)
    assert tensors["transformer.layers.0.attention.qkv.weight"].dtype == np.float32
    assert _config(tmp_path)["quantization"] == {"quant_algo": None, "kv_cache_quant_algo": None}


def test_a_fused_map_holds_codes_only_when_all_its_parts_do(tmp_path: Path) -> None:
    model = economize.apply(make_tiny_model(), _quant(exclude=("*k_proj",)))
    with pytest.warns(SimulatedPathWarning, match="fused"):
        export = to_trtllm(model, tmp_path, dtype="float32")
    assert export.supports == {"quant": "densified"}
    tensors = _load(export.path)
    assert tensors["transformer.layers.0.attention.qkv.weight"].dtype == np.float32
    assert tensors["transformer.layers.0.mlp.fc.weight"].dtype == np.int8
    excluded = _config(tmp_path)["quantization"]
    assert isinstance(excluded, dict) and "transformer.layers.0.attention.qkv" in excluded["exclude_modules"]


def test_fused_mismatches_names_the_maps_to_densify() -> None:
    from economize.integrations.compressed_tensors import quantized_modules

    recorded = dict(quantized_modules(economize.apply(make_tiny_model(), _quant())))
    assert fused_mismatches(recorded) == {}
    del recorded["model.layers.1.self_attn.v_proj"]
    assert set(fused_mismatches(recorded)) == {"model.layers.1.self_attn.q_proj", "model.layers.1.self_attn.k_proj"}
    assert all(trt_reason(weight) is None for weight in recorded.values())


def test_medusa_heads_go_into_the_checkpoint(tmp_path: Path) -> None:
    model = economize.apply(make_tiny_model(seed=0), MedusaConfig(num_heads=3, num_layers=2))
    export = to_trtllm(model, tmp_path, dtype="float32")
    assert export.supports == {"medusa": "native"}
    choices = [[0], [0, 0], [0, 0, 0]]
    assert export.settings["speculative_config"] == {"decoding_type": "Medusa", "max_draft_len": 3,
                                                     "num_medusa_heads": 3, "medusa_choices": choices}
    assert export.command[-4:] == ("--speculative_decoding_mode", "medusa", "--max_draft_len", "3")
    config = _config(tmp_path)
    assert config["architecture"] == "MedusaForCausalLM" and config["qwen_type"] == "qwen2"
    assert config["num_medusa_heads"] == 3 and config["num_medusa_layers"] == 2 and config["max_draft_len"] == 3
    tensors = _load(export.path)
    heads = list(getattr(model.techniques["medusa"], "heads"))  # noqa: B009
    for h in (0, 2):
        block = heads[h].blocks[1].linear
        assert np.array_equal(tensors[f"medusa_heads.{h}.medusa_layers.1.linear.weight"], block.weight.detach().numpy())
        assert np.array_equal(tensors[f"medusa_heads.{h}.medusa_layers.1.linear.bias"], block.bias.detach().numpy())
        assert np.array_equal(tensors[f"medusa_heads.{h}.lm_head.weight"], heads[h].proj.weight.detach().numpy())


def test_medusa_choices_follow_a_named_tree() -> None:
    choices = TrtMedusa.choices(MedusaConfig(num_heads=5, tree="small"))
    assert choices == [[0], [1], [0, 0], [0, 1], [0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0, 0]]


def test_draft_model_folder_and_settings(tmp_path: Path) -> None:
    draft = make_tiny_model(seed=1, num_layers=2)
    draft.save_pretrained(tmp_path / "draft_lm")
    model = economize.apply(make_tiny_model(), DraftModelConfig(model=str(tmp_path / "draft_lm")))
    folder = tmp_path / "out"
    export = to_trtllm(model, folder, num_draft_tokens=3)
    assert export.supports == {"draft_model": "native"}
    assert export.command[-4:] == ("--speculative_decoding_mode", "draft_tokens_external", "--max_draft_len", "3")
    assert export.settings["speculative_config"] == {"decoding_type": "DraftTarget", "max_draft_len": 3,
                                                     "speculative_model_dir": "draft/engine"}
    assert _config(folder / "draft")["num_hidden_layers"] == 2
    drafted = _load(folder / "draft" / "rank0.safetensors")["transformer.vocab_embedding.weight"]
    assert np.array_equal(drafted, draft.model.embed_tokens.weight.half().numpy())


def test_lookup_is_ngram_speculation(tmp_path: Path) -> None:
    export = to_trtllm(economize.apply(make_tiny_model(), LookupConfig(max_ngram=4)), tmp_path, num_draft_tokens=5)
    assert export.supports == {"lookup": "native"}
    assert export.settings["speculative_config"] == {"decoding_type": "NGram", "max_draft_len": 5,
                                                     "max_matching_ngram_size": 4, "is_use_oldest": True,
                                                     "is_keep_all": True, "is_public_pool": False}


def test_streaming_eviction_sets_sinks_and_window(tmp_path: Path) -> None:
    export = to_trtllm(economize.apply(make_tiny_model(), KVEvictConfig(sinks=2, recent=60)), tmp_path)
    assert export.supports == {"kv_evict": "native"}
    assert export.settings == {"kv_cache_config": {"sink_token_length": 2, "max_attention_window": [62]}}


def test_fp8_cache_sets_the_algorithm_and_unit_scales(tmp_path: Path) -> None:
    export = to_trtllm(economize.apply(make_tiny_model(), FP8KVConfig()), tmp_path)
    assert export.supports == {"kv_fp8": "native"}
    assert _config(tmp_path)["quantization"] == {"quant_algo": None, "kv_cache_quant_algo": "FP8"}
    tensors = _load(export.path)
    for i in range(6):
        scale = tensors[f"transformer.layers.{i}.attention.kv_cache_scaling_factor"]
        assert scale.dtype == np.float32 and scale.tolist() == [1.0]


def test_tokenizer_is_saved_beside_the_weights(tmp_path: Path) -> None:
    class Tokenizer:
        def save_pretrained(self, folder: Path) -> tuple[str, ...]:
            (Path(folder) / "tokenizer.json").write_text("{}")
            return (str(Path(folder) / "tokenizer.json"),)

    export = to_trtllm(make_tiny_model(), tmp_path, tokenizer=Tokenizer())
    assert tmp_path / "tokenizer.json" in export.files
