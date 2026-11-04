"""The MLX export (:mod:`economize.backends.mlx`): support levels, refusals and the ``mlx_lm`` folder layout.

MLX runs on Apple silicon only, so these tests check the format: the safetensors file is read back with
``safetensors`` and the packed codes are unpacked by an independent reference (integer arithmetic on the words, as
``mlx.core.dequantize`` reads them), then compared with economize's codes and dequantized weights.
"""

from __future__ import annotations

import dataclasses
import json
import warnings
from pathlib import Path

import numpy as np
import pytest

import economize
from economize.backends.exporters import UnsupportedBackendError
from economize.backends.mlx import MLX, check_mlx, mlx_config, mlx_reason, pack_mlx, to_mlx
from economize.backends.runtime_export import ModelShape
from economize.backends.vllm.serving.base import Role, Support
from economize.core.technique import listed_techniques
from economize.errors import ConfigError, IncompatibleTechniquesError, SimulatedPathWarning
from economize.techniques.draft_model.config import DraftModelConfig
from economize.techniques.kv_evict.technique import KVEvictConfig
from economize.techniques.kv_quant.fp8 import FP8KVConfig
from economize.techniques.lookup.config import LookupConfig
from economize.techniques.quant import QuantConfig
from economize.techniques.quant.grid import IntGrid
from helpers import make_family_model, make_tiny_model

safetensors = pytest.importorskip("safetensors.numpy")


def _quant(fmt: str = "int4", symmetric: bool = False, group_size: int = 32,
           backend: str = "compressed-tensors") -> QuantConfig:
    return QuantConfig(format=fmt, group_size=group_size, symmetric=symmetric, backend=backend)


def _unpack(words: np.ndarray, bits: int, count: int) -> np.ndarray:
    """Reference unpacker: code ``k`` of a row is bits ``bits*k .. bits*k + bits - 1`` of its little-endian words."""
    out = np.zeros((words.shape[0], count), dtype=np.int64)
    for row in range(words.shape[0]):
        stream = sum(int(word) << (32 * i) for i, word in enumerate(words[row]))
        out[row] = [(stream >> (bits * k)) & ((1 << bits) - 1) for k in range(count)]
    return out


def _load(path: Path) -> dict[str, np.ndarray]:
    return dict(safetensors.load_file(str(path)))


# ---- support levels and refusals ------------------------------------------------------------------------------------
def test_every_technique_has_an_mlx_status() -> None:
    rows = {name: (role, support, reason) for name, role, support, reason in MLX.support_table()}
    assert set(listed_techniques()) <= set(rows)
    assert all(reason for _, _, reason in rows.values())
    assert rows["draft_model"][1] == rows["kv_evict"][1] == rows["depth_prune"][1] == Support.NATIVE.value
    assert rows["quant"][1] == rows["lowrank"][1] == Support.DENSIFIED.value
    for refused in ("lookup", "eagle", "eagle3", "medusa", "kv_fp8", "teal", "calm", "frspec"):
        assert rows[refused][1] == Support.UNSUPPORTED.value, refused
    assert "jump" not in rows and MLX.support_table(["jump"])[0][2] == Support.UNSUPPORTED.value  # unlisted
    assert "prompt lookup" in rows["lookup"][2]


def test_backend_support_reports_mlx() -> None:
    assert economize.backend_support("draft_model")["mlx"].status.value == "real"
    assert economize.backend_support("medusa")["mlx"].status.value == "unsupported"
    assert "Medusa" in economize.backend_support("medusa")["mlx"].detail


def test_quant_level_follows_its_configuration() -> None:
    adapter = MLX.adapter_for("quant")
    assert adapter.level(_quant()) is Support.NATIVE
    assert adapter.level(_quant("int8", symmetric=True, backend="marlin")) is Support.NATIVE
    assert adapter.level(_quant(backend="fake")) is Support.DENSIFIED
    assert adapter.level(QuantConfig(format="fp8", backend="torchao", group_size=None)) is Support.DENSIFIED
    w8a8 = QuantConfig(format="fp8", backend="torchao", fp8_activations=True, group_size=None)
    assert adapter.level(w8a8) is Support.UNSUPPORTED


def test_kv_eviction_level_follows_its_configuration() -> None:
    adapter = MLX.adapter_for("kv_evict")
    assert adapter.level(KVEvictConfig()) is Support.NATIVE
    assert adapter.level(KVEvictConfig(sinks=4, recent=64)) is Support.NATIVE
    assert adapter.level(KVEvictConfig(sinks=8)) is Support.UNSUPPORTED
    assert adapter.level(KVEvictConfig(method="h2o")) is Support.UNSUPPORTED
    assert adapter.level(KVEvictConfig(layers=(1, 2))) is Support.UNSUPPORTED


def test_unsupported_techniques_are_refused_before_writing(tmp_path: Path) -> None:
    with pytest.raises(UnsupportedBackendError, match="fp8 KV cache"):
        to_mlx(economize.apply(make_tiny_model(), FP8KVConfig()), tmp_path / "a")
    with pytest.raises(UnsupportedBackendError, match="prompt lookup"):
        check_mlx(economize.apply(make_tiny_model(), LookupConfig()))
    with pytest.raises(UnsupportedBackendError, match="sinks=4"):
        check_mlx(economize.apply(make_tiny_model(), KVEvictConfig(sinks=2)))
    assert not (tmp_path / "a").exists()


def test_one_draft_model_and_one_cache_size_per_export() -> None:
    builder = check_mlx(make_tiny_model())
    builder.claim(Role.DRAFTER, "draft_model")
    with pytest.raises(IncompatibleTechniquesError, match="--draft-model"):
        builder.claim(Role.DRAFTER, "placed_draft_model")
    builder.claim(Role.KV_EVICTION, "kv_evict")
    with pytest.raises(IncompatibleTechniquesError, match="--max-kv-size"):
        builder.claim(Role.KV_EVICTION, "other")


def test_unknown_dtype_and_sliding_window_are_refused(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="dtype"):
        to_mlx(make_tiny_model(), tmp_path / "a", dtype="int8")
    with pytest.raises(ConfigError, match="sliding window"):
        to_mlx(make_family_model("mistral"), tmp_path / "b")


# ---- packing --------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("bits", [2, 3, 4, 5, 6, 8])
def test_pack_matches_the_reference_unpacker(bits: int) -> None:
    rng = np.random.default_rng(bits)
    codes = rng.integers(0, 2**bits, size=(3, 64), dtype=np.uint8)
    words = pack_mlx(codes, bits)
    assert words.dtype == np.uint32 and words.shape == (3, 64 * bits // 32)
    assert np.array_equal(_unpack(words, bits, 64), codes)


def test_pack_of_four_bits_puts_code_k_at_bit_4k() -> None:
    codes = np.arange(8, dtype=np.uint8)[None]
    assert pack_mlx(codes, 4)[0, 0] == sum(k << (4 * k) for k in range(8))


def test_pack_refuses_rows_that_do_not_fill_words() -> None:
    with pytest.raises(ConfigError, match="32-bit words"):
        pack_mlx(np.zeros((1, 10), dtype=np.uint8), 3)


# ---- the exported folder --------------------------------------------------------------------------------------------
def test_dense_export_layout(tmp_path: Path) -> None:
    model = make_tiny_model()
    export = to_mlx(model, tmp_path, dtype="float32")
    assert export.path == tmp_path / "model.safetensors" and not export.simulated
    assert export.command == ("mlx_lm.generate", "--model", str(tmp_path))
    assert json.loads((tmp_path / "economize_mlx.json").read_text())["runtime"] == "mlx"
    tensors = _load(export.path)
    assert "lm_head.weight" not in tensors  # tied: mlx_lm reuses the embedding
    state = model.state_dict()
    for name in ("model.embed_tokens.weight", "model.layers.2.self_attn.q_proj.bias", "model.norm.weight"):
        assert np.array_equal(tensors[name], state[name].numpy()), name
    config = json.loads((tmp_path / "config.json").read_text())
    assert config["model_type"] == "qwen2" and config["num_hidden_layers"] == 6
    assert config["rope_theta"] == ModelShape.of(model, "MLX").rope_theta and config["rope_scaling"] is None
    assert config["tie_word_embeddings"] is True and config["head_dim"] == 16 and "quantization" not in config


def test_float16_is_the_default_type(tmp_path: Path) -> None:
    tensors = _load(to_mlx(make_tiny_model(), tmp_path).path)
    assert tensors["model.layers.0.mlp.up_proj.weight"].dtype == np.float16


@pytest.mark.parametrize(("fmt", "symmetric"), [("int4", False), ("int4", True), ("int8", True), ("int3", False)])
def test_quant_codes_dequantize_to_economize_weights(tmp_path: Path, fmt: str, symmetric: bool) -> None:
    from economize.integrations.compressed_tensors import quantized_modules

    model = economize.apply(make_tiny_model(), _quant(fmt, symmetric))
    with warnings.catch_warnings():
        warnings.simplefilter("error", SimulatedPathWarning)
        export = to_mlx(model, tmp_path, dtype="float32")
    assert export.supports == {"quant": "native"}
    recorded = quantized_modules(model)
    tensors = _load(export.path)
    bits = int(fmt.removeprefix("int"))
    config = json.loads((tmp_path / "config.json").read_text())
    assert config["quantization"] == {"group_size": 32, "bits": bits}
    for name, weight in recorded.items():
        out, inputs = weight.codes.shape
        codes = _unpack(tensors[f"{name}.weight"], bits, inputs)
        assert np.array_equal(codes, weight.codes.numpy()), name
        scales, biases = tensors[f"{name}.scales"], tensors[f"{name}.biases"]
        assert scales.shape == biases.shape == (out, inputs // 32)
        stored = np.repeat(scales, 32, axis=1) * codes + np.repeat(biases, 32, axis=1)
        expected = (weight.scale.repeat_interleave(32, dim=1) * (weight.codes.float()
                    - weight.zero.repeat_interleave(32, dim=1))).numpy()
        np.testing.assert_allclose(stored, expected, rtol=1e-6, atol=1e-6)
    assert not any(f"{name}.weight" in tensors and tensors[f"{name}.weight"].dtype != np.uint32 for name in recorded)


def test_fake_quant_is_written_dense_with_a_warning(tmp_path: Path) -> None:
    model = economize.apply(make_tiny_model(), _quant(backend="fake"))
    with pytest.warns(SimulatedPathWarning, match="quant"):
        export = to_mlx(model, tmp_path, dtype="float32")
    assert export.supports == {"quant": "densified"} and export.simulated
    assert _load(export.path)["model.layers.0.self_attn.q_proj.weight"].dtype == np.float32


def test_groups_mlx_does_not_have_are_written_dense(tmp_path: Path) -> None:
    model = economize.apply(make_tiny_model(), _quant(group_size=16))
    with pytest.warns(SimulatedPathWarning):
        export = to_mlx(model, tmp_path, dtype="float32")
    assert export.supports == {"quant": "densified"} and "group size 16" in export.simulations["quant"]
    assert "quantization" not in json.loads((tmp_path / "config.json").read_text())


def test_mixed_settings_get_per_module_entries() -> None:
    from economize.integrations.compressed_tensors import quantized_modules

    model = economize.apply(make_tiny_model(), _quant())
    recorded = dict(quantized_modules(model))
    odd = "model.layers.0.mlp.down_proj"
    recorded[odd] = dataclasses.replace(recorded[odd], grid=IntGrid(8, False), group_size=64)
    assert mlx_reason(recorded[odd]) is None
    config = mlx_config(model, ModelShape.of(model, "MLX"), recorded)
    assert config["quantization"] == {"group_size": 32, "bits": 4, odd: {"group_size": 64, "bits": 8}}


def test_draft_model_folder_and_command(tmp_path: Path) -> None:
    draft = make_tiny_model(seed=1, num_layers=2)
    draft.save_pretrained(tmp_path / "draft_lm")
    model = economize.apply(make_tiny_model(), DraftModelConfig(model=str(tmp_path / "draft_lm")))
    folder = tmp_path / "out"
    export = to_mlx(model, folder, num_draft_tokens=3)
    assert export.supports == {"draft_model": "native"}
    assert export.command == ("mlx_lm.generate", "--model", str(folder), "--draft-model", str(folder / "draft"),
                              "--num-draft-tokens", "3")
    assert json.loads((folder / "draft" / "config.json").read_text())["num_hidden_layers"] == 2
    target = _load(folder / "model.safetensors")["model.embed_tokens.weight"]
    drafted = _load(folder / "draft" / "model.safetensors")["model.embed_tokens.weight"]
    assert np.array_equal(drafted, draft.model.embed_tokens.weight.half().numpy())
    assert not np.array_equal(drafted, target)


def test_streaming_eviction_sets_the_cache_size(tmp_path: Path) -> None:
    model = economize.apply(make_tiny_model(), KVEvictConfig(recent=60))
    export = to_mlx(model, tmp_path)
    assert export.supports == {"kv_evict": "native"}
    assert export.settings == {"--max-kv-size": 64}
    assert export.command[-2:] == ("--max-kv-size", "64")


def test_tokenizer_is_saved_beside_the_weights(tmp_path: Path) -> None:
    class Tokenizer:
        def save_pretrained(self, folder: Path) -> tuple[str, ...]:
            (Path(folder) / "tokenizer.json").write_text("{}")
            return (str(Path(folder) / "tokenizer.json"),)

    export = to_mlx(make_tiny_model(), tmp_path, tokenizer=Tokenizer())
    assert tmp_path / "tokenizer.json" in export.files
