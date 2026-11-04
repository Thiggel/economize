"""The llama.cpp export (:mod:`economize.backends.gguf`): support levels, refusals, the GGUF files and their outputs.

The support levels and refusals need neither ``gguf`` nor ``llama_cpp``. The file tests read the export back with
``gguf.GGUFReader`` and check that ``quant`` codes are packed bit for bit. The round trips load the files in
llama-cpp-python: with a float32 KV cache (``type_k=type_v=0``) the dense float32 file reproduces the Hugging Face
float32 logits, and prompt lookup and the draft model reproduce Hugging Face greedy decoding. llama.cpp quantizes the
activations of quantized matrix products to 8 bits, so quantized files are only checked for agreement of the argmax.
"""

from __future__ import annotations

import copy
import json
import warnings
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

import economize
from economize.backends.exporters import UnsupportedBackendError
from economize.backends.gguf import GGUF, check_gguf, code_type, llama_quantize_command, to_gguf
from economize.backends.vllm.serving.base import Role, Support
from economize.core.technique import listed_techniques
from economize.errors import ConfigError, IncompatibleTechniquesError, SimulatedPathWarning
from economize.techniques.draft_model.config import DraftModelConfig
from economize.techniques.kv_quant.fp8 import FP8KVConfig
from economize.techniques.lookup.config import LookupConfig
from economize.techniques.quant import QuantConfig
from helpers import make_family_model, make_tiny_model, reference_greedy

IDS = list(range(3, 20))
PROMPT = [5, 9, 17, 5, 9, 17, 5, 9, 33, 41, 5, 9]
#: llama-cpp-python settings of the round trips: a float32 KV cache (llama.cpp's default f16 cache moves the
#: logits of the tiny model by about 1e-2).
LLAMA = {"n_ctx": 128, "n_threads": 2, "verbose": False, "type_k": 0, "type_v": 0}


def _quant(fmt: str = "int4", symmetric: bool = True, backend: str = "compressed-tensors") -> QuantConfig:
    return QuantConfig(format=fmt, group_size=32, symmetric=symmetric, backend=backend)


def _logits(model: nn.Module, ids: list[int] = IDS) -> np.ndarray:
    with torch.no_grad():
        return model(torch.tensor([ids])).logits[0].float().numpy()


def _truncated(base: nn.Module, layers: int) -> nn.Module:
    """The first ``layers`` layers of ``base`` with its norm and head (a draft LM that often agrees with it)."""
    from transformers import Qwen2ForCausalLM

    config = copy.deepcopy(base.config)
    config.num_hidden_layers = layers
    config.layer_types = list(config.layer_types[:layers])
    draft = Qwen2ForCausalLM(config).float().eval()
    draft.load_state_dict(base.state_dict(), strict=False)
    return draft


# ---- support levels and refusals (no gguf needed) ------------------------------------------------------------------
def test_every_technique_has_a_llama_cpp_status() -> None:
    rows = {name: (role, support, reason) for name, role, support, reason in GGUF.support_table()}
    assert set(listed_techniques()) <= set(rows)
    assert all(reason for _, _, reason in rows.values())
    assert rows["draft_model"][1] == rows["lookup"][1] == Support.NATIVE.value
    assert rows["quant"][1] == rows["prune"][1] == Support.DENSIFIED.value
    for refused in ("eagle", "eagle3", "medusa", "kv_fp8", "kv_evict", "teal", "calm"):
        assert rows[refused][1] == Support.UNSUPPORTED.value, refused
    assert "jump" not in rows and GGUF.support_table(["jump"])[0][2] == Support.UNSUPPORTED.value  # unlisted
    assert "EAGLE" in rows["eagle"][2] and "fp8" in rows["kv_fp8"][2]


def test_backend_support_reports_llama_cpp() -> None:
    assert economize.backend_support("draft_model")["gguf"].status.value == "real"
    assert economize.backend_support("eagle")["gguf"].status.value == "unsupported"
    assert "llama.cpp has no EAGLE" in economize.backend_support("eagle")["gguf"].detail


def test_quant_level_follows_its_configuration() -> None:
    adapter = GGUF.adapter_for("quant")
    assert adapter.level(_quant()) is Support.NATIVE
    assert adapter.level(_quant("int8")) is Support.NATIVE
    assert adapter.level(_quant(backend="fake")) is Support.DENSIFIED
    w8a8 = QuantConfig(format="fp8", backend="torchao", fp8_activations=True, group_size=None)
    assert adapter.level(w8a8) is Support.UNSUPPORTED


def test_unsupported_techniques_are_refused_with_the_reason() -> None:
    with pytest.raises(UnsupportedBackendError, match="fp8 KV cache"):
        check_gguf(economize.apply(make_tiny_model(), FP8KVConfig()))
    with pytest.raises(UnsupportedBackendError, match="match='latest'"):
        check_gguf(economize.apply(make_tiny_model(), LookupConfig(match="latest")))


def test_one_drafter_and_one_cache_type_per_export() -> None:
    builder = check_gguf(make_tiny_model())
    builder.claim(Role.DRAFTER, "lookup")
    builder.claim(Role.FORWARD, "a")
    builder.claim(Role.FORWARD, "b")  # not exclusive
    with pytest.raises(IncompatibleTechniquesError, match="--model-draft"):
        builder.claim(Role.DRAFTER, "draft_model")
    builder.claim(Role.KV_CACHE, "kivi")
    with pytest.raises(IncompatibleTechniquesError, match="KV-cache type"):
        builder.claim(Role.KV_CACHE, "kv_fp8")


def test_llama_quantize_command() -> None:
    assert llama_quantize_command("a.gguf", "b.gguf", "Q4_K_M") == ("llama-quantize", "a.gguf", "b.gguf", "Q4_K_M")


# ---- the GGUF files ------------------------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def gguf_reader() -> type:
    return pytest.importorskip("gguf").GGUFReader


def _tensors(path: Path, reader: type) -> dict[str, object]:
    return {tensor.name: tensor for tensor in reader(str(path)).tensors}


def test_dense_export_reads_back(tmp_path: Path, gguf_reader: type) -> None:
    model = make_tiny_model()
    export = to_gguf(model, tmp_path, outtype="f32")
    assert export.path == tmp_path / "model.gguf" and export.command == ("llama-server", "-m", "model.gguf")
    assert not export.simulated
    manifest = json.loads((tmp_path / "economize_gguf.json").read_text())
    assert manifest["runtime"] == "gguf"
    reader = gguf_reader(str(export.path))
    fields = {name: field.contents() for name, field in reader.fields.items() if not name.startswith("tokenizer")}
    assert fields["general.architecture"] == "qwen2" and fields["qwen2.block_count"] == 6
    assert fields["qwen2.attention.head_count_kv"] == 2 and fields["qwen2.embedding_length"] == 64
    tensors = {tensor.name: tensor for tensor in reader.tensors}
    assert "output.weight" not in tensors  # tied to token_embd
    embed = model.model.embed_tokens.weight.detach().numpy()
    assert np.array_equal(np.asarray(tensors["token_embd.weight"].data).reshape(embed.shape), embed)
    q_bias = model.model.layers[2].self_attn.q_proj.bias.detach().numpy()
    assert np.array_equal(np.asarray(tensors["blk.2.attn_q.bias"].data), q_bias)


def test_unknown_outtype_and_sliding_window_are_refused(tmp_path: Path, gguf_reader: type) -> None:
    with pytest.raises(ConfigError, match="outtype"):
        to_gguf(make_tiny_model(), tmp_path / "a", outtype="q4_k")
    with pytest.raises(ConfigError, match="sliding window"):
        to_gguf(make_family_model("mistral"), tmp_path / "b")


@pytest.mark.parametrize(("fmt", "symmetric", "qtype"), [("int4", True, "Q4_0"), ("int4", False, "Q4_1"),
                                                         ("int8", True, "Q8_0")])
def test_quant_codes_are_packed_bit_for_bit(tmp_path: Path, gguf_reader: type, fmt: str, symmetric: bool,
                                            qtype: str) -> None:
    import gguf

    from economize.integrations.compressed_tensors import quantized_modules

    model = economize.apply(make_tiny_model(), _quant(fmt, symmetric))
    with warnings.catch_warnings():
        warnings.simplefilter("error", SimulatedPathWarning)
        export = to_gguf(model, tmp_path, outtype="f32")
    assert export.supports == {"quant": "native"} and not export.simulated
    recorded = quantized_modules(model)
    tensors = _tensors(export.path, gguf_reader)
    weight = recorded["model.layers.1.mlp.down_proj"]
    assert code_type(weight) == (qtype, "")
    tensor = tensors["blk.1.ffn_down.weight"]
    assert tensor.tensor_type.name == qtype
    stored = gguf.quants.dequantize(tensor.data, tensor.tensor_type).reshape(weight.codes.shape)
    codes = weight.codes.double().numpy()
    scale = np.repeat(weight.scale.numpy().astype(np.float16).astype(np.float64), weight.group_size, axis=1)
    zero = np.repeat(weight.zero.double().numpy(), weight.group_size, axis=1)
    if qtype == "Q4_1":  # d * q + m with m = -scale * zero rounded to float16
        offset = np.repeat((-weight.scale * weight.zero).numpy().astype(np.float16).astype(np.float64),
                           weight.group_size, axis=1)
        expected = scale * codes + offset
    else:
        expected = scale * (codes - zero)
    np.testing.assert_allclose(stored, expected, rtol=1e-6, atol=1e-7)
    assert len([name for name, t in tensors.items() if t.tensor_type.name == qtype]) == len(recorded)


def test_fake_quant_is_written_dense_with_a_warning(tmp_path: Path, gguf_reader: type) -> None:
    model = economize.apply(make_tiny_model(), _quant(backend="fake"))
    with pytest.warns(SimulatedPathWarning, match="quant"):
        export = to_gguf(model, tmp_path, outtype="f32")
    assert export.supports == {"quant": "densified"} and export.simulated
    tensors = _tensors(export.path, gguf_reader)
    assert tensors["blk.0.attn_q.weight"].tensor_type.name == "F32"


def test_groups_that_are_not_blocks_of_32_are_written_dense(tmp_path: Path, gguf_reader: type) -> None:
    model = economize.apply(make_tiny_model(), QuantConfig(format="int4", group_size=16,
                                                           backend="compressed-tensors"))
    with pytest.warns(SimulatedPathWarning):
        export = to_gguf(model, tmp_path, outtype="f32")
    assert export.supports == {"quant": "densified"}
    assert "multiples of 32" in export.simulations["quant"]


# ---- llama.cpp round trips -----------------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def llama() -> type:
    pytest.importorskip("gguf")
    return pytest.importorskip("llama_cpp").Llama


def _scores(llm: object, ids: list[int] = IDS) -> np.ndarray:
    llm.eval(ids)  # type: ignore[attr-defined]  # a llama_cpp.Llama
    return np.asarray(llm.scores[: len(ids)])  # type: ignore[attr-defined]


def _greedy(llm: object, n: int = 20) -> list[int]:
    tokens: Iterator[int] = llm.generate(PROMPT, top_k=1, temp=0.0, reset=True)  # type: ignore[attr-defined]
    out: list[int] = []
    for token in tokens:
        out.append(int(token))
        if len(out) == n:
            break
    return out


@pytest.mark.parametrize("family", ["qwen2", "llama", "qwen3"])
def test_dense_float32_file_reproduces_the_logits(tmp_path: Path, llama: type, family: str) -> None:
    model = make_tiny_model() if family == "qwen2" else make_family_model(family)
    export = to_gguf(model, tmp_path, outtype="f32")
    llm = llama(str(export.path), logits_all=True, **LLAMA)
    try:
        np.testing.assert_allclose(_scores(llm), _logits(model), atol=1e-4)
    finally:
        llm.close()


def test_quantized_file_keeps_the_argmax(tmp_path: Path, llama: type) -> None:
    model = economize.apply(make_tiny_model(), _quant(symmetric=False))
    export = to_gguf(model, tmp_path, outtype="f32")
    llm = llama(str(export.path), logits_all=True, **LLAMA)
    try:
        got, want = _scores(llm), _logits(model)
    finally:
        llm.close()
    assert (got.argmax(-1) == want.argmax(-1)).mean() >= 0.75  # llama.cpp also quantizes the activations to 8 bits
    assert np.abs(got - want).max() < 1.0


def test_prompt_lookup_reproduces_greedy_decoding(tmp_path: Path, llama: type) -> None:
    from economize.backends.gguf import load_llama

    model = economize.apply(make_tiny_model(), LookupConfig())
    export = to_gguf(model, tmp_path, outtype="f32")
    assert export.settings["llama_cpp.LlamaPromptLookupDecoding"] == {"max_ngram_size": 3, "num_pred_tokens": 4}
    llm = load_llama(export, **LLAMA)
    try:
        assert type(llm.draft_model).__name__ == "LlamaPromptLookupDecoding"
        assert _greedy(llm) == reference_greedy(make_tiny_model(), torch.tensor(PROMPT), 20).tolist()
    finally:
        llm.close()


def test_draft_model_reproduces_greedy_decoding(tmp_path: Path, llama: type) -> None:
    from economize.backends.gguf import GgufGreedyDraft, load_llama

    _truncated(make_tiny_model(), 5).save_pretrained(tmp_path / "draft_lm")
    model = economize.apply(make_tiny_model(), DraftModelConfig(model=str(tmp_path / "draft_lm")))
    export = to_gguf(model, tmp_path / "out", outtype="f32", num_draft_tokens=3)
    assert [path.name for path in export.files] == ["model.gguf", "draft.gguf"]
    assert export.command == ("llama-server", "-m", "model.gguf", "--model-draft", "draft.gguf", "--draft-max", "3",
                              "--draft-min", "0")
    llm = load_llama(export, **LLAMA)
    try:
        assert isinstance(llm.draft_model, GgufGreedyDraft)
        assert _greedy(llm) == reference_greedy(make_tiny_model(), torch.tensor(PROMPT), 20).tolist()
    finally:
        llm.draft_model.draft.close()
        llm.close()


def test_byte_level_bpe_vocabulary_tokenizes_like_hugging_face(tmp_path: Path, llama: type) -> None:
    tokenizers = pytest.importorskip("tokenizers")
    from transformers import PreTrainedTokenizerFast, Qwen2Config, Qwen2ForCausalLM

    byte_level = tokenizers.pre_tokenizers.ByteLevel
    backend = tokenizers.Tokenizer(tokenizers.models.BPE())
    backend.pre_tokenizer = byte_level(add_prefix_space=False)
    backend.decoder = tokenizers.decoders.ByteLevel()
    trainer = tokenizers.trainers.BpeTrainer(vocab_size=300, initial_alphabet=byte_level.alphabet(),
                                             special_tokens=["<|endoftext|>"])
    backend.train_from_iterator(["the quick brown fox jumps over the lazy dog", "hello world 123 456"] * 20, trainer)
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, eos_token="<|endoftext|>")
    torch.manual_seed(0)
    config = Qwen2Config(vocab_size=len(tokenizer), hidden_size=64, intermediate_size=96, num_hidden_layers=2,
                         num_attention_heads=4, num_key_value_heads=2, tie_word_embeddings=True)
    export = to_gguf(Qwen2ForCausalLM(config).eval(), tmp_path, tokenizer=tokenizer, outtype="f32")
    llm = llama(str(export.path), **LLAMA)
    try:
        for text in ["hello world", "the lazy dog jumps 123", " two  spaces"]:
            assert llm.tokenize(text.encode(), add_bos=False, special=False) == tokenizer.encode(text), text
        assert llm.token_eos() == tokenizer.eos_token_id
    finally:
        llm.close()
