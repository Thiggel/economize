"""DFlash: configuration, lossless decoding on a tiny Gemma-4 and a tiny Llama, costs, the official checkpoint
layouts against a reference forward, and CUDA graphs."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pytest
import torch
import torch.nn.functional as F

import economize
from economize import ConfigError, DecodeConfig
from economize.core.serialization import FORMATS, read_checkpoint
from economize.core.technique import TECHNIQUES
from economize.errors import CheckpointFormatError
from economize.techniques.dflash import DFlashConfig, DFlashDrafter, DFlashTechnique, default_target_layers
from economize.techniques.dflash.formats import config_from_dflash, config_from_speculators, state_from_dflash
from helpers import PROMPT_LEN, VOCAB, family_skip_reason, make_family_model, make_tiny_model, perturb, reference_greedy

GEMMA = "gemma4_text"
SMALL = dict(num_layers=2, intermediate_size=48, num_heads=4, num_kv_heads=2, head_dim=16, rope_theta=1e4,
             mask_token_id=4)


def _base(family: str) -> torch.nn.Module:
    if family == GEMMA:
        reason = family_skip_reason(GEMMA)
        if reason:
            pytest.skip(reason)
        return make_family_model(GEMMA)
    return make_tiny_model()


def _model(config: DFlashConfig, base: torch.nn.Module, std: float = 0.1, **decoding: Any) -> Any:
    model = economize.apply(base, config, decoding=DecodeConfig(**decoding) if decoding else None)
    if std:
        perturb(model.techniques["dflash"], std, seed=1)
    return model


def _prompts(seed: int = 1, lengths: tuple[int, ...] = (PROMPT_LEN, 7, 20)) -> list[torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    return [torch.randint(1, VOCAB, (n,), generator=generator) for n in lengths]


# ---- configuration -------------------------------------------------------------------------------------------------
def test_registration_and_config() -> None:
    assert TECHNIQUES.get("dflash") is DFlashTechnique
    assert "dflash" in FORMATS.names() and "speculators_dflash" in FORMATS.names()
    assert DFlashConfig().types() == ("sliding_attention",) * 4 + ("full_attention",)
    # the reference's spread for Gemma-4-31B (60 layers): ids 1, 15, 29, 43, 57 read blocks 2, 16, 30, 44, 58
    assert default_target_layers(60) == (2, 16, 30, 44, 58)
    assert default_target_layers(4) == (2,) and default_target_layers(8) == (2, 3, 4, 5, 6)  # no repeats
    assert DFlashConfig(target_layers=[3, 1]).target_layers == (3, 1)
    for bad in (dict(block_size=1), dict(num_layers=0), dict(mask_token_id=-1), dict(num_heads=6, num_kv_heads=4),
                dict(layer_types=("full_attention",)), dict(num_layers=1, layer_types=("local",)),
                dict(sliding_window=0), dict(rope_theta=0), dict(embed_scale=0), dict(logit_softcap=-1.0),
                dict(param_dtype="float8"), dict(target_layers=()), dict(rope_scaling={"rope_type": "nope"})):
        with pytest.raises(ConfigError):
            DFlashConfig(**bad)


def test_validation_against_the_model() -> None:
    for bad in (dict(target_layers=(7,)), dict(mask_token_id=VOCAB), dict(num_kv_heads=3)):
        with pytest.raises(ConfigError):
            economize.apply(make_tiny_model(), DFlashConfig(**bad))


def test_parameters_are_named_as_in_the_official_checkpoints() -> None:
    model = economize.apply(make_tiny_model(), DFlashConfig(target_layers=(1, 3, 4), **SMALL))
    names = set(model.techniques["dflash"].state_dict())
    assert {"fc.weight", "hidden_norm.weight", "norm.weight", "layers.1.self_attn.k_norm.weight",
            "layers.0.mlp.down_proj.weight"} <= names
    assert not any(name.startswith(("embed_tokens", "lm_head")) for name in names)
    assert model.techniques["dflash"].fc.weight.shape == (64, 3 * 64)


# ---- lossless decoding ---------------------------------------------------------------------------------------------
@pytest.mark.parametrize("family", [GEMMA, "llama"])
@pytest.mark.parametrize("block_size", [4, 8])
@pytest.mark.parametrize("cycle_length", [1, 3, 6])
@torch.no_grad()
def test_greedy_blocks_equal_plain_greedy(family: str, block_size: int, cycle_length: int) -> None:
    """fp32 greedy decoding with block drafts equals greedy decoding by full recomputation."""
    base = _base(family)
    config = DFlashConfig(target_layers=(1, 3, 4), block_size=block_size, sliding_window=4, **SMALL)
    model = _model(config, base, cycle_length=cycle_length)
    prompts = _prompts()
    for prompt, result in zip(prompts, model.decode_batch(prompts, 24)):
        assert torch.equal(result.tokens, reference_greedy(base, prompt, 24))
    ragged = model.decode_batch(prompts, [5, 24, 11])
    for prompt, n, result in zip(prompts, [5, 24, 11], ragged):
        assert torch.equal(result.tokens, reference_greedy(base, prompt, n))


@torch.no_grad()
def test_lossy_decoding_keeps_whole_blocks() -> None:
    base = make_tiny_model()
    model = _model(DFlashConfig(target_layers=(2, 4), block_size=6, **SMALL), base)
    result = model.decode(_prompts()[0], 21, decoding=DecodeConfig(cycle_length=6, lossless=False))
    assert result.tokens.shape == (21,) and all(a == k for k, a in result.acceptance)
    assert {k for k, _ in result.acceptance} <= {5, 4, 3, 2, 1, 0}


@torch.no_grad()
def test_cost_log_counts_drafts_catchup_and_observe() -> None:
    base = make_tiny_model()
    block = 6
    model = _model(DFlashConfig(target_layers=(2, 4), block_size=block, **SMALL), base, cycle_length=4)
    result = model.decode(_prompts()[0], 30)
    events = result.cost.events
    assert {e.kind for e in events} <= {"exact", "draft", "draft_wasted", "draft_observe", "draft_catchup"}
    drafts = [e for e in events if e.kind in ("draft", "draft_wasted")]
    assert all((e.tokens, e.module) == (1, "dflash") for e in drafts)
    assert len(drafts) == sum(k for k, _ in result.acceptance)
    # every cycle runs the block at its full width: the 3 drafts plus the anchor and 2 unused masks
    catchup = [e for e in events if e.kind == "draft_catchup"]
    assert catchup and all(e.module == "dflash" and e.tokens == block - 3 for e in catchup)
    observe = [e for e in events if e.kind == "draft_observe"]
    assert observe and all(e.module == "dflash" for e in observe)


def test_save_and_load_round_trip(tmp_path: Path) -> None:
    base = make_tiny_model()
    model = _model(DFlashConfig(target_layers=(1, 4), block_size=5, **SMALL), base, cycle_length=5)
    economize.save(model, tmp_path / "ckpt")
    loaded = economize.load(make_tiny_model(), tmp_path / "ckpt")
    assert loaded.drafting_technique.config == model.drafting_technique.config
    for (name, a), (_, b) in zip(model.techniques["dflash"].state_dict().items(),
                                 loaded.techniques["dflash"].state_dict().items()):
        assert torch.equal(a, b), name
    prompt = _prompts()[2]
    first, second = model.decode(prompt, 20), loaded.decode(prompt, 20)
    assert torch.equal(first.tokens, second.tokens) and first.acceptance == second.acceptance


# ---- official checkpoints ------------------------------------------------------------------------------------------
_LAYER = dict(hidden_size=64, intermediate_size=48, num_attention_heads=4, num_key_value_heads=2, head_dim=16,
              rms_norm_eps=1e-6, attention_bias=False, sliding_window=4, max_position_embeddings=256)
ZLAB = dict(  # the layout of z-lab/gemma-4-31B-it-DFlash (Qwen3 decoder layers, 0-indexed target ids)
    _LAYER, architectures=["DFlashDraftModel"], model_type="qwen3", num_hidden_layers=3, block_size=8,
    layer_types=["sliding_attention", "sliding_attention", "full_attention"], rope_theta=1e6, vocab_size=VOCAB,
    dflash_config={"mask_token_id": 4, "target_layer_ids": [0, 2, 3]}, final_logit_softcapping=3.0,
    tie_word_embeddings=True,
)
SPECULATORS = dict(  # the layout of RedHatAI/gemma-4-31B-it-speculator.dflash
    architectures=["DFlashDraftModel"], speculators_model_type="dflash", block_size=8, mask_token_id=4,
    aux_hidden_state_layer_ids=[1, 3, 4], draft_vocab_size=VOCAB, sliding_window_non_causal=False,
    tie_word_embeddings=False,
    speculators_config={"verifier": {"name_or_path": "google/gemma-4-31b-it"}, "proposal_methods": []},
    transformer_layer_config=dict(_LAYER, num_hidden_layers=3, vocab_size=VOCAB, layer_types=["sliding_attention"] * 3,
                                  rope_parameters={"rope_theta": 10000.0, "rope_type": "default"}),
)


def _official_state(layers: int, sources: int, seed: int = 0) -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    d, inter, heads, kv, dim = 64, 48, 4, 2, 16

    def w(*shape: int, std: float = 0.1) -> torch.Tensor:
        return torch.randn(*shape, generator=generator) * std

    def norm(n: int) -> torch.Tensor:
        return 1 + w(n)

    state = {"fc.weight": w(d, sources * d), "hidden_norm.weight": norm(d), "norm.weight": norm(d),
             "embed_tokens.weight": w(VOCAB, d)}
    for i in range(layers):
        p = f"layers.{i}."
        state |= {p + "input_layernorm.weight": norm(d), p + "post_attention_layernorm.weight": norm(d),
                  p + "self_attn.q_proj.weight": w(heads * dim, d), p + "self_attn.k_proj.weight": w(kv * dim, d),
                  p + "self_attn.v_proj.weight": w(kv * dim, d), p + "self_attn.o_proj.weight": w(d, heads * dim),
                  p + "self_attn.q_norm.weight": norm(dim), p + "self_attn.k_norm.weight": norm(dim),
                  p + "mlp.gate_proj.weight": w(inter, d), p + "mlp.up_proj.weight": w(inter, d),
                  p + "mlp.down_proj.weight": w(d, inter)}
    return state


def _write(path: Path, config: dict[str, Any], state: dict[str, torch.Tensor]) -> None:
    from safetensors.torch import save_file

    path.mkdir(parents=True)
    (path / "config.json").write_text(json.dumps(config))
    save_file(state, path / "model.safetensors")


def _reference_block(state: dict[str, torch.Tensor], layer: dict[str, Any], types: list[str], causal_sliding: bool,
                     theta: float, context: torch.Tensor, block: torch.Tensor, start: int) -> torch.Tensor:
    """The draft layers' outputs ``[1, B, d]`` written out from the raw tensors after z-lab/dflash ``model.py``:
    queries of the block at ``start .. start + B - 1``, keys and values of ``[context; block]`` at ``0 .. start +
    B - 1``, the window on absolute positions."""
    eps, heads, kv, dim, window = (layer["rms_norm_eps"], layer["num_attention_heads"],
                                   layer["num_key_value_heads"], layer["head_dim"], layer["sliding_window"])

    def rms(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        return weight * x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)

    def rope(x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        angles = positions.float()[:, None] / theta ** (torch.arange(0, dim, 2).float() / dim)
        cos, sin = torch.cat([angles, angles], -1).cos(), torch.cat([angles, angles], -1).sin()
        return x * cos + torch.cat([-x[..., dim // 2:], x[..., : dim // 2]], -1) * sin

    width, n = block.shape[1], context.shape[1]
    ctx = rms(context @ state["fc.weight"].T, state["hidden_norm.weight"])
    query_pos, key_pos = torch.arange(start, start + width), torch.arange(n + width)
    h = block
    for i, kind in enumerate(types):
        p = f"layers.{i}."
        x = rms(h, state[p + "input_layernorm.weight"])
        both = torch.cat([ctx, x], 1)
        q = (x @ state[p + "self_attn.q_proj.weight"].T).view(1, width, heads, dim).transpose(1, 2)
        k = (both @ state[p + "self_attn.k_proj.weight"].T).view(1, n + width, kv, dim).transpose(1, 2)
        v = (both @ state[p + "self_attn.v_proj.weight"].T).view(1, n + width, kv, dim).transpose(1, 2)
        q = rope(rms(q, state[p + "self_attn.q_norm.weight"]), query_pos)
        k = rope(rms(k, state[p + "self_attn.k_norm.weight"]), key_pos)
        k, v = k.repeat_interleave(heads // kv, 1), v.repeat_interleave(heads // kv, 1)
        visible = torch.ones(width, n + width, dtype=torch.bool)
        sliding = kind == "sliding_attention"
        if sliding and causal_sliding:
            visible &= key_pos[None] <= query_pos[:, None]
        if sliding:
            visible &= (query_pos[:, None] - key_pos[None]).abs() < window
        scores = (q @ k.transpose(-1, -2)) / math.sqrt(dim)
        attended = (scores.masked_fill(~visible, float("-inf")).softmax(-1) @ v).transpose(1, 2).reshape(1, width, -1)
        h = h + attended @ state[p + "self_attn.o_proj.weight"].T
        y = rms(h, state[p + "post_attention_layernorm.weight"])
        h = h + (F.silu(y @ state[p + "mlp.gate_proj.weight"].T) * (y @ state[p + "mlp.up_proj.weight"].T)) @ \
            state[p + "mlp.down_proj.weight"].T
    return rms(h, state["norm.weight"])


@pytest.mark.parametrize("style", ["zlab", "speculators"])
@torch.no_grad()
def test_official_drafters_load_and_draft_as_the_reference(style: str, tmp_path: Path) -> None:
    """A fake official drafter of each layout loads on a tiny Gemma-4, drafts in lossy mode exactly the reference
    forward's argmax tokens, and decodes losslessly."""
    base = _base(GEMMA)
    config = ZLAB if style == "zlab" else SPECULATORS
    name = "dflash" if style == "zlab" else "speculators_dflash"
    state = _official_state(3, 3)
    folder = tmp_path / style
    _write(folder, config, state)
    assert FORMATS.get(name).detect(folder)
    expected = dict(target_layers=(1, 3, 4), block_size=8, mask_token_id=4, num_layers=3, intermediate_size=48,
                    num_heads=4, num_kv_heads=2, head_dim=16, sliding_window=4, rms_norm_eps=1e-6)
    checkpoint = read_checkpoint(folder, format=name)
    if style == "zlab":
        assert checkpoint.configs == [DFlashConfig(**expected, layer_types=tuple(ZLAB["layer_types"]),
                                                   rope_theta=1e6, logit_softcap=3.0)]
    else:
        assert checkpoint.configs == [DFlashConfig(**expected, layer_types=("sliding_attention",) * 3,
                                                   rope_theta=1e4, logit_softcap=0.0)]
        assert checkpoint.base_model == "google/gemma-4-31b-it"
    model = economize.load(base, folder, format=name, decoding=DecodeConfig(cycle_length=8))
    technique = model.techniques["dflash"]
    assert technique.target_layers == (1, 3, 4)

    prompt = _prompts()[0]
    first = reference_greedy(base, prompt, 1)
    lossy = model.decode(prompt, 8, decoding=DecodeConfig(cycle_length=8, lossless=False))
    assert lossy.tokens[0] == first[0]
    states = base(prompt[None], output_hidden_states=True).hidden_states
    context = torch.cat([states[layer] for layer in (1, 3, 4)], -1)
    embed = base.get_input_embeddings().weight
    ids = torch.full((1, 8), 4)
    ids[0, 0] = first[0]
    layer = config["transformer_layer_config"] if style == "speculators" else config
    theta = 1e6 if style == "zlab" else 1e4
    hidden = _reference_block(state, layer, list(layer["layer_types"]), True, theta, context, embed[ids],
                              prompt.numel())
    reference = base.lm_head(hidden)[0, 1:].argmax(-1)
    assert lossy.tokens[1:].tolist() == reference.tolist()
    for prompt, result in zip(_prompts(), model.decode_batch(_prompts(), 20)):
        assert torch.equal(result.tokens, reference_greedy(base, prompt, 20))


def test_round_trip_of_an_official_drafter(tmp_path: Path) -> None:
    _write(tmp_path / "zlab", ZLAB, _official_state(3, 3))
    model = economize.load(make_tiny_model(), tmp_path / "zlab", format="dflash")
    economize.save(model, tmp_path / "ckpt")
    loaded = economize.load(make_tiny_model(), tmp_path / "ckpt")
    assert loaded.drafting_technique.config == model.drafting_technique.config
    for (name, a), (_, b) in zip(model.techniques["dflash"].state_dict().items(),
                                 loaded.techniques["dflash"].state_dict().items()):
        assert torch.equal(a, b), name


def test_layout_helpers() -> None:
    assert config_from_dflash(ZLAB).num_layers == 3 and config_from_dflash(ZLAB, 2).layer_types == (
        "sliding_attention", "sliding_attention")
    assert config_from_speculators(SPECULATORS).sliding_causal
    assert not config_from_speculators({**SPECULATORS, "sliding_window_non_causal": True}).sliding_causal
    assert "embed_tokens.weight" not in state_from_dflash(_official_state(1, 3))


def test_unsupported_official_variants_are_refused(tmp_path: Path) -> None:
    state = _official_state(3, 3)
    cases = {
        "vocab": (SPECULATORS, "speculators_dflash", state, {"draft_vocab_size": 64}),
        "width": (ZLAB, "dflash", state, {"target_hidden_size": 128}),
        "multiplier": (ZLAB, "dflash", state, {"output_multiplier": 2.0}),
        "causal": (ZLAB, "dflash", state, {"is_causal": True}),
        "kind": (ZLAB, "dflash", state, {"layer_types": ["chunked_attention"] * 3}),
        "head": (ZLAB, "dflash", {**state, "lm_head.weight": torch.zeros(VOCAB, 64)}, {}),
        "unknown": (ZLAB, "dflash", {**state, "layers.0.extra.weight": torch.zeros(2)}, {}),
        "missing": (ZLAB, "dflash", {k: v for k, v in state.items() if k != "hidden_norm.weight"}, {}),
    }
    for case, (config, name, tensors, change) in cases.items():
        folder = tmp_path / case
        _write(folder, {**config, **change}, tensors)
        with pytest.raises(CheckpointFormatError):
            read_checkpoint(folder, format=name)
    with pytest.raises(CheckpointFormatError):
        FORMATS.get("dflash").read(tmp_path / "missing_dir")


def test_the_zlab_layout_is_detected_by_itself_only(tmp_path: Path) -> None:
    _write(tmp_path / "zlab", ZLAB, _official_state(3, 3))
    assert [name for name in FORMATS.names() if FORMATS.get(name).detect(tmp_path / "zlab")] == ["dflash"]
    assert read_checkpoint(tmp_path / "zlab").configs == read_checkpoint(tmp_path / "zlab", format="dflash").configs


# ---- CUDA graphs ---------------------------------------------------------------------------------------------------
@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
@pytest.mark.parametrize("cycle_length", [3, 6])
@torch.no_grad()
def test_graphed_blocks_match_eager_and_are_replayed(cycle_length: int) -> None:
    """Graphed blocks (persistent cache, full-capacity reads) draft exactly what eager blocks draft, also when rows
    leave the batch and when a longer generation outgrows the cache."""
    base = make_tiny_model().cuda()
    model = _model(DFlashConfig(target_layers=(1, 3, 4), block_size=6, sliding_window=4, **SMALL), base)
    prompts = [p.cuda() for p in _prompts(lengths=(PROMPT_LEN, 7, 20, 5))]
    limits = [24, 3, 17, 9]
    greedy = [reference_greedy(base, p, n).tolist() for p, n in zip(prompts, limits)]
    eager = model.decode_batch(prompts, limits, decoding=DecodeConfig(cycle_length=cycle_length))
    graphed = DecodeConfig(cycle_length=cycle_length, cuda_graphs=True)
    for _ in range(3):  # eager warm-up, capture, replay
        results = model.decode_batch(prompts, limits, decoding=graphed)
        assert [r.tokens.tolist() for r in results] == greedy
        assert [r.acceptance for r in results] == [r.acceptance for r in eager]
    decoder = model.decoder(graphed)
    drafter = decoder.active_drafter
    assert isinstance(drafter, DFlashDrafter) and drafter._step_fn.enabled
    for p, n in zip(prompts, limits):
        assert model.decode(p, n, decoding=graphed).tokens.tolist() == reference_greedy(base, p, n).tolist()
    long = model.decode(prompts[0], 300, decoding=graphed)
    assert long.tokens.tolist() == reference_greedy(base, prompts[0], 300).tolist()
    assert long.acceptance == model.decode(prompts[0], 300, decoding=DecodeConfig(cycle_length=cycle_length)).acceptance
