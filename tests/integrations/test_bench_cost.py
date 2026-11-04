"""The analytic cost model: parameter counts of real ``transformers`` models, FLOPs against PyTorch's FLOP counter,
the attention and expert-routing formulas, and pricing of real cost logs."""

from __future__ import annotations

import math
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from torch.utils.flop_counter import FlopCounterMode
from transformers import (
    DynamicCache,
    GPT2Config,
    Llama4TextConfig,
    LlamaConfig,
    LlamaForCausalLM,
    MistralConfig,
    MixtralConfig,
    MixtralForCausalLM,
    NemotronHConfig,
    NemotronHForCausalLM,
    Qwen2Config,
    Qwen2MoeConfig,
    Qwen2MoeForCausalLM,
)

from economize.arch.spec import AttentionSpec
from economize.bench import (
    HARDWARE,
    SHAPE_READERS,
    CallSpec,
    CostModel,
    Hardware,
    ModelShape,
    count_tokens,
    get_hardware,
)
from economize.compat.tskip import cost_events
from economize.decoding import CostEvent, CostLog
from economize.distributed import AsyncSpeculativeDecoder, SSDConfig
from economize.errors import ConfigError, UnknownNameError
from economize.runtime.mask import MASKS
from helpers import PROMPT_LEN, VOCAB, make_tiny_model

TINY = dict(vocab_size=128, hidden_size=64, intermediate_size=96, num_hidden_layers=3, num_attention_heads=4,
            num_key_value_heads=2)


def _matrix_params(module: torch.nn.Module) -> int:
    return sum(p.numel() for p in module.parameters() if p.dim() >= 2)


@pytest.mark.parametrize("build", [
    lambda: LlamaForCausalLM(LlamaConfig(**TINY, attention_bias=True)),
    lambda: MixtralForCausalLM(MixtralConfig(**TINY, num_local_experts=4, num_experts_per_tok=2)),
    lambda: Qwen2MoeForCausalLM(Qwen2MoeConfig(**TINY, moe_intermediate_size=32, shared_expert_intermediate_size=48,
                                               num_experts=4, num_experts_per_tok=2)),
], ids=["llama", "mixtral", "qwen2-moe"])
def test_layer_params_match_the_model(build) -> None:
    model = build()
    shape = ModelShape.from_config(model.config)
    for layer, module in enumerate(model.model.layers, start=1):
        assert shape.layer_params(layer) == _matrix_params(module)


def test_total_params_of_a_tied_model() -> None:
    model = make_tiny_model()
    shape = ModelShape.from_config(model.config)
    assert shape.total_params == _matrix_params(model)  # the tied embedding is the head
    assert shape.head_params == VOCAB * 64


def test_qwen25_half_billion() -> None:
    """Qwen2.5-0.5B has 494.0M parameters; weight matrices are all but 0.1 % of them."""
    config = Qwen2Config(vocab_size=151936, hidden_size=896, intermediate_size=4864, num_hidden_layers=24,
                         num_attention_heads=14, num_key_value_heads=2, tie_word_embeddings=True)
    shape = ModelShape.from_config(config)
    assert shape.head_dim == 64 and shape.layer_types == ("full_attention",) * 24 and shape.sliding_window is None
    assert shape.total_params == pytest.approx(494.03e6, rel=1e-3)


def test_expert_routing() -> None:
    shape = ModelShape.from_config(Qwen2MoeConfig(**TINY, moe_intermediate_size=32, num_experts=8,
                                                  num_experts_per_tok=2, shared_expert_intermediate_size=0,
                                                  mlp_only_layers=[0], decoder_sparse_step=1))
    expert = 3 * 64 * 32
    assert shape.dense_layers == frozenset({1}) and not shape.is_moe(1) and shape.is_moe(2)
    assert shape.layer_params(2) - shape.active_params(2) == 6 * expert
    assert shape.touched_params(2, 1) == shape.active_params(2)
    assert shape.touched_params(2, 0) == 0
    assert shape.active_params(2) < shape.touched_params(2, 4) < shape.layer_params(2)
    assert shape.touched_params(2, 10_000) == pytest.approx(shape.layer_params(2))
    assert shape.touched_params(1, 4) == shape.layer_params(1)  # a dense layer reads everything



def test_family_shape_readers() -> None:
    """A family whose config the generic reading gets wrong registers a reader; ``from_config`` stays unchanged."""
    gpt2 = ModelShape.from_config(GPT2Config(n_embd=64, n_layer=2, n_head=4, vocab_size=100))
    assert "gpt2" in SHAPE_READERS and not gpt2.gated and gpt2.intermediate == 256 and gpt2.num_layers == 2
    assert ModelShape.from_config(LlamaConfig(**TINY)).gated

    def wide(config, fields):
        return {"intermediate": 2 * fields["intermediate"]}

    SHAPE_READERS.register("llama", wide)
    try:
        assert ModelShape.from_config(LlamaConfig(**TINY)).intermediate == 2 * TINY["intermediate_size"]
        assert ModelShape.from_config(LlamaConfig(**TINY), intermediate=8).intermediate == 8  # overrides win
    finally:
        SHAPE_READERS.unregister("llama")

def _nemotron_h(**kwargs: object) -> NemotronHConfig:
    """Blocks Mamba-2, MoE, attention, MLP (Nemotron-H keeps one mixer per block)."""
    return NemotronHConfig(**dict(TINY, num_hidden_layers=None), head_dim=16,
                           layers_block_type=["linear_attention", "moe", "full_attention", "mlp"], mamba_num_heads=4,
                           mamba_head_dim=8, ssm_state_size=8, n_groups=2, conv_kernel=4, n_routed_experts=4,
                           num_experts_per_tok=2, moe_intermediate_size=32, moe_shared_expert_intermediate_size=48,
                           use_mamba_kernels=False, tie_word_embeddings=False, **kwargs)


def test_nemotron_h_against_a_hand_count() -> None:
    """One part per block, ungated relu^2 MLPs and experts, a Mamba-2 mixer priced by its weights."""
    shape = ModelShape.from_config(_nemotron_h())
    mamba = 64 * (32 + 64 + 4) + 64 * 4 + 32 * 64  # in_proj (gate, x B C, dt), conv1d over x B C, out_proj
    router, expert, shared = 4 * 64, 2 * 64 * 32, 2 * 64 * 48
    attention = 64 * 64 + 2 * 64 * 32 + 64 * 64  # q, k, v, o with 2 of 4 heads for keys and values
    mlp = 2 * 64 * 96
    assert [shape.layer_params(i) for i in range(1, 5)] == [mamba, router + shared + 4 * expert, attention, mlp]
    assert [shape.active_params(i) for i in range(1, 5)] == [mamba, router + shared + 2 * expert, attention, mlp]
    assert shape.total_params == 8704 + 22784 + 12288 + 12288 + 128 * 64 == 64256
    position = 9
    flops = CostModel(shape).event(CostEvent("exact", 1, position, 1, 4)).flops
    attended = 4 * 4 * 16 * (position + 1)  # scores and values over the keys, in the attention block only
    assert flops == 2 * (mamba + router + shared + 2 * expert + attention + mlp) + attended + 2 * 64 * 128


@pytest.mark.parametrize("latent", [None, 16], ids=["moe", "latent-moe"])
def test_nemotron_h_layer_params_match_the_model(latent: int | None) -> None:
    model = NemotronHForCausalLM(_nemotron_h(moe_latent_size=latent))
    shape = ModelShape.from_config(model.config)
    for layer, module in enumerate(model.model.layers, start=1):
        assert shape.layer_params(layer) == _matrix_params(module)
    assert shape.total_params == _matrix_params(model) - _matrix_params(model.model.embeddings)


@pytest.mark.parametrize("position", [0, 6, 29])
def test_decode_step_flops_match_the_flop_counter(position: int) -> None:
    """A one-token step: every matmul the model runs is a weight or an attention product."""
    model = make_tiny_model(attn="eager")
    shape = ModelShape.from_config(model.config)
    cache = DynamicCache(config=model.config)
    with torch.no_grad():
        if position:
            model(torch.arange(3, 3 + position).view(1, -1), past_key_values=cache, use_cache=True)
        with FlopCounterMode(display=False) as counter:
            model(torch.tensor([[5]]), past_key_values=cache, use_cache=True)
    flops = CostModel(shape).event(CostEvent("exact", 1, position, 1, shape.num_layers)).flops
    rotary = counter.get_total_flops() - flops  # the rotary embedding's frequencies times the position
    assert 0 <= rotary <= shape.head_dim


@pytest.mark.parametrize(("window", "position", "tokens"), [(None, 0, 7), (None, 9, 3), (8, 0, 5), (8, 0, 12),
                                                             (8, 5, 6), (8, 20, 4), (1, 3, 3)])
def test_attended_keys(window: int | None, position: int, tokens: int) -> None:
    layer_type = "sliding_attention" if window else "full_attention"
    shape = ModelShape(2, 8, 2, 1, 4, 16, 10, layer_types=(layer_type, "linear_attention"), sliding_window=window)
    expected = sum(min(j, window or j) for j in range(position + 1, position + tokens + 1))
    assert shape.attended_keys(1, position, tokens) == expected
    assert shape.cached_keys(1, position + tokens) == min(position + tokens, window or position + tokens)
    assert shape.attended_keys(2, position, tokens) == 0 and shape.cached_keys(2, 99) == 0


def test_sliding_window_from_config() -> None:
    shape = ModelShape.from_config(MistralConfig(**TINY, sliding_window=8))
    assert shape.layer_types == ("sliding_attention",) * 3 and shape.sliding_window == 8
    long = CostModel(shape).event(CostEvent("exact", 1, 1000, 1, 3))
    short = CostModel(shape).event(CostEvent("exact", 1, 10, 1, 3))
    assert long == short  # past the window, the context no longer costs anything


def test_logits_per_kind() -> None:
    shape = ModelShape.from_config(make_tiny_model().config)
    head = 2 * shape.hidden * shape.vocab
    costs, last = CostModel(shape), CostModel(shape, exact_logits="last")

    def head_flops(model: CostModel, kind: str, tokens: int, position: int) -> float:
        with_head = model.event(CostEvent(kind, tokens, position, 1, 6)).flops
        without = CostModel(shape, specs={kind: CallSpec(logits="none")}).event(
            CostEvent(kind, tokens, position, 1, 6)).flops
        return (with_head - without) / head

    assert head_flops(costs, "exact", 12, 0) == 1  # the prompt: its last token only
    assert head_flops(costs, "exact", 4, 12) == 4  # a verification: every token
    assert head_flops(last, "exact", 4, 12) == 1  # a lossy refresh
    assert head_flops(costs, "draft", 1, 12) == 1
    assert costs.event(CostEvent("replay", 3, 12, 1, 6)).flops == CostModel(
        shape, specs={"replay": CallSpec(logits="none")}).event(CostEvent("replay", 3, 12, 1, 6)).flops
    reduced = CostModel(shape, draft=CallSpec(head_rows=32))
    assert costs.event(CostEvent("draft", 1, 12, 3, 4)).flops - reduced.event(
        CostEvent("draft", 1, 12, 3, 4)).flops == 2 * shape.hidden * (shape.vocab - 32)


def test_jump_cycle_against_plain() -> None:
    """A jump drafter's step runs layers m+1..r plus its predictor; a cycle of N tokens replaces N-1 full steps."""
    shape = ModelShape.from_config(make_tiny_model().config)
    extra = 2 * 64 * 64
    costs = CostModel(shape, draft=CallSpec(extra_params=extra), exact_logits="last")
    draft = costs.event(CostEvent("draft", 1, 20, 3, 4))
    full = costs.event(CostEvent("exact", 1, 20, 1, 6))
    attention = 4 * shape.heads * shape.head_dim * 21
    assert draft.flops == 2 * (2 * shape.active_params() + extra) + 2 * attention + 2 * shape.hidden * shape.vocab
    assert full.flops == 6 * (2 * shape.active_params() + attention) + 2 * shape.hidden * shape.vocab

    log = CostLog()
    log.record("exact", PROMPT_LEN, 0, 1, 6)
    position = PROMPT_LEN
    for _ in range(5):  # N = 4: three draft steps, then one exact pass over the three pending tokens
        for _ in range(3):
            log.record("draft", 1, position, 3, 4)
            position += 1
        log.record("exact", 3, position - 3, 1, 6)
    generated = position - PROMPT_LEN + 1
    relative = costs.relative(log, PROMPT_LEN, generated, "a100-80gb")
    # the exact pass recomputes the pending tokens, so the cycle spends more FLOPs and reads the weights less often
    assert relative["flops"] > 1 and relative["bytes"] < 1 and relative["speedup"] > 1
    assert relative["seconds"] == pytest.approx(1 / relative["speedup"])
    assert CostModel(shape).relative(costs.plain_log(PROMPT_LEN, generated), PROMPT_LEN, generated, "a40") == {
        "flops": 1.0, "bytes": 1.0, "calls": 1.0, "seconds": 1.0, "speedup": 1.0}


def test_report_totals_and_roofline() -> None:
    shape = ModelShape.from_config(make_tiny_model().config)
    costs = CostModel(shape)
    report = costs.plain(PROMPT_LEN, 9)
    assert len(report.calls) == 9 and set(report.by_kind()) == {"exact"}
    assert report.by_kind()["exact"].calls == 9
    assert report.flops == pytest.approx(sum(c.flops for _, c in report.calls))
    hw = Hardware("toy", flops=1e9, bandwidth=1e9, call_overhead=1e-3)
    step = costs.event(CostEvent("exact", 1, 20, 1, 6))
    assert step.seconds(hw) == pytest.approx(max(step.flops, step.bytes) / 1e9 + 1e-3)
    assert report.seconds(hw) == pytest.approx(sum(c.seconds(hw) for _, c in report.calls))
    assert step.intensity == step.flops / step.bytes
    assert report.to_dict("a40")["by_kind"]["exact"]["calls"] == 9 and report.to_dict()["calls"] == 9


def test_quantized_weights_move_fewer_bytes() -> None:
    shape = ModelShape.from_config(make_tiny_model().config)
    event = CostEvent("exact", 1, 30, 1, 6)
    bf16, int4 = CostModel(shape).event(event), CostModel(shape, weight_bytes=0.5).event(event)
    assert int4.flops == bf16.flops and int4.bytes < bf16.bytes / 2


def test_hardware_presets_and_extension() -> None:
    a100 = get_hardware("a100-80gb")
    assert a100.flops == 312e12 and a100.bandwidth == 2039e9 and a100.ridge == pytest.approx(153.0, rel=1e-2)
    assert get_hardware(a100) is a100 and "h100-sxm" in HARDWARE.names()
    with pytest.raises(UnknownNameError):
        get_hardware("abacus")
    HARDWARE.register("test-chip", Hardware("test-chip", 1e15, 1e12))
    try:
        assert get_hardware("test-chip").ridge == 1000
    finally:
        HARDWARE.unregister("test-chip")


def test_invalid_inputs() -> None:
    shape = ModelShape.from_config(make_tiny_model().config)
    with pytest.raises(ConfigError, match="layer 9"):
        CostModel(shape).event(CostEvent("exact", 1, 0, 1, 9))
    with pytest.raises(ConfigError):
        CostModel(shape, exact_logits="some")  # type: ignore[arg-type]
    with pytest.raises(ConfigError):
        CallSpec(logits="first")  # type: ignore[arg-type]
    with pytest.raises(ConfigError):
        ModelShape(2, 8, 2, 1, 4, 16, 10, layer_types=("full_attention",))
    with pytest.raises(ConfigError):
        ModelShape(2, 8, 2, 1, 4, 16, 10, experts=4, active_experts=5)


def test_prices_a_two_model_generation() -> None:
    """A separate drafter: its calls are priced with its own shape, and the target's with the target's."""
    target, drafter = make_tiny_model(seed=0), make_tiny_model(seed=0, num_layers=2)
    prompt = torch.randint(0, VOCAB, (PROMPT_LEN,), generator=torch.Generator().manual_seed(1))
    result = AsyncSpeculativeDecoder(target, drafter, SSDConfig(draft_length=3, mode="sync")).generate(prompt, 16)
    shape, small = ModelShape.from_config(target.config), ModelShape.from_config(drafter.config)
    costs = CostModel(shape, draft=CallSpec(shape=small))
    target_report, draft_report = costs.log(result.cost), costs.log(result.draft_cost)
    assert set(target_report.by_kind()) == {"exact"} and set(draft_report.by_kind()) == {"draft"}
    assert draft_report.flops < target_report.flops
    with pytest.raises(ConfigError):  # the drafter's events do not fit a 1-layer shape
        CostModel(shape, draft=CallSpec(shape=ModelShape(1, 64, 4, 2, 16, 128, VOCAB))).log(result.draft_cost)


def test_prices_an_economize_generation(jump_model) -> None:
    model = jump_model(cycle_length=4)
    prompt = torch.randint(0, VOCAB, (PROMPT_LEN,), generator=torch.Generator().manual_seed(2))
    result = model.decode(prompt, 20)
    shape = ModelShape.from_config(model.base_model.config)
    extra = sum(p.numel() for t in model.techniques.values() for p in t.parameters() if p.dim() >= 2)
    costs = CostModel(shape, draft=CallSpec(extra_params=extra))
    report = costs.log(result.cost)
    assert len(report.calls) == len(result.cost)
    relative = costs.relative(result.cost, PROMPT_LEN, result.tokens.numel(), "a40")
    assert math.isfinite(relative["speedup"]) and relative["flops"] > 0 and relative["bytes"] > 0


# ---- layer sets and modules ----------------------------------------------------------------------------------------
def test_prices_exactly_the_layers_an_event_ran() -> None:
    """A draft that skips layers costs the layers it ran: layers {1, 3} of a uniform model cost twice one layer."""
    shape = ModelShape.from_config(make_tiny_model().config)
    costs = CostModel(shape, specs={"draft": CallSpec(logits="none")})
    one = costs.event(CostEvent("draft", 1, 10, 2, 2))
    split = costs.event(CostEvent("draft", 1, 10, 1, 3, layers=(1, 3)))
    span = costs.event(CostEvent("draft", 1, 10, 1, 3))
    assert split.flops == pytest.approx(2 * one.flops) and split.bytes == pytest.approx(2 * one.bytes)
    assert span.flops == pytest.approx(3 * one.flops)


def test_prices_module_events_by_their_module() -> None:
    shape = ModelShape.from_config(make_tiny_model().config)
    small = ModelShape.from_config(make_tiny_model(num_layers=2).config)
    spec = CallSpec(shape=small)
    log = CostLog()
    log.record("draft", 1, 10, 1, 2, module="eagle3")
    log.record("draft_observe", 3, 11, 1, 2, module="eagle3")
    with pytest.raises(ConfigError, match="'eagle3'"):
        CostModel(shape).log(log)
    report = CostModel(shape, modules={"eagle3": spec}).log(log)
    plain = CostModel(small)
    draft, observe = (cost for _, cost in report.calls)
    assert draft.flops == pytest.approx(plain.event(CostEvent("draft", 1, 10, 1, 2)).flops)
    # observing is the module's call without logits
    assert observe.flops == pytest.approx(plain.event(CostEvent("replay", 3, 11, 1, 2)).flops)
    assert observe.flops < plain.event(CostEvent("draft", 3, 11, 1, 2)).flops


def test_draft_is_the_draft_model_module() -> None:
    """``draft=`` prices the untagged draft events of older logs and the ``"draft_model"`` module alike, and its
    catch-up calls without logits."""
    shape = ModelShape.from_config(make_tiny_model().config)
    small = ModelShape.from_config(make_tiny_model(num_layers=2).config)
    costs = CostModel(shape, draft=CallSpec(shape=small))
    tagged, untagged = CostLog(), CostLog()
    for log, module in ((tagged, "draft_model"), (untagged, None)):
        log.record("draft", 1, 10, 1, 2, module=module)
        log.record("draft_catchup", 2, 8, 1, 2, module=module)
    assert costs.log(tagged).flops == pytest.approx(costs.log(untagged).flops)
    catchup = costs.event(CostEvent("draft_catchup", 2, 8, 1, 2, module="draft_model"))
    assert catchup.flops == pytest.approx(CostModel(small, specs={"x": CallSpec(logits="none")}).event(
        CostEvent("x", 2, 8, 1, 2)).flops)
    explicit = CostModel(shape, modules={"draft_model": CallSpec(shape=small)})
    assert explicit.log(tagged).flops == pytest.approx(costs.log(tagged).flops)


def test_summary_splits_the_decoded_model_from_modules() -> None:
    log = CostLog()
    log.record("exact", 4, 0, 1, 4)
    log.record("draft", 1, 4, 1, 3, layers=(1, 3))
    log.record("draft", 1, 4, 1, 2, module="draft_model")
    log.record("draft_observe", 2, 4, 1, 1, module="eagle3")
    summary = log.summary(num_layers=4, prompt_tokens=4, generated=2)
    assert summary["layer_tokens"] == 4 * 4 + 2
    assert summary["module_layer_tokens"] == {"draft_model": 2, "eagle3": 2}
    assert summary["draft_steps"] == 2
    assert cost_events(log)[-1] == ("pred_ctx", 2, 4)


# ---- token counting, sliding-window reads and chunked attention ----
def _visible(kind: str, window: int, position: int, tokens: int) -> torch.Tensor:
    """``[T, K]`` visibility of the keys ``0 .. position+tokens-1`` for the call's queries."""
    queries = torch.arange(position, position + tokens)[:, None]
    keys = torch.arange(position + tokens)[None, :]
    return MASKS.get(kind)(queries, keys, AttentionSpec(kind, window, None))


def _shape(kind: str, window: int) -> ModelShape:
    extra = {"sliding_window": window} if kind == "sliding_attention" else {"chunk_size": window}
    return ModelShape(1, 8, 2, 1, 4, 16, 10, layer_types=(kind,), **extra)


def test_count_tokens_of_batched_outputs() -> None:
    assert count_tokens(torch.zeros(2, 6, dtype=torch.long)) == 12
    results = [SimpleNamespace(tokens=torch.arange(5)), SimpleNamespace(tokens=torch.arange(7))]
    assert count_tokens(results) == 12  # the list decode_batch returns
    assert count_tokens([torch.arange(3), torch.arange(4)]) == 7
    assert count_tokens([[1, 2], [3, 4, 5]]) == 5
    assert count_tokens([1, 2, 3]) == 3
    assert count_tokens(SimpleNamespace(tokens=[4, 5])) == 2
    with pytest.raises(TypeError):
        count_tokens("generated text")


@pytest.mark.parametrize(("kind", "rule"), [("sliding_attention", "sliding"), ("chunked_attention", "chunked")])
@pytest.mark.parametrize("window", [1, 4, 5])
@pytest.mark.parametrize("position", [0, 1, 3, 4, 7, 9, 12])
@pytest.mark.parametrize("tokens", [1, 2, 4, 6, 11])
def test_windowed_keys_match_the_mask(kind: str, rule: str, window: int, position: int, tokens: int) -> None:
    shape = _shape(kind, window)
    visible = _visible(rule, window, position, tokens)
    assert shape.attended_keys(1, position, tokens) == int(visible.sum())
    assert shape.cached_keys(1, position + tokens, tokens) == int(visible.any(0).sum())


def test_sliding_prefill_reads_every_key_it_attends_to() -> None:
    shape, window = _shape("sliding_attention", 4), 4
    costs = CostModel(shape, weight_bytes=0)
    kv_width = 2 * shape.kv_heads * shape.head_dim * costs.kv_bytes
    prefill = costs.event(CostEvent("exact", 10, 0, 1, 1))
    head = shape.hidden * shape.vocab * costs.weight_bytes
    assert prefill.bytes - head == kv_width * (10 + 10)  # all ten keys are read by some query, ten are written
    step = costs.event(CostEvent("exact", 1, 20, 1, 1))
    assert step.bytes - head == kv_width * (window + 1)  # unchanged for single-token calls


def test_llama4_chunked_layers_are_priced() -> None:
    config = Llama4TextConfig(num_hidden_layers=4, hidden_size=64, num_attention_heads=4, num_key_value_heads=2,
                              head_dim=16, intermediate_size=32, intermediate_size_mlp=64, vocab_size=100,
                              num_local_experts=4, attention_chunk_size=8)
    shape = ModelShape.from_config(config)
    assert shape.chunk_size == 8
    assert shape.layer_types.count("chunked_attention") == 3
    chunked = [layer for layer in range(1, 5) if shape.layer_type(layer) == "chunked_attention"]
    assert all(shape.attended_keys(layer, 100, 1) == 100 % 8 + 1 for layer in chunked)
    assert all(shape.cached_keys(layer, 101) == 100 % 8 + 1 for layer in chunked)
    # A chunked layer reads its chunk, a full layer the whole context.
    costs = CostModel(shape)
    full = next(layer for layer in range(1, 5) if layer not in chunked)
    one = lambda layer: costs.event(CostEvent("draft", 1, 100, layer, layer))
    assert 0 < one(chunked[0]).flops < one(full).flops
    assert 0 < one(chunked[0]).bytes < one(full).bytes


# ---- bench ---------------------------------------------------------------------------------------------------------
def _moe_configs() -> list[Any]:
    from transformers import Llama4TextConfig, MixtralConfig, Qwen2MoeConfig, Qwen3MoeConfig

    common = dict(num_hidden_layers=4, hidden_size=64, num_attention_heads=4, num_key_value_heads=2, vocab_size=101)
    llama4 = dict(common, head_dim=16, intermediate_size=48, intermediate_size_mlp=96, num_local_experts=4,
                  num_experts_per_tok=1, attention_chunk_size=8)
    return [
        Llama4TextConfig(**llama4),                                   # every layer MoE, one shared expert
        Llama4TextConfig(**llama4, interleave_moe_layer_step=2),      # Maverick: dense layers of intermediate_size_mlp
        Qwen2MoeConfig(**common, intermediate_size=96, moe_intermediate_size=32, shared_expert_intermediate_size=48,
                       num_experts=4, num_experts_per_tok=2, decoder_sparse_step=2),
        Qwen3MoeConfig(**common, head_dim=16, intermediate_size=96, moe_intermediate_size=32, num_experts=4,
                       num_experts_per_tok=2),
        MixtralConfig(**common, intermediate_size=48, num_local_experts=4, num_experts_per_tok=2),
    ]


@pytest.mark.parametrize("index", range(5), ids=["llama4", "llama4-interleaved", "qwen2-moe", "qwen3-moe", "mixtral"])
def test_moe_shapes_count_the_weights_of_the_real_model(index: int) -> None:
    """``ModelShape.from_config`` counts the weight matrices (no norms or biases, embedding not counted, untied
    head counted) of mixture-of-experts models exactly: Llama-4's shared expert and its dense layers included."""
    from transformers import AutoModelForCausalLM

    from economize.bench import ModelShape

    config = _moe_configs()[index]
    model = AutoModelForCausalLM.from_config(config)
    real = sum(p.numel() for name, p in model.named_parameters() if p.dim() >= 2 and "embed_tokens" not in name)
    assert ModelShape.from_config(config).total_params == real
