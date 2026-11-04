"""FlashAttention for short queries over a declared prefix mask (``economize.attention.flash.flash_decode``).

The declaration (:func:`~economize.attention.flash.declare_prefix`) and which contexts attach it run anywhere. The
kernel tests need vLLM's FlashAttention build on Hopper or Blackwell (``-m gpu``): FA3 (head dim <= 256) and FA4
(head dim 512) over the static buffer are compared with ``sdpa`` and with the split-K kernel on the same boolean mask
for causal and sliding-window layers, GQA, soft capping, several rows at different offsets, rows that end at the
last real slot of the buffer, replay from a CUDA graph, and a decoder whose verification passes run on them.
"""

from __future__ import annotations

import pytest
import torch

from economize.attention import flash, interface
from economize.attention.decode import decode_attention
from economize.attention.flash import _decode_version, declare_prefix, flash_decode, prefix_hint
from economize.runtime import context as context_module


def _mask(queries: int, keys: int, starts: list[int], window: int | None, device: str = "cpu") -> torch.Tensor:
    slot = torch.arange(keys, device=device)
    position = torch.tensor(starts, device=device)[:, None] + torch.arange(queries, device=device)
    visible = slot <= position[..., None]
    if window is not None:
        visible &= slot > position[..., None] - window
    return visible[:, None]


# ---- the declaration (no kernels) --------------------------------------------------------------------------------


def test_declare_prefix_builds_the_varlen_arguments() -> None:
    mask = _mask(5, 64, [0, 30, 59], None)
    assert declare_prefix(mask, torch.tensor([0, 30, 59]), None) is mask
    hint = prefix_hint(mask)
    assert hint is not None and (hint.queries, hint.keys, hint.window) == (5, 64, None)
    assert hint.used.tolist() == [5, 35, 64] and hint.used.dtype == torch.int32
    assert hint.table.tolist() == [[0], [1], [2]]
    assert hint.cu_queries.tolist() == [0, 5, 10, 15]
    assert prefix_hint(mask & mask) is None  # a modified mask is a new tensor without the declaration
    assert prefix_hint(None) is None


def test_refuses_without_cuda() -> None:
    q = torch.randn(1, 4, 3, 16)
    mask = declare_prefix(_mask(3, 40, [10], None), torch.tensor([10]), None)
    out, reason = flash_decode(q, q[:, :2], q[:, :2], prefix_hint(mask), 0.25, force=True)
    assert out is None and reason == "cpu"


def test_small_batches_never_use_flash(monkeypatch: pytest.MonkeyPatch) -> None:
    """Below FLASH_MIN_BATCH cache rows the split-K kernel always runs and no hint is built, whatever the shape."""
    for version, dim in ((3, 256), (4, 512)):
        for batch in range(1, flash.FLASH_MIN_BATCH):
            assert not flash.decode_prefers_flash(version, dim, None, batch, 64, 8192)
            assert not flash.decode_hints_wanted(torch.device("cuda", 0), batch)
    assert flash.decode_prefers_flash(3, 256, None, 16, 9, 4352)
    assert flash.decode_prefers_flash(3, 128, None, 4, 1, 1536)
    assert not flash.decode_prefers_flash(3, 64, None, 4, 9, 1536)
    assert flash.decode_prefers_flash(3, 64, None, 8, 9, 1536)
    assert not flash.decode_prefers_flash(4, 512, None, 8, 4, 8192)
    assert flash.decode_prefers_flash(4, 512, None, 4, 9, 1536)
    assert flash.decode_prefers_flash(4, 512, None, 16, 1, 1536)
    monkeypatch.setattr(flash, "FLASH_DECODE", "1")  # forcing builds the hint at any batch size (tests)
    monkeypatch.setattr(flash, "_hints_wanted", lambda index: True)
    assert flash.decode_hints_wanted(torch.device("cuda", 0), 1)


def test_only_consecutive_chain_contexts_declare_their_masks(monkeypatch: pytest.MonkeyPatch) -> None:
    from economize.arch import adapter_for
    from economize.runtime import LayerRunner
    from helpers import make_tiny_model

    interface.install()
    monkeypatch.setattr(context_module, "decode_hints_wanted", lambda device, batch: True)
    model = make_tiny_model("economize")
    runner = LayerRunner(adapter_for(model))
    cache = runner.new_cache(32, batch_size=2)
    spec = runner.adapter.attention_spec(0)
    positions = torch.tensor([[3, 4, 5], [10, 11, 12]])
    chain = runner.context(cache, positions, trusted=True, consecutive=True)
    hint = prefix_hint(chain.mask(spec, torch.device("cpu")))
    assert hint is not None and hint.used.tolist() == [6, 13]
    assert chain.layout.consecutive
    assert prefix_hint(runner.context(cache, positions, trusted=True).mask(spec, torch.device("cpu"))) is None
    single = runner.context(cache, positions[:, :1], trusted=True)  # one query per row is consecutive by itself
    assert prefix_hint(single.mask(spec, torch.device("cpu"))) is not None
    picked = chain.select(torch.tensor([[0, 2], [1, 2]]), torch.ones(2, 2, dtype=torch.bool))
    assert prefix_hint(picked.mask(spec, torch.device("cpu"))) is None
    model.set_attn_implementation("sdpa")
    sdpa = runner.context(cache, positions, trusted=True, consecutive=True)
    assert prefix_hint(sdpa.mask(spec, torch.device("cpu"))) is None  # only the "economize" kernel reads it


# ---- kernels ------------------------------------------------------------------------------------------------------


def _version(dim: int) -> int | None:
    if not torch.cuda.is_available():
        return None
    return _decode_version(torch.device("cuda", torch.cuda.current_device()), dim)


requires_fa3 = pytest.mark.skipif(_version(256) != 3, reason="needs FA3 (vLLM's FlashAttention build on Hopper)")
requires_fa4 = pytest.mark.skipif(_version(512) != 4, reason="needs FA4 (vLLM's FlashAttention build, sm90/sm100)")


def _reference(query, key, value, mask, scale, softcap=None):
    from economize.attention import reference_attention

    return reference_attention(query.float(), key.float(), value.float(), mask, scale, softcap=softcap)


def _inputs(batch: int, q_heads: int, kv_heads: int, queries: int, keys: int, dim: int, seed: int = 0):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    shape = (batch, kv_heads, keys, dim)
    query = torch.randn(batch, q_heads, queries, dim, device="cuda", dtype=torch.bfloat16, generator=gen)
    key = torch.randn(*shape, device="cuda", dtype=torch.bfloat16, generator=gen)
    value = torch.randn(*shape, device="cuda", dtype=torch.bfloat16, generator=gen)
    return query, key, value


CASES = [  # dim, q heads, kv heads, window, softcap, queries, starts
    pytest.param(128, 8, 2, None, None, 1, [700], marks=requires_fa3, id="d128-q1"),
    pytest.param(128, 8, 8, None, None, 9, [0, 333, 1000, 1142], marks=requires_fa3, id="d128-mha-q9-end"),
    pytest.param(64, 8, 2, None, None, 5, [17, 900], marks=requires_fa3, id="d64-q5"),
    pytest.param(256, 16, 8, 300, None, 9, [40, 299, 1100, 1142], marks=requires_fa3, id="d256-window-q9"),
    pytest.param(256, 16, 8, 300, None, 1, [5, 1150], marks=requires_fa3, id="d256-window-q1"),
    pytest.param(256, 8, 2, None, 30.0, 4, [100, 0], marks=requires_fa3, id="d256-softcap-q4"),
    pytest.param(512, 16, 2, None, None, 1, [1000], marks=requires_fa4, id="d512-q1"),
    pytest.param(512, 16, 2, None, None, 9, [0, 512, 1142, 77], marks=requires_fa4, id="d512-q9-end"),
    pytest.param(512, 32, 4, None, None, 16, [64, 900], marks=requires_fa4, id="d512-q16"),
]
KEYS = 1153  # a graph cache's capacity (a multiple of 128) plus the trash slot: never a multiple of 256


@pytest.mark.gpu
@pytest.mark.parametrize(("dim", "q_heads", "kv_heads", "window", "softcap", "queries", "starts"), CASES)
def test_matches_sdpa_and_the_split_kernel(dim, q_heads, kv_heads, window, softcap, queries, starts) -> None:
    batch = len(starts)
    query, key, value = _inputs(batch, q_heads, kv_heads, queries, KEYS, dim)
    mask = declare_prefix(_mask(queries, KEYS, starts, window, "cuda"), torch.tensor(starts, device="cuda"), window)
    scale = dim**-0.5
    out, path = flash_decode(query, key, value, prefix_hint(mask), scale, softcap, force=True)
    assert out is not None, path
    assert path == f"flash{_version(dim)}" and out.shape == (batch, queries, q_heads, dim)
    expected = _reference(query, key, value, mask, scale, softcap)
    torch.testing.assert_close(out.float(), expected, atol=2e-2, rtol=2e-2)
    split = decode_attention(query, key, value, mask, scale, softcap=softcap, window=window)
    torch.testing.assert_close(out.float(), split.float(), atol=2e-2, rtol=2e-2)


@pytest.mark.gpu
@requires_fa3
def test_refuses_what_it_cannot_express() -> None:
    query, key, value = _inputs(2, 8, 2, 4, KEYS, 128)
    mask = declare_prefix(_mask(4, KEYS, [5, 9], None, "cuda"), torch.tensor([5, 9], device="cuda"), None)
    hint = prefix_hint(mask)
    assert flash_decode(query.float(), key.float(), value.float(), hint, 0.1, force=True)[1] == "flash_dtype"
    assert flash_decode(query, key.to(torch.float8_e4m3fn), value, hint, 0.1, force=True)[1] == "flash_dtype"
    assert flash_decode(query[:, :, :3], key, value, hint, 0.1, force=True)[1] == "flash_hint"
    assert flash_decode(query, key[..., :-1, :], value[..., :-1, :], hint, 0.1, force=True)[1] == "flash_hint"
    assert flash_decode(query[:1], key[:1], value[:1], hint, 0.1, force=True)[1] == "flash_hint"


@pytest.mark.gpu
@requires_fa3
@pytest.mark.parametrize("dim", [128, pytest.param(512, marks=requires_fa4)])
def test_replays_from_a_cuda_graph(dim: int) -> None:
    """The declaration is built inside the capture, so a replay at new positions reads the new ``seqused_k``."""
    queries, window = 5, None
    query, key, value = _inputs(2, 8, 2, queries, KEYS, dim)
    first = torch.tensor([3, 40], device="cuda")
    scale = dim**-0.5

    def step() -> torch.Tensor:
        mask = _mask_like(first, queries, window)
        hint = prefix_hint(declare_prefix(mask, first, window))
        out, path = flash_decode(query, key, value, hint, scale, force=True)
        assert out is not None, path
        return out

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(2):
            step()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        static = step()
    for starts in ([3, 40], [700, 11], [1147, 0]):
        first.copy_(torch.tensor(starts, device="cuda"))
        graph.replay()
        mask = _mask(queries, KEYS, starts, window, "cuda")
        torch.testing.assert_close(static.float(), _reference(query, key, value, mask, scale), atol=2e-2, rtol=2e-2)


def _mask_like(first: torch.Tensor, queries: int, window: int | None) -> torch.Tensor:
    slot = torch.arange(KEYS, device=first.device)
    position = first[:, None] + torch.arange(queries, device=first.device)
    visible = slot <= position[..., None]
    if window is not None:
        visible &= slot > position[..., None] - window
    return visible[:, None]


@pytest.mark.gpu
@requires_fa3
def test_forward_takes_flash_only_for_declared_masks(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(flash, "FLASH_DECODE", "1")
    query, key, value = _inputs(2, 8, 2, 4, KEYS, 128, seed=2)
    module = torch.nn.Module()
    module.layer_idx = 1
    plain = _mask(4, KEYS, [5, 300], None, "cuda")
    declared = declare_prefix(plain.clone(), torch.tensor([5, 300], device="cuda"), None)
    interface.kernel_usage(reset=True)
    out, _ = interface.economize_attention_forward(module, query, key, value, declared, scaling=0.1)
    ref, _ = interface.economize_attention_forward(module, query, key, value, plain, scaling=0.1)
    paths = {r["path"] for r in interface.kernel_usage(reset=True)}
    assert paths == {"flash3", "kernel"}
    torch.testing.assert_close(out.float(), ref.float(), atol=2e-2, rtol=2e-2)
    monkeypatch.setattr(flash, "FLASH_DECODE", "0")
    interface.economize_attention_forward(module, query, key, value, declared, scaling=0.1)
    assert {r["path"] for r in interface.kernel_usage(reset=True)} == {"kernel"}


def _bf16_model():
    """A random 4-layer Qwen2 at head dim 64 in bf16 (FlashAttention needs a 16-bit dtype)."""
    from transformers import Qwen2Config, Qwen2ForCausalLM

    from helpers import VOCAB

    config = Qwen2Config(
        vocab_size=VOCAB, hidden_size=256, intermediate_size=512, num_hidden_layers=4, num_attention_heads=4,
        num_key_value_heads=2, max_position_embeddings=4096, tie_word_embeddings=True, initializer_range=0.05,
        attn_implementation="economize",
    )  # fmt: skip
    torch.manual_seed(0)
    model = Qwen2ForCausalLM(config).to("cuda", torch.bfloat16).eval()
    model.requires_grad_(False)
    model.generation_config.eos_token_id = None
    model.generation_config.pad_token_id = 0
    return model


@pytest.mark.gpu
@requires_fa3
@pytest.mark.parametrize("graphs", [False, True])
@torch.no_grad()
def test_decoder_verifies_on_flash(graphs: bool, monkeypatch: pytest.MonkeyPatch) -> None:
    """Self-speculative decoding with every verification pass on FA3: each emitted token is the verifier's greedy
    choice up to bf16 rounding (checked against one full-recompute ``sdpa`` pass over the output)."""
    import economize
    from economize import DecodeConfig, JumpConfig
    from helpers import VOCAB, perturb

    interface.install()
    monkeypatch.setattr(flash, "FLASH_DECODE", "1")
    model = economize.apply(_bf16_model(), JumpConfig(target_layer=1, source_layer=2))
    perturb(model.techniques["jump"], 0.02)
    gen = torch.Generator().manual_seed(5)
    prompts = [torch.randint(3, VOCAB, (n,), generator=gen).cuda() for n in (21, 6, 13)]
    interface.kernel_usage(reset=True)
    decoding = DecodeConfig(cycle_length=4, cuda_graphs=graphs)
    for _ in range(2 if graphs else 1):  # capture, then replay
        results = model.decode_batch(prompts, 24, decoding=decoding)
    paths = {r["path"] for r in interface.kernel_usage(reset=True)}
    assert "flash3" in paths, paths
    reference = _bf16_model()
    reference.set_attn_implementation("sdpa")
    for prompt, result in zip(prompts, results):
        seq = torch.cat([prompt, result.tokens.to(prompt.device)])[None]
        logits = reference(seq).logits[0, prompt.numel() - 1 : -1].float()
        chosen = logits.gather(1, result.tokens.to(logits.device)[:, None])[:, 0]
        gap = logits.max(-1).values - chosen  # 0 where the token is the argmax
        assert result.tokens.numel() == 24
        assert float(gap.max()) < 0.1 * float(logits.std()), gap


#: A Gemma-4 laid out like gemma-4-E2B's first ten layers (window 512, head dims 256 and 512, the last four layers
#: reading shared K/V, proportional partial RoPE on the global layers), at a tiny width.
_LONG_WINDOW = dict(
    num_hidden_layers=10, layer_types=(["sliding_attention"] * 4 + ["full_attention"]) * 2, sliding_window=512,
    num_kv_shared_layers=4, hidden_size=128, intermediate_size=192, num_attention_heads=8, num_key_value_heads=1,
    head_dim=256, global_head_dim=512, num_global_key_value_heads=1, max_position_embeddings=4096,
    final_logit_softcapping=30.0, initializer_range=0.05,
    rope_parameters={
        "full_attention": {"partial_rotary_factor": 0.25, "rope_theta": 1e6, "rope_type": "proportional"},
        "sliding_attention": {"rope_theta": 1e4, "rope_type": "default"},
    },
)  # fmt: skip
PROMPT, CHUNK_SIZES = 654, (1, 9, 1, 1, 9, 4)  # a prompt longer than the window, then verify-sized passes


def _long_window_model(dtype: torch.dtype):
    import transformers

    from helpers import _GEMMA4

    cls = transformers.Gemma4ForCausalLM
    torch.manual_seed(0)
    model = cls._from_config(cls.config_class(**{**_GEMMA4, **_LONG_WINDOW}), attn_implementation="economize")
    return model.to("cuda", dtype).eval().requires_grad_(False)


@torch.no_grad()
def _chunked_logits(model, tokens: torch.Tensor) -> torch.Tensor:
    """Logits of one row from cached passes: a prefill of ``PROMPT`` tokens, then consecutive chunks."""
    from economize.arch import adapter_for
    from economize.runtime import LayerRunner

    runner = LayerRunner(adapter_for(model))
    cache = runner.new_cache(1024)
    rows, done = [], 0
    for size in (PROMPT, *CHUNK_SIZES):
        positions = torch.arange(done, done + size, device="cuda")[None]
        ctx = runner.context(cache, positions, trusted=True, consecutive=True)
        rows.append(runner.logits(runner.forward(tokens[None, done : done + size], ctx).final)[0].float())
        done += size
    return torch.cat(rows)


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16], ids=["fp32", "bf16"])
@torch.no_grad()
def test_window_shorter_than_the_prompt(dtype: torch.dtype, monkeypatch: pytest.MonkeyPatch) -> None:
    """Cached passes after a prompt longer than the sliding window match a full ``sdpa`` recompute.

    In fp32 the logits must agree to reduction noise, which this random 10-layer model amplifies to about 4e-3 of
    their spread with or without a window (a key outside the window, or one missing inside it, moves them by
    their whole spread). bf16 runs the FlashAttention decode path where
    vLLM's build is present (else the split kernel) and must stay as close to the fp32 recompute as the split kernel
    in bf16 does.
    """
    from helpers import VOCAB

    interface.install()
    total = PROMPT + sum(CHUNK_SIZES)
    tokens = torch.randint(3, VOCAB, (total,), generator=torch.Generator().manual_seed(1)).cuda()
    model = _long_window_model(dtype)
    if dtype == torch.float32:
        got = _chunked_logits(model, tokens)
        model.set_attn_implementation("sdpa")
        want = model(tokens[None]).logits[0].float()
        torch.testing.assert_close(got, want, atol=2e-2 * float(want.std()), rtol=0)
        return
    flash_on = _version(256) == 3
    monkeypatch.setattr(flash, "FLASH_DECODE", "1" if flash_on else "0")
    interface.kernel_usage(reset=True)
    got = _chunked_logits(model, tokens)
    paths = {r["path"] for r in interface.kernel_usage(reset=True)}
    assert ("flash3" in paths) == flash_on, paths
    monkeypatch.setattr(flash, "FLASH_DECODE", "0")
    split = _chunked_logits(model, tokens)
    model.float().set_attn_implementation("sdpa")
    want = model(tokens[None]).logits[0]
    decode = slice(PROMPT, None)  # the cached passes; the prefill is the same pass on both paths
    err, split_err = ((x[decode] - want[decode]).abs().mean() for x in (got, split))
    assert float(err) <= 1.5 * float(split_err) + 1e-3 * float(want.std()), (float(err), float(split_err))
