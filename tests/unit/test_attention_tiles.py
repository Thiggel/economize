"""Launch shapes of the split-K decode kernels (``economize.attention.decode.DecodePlan``).

Every launch shape computes the same attention: row tiles of any height, value tiles (programs that recompute the
scores of their rows and accumulate one slice of the value dimension), more warps and pipelining stages. The tests
force each shape on the Gemma-4-31B sliding and global layers, a GQA geometry with a soft cap and a value head size
that is no power of two, over bf16, fp32 and fp8 K/V and over the gathered (compacted) cache, and compare with
:func:`~economize.attention.reference_attention` at the kernel tests' tolerances.
"""

from __future__ import annotations

import pytest
import torch

from economize.attention.decode import DecodePlan, decode_attention, plan, reference_attention
from economize.errors import ConfigError

pytestmark = [pytest.mark.gpu, pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")]

triton = pytest.importorskip("triton")


def _launch(
    batch: int, kv_heads: int, rows: int, keys: int, head_dim: int, block_m: int, block_n: int, dv_tiles: int,
    num_warps: int = 4, num_stages: int = 1, splits: int = 7,
) -> DecodePlan:
    block_d = max(16, triton.next_power_of_2(head_dim))
    blocks = -(-keys // block_n)
    split_len = -(-blocks // min(splits, blocks)) * block_n
    return DecodePlan(
        block_m, block_n, block_d, -(-rows // block_m), split_len, -(-keys // split_len), block_d // dv_tiles,
        dv_tiles, num_warps, num_stages,
    )


def _mask(batch: int, heads: int, queries: int, keys: int, window: int | None) -> torch.Tensor:
    pos = torch.arange(keys, device="cuda")
    end = keys - 300 - 5 * torch.arange(batch, device="cuda")  # rows at different positions
    qp = end[:, None] + torch.arange(queries, device="cuda")[None, :]
    visible = pos[None, None, :] <= qp[..., None]
    if window is not None:
        visible &= pos[None, None, :] > qp[..., None] - window
    visible = visible[:, None].expand(batch, heads, queries, keys).clone()
    if heads > 1:
        visible[:, ::3, :, 20:90] = False  # evicted slots of some heads
    return visible


GEOMETRIES = {  # q heads, kv heads, head dim, value head dim, window, softcap, mask heads
    "sliding": (32, 16, 256, 256, 600, None, 1),  # Gemma-4-31B sliding layer
    "global": (32, 4, 512, 512, None, None, 1),  # Gemma-4-31B global layer
    "gqa": (14, 2, 64, 48, None, 30.0, 14),  # Qwen-like, soft cap, per-head mask, value dim no power of two
}
LAUNCHES = [  # block_m, block_n, dv_tiles, num_warps, num_stages
    (16, 32, 1, 4, 1),
    (16, 64, 2, 4, 2),
    (32, 32, 2, 8, 2),
    (64, 32, 4, 8, 1),
    (64, 16, 1, 4, 2),
]


def _inputs(name: str, batch: int, queries: int, keys: int, dtype: torch.dtype, kv_dtype: torch.dtype | None = None):
    hq, hkv, d, dv, window, softcap, mask_heads = GEOMETRIES[name]
    gen = torch.Generator(device="cuda").manual_seed(0)
    q = torch.randn(batch, queries, hq, d, device="cuda", generator=gen).to(dtype).transpose(1, 2)  # HF strides
    k = torch.randn(batch, hkv, keys, d, device="cuda", generator=gen).to(kv_dtype or dtype)
    v = torch.randn(batch, hkv, keys, dv, device="cuda", generator=gen).to(kv_dtype or dtype)
    return q, k, v, _mask(batch, mask_heads, queries, keys, window), d**-0.5, softcap


def _skip_on_shared_memory(call):
    try:
        return call()
    except Exception as error:  # Turing has 64 KiB of shared memory, too little for some tiles at head dim 512
        if type(error).__name__ == "OutOfResources":
            pytest.skip(str(error))
        raise


@pytest.mark.parametrize("launch", LAUNCHES, ids=[f"m{c[0]}n{c[1]}dv{c[2]}w{c[3]}s{c[4]}" for c in LAUNCHES])
@pytest.mark.parametrize("name", list(GEOMETRIES))
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_every_launch_shape_matches_the_reference(name: str, launch: tuple, dtype: torch.dtype) -> None:
    batch, queries, keys = 3, 9, 1500
    q, k, v, mask, scale, softcap = _inputs(name, batch, queries, keys, dtype)
    hq, hkv, d, dv = GEOMETRIES[name][:4]
    p = _launch(batch, hkv, hq // hkv * queries, keys, max(d, dv), *launch)
    out = _skip_on_shared_memory(lambda: decode_attention(q, k, v, mask, scale, softcap=softcap, launch=p))
    ref = reference_attention(q, k, v, mask, scale, softcap=softcap)
    tol = 1e-5 if dtype == torch.float32 else 1.6e-2
    torch.testing.assert_close(out.float(), ref.float(), atol=tol, rtol=0)


@pytest.mark.skipif(not hasattr(torch, "float8_e4m3fn"), reason="needs fp8 dtypes")
@pytest.mark.parametrize("fp8", ["float8_e4m3fn", "float8_e5m2"])
@pytest.mark.parametrize("name", ["sliding", "global"])
def test_value_tiles_read_fp8_bytes(name: str, fp8: str) -> None:
    q, k, v, mask, scale, _ = _inputs(name, 2, 4, 1100, torch.bfloat16, getattr(torch, fp8))
    hq, hkv, d = GEOMETRIES[name][:3]
    ref = reference_attention(q, k.to(torch.bfloat16), v.to(torch.bfloat16), mask, scale)
    for launch in LAUNCHES[1:4]:
        p = _launch(2, hkv, hq // hkv * 4, 1100, d, *launch)
        out = _skip_on_shared_memory(lambda p=p: decode_attention(q, k, v, mask, scale, launch=p))
        torch.testing.assert_close(out.float(), ref.float(), atol=1.6e-2, rtol=0)


@pytest.mark.parametrize("sinks", [False, True])
def test_value_tiles_write_the_row_statistics_once(sinks: bool) -> None:
    """Rows that see no key stay zero and sinks join every row's denominator once, with any number of value tiles."""
    q, k, v, mask, scale, _ = _inputs("global", 2, 3, 400, torch.float32)
    mask[1, :, 2] = False  # a query that sees nothing
    s = torch.randn(32, device="cuda") if sinks else None
    ref = reference_attention(q, k, v, mask, scale, sinks=s)
    for dv_tiles in (1, 2, 4, 8):
        p = _launch(2, 4, 8 * 3, 400, 512, 32, 16, dv_tiles)
        out = _skip_on_shared_memory(lambda p=p: decode_attention(q, k, v, mask, scale, sinks=s, launch=p))
        torch.testing.assert_close(out, ref, atol=1e-5, rtol=0)
        if not sinks:
            assert out[1, 2].abs().max() == 0


@pytest.mark.parametrize("launch", LAUNCHES[1:4], ids=[f"m{c[0]}n{c[1]}dv{c[2]}" for c in LAUNCHES[1:4]])
@pytest.mark.parametrize("name", ["sliding", "global"])
def test_gathered_cache_takes_every_launch_shape(name: str, launch: tuple) -> None:
    from economize.attention.gathered import GatheredKV, gathered_decode_attention

    batch, queries, keys, prompt, kept = 2, 5, 900, 600, 200
    q, k, v, mask, scale, _ = _inputs(name, batch, queries, keys, torch.bfloat16)
    hq, hkv, d = GEOMETRIES[name][:3]
    gen = torch.Generator().manual_seed(1)
    index = torch.stack([
        torch.stack([torch.randperm(prompt, generator=gen)[:kept].sort().values for _ in range(hkv)])
        for _ in range(batch)
    ]).to(device="cuda", dtype=torch.int32)
    gather = index.long()[..., None]
    gathered = GatheredKV(
        key=k.gather(2, gather.expand(-1, -1, -1, d)).contiguous(),
        value=v.gather(2, gather.expand(-1, -1, -1, v.shape[-1])).contiguous(),
        index=index,
        count=torch.full((batch, hkv), kept, device="cuda", dtype=torch.int32),
        start=torch.full((batch,), prompt, device="cuda", dtype=torch.int32),
        dropped=prompt - kept,
    )
    keep = torch.zeros(batch, hkv, keys, dtype=torch.bool, device="cuda")
    keep[:, :, prompt:] = True
    keep.scatter_(2, index.long(), True)
    dense_mask = mask & keep.repeat_interleave(hq // hkv, dim=1)[:, :, None]
    ref = reference_attention(q, k, v, dense_mask, scale)
    p = _launch(batch, hkv, hq // hkv * queries, keys - gathered.dropped, d, *launch)
    k_nan, v_nan = k.clone(), v.clone()
    k_nan[:, :, :prompt], v_nan[:, :, :prompt] = float("nan"), float("nan")  # read only through the compact entries
    out = _skip_on_shared_memory(lambda: gathered_decode_attention(q, k_nan, v_nan, mask, scale, gathered, launch=p))
    assert torch.isfinite(out).all()
    torch.testing.assert_close(out.float(), ref.float(), atol=1.6e-2, rtol=0)


def test_default_plan_is_a_function_of_the_shapes() -> None:
    a = plan(4, 4, 72, 4352, 512, torch.device("cuda"))
    assert a == plan(4, 4, 72, 4352, 512, torch.device("cuda"))
    assert a.dv_tiles * (a.block_dv or a.block_d) == a.block_d
    assert a.split_len % a.block_n == 0 and a.splits * a.split_len >= 4352


@pytest.mark.parametrize("batch", [1, 4, 16, 64])
@pytest.mark.parametrize("queries", [1, 4, 9, 16])
@pytest.mark.parametrize("head", [(512, 512), (576, 512), (256, 256), (128, 128), (64, 64)])
def test_tuned_plans_cover_the_value_width(batch: int, queries: int, head: tuple) -> None:
    d, dv = head
    dev = torch.device("cuda")
    block_dv = max(16, 1 << (dv - 1).bit_length())
    for keys in (1, 1000, 8192):
        p = plan(batch, 4, 8 * queries, keys, max(d, dv), dev, value_dim=dv)
        assert p.dv_tiles * (p.block_dv or block_dv) == block_dv
        assert p.split_len % p.block_n == 0 and p.splits * p.split_len >= keys > (p.splits - 1) * p.split_len
        assert p.m_tiles * p.block_m >= 8 * queries
        if max(d, dv) not in range(257, 513):  # other heads keep the original tiles and only launch fewer splits
            legacy = plan(batch, 4, 8 * queries, keys, max(d, dv), dev, value_dim=dv, tuned=False)
            assert (p.block_m, p.block_n, p.block_d, p.dv_tiles) == (legacy.block_m, legacy.block_n, legacy.block_d, 1)
            assert p.splits <= legacy.splits


def test_untuned_plan_is_the_original_rule() -> None:
    p = plan(4, 4, 72, 4352, 512, torch.device("cuda"), tuned=False)
    assert (p.block_m, p.block_n, p.block_dv, p.dv_tiles, p.num_warps) == (16, 16, 0, 1, 0)
    assert plan(4, 4, 72, 4352, 512, torch.device("cuda"), element_size=4).dv_tiles == 1  # fp32 keeps the old tiles


@pytest.mark.parametrize(
    "batch,queries", [(1, 1), (1, 4), (4, 1), (4, 9), (4, 16), (16, 4), (16, 16), (64, 1), (64, 4)]
)
@pytest.mark.parametrize("fp8", [None, "float8_e4m3fn"])
def test_tuned_wide_plans_match_the_reference(batch: int, queries: int, fp8: str | None) -> None:
    """Every regime of the 512-wide dispatch (value tiles, 32-row and 16-key tiles, one to four waves) on a global
    layer."""
    torch.manual_seed(batch * 100 + queries)
    hq, hkv, d, keys = 32, 4, 512, 700
    q = torch.randn(batch, hq, queries, d, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(batch, hkv, keys, d, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(batch, hkv, keys, d, device="cuda", dtype=torch.bfloat16)
    if fp8 is not None:
        k, v = k.to(getattr(torch, fp8)), v.to(getattr(torch, fp8))
    lengths = torch.randint(keys // 2, keys + 1, (batch,), device="cuda")
    pos = torch.arange(keys, device="cuda")
    last = lengths[:, None, None] - queries + torch.arange(queries, device="cuda")[None, :, None]
    causal = pos[None, None, :] <= last
    mask = causal[:, None]
    scale = d**-0.5
    out = _skip_on_shared_memory(lambda: decode_attention(q, k, v, mask, scale))
    ref = reference_attention(q, k.to(torch.bfloat16), v.to(torch.bfloat16), mask, scale)
    torch.testing.assert_close(out.float(), ref.float(), atol=1.6e-2, rtol=0)


def test_visible_keys_size_the_splits_of_a_long_buffer() -> None:
    """A window over a long buffer gets the split length of a buffer holding only the window, over all its keys."""
    dev = torch.device("cuda")
    for d in (256, 512, 64):
        short = plan(1, 16, 2, 1025, d, dev)
        long = plan(1, 16, 2, 8192, d, dev, visible=1025)
        assert long.split_len == short.split_len and long.splits == -(-8192 // short.split_len)
        assert plan(1, 16, 2, 8192, d, dev, visible=1025, tuned=False) == plan(1, 16, 2, 8192, d, dev, tuned=False)
        assert plan(1, 16, 2, 800, d, dev, visible=1025) == plan(1, 16, 2, 800, d, dev)


def test_a_batch_that_fills_the_gpu_reads_the_window_with_one_split() -> None:
    """With more programs than SMs the hidden part of a long buffer gets one split of its own, a short one none."""
    dev = torch.device("cuda")
    for d in (256, 512, 64):
        long = plan(64, 16, 2, 8192, d, dev, visible=1025)
        assert (long.splits, long.split_len) == (2, 4096)
        short = plan(64, 16, 2, 2048, d, dev, visible=1025)
        assert (short.splits, short.split_len) == (1, 2048)


@pytest.mark.parametrize("batch,queries", [(1, 1), (4, 4), (16, 9), (64, 1)])
def test_window_sized_splits_match_the_reference(batch: int, queries: int) -> None:
    q, k, v, mask, scale, _ = _inputs("sliding", batch, queries, 5000, torch.bfloat16)
    out = decode_attention(q, k, v, mask, scale, window=GEOMETRIES["sliding"][4])
    ref = reference_attention(q, k, v, mask, scale)
    torch.testing.assert_close(out.float(), ref.float(), atol=1.6e-2, rtol=0)


def test_value_tiles_that_miss_the_value_width_are_refused() -> None:
    q, k, v, mask, scale, _ = _inputs("global", 1, 1, 64, torch.bfloat16)
    p = plan(1, k.shape[1], q.shape[1] // k.shape[1], 64, 512, q.device, tuned=False)
    bad = DecodePlan(p.block_m, p.block_n, p.block_d, p.m_tiles, p.split_len, p.splits, 128, 2)
    with pytest.raises(ConfigError, match="value tiles"):
        decode_attention(q, k, v, mask, scale, launch=bad)
