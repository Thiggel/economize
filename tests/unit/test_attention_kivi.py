"""The KIVI decode kernel (``economize.attention.kivi``).

The kernel reads the oldest keys and values of every row as packed codes and the rest from the dense buffers. The
tests fill the quantized slots of the dense buffers with NaN, so any read of them would show, and compare with
:func:`economize.attention.decode_attention` over the correctly dequantized dense buffers: the two kernels share
their arithmetic, so outputs must be bit-identical (which also shows that the in-register dequantization equals
``torch``'s). They also agree with :func:`~economize.attention.reference_attention`. Kernel tests need ``-m gpu``.
"""

from __future__ import annotations

import pytest
import torch

from economize.attention import KIVIPacked, decode_attention, kivi_decode_attention, reference_attention
from economize.attention.kivi import dequantize, kivi_plan, pack_codes, unpack_codes
from economize.techniques.kv_quant import quantize


def _dense(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, mask, scale: float, **kw) -> torch.Tensor:
    """The dense kernel under the KIVI kernel's launch plan (bit-equal outputs)."""
    batch, q_heads, queries, d = query.shape
    kv_heads, keys, dv = key.shape[1], key.shape[2], value.shape[-1]
    p = kivi_plan(batch, kv_heads, q_heads // kv_heads * queries, keys, max(d, dv), query.device,
                  query.element_size())
    return decode_attention(query, key, value, mask, scale, launch=p, **kw)

pytestmark = [pytest.mark.gpu, pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")]


def test_pack_round_trip() -> None:
    for bits in (2, 4, 8):
        codes = torch.randint(0, 2**bits, (3, 2, 5, 64))
        assert torch.equal(unpack_codes(pack_codes(codes, bits), bits), codes)


def _packed(
    key: torch.Tensor, value: torch.Tensor, bits: int, group: int, kq: list[int], vq: list[int]
) -> tuple[KIVIPacked, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Packed codes of ``key`` / ``value``, the dense buffers with the packed slots dequantized, and the same
    buffers with NaN in the packed slots."""
    b, h, s, d = key.shape
    dv = value.shape[3]
    vg = group if dv % group == 0 else dv
    full = s // group * group
    codes, scale, low = quantize(key[:, :, :full].reshape(b, h, full // group, group, d), bits, dim=3)
    key_codes = torch.zeros(b, h, s, d // (8 // bits), dtype=torch.uint8, device=key.device)
    key_codes[:, :, :full] = pack_codes(codes.reshape(b, h, full, d), bits)
    key_scale = torch.zeros(b, h, -(-s // group), d, dtype=torch.float16, device=key.device)
    key_zero = torch.zeros_like(key_scale)
    key_scale[:, :, : full // group] = scale.squeeze(3).half()
    key_zero[:, :, : full // group] = low.squeeze(3).half()
    vcodes, vscale, vlow = quantize(value.reshape(b, h, s, dv // vg, vg), bits, dim=4)
    packed = KIVIPacked(
        bits=bits,
        key_group=group,
        value_group=vg,
        key_codes=key_codes,
        key_scale=key_scale,
        key_zero=key_zero,
        value_codes=pack_codes(vcodes.reshape(b, h, s, dv), bits),
        value_scale=vscale.squeeze(4).half().contiguous(),
        value_zero=vlow.squeeze(4).half().contiguous(),
        keys_quantized=torch.tensor(kq, dtype=torch.int32, device=key.device),
        values_quantized=torch.tensor(vq, dtype=torch.int32, device=key.device),
    )
    k_deq = dequantize(packed.key_codes, key_scale, key_zero, bits, "key", group, key.dtype)
    v_deq = dequantize(packed.value_codes, packed.value_scale, packed.value_zero, bits, "value", vg, value.dtype)
    slots = torch.arange(s, device=key.device)
    k_in = slots[None, :] < packed.keys_quantized[:, None]
    v_in = slots[None, :] < packed.values_quantized[:, None]
    key_ok = torch.where(k_in[:, None, :, None], k_deq, key)
    value_ok = torch.where(v_in[:, None, :, None], v_deq, value)
    key_nan = key_ok.masked_fill(k_in[:, None, :, None], float("nan"))
    value_nan = value_ok.masked_fill(v_in[:, None, :, None], float("nan"))
    return packed, key_ok, value_ok, key_nan, value_nan


def _mask(batch: int, heads: int, queries: int, keys: int, filled: list[int], tree: bool) -> torch.Tensor:
    pos = torch.arange(keys)
    qp = torch.tensor(filled)[:, None] + torch.arange(queries)[None, :]
    visible = pos[None, None, :] <= qp[..., None]
    if tree:  # a draft tree: query i sees the committed prefix and its ancestors only
        start = torch.tensor(filled)[:, None, None]
        drafted = (pos[None, None, :] >= start) & visible
        parent_ok = (pos[None, None, :] - start) % 2 == (torch.arange(queries)[None, :, None] % 2)
        visible &= ~drafted | parent_ok | (pos[None, None, :] == qp[..., None])
    visible = visible[:, None].expand(batch, heads, queries, keys).clone()
    if heads > 1:
        visible[:, ::2, :, 10:40] = False  # per-head eviction
    return visible.cuda()


CASES = [
    # bits, dtype, queries, mask kind, softcap, sinks, head dim, value head dim
    (2, torch.float32, 1, None, None, False, 64, 64),
    (2, torch.bfloat16, 1, "chain", None, False, 128, 128),
    (4, torch.bfloat16, 5, "chain", None, False, 128, 128),
    (2, torch.bfloat16, 6, "tree", 30.0, False, 64, 64),
    (8, torch.float16, 3, "per_head", None, True, 64, 64),
    (4, torch.float32, 4, "tree", None, True, 128, 128),
    (2, torch.bfloat16, 2, "chain", None, False, 64, 48),  # value groups of the whole head (48 % 32 != 0)
]


@pytest.mark.parametrize("case", CASES, ids=[f"b{c[0]}-{str(c[1])[6:]}-q{c[2]}-{c[3]}" for c in CASES])
def test_kernel_reads_codes_and_matches_dense(case: tuple) -> None:
    _check_against_dense(*case, group=32)


GROUPS = [
    # bits, dtype, head dim, value head dim, group: groups that split a key block (16), that hold one (128), that
    # neither divide nor are divided by it and do not tile the value row (24, the per-element loads)
    (2, torch.bfloat16, 128, 128, 16),
    (4, torch.bfloat16, 64, 64, 128),
    (2, torch.float16, 48, 48, 24),
    (8, torch.float32, 64, 64, 16),
]


@pytest.mark.parametrize("case", GROUPS, ids=[f"b{c[0]}-{str(c[1])[6:]}-d{c[2]}-g{c[4]}" for c in GROUPS])
def test_kernel_matches_dense_at_any_group_size(case: tuple) -> None:
    """The broadcast loads (whole key groups per block, value groups that tile the row) and the per-element
    loads of other group sizes."""
    bits, dtype, d, dv, group = case
    _check_against_dense(bits, dtype, 3, "chain", None, False, d, dv, group=group)


def _check_against_dense(
    bits: int, dtype: torch.dtype, queries: int, mask_kind: str | None, softcap: float | None, use_sinks: bool,
    d: int, dv: int, group: int,
) -> None:
    torch.manual_seed(0)
    batch, q_heads, kv_heads, keys = 3, 8, 2, 300
    query = torch.randn(batch, q_heads, queries, d, device="cuda", dtype=dtype)
    key = (torch.randn(batch, kv_heads, keys, d, device="cuda") * 2).to(dtype)
    key[..., 3] += 20  # an outlier channel, the reason keys are quantized per channel
    value = torch.randn(batch, kv_heads, keys, dv, device="cuda").to(dtype)
    filled = [260, 140, 30]
    kq = [256, 128, 0]
    vq = [260 - 128 + 1, 12, 0]
    packed, key_ok, value_ok, key_nan, value_nan = _packed(key, value, bits, group, kq, vq)
    mask = None
    if mask_kind is not None:
        heads = q_heads if mask_kind == "per_head" else 1
        mask = _mask(batch, heads, queries, keys, filled, mask_kind == "tree")
    sinks = torch.randn(q_heads, device="cuda") if use_sinks else None
    scale = d**-0.5
    got = kivi_decode_attention(query, key_nan, value_nan, mask, scale, packed, softcap=softcap, sinks=sinks)
    dense = _dense(query, key_ok, value_ok, mask, scale, softcap=softcap, sinks=sinks)
    assert torch.isfinite(got).all()
    assert torch.equal(got, dense)
    ref = reference_attention(query, key_ok, value_ok, mask, scale, softcap=softcap, sinks=sinks)
    tol = 1e-5 if dtype == torch.float32 else 1.6e-2
    torch.testing.assert_close(got.float(), ref.float(), atol=tol, rtol=tol)


def test_kernel_matches_a_cache_without_codes() -> None:
    """Rows with nothing quantized read the dense buffers only (the first pass of a generation)."""
    torch.manual_seed(1)
    query = torch.randn(2, 4, 1, 64, device="cuda", dtype=torch.bfloat16)
    key = torch.randn(2, 2, 96, 64, device="cuda", dtype=torch.bfloat16)
    value = torch.randn(2, 2, 96, 64, device="cuda", dtype=torch.bfloat16)
    packed, *_ = _packed(key, value, 2, 32, [0, 0], [0, 0])
    got = kivi_decode_attention(query, key, value, None, 0.125, packed)
    assert torch.equal(got, _dense(query, key, value, None, 0.125))
