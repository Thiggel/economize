"""FlashAttention for long queries (``economize.attention.flash``).

The mask check runs anywhere. The kernel tests need vLLM's FlashAttention build and an Ampere or newer GPU
(``-m gpu``): FlashAttention over the static buffer is compared with ``sdpa`` on the same boolean mask for causal and
sliding-window prefills, batched rows at different offsets, GQA, soft capping and head dims up to 256, and every
mask it cannot express (padding, per-head masks, trees) must fall back to ``sdpa`` per call.
"""

from __future__ import annotations

import threading

import pytest
import torch

from economize.attention import flash, interface
from economize.attention.flash import causal_offsets, flash_prefill, flash_version


def _mask(batch: int, queries: int, keys: int, starts: list[int], window: int | None) -> torch.Tensor:
    slot = torch.arange(keys)
    position = torch.tensor(starts)[:, None] + torch.arange(queries)
    visible = slot <= position[..., None]
    if window is not None:
        visible &= slot > position[..., None] - window
    return visible[:, None]


# ---- mask check (no kernels) --------------------------------------------------------------------------------------


@pytest.mark.parametrize("window", [None, 5])
def test_offsets_of_causal_masks(window: int | None) -> None:
    mask = torch.cat([_mask(2, 8, 40, [0, 12], window), torch.zeros(2, 1, 8, 9, dtype=torch.bool)], -1)
    assert causal_offsets(mask, 40, window).tolist() == [0, 12]


def test_concurrent_decoders_keep_their_own_offsets() -> None:
    """Async SSD decodes in two threads at once: neither may read the offsets of the other's mask."""
    masks = [_mask(1, 4, 40, [start], None) for start in (3, 17)]
    barrier = threading.Barrier(2, timeout=30)
    seen: dict[int, list[list[int]]] = {0: [], 1: []}

    def decode(index: int) -> None:
        for _ in range(200):
            barrier.wait()
            offsets = flash._offsets(masks[index], 1, 40, None)
            assert offsets is not None
            seen[index].append(offsets.tolist())

    threads = [threading.Thread(target=decode, args=(index,)) for index in (0, 1)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert seen == {0: [[3]] * 200, 1: [[17]] * 200}


def test_rejects_masks_flash_cannot_express() -> None:
    mask = _mask(2, 8, 40, [3, 12], None)
    padded = mask.clone()
    padded[0, :, :, :2] = False  # left padding
    assert causal_offsets(padded, 40, None) is None
    tree = mask.clone()
    tree[1, 0, 5, 16] = False  # a draft tree hides a sibling
    assert causal_offsets(tree, 40, None) is None
    assert causal_offsets(mask, 40, 6) is None  # a full causal mask is no window
    assert causal_offsets(_mask(1, 8, 40, [35], None), 40, None) is None  # queries past the buffer


@pytest.mark.parametrize(
    ("major", "build", "fa4", "dim", "window", "softcap", "expected"),
    [
        (9, 3, True, 128, None, None, (3, "")),
        (9, 3, True, 256, 1024, None, (4, "")),  # FA3's local-window kernel is wrong: windows go to FA4
        (9, 3, False, 256, 1024, None, (None, "flash_window")),
        (8, 2, False, 256, 1024, None, (2, "")),
        (9, 3, True, 512, None, None, (4, "")),  # head dim 512 only on FA4
        (9, 3, True, 512, None, 30.0, (None, "flash_head_dim")),  # FA4 has no soft cap
        (8, 2, False, 512, None, None, (None, "flash_head_dim")),
    ],
)
def test_prefill_version(
    major: int,
    build: int,
    fa4: bool,
    dim: int,
    window: int | None,
    softcap: float | None,
    expected: tuple[int | None, str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(flash, "flash_version", lambda index: build)
    monkeypatch.setattr(flash, "_fa4_supported", lambda: fa4)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda index=None: (major, 0))
    assert flash._prefill_version(0, dim, window, softcap) == expected


def test_refuses_without_cuda() -> None:
    q = torch.randn(1, 4, 80, 16)
    out, reason = flash_prefill(q, q[:, :2], q[:, :2], _mask(1, 80, 80, [0], None), 0.25)
    assert out is None and reason == "cpu"


# ---- kernels ------------------------------------------------------------------------------------------------------


def _has_flash() -> bool:
    return torch.cuda.is_available() and flash_version(torch.cuda.current_device()) is not None


requires_flash = pytest.mark.skipif(not _has_flash(), reason="needs vLLM's FlashAttention build")


def _has_fa4() -> bool:
    return _has_flash() and torch.cuda.get_device_capability()[0] in (9, 10) and flash._fa4_supported()


def _sdpa(query, key, value, mask, scale, softcap=None):
    if softcap is None:
        groups = query.shape[1] // key.shape[1]
        out = torch.nn.functional.scaled_dot_product_attention(
            query.float(), key.float().repeat_interleave(groups, 1), value.float().repeat_interleave(groups, 1),
            attn_mask=mask[..., : key.shape[2]], scale=scale,
        )  # fmt: skip
        return out.transpose(1, 2)
    from economize.attention import reference_attention

    return reference_attention(query.float(), key.float(), value.float(), mask, scale, softcap=softcap)


@pytest.mark.gpu
@requires_flash
@pytest.mark.parametrize(
    ("dim", "kv_heads", "window", "softcap", "starts"),
    [
        (128, 2, None, None, [0]),
        (64, 4, None, None, [0, 256, 700]),
        (256, 2, 300, None, [0, 400]),
        pytest.param(512, 2, None, None, [0, 300], marks=pytest.mark.skipif("not _has_fa4()")),
        pytest.param(512, 4, 300, None, [17, 0], marks=pytest.mark.skipif("not _has_fa4()")),
        (128, 8, None, 30.0, [100, 0]),
    ],
)
def test_matches_sdpa(
    dim: int,
    kv_heads: int,
    window: int | None,
    softcap: float | None,
    starts: list[int],
) -> None:
    gen = torch.Generator(device="cuda").manual_seed(0)
    batch, q_heads, queries, keys = len(starts), 8, 300, 1024
    query = torch.randn(batch, q_heads, queries, dim, device="cuda", dtype=torch.bfloat16, generator=gen)
    key = torch.randn(batch, kv_heads, keys, dim, device="cuda", dtype=torch.bfloat16, generator=gen)
    value = torch.randn(batch, kv_heads, keys, dim, device="cuda", dtype=torch.bfloat16, generator=gen)
    mask = _mask(batch, queries, keys, starts, window).cuda()
    scale = dim**-0.5
    out, path = flash_prefill(query, key, value, mask, scale, softcap=softcap, window=window)
    assert path.startswith("flash") and out.shape == (batch, queries, q_heads, dim)
    expected = _sdpa(query, key, value, mask, scale, softcap)
    torch.testing.assert_close(out.float(), expected, atol=2e-2, rtol=2e-2)


@pytest.mark.gpu
@requires_flash
def test_forward_logs_flash_and_falls_back_per_call() -> None:
    gen = torch.Generator(device="cuda").manual_seed(1)
    query = torch.randn(2, 8, 96, 64, device="cuda", dtype=torch.bfloat16, generator=gen)
    key = torch.randn(2, 2, 512, 64, device="cuda", dtype=torch.bfloat16, generator=gen)
    value = torch.randn_like(key)
    module = torch.nn.Module()
    module.layer_idx = 3
    module.num_key_value_groups = 4  # read by the sdpa fallback to repeat the KV heads
    causal = _mask(2, 96, 512, [0, 40], None).cuda()
    padded = causal.clone()
    padded[0, :, :, :3] = False
    interface.kernel_usage(reset=True)
    before = interface.kernel_calls()
    out, _ = interface.economize_attention_forward(module, query, key, value, causal, scaling=0.125)
    ref, _ = interface.economize_attention_forward(module, query, key, value, padded, scaling=0.125)
    after = interface.kernel_calls()
    assert after["flash"] == before["flash"] + 1 and after["fallback"] == before["fallback"] + 1
    paths = {(r["path"], r["reason"]) for r in interface.kernel_usage(reset=True)}
    assert paths == {(f"flash{flash_version(torch.cuda.current_device())}", ""), ("sdpa", "long_query")}
    expected = _sdpa(query, key, value, causal, 0.125)
    torch.testing.assert_close(out.float(), expected, atol=2e-2, rtol=2e-2)
    assert out.shape == ref.shape


@pytest.mark.gpu
@requires_flash
@pytest.mark.parametrize("fp8", [torch.float8_e4m3fn, torch.float8_e5m2])
def test_forward_runs_fp8_prefill_on_flash(fp8: torch.dtype) -> None:
    # kv_fp8_fused hands its fp8 buffers to attention; a prefill must not fall back to sdpa over the whole cache
    gen = torch.Generator(device="cuda").manual_seed(2)
    query = torch.randn(1, 8, 96, 64, device="cuda", dtype=torch.bfloat16, generator=gen)
    key = torch.randn(1, 2, 512, 64, device="cuda", dtype=torch.bfloat16, generator=gen).to(fp8)
    value = torch.randn(1, 2, 512, 64, device="cuda", dtype=torch.bfloat16, generator=gen).to(fp8)
    module = torch.nn.Module()
    module.layer_idx = 0
    mask = _mask(1, 96, 512, [0], None).cuda()
    interface.kernel_usage(reset=True)
    out, _ = interface.economize_attention_forward(module, query, key, value, mask, scaling=0.125)
    paths = {(r["path"], r["reason"]) for r in interface.kernel_usage(reset=True)}
    assert paths == {(f"flash{flash_version(torch.cuda.current_device())}", "")}
    expected = _sdpa(query, key.to(torch.bfloat16), value.to(torch.bfloat16), mask, 0.125)
    torch.testing.assert_close(out.float(), expected, atol=2e-2, rtol=2e-2)


class _Packed:
    """A packed attachment as KIVI's: the dense buffers hold the values it encodes."""

    format = "test"


class _Sparse(_Packed):
    """A sparse attachment as Quest's: it hides the first 64 keys."""

    def restrict_mask(self, query, key, mask, is_causal=False):
        restricted = mask.clone()
        restricted[..., :64] = False
        return restricted


@pytest.mark.gpu
@requires_flash
def test_forward_runs_packed_prefill_on_flash_and_applies_restrictions() -> None:
    # a prefill chunk over a KIVI cache must not fall back to sdpa; a sparse cache's restriction must still apply
    from economize.attention.packed import attach_packed

    gen = torch.Generator(device="cuda").manual_seed(3)
    query = torch.randn(1, 8, 96, 64, device="cuda", dtype=torch.bfloat16, generator=gen)
    key = torch.randn(1, 2, 512, 64, device="cuda", dtype=torch.bfloat16, generator=gen)
    value = torch.randn_like(key)
    module = torch.nn.Module()
    module.layer_idx = 0
    module.num_key_value_groups = 4
    mask = _mask(1, 96, 512, [0], None).cuda()
    version = flash_version(torch.cuda.current_device())
    interface.kernel_usage(reset=True)
    packed = attach_packed(key, _Packed())
    out, _ = interface.economize_attention_forward(module, query, packed, value, mask, scaling=0.125)
    assert {(r["path"], r["reason"]) for r in interface.kernel_usage(reset=True)} == {(f"flash{version}", "")}
    torch.testing.assert_close(out.float(), _sdpa(query, key, value, mask, 0.125), atol=2e-2, rtol=2e-2)
    packed = attach_packed(key, _Sparse())
    out, _ = interface.economize_attention_forward(module, query, packed, value, mask, scaling=0.125)
    assert {r["path"] for r in interface.kernel_usage(reset=True)} == {"sdpa"}
    restricted = mask.clone()
    restricted[..., :64] = False
    torch.testing.assert_close(out.float(), _sdpa(query, key, value, restricted, 0.125), atol=2e-2, rtol=2e-2)
