"""Band masks attended by their structure (``economize.attention.banded``).

A :class:`BandMask` must give every query the keys of its dense form: causal, a sliding window, or sinks plus a
recent window (DuoAttention's streaming heads). The checks compare the blocked attention with one dense-masked
``sdpa`` call, in the outputs and in the gradients, across lengths that do not divide into blocks, windows longer
than the pass, sinks inside and outside the window, and grouped K/V heads.
"""

from __future__ import annotations

from typing import Any

import pytest
import torch
import torch.nn.functional as F

import economize
from economize.arch.spec import AttentionSpec
from economize.attention import ATTENTION_KERNELS, BandMask, StructuredMask, accepts_structured_masks, interface
from economize.attention.sdpa_guard import sdpa_attention_forward
from economize.errors import ConfigError
from economize.runtime import LayerContext
from economize.runtime.mask import STRUCTURED_MASKS, visibility
from economize.techniques.duo_attention.gates import SinglePassCache, streaming_visibility
from helpers import make_family_model, make_tiny_model


class _Attention(torch.nn.Module):
    def __init__(self, groups: int) -> None:
        super().__init__()
        self.num_key_value_groups = groups
        self.is_causal = True
        self.layer_idx = 0


def _inputs(length: int, heads: int = 4, kv_heads: int = 2, dim: int = 16) -> tuple[torch.Tensor, ...]:
    g = torch.Generator().manual_seed(length)
    q = torch.randn(2, heads, length, dim, generator=g, dtype=torch.float64)
    k = torch.randn(2, kv_heads, length, dim, generator=g, dtype=torch.float64)
    v = torch.randn(2, kv_heads, length, dim, generator=g, dtype=torch.float64)
    return q, k, v


def _dense(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    groups = q.shape[1] // k.shape[1]
    k, v = (x.repeat_interleave(groups, 1) for x in (k, v))
    return F.scaled_dot_product_attention(q, k, v, attn_mask=mask).transpose(1, 2)


BANDS = [
    BandMask(37),  # causal
    BandMask(37, window=64),  # a window longer than the pass is causal
    BandMask(100, window=7),
    BandMask(100, window=7, block=5),  # blocks shorter than the window
    BandMask(100, window=16, block=32),
    BandMask(130, window=16, sinks=4),  # the streaming heads
    BandMask(130, window=16, sinks=20, block=8),  # sinks reaching past the first window
    BandMask(3, window=2, sinks=1),
    BandMask(1, window=1, sinks=3),
]


@pytest.mark.parametrize("band", BANDS, ids=repr)
def test_band_matches_its_dense_mask(band: BandMask) -> None:
    q, k, v = _inputs(band.length)
    dense = band.dense(q.device)
    positions = torch.arange(band.length)
    rule = streaming_visibility(positions[None], positions, band.sinks, band.window or band.length)
    assert torch.equal(dense[0, 0], rule[0])
    expected = _dense(q, k, v, dense)
    out, _ = band.attend(sdpa_attention_forward, _Attention(2), q, k, v, dropout=0.0, scaling=None)
    torch.testing.assert_close(out, expected, rtol=1e-12, atol=1e-12)


def test_band_gradients_match_the_dense_mask() -> None:
    band = BandMask(90, window=12, sinks=5, block=16)
    q, k, v = (x.requires_grad_() for x in _inputs(band.length))
    weights = torch.randn(2, band.length, 4, 16, dtype=torch.float64, generator=torch.Generator().manual_seed(1))
    out, _ = band.attend(sdpa_attention_forward, _Attention(2), q, k, v, dropout=0.0, scaling=None)
    blocked = torch.autograd.grad((out * weights).sum(), (q, k, v))
    dense = torch.autograd.grad((_dense(q, k, v, band.dense(q.device)) * weights).sum(), (q, k, v))
    for a, b in zip(blocked, dense, strict=True):
        torch.testing.assert_close(a, b, rtol=1e-12, atol=1e-12)


def test_band_validation_and_shape_check() -> None:
    for bad in ({"length": 0}, {"length": 4, "window": 0}, {"length": 4, "sinks": -1}, {"length": 4, "block": 0}):
        with pytest.raises(ConfigError):
            BandMask(**bad)
    q, k, v = _inputs(10)
    with pytest.raises(ConfigError, match="10 queries"):
        BandMask(12, window=4).attend(sdpa_attention_forward, _Attention(2), q, k, v)
    assert BandMask(8).causal and BandMask(8, window=8).causal and not BandMask(8, window=7).causal
    assert BandMask(8, window=1000).block_size == 512 and BandMask(8, window=4).block_size == 128


# ---- dispatch -------------------------------------------------------------------------------------------------------
def test_sdpa_guard_and_economize_kernel_take_band_masks() -> None:
    band = BandMask(70, window=9, sinks=3)
    q, k, v = (x.float() for x in _inputs(band.length))
    expected = _dense(q, k, v, band.dense(q.device))
    module = _Attention(2)
    guarded, _ = sdpa_attention_forward(module, q, k, v, band, 0.0, None)  # transformers passes dropout positionally
    torch.testing.assert_close(guarded, expected)
    interface.kernel_usage(reset=True)
    kernel, _ = interface.economize_attention_forward(module, q, k, v, band, dropout=0.0, scaling=None)
    torch.testing.assert_close(kernel, expected)
    assert interface.kernel_usage(reset=True) == [
        {"layer": 0, "queries": band.length, "path": "sdpa", "reason": "structured", "calls": 1}
    ]


def test_which_implementations_accept_structured_masks() -> None:
    economize.apply(make_tiny_model())  # installs economize's sdpa function
    assert accepts_structured_masks("sdpa")
    assert accepts_structured_masks("economize") and ATTENTION_KERNELS.get("economize").structured_masks
    assert not accepts_structured_masks("eager")
    assert not accepts_structured_masks("flash_attention_2")


class _Shifted(StructuredMask):
    """A layout added by extension: attends causally and records the call."""

    calls = 0

    def attend(self, forward: Any, module: torch.nn.Module, query: torch.Tensor, key: torch.Tensor,
               value: torch.Tensor, **kwargs: Any) -> tuple[torch.Tensor, None]:
        type(self).calls += 1
        kwargs.pop("is_causal", None)
        out, _ = forward(module, query, key, value, None, is_causal=True, **kwargs)
        return out, None

    def dense(self, device: torch.device) -> torch.Tensor:
        raise NotImplementedError


def test_new_layouts_plug_in_by_subclassing() -> None:
    q, k, v = (x.float() for x in _inputs(20))
    out, _ = sdpa_attention_forward(_Attention(2), q, k, v, _Shifted(), 0.0, None)
    torch.testing.assert_close(out, _dense(q, k, v, BandMask(20).dense(q.device)))
    assert _Shifted.calls == 1


# ---- the runtime ----------------------------------------------------------------------------------------------------
def test_whole_sequence_passes_get_band_masks() -> None:
    model = economize.apply(make_family_model("gemma4_text"))
    adapter = model.adapter
    length = 40
    cache = SinglePassCache(length, 1)
    positions = torch.arange(length)
    whole = LayerContext.chain(adapter, cache, positions, trusted=True, whole_sequences=True)
    plain = LayerContext.chain(adapter, cache, positions, trusted=True)
    seen = set()
    for index in range(1, adapter.num_layers + 1):
        spec = adapter.attention_spec(index)
        mask = whole.mask(spec, torch.device("cpu"))
        assert isinstance(mask, BandMask)
        seen.add(spec.kind)
        dense = plain.mask(spec, torch.device("cpu"))
        assert isinstance(dense, torch.Tensor) and torch.equal(mask.dense(torch.device("cpu"))[0, 0], dense[0, 0])
    assert seen == {"full", "sliding"}
    with pytest.raises(ConfigError, match="padding"):
        LayerContext.chain(adapter, cache, positions, torch.ones(1, length, dtype=torch.bool), whole_sequences=True)


def test_kinds_without_a_structured_rule_keep_dense_masks() -> None:
    spec = AttentionSpec("chunked", window=8)
    assert "chunked" not in STRUCTURED_MASKS
    model = economize.apply(make_tiny_model())
    ctx = LayerContext.chain(model.adapter, SinglePassCache(16, 1), torch.arange(16), whole_sequences=True)
    mask = ctx.mask(spec, torch.device("cpu"))
    assert isinstance(mask, torch.Tensor)
    positions = torch.arange(16)
    assert torch.equal(mask[0, 0], visibility(spec, positions[None], positions)[0])
