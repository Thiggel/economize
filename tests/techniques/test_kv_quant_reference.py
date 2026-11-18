"""KIVI against a port of the reference schedule (``LlamaAttention_KIVI`` of https://github.com/jy-yuan/KIVI).

The reference keeps keys in full precision until a block of ``R`` (``residual_length``) new keys is complete and
then quantizes the block per channel in groups of ``group_size`` tokens; it keeps the last ``R`` values in full
precision and quantizes each older value per token in groups of ``group_size`` channels. The prompt is quantized
after its attention (all keys but the last ``T % R``, all values but the last ``R``), and every quantization of a
decode step happens after that step's attention. The quantizer is ``triton_quantize_and_pack_along_last_dim``:
``codes = round(clamp((x - min) / scale, 0, 2^bits - 1))`` with ``scale = (max - min) / (2^bits - 1)``, and the
scale and minimum are stored in the model dtype, fp16 in the paper (here rounded to fp16 on the fp32 model).

The port fake-quantizes the entries of an HF :class:`~transformers.DynamicCache` in place on that schedule and
decodes teacher-forced through the HF model, independently of :mod:`economize.techniques.kv_quant`. The two
compute the keys and values with different kernels, so an entry within float32 noise of a rounding boundary may
take the neighbouring code; at 2 bits such a flip moves the logits by up to about ``1e-3``. Quantizing one step
early or late, or with a different grouping, moves them by more than ``0.5`` on this model, and the quantization
itself by more than ``1``.
"""

from __future__ import annotations

import pytest
import torch
from transformers import DynamicCache

import economize
from economize.runtime import phase
from economize.techniques.kv_quant import KIVIConfig
from helpers import VOCAB, make_tiny_model

PREFIX, TOTAL = 12, 44


def _ref_quantize_last_dim(data: torch.Tensor, group_size: int, bits: int) -> torch.Tensor:
    """``triton_quantize_and_pack_along_last_dim`` followed by its dequantization."""
    shape = data.shape
    data = data.reshape(*shape[:-1], shape[-1] // group_size, group_size)
    mn = data.amin(-1, keepdim=True).half().float()
    mx = data.amax(-1, keepdim=True)
    max_int = 2**bits - 1
    scale = ((mx - data.amin(-1, keepdim=True)) / max_int).half().float()
    safe = torch.where(scale > 0, scale, torch.ones_like(scale))
    codes = ((data - mn) / safe).clamp(0, max_int).round()
    return (codes * scale + mn).reshape(shape)


class _RefKIVI:
    """The quantization schedule of one layer: how many keys and values are quantized so far."""

    def __init__(self, config: KIVIConfig) -> None:
        self.config = config
        self.keys = 0
        self.values = 0

    def quantize(self, layer: object, committed: int) -> None:
        """Quantize what the reference has quantized once ``committed`` entries went through attention."""
        R, g, bits = self.config.residual_length, self.config.group_size, self.config.bits
        keys, values = layer.keys, layer.values  # [1, H, T, D], post-RoPE keys
        key_end = committed // R * R
        if key_end > self.keys:  # complete blocks of R keys, per channel in groups of g tokens
            block = keys[:, :, self.keys : key_end].transpose(2, 3)
            keys[:, :, self.keys : key_end] = _ref_quantize_last_dim(block, g, bits).transpose(2, 3)
            self.keys = key_end
        value_end = max(committed - R, 0)
        if value_end > self.values:  # values older than the last R, per token in groups of g channels
            values[:, :, self.values : value_end] = _ref_quantize_last_dim(
                values[:, :, self.values : value_end], g, bits
            )
            self.values = value_end


@torch.no_grad()
def _ref_kivi_logits(base: torch.nn.Module, seq: torch.Tensor, config: KIVIConfig) -> torch.Tensor:
    cache = DynamicCache(config=base.config)
    schedules = [_RefKIVI(config) for _ in range(base.config.num_hidden_layers)]
    out = [base(seq[:, :PREFIX], past_key_values=cache).logits[0]]
    for t in range(PREFIX, seq.shape[1]):
        for layer, schedule in zip(cache.layers, schedules, strict=True):
            schedule.quantize(layer, t)
        out.append(base(seq[:, t : t + 1], past_key_values=cache).logits[0])
    return torch.cat(out).float()


def _teacher_forced(model: economize.EfficientModel, seq: torch.Tensor) -> torch.Tensor:
    runner = model.runner
    cache = model.techniques["kivi"].wrap_cache(runner.new_cache(64))
    ctx = runner.context(cache, torch.arange(PREFIX)[None], trusted=True)
    with phase("prefill"):
        out = [runner.logits(runner.forward(seq[:, :PREFIX], ctx).final)[0]]
    for t in range(PREFIX, seq.shape[1]):
        ctx = runner.context(cache, torch.tensor([[t]]), trusted=True)
        with phase("decode"):
            out.append(runner.logits(runner.forward(seq[:, t : t + 1], ctx).final)[0])
    return torch.cat(out).float()


@pytest.mark.parametrize("storage", ["simulated", "packed"])
@pytest.mark.parametrize(("bits", "group_size", "residual_length"), [(2, 4, 8), (4, 8, 8), (2, 4, 4)])
def test_kivi_matches_the_reference_schedule(storage: str, bits: int, group_size: int, residual_length: int) -> None:
    seq = torch.randint(0, VOCAB, (1, TOTAL), generator=torch.Generator().manual_seed(0))
    config = KIVIConfig(bits=bits, group_size=group_size, residual_length=residual_length, storage=storage)
    base = make_tiny_model(attn="eager")
    expected = _ref_kivi_logits(base, seq, config)
    dense = base(seq).logits[0].float()
    logits = _teacher_forced(economize.apply(make_tiny_model(attn="eager"), config), seq)
    assert (expected - dense).abs().max() > 1.0  # the quantization changes the logits
    torch.testing.assert_close(logits, expected, rtol=0, atol=5e-3)
