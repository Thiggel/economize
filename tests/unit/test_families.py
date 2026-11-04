"""Every supported family: the layer runner reproduces the model's own logits.

Each case builds a tiny random float32 model and runs a ragged batch of three sequences through
:class:`~economize.runtime.LayerRunner`: a left-padded prefill longer than the sliding window, then right-padded chunks
of different lengths per row (rows with no tokens included), then single tokens. Logits of every real token must
match those of the Hugging Face model run on each unpadded sequence alone. Hybrid models restore their recurrent
states after each padded chunk and replay the real tokens, as the speculative loop does.
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

from economize.arch import adapter_for
from economize.errors import UnsupportedModelError
from economize.runtime import HybridCache, LayerRunner
from helpers import FAMILY_MODELS, VOCAB, WINDOW, family_skip_reason, make_family_model

PROMPTS = (11, 4, 8)
#: Tokens per row of each incremental pass after the prefill.
CHUNKS = ((3, 1, 2), (1, 1, 1), (0, 4, 2), (2, 2, 0), (1, 1, 1), (1, 1, 1))
ATOL = 1e-4


def _sequences() -> list[torch.Tensor]:
    generator = torch.Generator().manual_seed(0)
    totals = [p + sum(chunk[b] for chunk in CHUNKS) for b, p in enumerate(PROMPTS)]
    assert min(totals) > 2 * WINDOW
    return [torch.randint(3, VOCAB, (n,), generator=generator) for n in totals]


def _ragged(rows: list[torch.Tensor], starts: list[int], left: bool) -> tuple[torch.Tensor, ...]:
    width = max(len(r) for r in rows)
    tokens = torch.zeros(len(rows), width, dtype=torch.long)
    positions = torch.zeros_like(tokens)
    valid = torch.zeros_like(tokens, dtype=torch.bool)
    for b, (row, start) in enumerate(zip(rows, starts)):
        cols = slice(width - len(row), width) if left else slice(0, len(row))
        tokens[b, cols] = row
        positions[b, cols] = torch.arange(start, start + len(row))
        valid[b, cols] = True
    return tokens, positions, valid


@torch.no_grad()
def check_exact_path(model: nn.Module) -> None:
    """Assert that runner logits match the model's for every real token of a ragged incremental schedule."""
    runner = LayerRunner(adapter_for(model))
    sequences = _sequences()
    reference = [model(input_ids=seq[None]).logits[0] for seq in sequences]
    cache = runner.new_cache(capacity=48, batch_size=len(sequences))
    done = [0] * len(sequences)
    for step, sizes in enumerate((PROMPTS, *CHUNKS)):
        rows = [seq[d : d + n] for seq, d, n in zip(sequences, done, sizes)]
        tokens, positions, valid = _ragged(rows, done, left=step == 0)
        snapshot = cache.snapshot() if isinstance(cache, HybridCache) else None
        ctx = runner.context(cache, positions, None if bool(valid.all()) else valid)
        logits = runner.logits(runner.forward(tokens, ctx).final)
        for b, (d, n) in enumerate(zip(done, sizes)):
            got = logits[b][valid[b]]
            torch.testing.assert_close(
                got, reference[b][d : d + n], atol=ATOL, rtol=0,
                msg=lambda m, step=step, b=b: f"pass {step}, row {b}: {m}",
            )
        if snapshot is not None and step > 0:
            start = torch.tensor(done)
            runner.replay(cache, snapshot, tokens, start, torch.tensor(sizes))
        done = [d + n for d, n in zip(done, sizes)]


@pytest.mark.parametrize("family", sorted(FAMILY_MODELS))
def test_runner_matches_model(family: str) -> None:
    reason = family_skip_reason(family)
    if reason is not None:
        pytest.skip(reason)
    check_exact_path(make_family_model(family))


@pytest.mark.parametrize("model_type", ["mamba", "rwkv", "jamba", "falcon_h1"])
def test_state_space_models_are_rejected(model_type: str) -> None:
    from transformers import AutoConfig, AutoModelForCausalLM

    config = AutoConfig.for_model(
        model_type, vocab_size=VOCAB, hidden_size=32, num_hidden_layers=2, num_attention_heads=2, intermediate_size=64
    )
    with pytest.raises(UnsupportedModelError, match=model_type):
        adapter_for(AutoModelForCausalLM.from_config(config))


@pytest.mark.parametrize("family", sorted(FAMILY_MODELS))
def test_removed_layers_leave_a_consistent_model(family: str) -> None:
    """After :meth:`ArchAdapter.remove_layers` the Hugging Face model and the runner agree on the shallower model."""
    reason = family_skip_reason(family)
    if reason is not None:
        pytest.skip(reason)
    model = make_family_model(family)
    adapter = adapter_for(model)
    count = adapter.num_layers
    if any(adapter.kv_source(i) != i for i in range(1, count + 1)):
        with pytest.raises(UnsupportedModelError, match="shares K/V"):
            adapter.remove_layers([2])
        assert adapter.num_layers == count
        return
    with pytest.raises(ValueError, match="cannot remove"):
        adapter.remove_layers(range(1, count + 1))
    with pytest.raises(ValueError, match="cannot remove"):
        adapter.remove_layers([count + 1])
    adapter.remove_layers([2])
    assert adapter.num_layers == adapter_for(model).num_layers == count - 1
    assert adapter.config.num_hidden_layers == count - 1
    check_exact_path(model)


@pytest.mark.parametrize("family", sorted(FAMILY_MODELS))
def test_head_geometry_matches_attention_modules(family: str) -> None:
    """The adapter's head width and key/value heads are those of every attention layer that projects its own K/V
    (Gemma-4's full-attention layers are wider than its sliding ones)."""
    reason = family_skip_reason(family)
    if reason is not None:
        pytest.skip(reason)
    adapter = adapter_for(make_family_model(family))
    checked = 0
    for index in range(1, adapter.num_layers + 1):
        attention = getattr(adapter.layer(index), "self_attn", None)
        kv = getattr(attention, "k_proj", None) or getattr(attention, "v_proj", None)
        if not isinstance(getattr(attention, "head_dim", None), int) or not isinstance(kv, nn.Linear):
            continue
        geometry = adapter.head_geometry(index)
        assert geometry.head_dim == attention.head_dim, index
        assert geometry.kv_heads * geometry.head_dim == kv.out_features, index
        checked += 1
    if family.startswith("gemma4"):
        assert checked > 0  # fused (phi3) and latent (deepseek_v3) projections have nothing to compare against


class _HeterogeneousConfig:
    """Stands in for a transformers >= 5.17 heterogeneous config: per-layer attributes refuse plain reads."""

    is_heterogeneous = True
    _globals = {"head_dim": 256, "num_key_value_heads": 16}

    def __getattr__(self, name: str) -> object:
        if name in self._globals:
            raise RuntimeError(f"'{name}' is a per-layer attribute")
        raise AttributeError(name)

    def _getattr_without_heterogeneous_validation(self, name: str, default: object = None) -> object:
        return self._globals.get(name, default)


def test_config_value_reads_the_global_value_of_heterogeneous_configs() -> None:
    from economize.arch import config_value

    config = _HeterogeneousConfig()
    assert config_value(config, "num_key_value_heads") == 16
    assert config_value(config, "head_dim") == 256
    assert config_value(config, "missing", 3) == 3
    plain = nn.Module()
    plain.head_dim = 64
    assert config_value(plain, "head_dim") == 64
    assert config_value(plain, "num_key_value_heads") is None
